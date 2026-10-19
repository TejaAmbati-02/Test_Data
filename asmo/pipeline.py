import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from .config import Config
from .auth import AuthClient
from .products import ProductsClient, Product
from .prices import PricesClient, Price
from .exporter import Exporter, ExportPaths
from .transformer import transform
from .images import ImageDownloader
from .storage import ObjectStorageClient, Folder
from .logging_service import logger, log_function


@dataclass
class PipelineResult:
    """Holds the combined output of a pipeline run."""

    products: list[Product]
    prices: list[Price]
    export_paths: ExportPaths | None = field(default=None, repr=False)

    def summary(self) -> str:
        return f"Products: {len(self.products)}, Prices: {len(self.prices)}"


class Pipeline:
    """Top-level orchestrator — the only class callers need to import."""

    def __init__(self, config: Config | None = None, output_dir: Path = Path(".")) -> None:
        self._config = config or Config()
        self._auth = AuthClient(self._config)
        self._products = ProductsClient(self._config, self._auth)
        self._prices = PricesClient(self._config, self._auth)
        self._exporter = Exporter(base_dir=output_dir)
        self._images = ImageDownloader.instance(base_dir=output_dir)
        self._storage = ObjectStorageClient.instance()

    @log_function
    async def run(self) -> PipelineResult:
        """Authenticate, fetch products + prices concurrently, transform, then export."""
        logger.info("Step 1/5 — authenticating", extra={"function_name": "run"})
        await self._auth.authenticate()

        logger.info("Step 2/5 — fetching products and prices concurrently", extra={"function_name": "run"})
        products, prices = await asyncio.gather(
            self._products.get_all(),
            self._prices.get_all(),
        )
        logger.info(
            f"Fetch complete — {len(products)} products, {len(prices)} prices",
            extra={"function_name": "run", "input_data": {"products": len(products), "prices": len(prices)}},
        )

        logger.info("Step 3/5 — transforming", extra={"function_name": "run"})
        transformed = await transform(products, prices, self._config.channel_id)

        # Per-SKU storage callbacks — only wired up when uploads are enabled.
        # When enabled, each SKU's JSON and image are uploaded to object storage
        # in parallel, as they are produced, inside the per-SKU pipeline below.
        upload_json = self._make_upload_json() if self._config.upload_enabled else None
        upload_image = self._make_upload_image() if self._config.upload_enabled else None

        # Image embeddings for the nested images.image_vector kNN field are only
        # needed when indexing. One embedder is shared by the image step (which
        # embeds each SKU's downloaded images) and the indexer (which embeds the
        # text content), so a single OCI client is reused. The image step runs to
        # completion before indexing, so their embed calls never overlap.
        embedder = None
        embed_image = None
        if self._config.index_enabled:
            from .embeddings import OCIEmbedder

            embedder = OCIEmbedder()

            async def embed_image(images: list[tuple[bytes, str]]) -> list[list[float]]:
                return await embedder.embed_images(images)

        # Local disk writes are opt-in (SAVE_LOCAL). When off, nothing is written
        # to out_products/out_prices/out_transformed/out_images — data streams
        # straight to storage/index, so a full disk can't break the run.
        save_local = self._config.save_local
        write_json = self._exporter.write_transformed_row if save_local else None

        logger.info(
            "Step 4/5 — per SKU, "
            + ("writing transformed JSON locally and " if save_local else "")
            + ("downloading each image and uploading JSON + image to object storage"
               if self._config.upload_enabled
               else "downloading each image")
            + (" (also saving raw dumps + images locally)" if save_local else " (no local files — SAVE_LOCAL is off)"),
            extra={"function_name": "run", "input_data": {"upload_enabled": self._config.upload_enabled, "save_local": save_local}},
        )

        async def _export_and_images():
            images_coro = self._images.download_all(
                transformed,
                write_json=write_json,
                upload_json=upload_json,
                upload_image=upload_image,
                embed_image=embed_image,
                write_images=save_local,
                write_image_content_local=save_local and self._config.image_content_local,
            )
            if save_local:
                return await asyncio.gather(
                    self._exporter.export_raw(products, prices),
                    images_coro,
                )
            # No local raw dumps — only run the image/upload work.
            return None, await images_coro

        if self._config.upload_enabled:
            # Open the shared S3 client once for the whole run and make sure the
            # target folders exist before the per-SKU uploads start.
            async with self._storage:
                await asyncio.gather(
                    self._storage.ensure_folder(Folder.PRODUCTS),
                    self._storage.ensure_folder(Folder.IMAGES),
                )
                paths, image_summary = await _export_and_images()
        else:
            paths, image_summary = await _export_and_images()

        logger.info(
            f"Export + storage — images: {image_summary['downloaded']} downloaded, "
            f"{image_summary['skipped']} skipped, {image_summary['failed']} failed; "
            f"storage: {image_summary['json_uploaded']} JSON + "
            f"{image_summary['image_uploaded']} images uploaded, "
            f"{image_summary['upload_failed']} upload failures; "
            f"local writes: {image_summary['json_write_failed']} JSON + "
            f"{image_summary['image_save_failed']} image write failures",
            extra={"function_name": "run", "status": "success", "input_data": image_summary},
        )

        if self._config.index_enabled:
            logger.info(
                "Step 5/5 — embedding content and indexing transformed rows into OpenSearch",
                extra={"function_name": "run"},
            )
            # Imported lazily so the package imports without OCI/OpenSearch installed.
            from .search_index import OpenSearchIndexer

            indexer = OpenSearchIndexer(embedder=embedder)
            try:
                index_summary = await indexer.index_rows(transformed)
                logger.info(
                    f"Indexing complete — {index_summary['indexed']} indexed, {index_summary['failed']} failed",
                    extra={"function_name": "run", "status": "success", "input_data": index_summary},
                )
            finally:
                await indexer.close()
        else:
            logger.info("Step 5/5 — indexing disabled (INDEX_ENABLED=false)", extra={"function_name": "run"})

        logger.info(
            f"Pipeline run complete — {len(products)} products, {len(prices)} prices, {len(transformed)} rows exported",
            extra={"function_name": "run", "status": "success"},
        )
        return PipelineResult(products=products, prices=prices, export_paths=paths)

    def _make_upload_json(self):
        """Build the async callback that uploads one SKU's JSON to storage.

        The returned callback serialises the transformed row to JSON (matching
        the local file's formatting and ``{sku-without-dashes}.json`` naming)
        and uploads it under :data:`Folder.PRODUCTS`.
        """
        async def upload_json(sku: str, row: dict) -> None:
            data = json.dumps(row, indent=2, ensure_ascii=False).encode("utf-8")
            await self._storage.upload_bytes(
                folder=Folder.PRODUCTS,
                filename=f"{sku.replace('-', '')}.json",
                data=data,
                content_type="application/json",
            )
        return upload_json

    def _make_upload_image(self):
        """Build the async callback that uploads one SKU's image to storage.

        The returned callback uploads the already-downloaded image bytes under
        :data:`Folder.IMAGES`, inferring the content type from the filename's
        extension.
        """
        content_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
            ".svg": "image/svg+xml", ".tiff": "image/tiff",
        }

        async def upload_image(sku: str, filename: str, data: bytes) -> None:
            content_type = content_types.get(Path(filename).suffix.lower(), "application/octet-stream")
            await self._storage.upload_bytes(
                folder=Folder.IMAGES,
                filename=filename,
                data=data,
                content_type=content_type,
            )
        return upload_image
