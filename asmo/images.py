"""Concurrent product-image downloading for the ASMO Data pipeline.

This module exposes :class:`ImageDownloader`, which takes the transformed SKU
rows produced upstream and downloads each row's ``imageUrl`` concurrently,
writing one image file per SKU into an output directory. Downloads are bounded
by a semaphore, identical URLs are fetched only once and shared, and failures
are tolerated and tallied rather than aborting the run.

Network calls use :mod:`aiohttp` with the project's shared SSL context and are
logged through the structured logger.
"""

import asyncio
import base64
from pathlib import Path
from typing import Awaitable, Callable, Optional
from urllib.parse import urlparse

import aiohttp

from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


class ImageDownloader:
    """Downloads product images concurrently, one file per SKU.

    Each image is saved into ``output_dir`` using the same stem as the SKU's
    transformed JSON file, so ``001000-002-chemicals`` → ``001000002chemicals.png``.
    Identical URLs are fetched only once and reused across SKUs.

    A single shared instance is created on first use via :meth:`instance` and
    reused for the lifetime of the application.
    """

    OUTPUT_DIR = Path("out_images")

    _instance: "ImageDownloader | None" = None

    def __init__(self, base_dir: Path = Path("."), concurrency: int = 10) -> None:
        """Configure the output location and concurrency limit.

        Args:
            base_dir: Directory under which the :data:`OUTPUT_DIR` image folder
                is created (defaults to the current directory).
            concurrency: Maximum number of image downloads allowed to run at
                once.
        """
        self._output_dir = base_dir / self.OUTPUT_DIR
        self._concurrency = concurrency

    @classmethod
    def instance(cls, base_dir: Path = Path("."), concurrency: int = 10) -> "ImageDownloader":
        """Return the shared downloader, creating it on the first call.

        Later calls reuse the object built on the first call and ignore their
        arguments, so the whole application shares one downloader.
        """
        if cls._instance is None:
            cls._instance = cls(base_dir=base_dir, concurrency=concurrency)
        return cls._instance

    @staticmethod
    def _filename_for(sku: str, url: str, index: int = 0) -> str:
        """Build the image filename for a SKU, mirroring the exporter's JSON naming.

        The transformed JSON for a SKU is written as ``{sku without dashes}.json`` by
        the Exporter, so the primary image (``index`` 0) gets the same stem with
        whatever extension the URL carries (``.jpg`` when the URL path has none).
        Additional images for the same SKU are suffixed with their index
        (``{stem}_1.jpg``, ``{stem}_2.jpg``, …) so they don't collide.
        """
        stem = sku.replace("-", "")
        ext = Path(urlparse(url).path).suffix.lower() or ".jpg"
        return f"{stem}_{index}{ext}" if index else f"{stem}{ext}"

    _MIME_BY_EXT = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }

    @classmethod
    def _mime_for(cls, filename: str) -> str:
        """Return the image MIME type for a filename's extension (png fallback)."""
        return cls._MIME_BY_EXT.get(Path(filename).suffix.lower(), "image/png")

    @log_function
    async def download_all(
        self,
        transformed: dict[str, dict],
        write_json: Optional[Callable[[str, dict], Awaitable]] = None,
        upload_json: Optional[Callable[[str, dict], Awaitable]] = None,
        upload_image: Optional[Callable[[str, str, bytes], Awaitable]] = None,
        embed_image: Optional[Callable[[list[tuple[bytes, str]]], Awaitable[list[list[float]]]]] = None,
        write_images: bool = True,
        write_image_content_local: bool = False,
        max_images: int | None = None,
    ) -> dict:
        """Persist every transformed row's JSON and image, optionally to storage.

        Each SKU is handled by its own coroutine; all SKUs run concurrently,
        bounded end-to-end by the configured semaphore (so at most
        ``concurrency`` SKUs — and therefore their in-memory image bytes and
        open upload clients — are ever in flight, which caps memory use).

        Per SKU the order is:

        1. Download every image for the SKU (``imageUrls``, falling back to the
           single ``imageUrl``), saving each locally and uploading each via
           ``upload_image(sku, filename, data)``. Each image's bytes are also
           base64-encoded when something downstream will consume them.
        2. ``write_json(sku, row)`` — create the SKU's local JSON file (if given).
           ``image_content`` is embedded here only when
           ``write_image_content_local`` is set; otherwise the local file stays
           lean.
        3. ``upload_json(sku, row + image_content)`` — upload the JSON to object
           storage with the base64 images attached as ``image_content`` (a list,
           one entry per successfully downloaded image).

        ``image_content`` is always included in the uploaded storage JSON and is
        never indexed into OpenSearch (indexing uses the plain transformed dict).
        It reaches the local JSON file only when ``write_image_content_local`` is
        enabled. Identical image URLs are fetched only once via a shared per-run
        cache. Failures for one SKU are logged and counted but never abort the run.

        Args:
            transformed: Mapping of SKU to transformed row. Rows without an
                ``imageUrl`` still have their JSON written/uploaded but are
                counted as skipped for the image download.
            write_json: Optional async callback ``(sku, row)`` that persists the
                SKU's JSON locally.
            upload_json: Optional async callback ``(sku, row)`` that uploads the
                SKU's JSON to object storage. When ``None``, JSON is not uploaded.
            upload_image: Optional async callback ``(sku, filename, data)`` that
                uploads the image bytes to object storage. When ``None``, images
                are not uploaded.
            embed_image: Optional async callback ``(list[(bytes, mime)]) ->
                list[vector]`` that embeds a SKU's downloaded images. When given,
                each row gains a nested ``images`` list of
                ``{image_url, image_base64, image_vector}`` (one entry per
                successfully embedded image) for kNN indexing. When ``None``, no
                image vectors are produced.
            write_images: Whether each downloaded image is saved to the local
                output directory.
            write_image_content_local: Whether the base64 ``image_content`` is
                also embedded in the local JSON written by ``write_json``. Has no
                effect when ``write_json`` is ``None``.
            max_images: Optional cap on the number of images handled per SKU
                (download, store and embed). ``None`` (the default) means no
                cap — every image a SKU has is processed and embedded into its
                nested ``images`` array.

        Returns:
            A summary dict with ``downloaded``, ``skipped``, ``failed`` (image
            counts); ``json_uploaded``, ``image_uploaded`` and ``upload_failed``
            (storage counts); and ``json_write_failed`` / ``image_save_failed``
            (local-disk write failures, tolerated and counted rather than
            aborting the run).
        """
        if write_images:
            self._output_dir.mkdir(parents=True, exist_ok=True)

        with_url = sum(1 for row in transformed.values() if row.get("image_urls"))
        without_url = len(transformed) - with_url
        uploading = upload_json is not None or upload_image is not None
        logger.info(
            f"Processing {len(transformed)} SKUs — {with_url} with imageUrl, {without_url} without"
            + (" (per SKU: write+upload JSON and download+upload image in parallel)" if uploading
               else " (writing JSON then downloading each image per SKU)" if write_json else ""),
            extra={
                "function_name": "download_all",
                "input_data": {"total": len(transformed), "with_url": with_url, "without_url": without_url, "uploading": uploading},
            },
        )

        sem = asyncio.Semaphore(self._concurrency)
        url_cache: dict[str, asyncio.Task] = {}
        counters = {
            "downloaded": 0, "skipped": 0, "failed": 0,
            "json_uploaded": 0, "image_uploaded": 0, "upload_failed": 0,
            "json_write_failed": 0, "image_save_failed": 0,
            "images_embedded": 0, "embed_failed": 0,
        }

        # trust_env=True so aiohttp honours the standard HTTP(S)_PROXY / NO_PROXY
        # env vars — required for outbound image fetches to reach public CDNs
        # (e.g. images.unsplash.com) through the corporate proxy.
        async with aiohttp.ClientSession(trust_env=True) as session:

            async def fetch(url: str) -> bytes | None:
                """Fetch a URL's bytes once; concurrent callers share the result."""
                task = url_cache.get(url)
                if task is None:
                    task = asyncio.create_task(self._get_bytes(session, url))
                    url_cache[url] = task
                return await task

            async def json_to_storage(sku: str, row: dict) -> None:
                """Upload the SKU's JSON to object storage (tolerant of failure)."""
                if upload_json is None:
                    return
                try:
                    await upload_json(sku, row)
                    counters["json_uploaded"] += 1
                    logger.debug(
                        f"Uploaded JSON for sku={sku!r}",
                        extra={"request_id": sku, "function_name": "download_all", "input_data": {"sku": sku}},
                    )
                except Exception as exc:
                    counters["upload_failed"] += 1
                    logger.error(
                        f"Failed to upload JSON for sku={sku!r}: {exc}",
                        extra={"request_id": sku, "function_name": "download_all", "status": "error",
                               "error_details": str(exc), "input_data": {"sku": sku}},
                    )

            async def image_to_storage(sku: str, row: dict, want_base64: bool) -> tuple[list[str], dict | None]:
                """Download a SKU's images, save + upload + embed each (tolerant).

                Iterates the SKU's image URLs (``image_urls``, falling back to the
                single ``image_url``), optionally capped at ``max_images``. Each image is
                fetched once (shared cache), optionally written to disk and
                uploaded to storage, with every per-image failure logged and
                counted but never raised.

                When ``want_base64`` is set, the raw bytes of each successfully
                fetched image are base64-encoded and returned in order, for the
                caller to attach as ``image_content`` on the storage JSON only.

                When ``embed_image`` is configured, every successfully downloaded
                image is embedded (one batched call per SKU) and returned as a
                parallel-array object ``{image_url: [...], image_base64: [...],
                image_vector: [...]}`` (slot ``i`` describes image ``i``) for the
                caller to attach to the transformed row as the ``images`` field.

                Returns:
                    ``(base64_list, images_obj)`` — the per-image base64 blobs and
                    the parallel-array images object (``images_obj`` is ``None``
                    when no image was embedded).
                """
                urls = row.get("image_urls") or []
                if max_images is not None:
                    urls = urls[:max_images]
                if not urls:
                    counters["skipped"] += 1
                    return [], []

                b64: list[str] = []
                # (filename, url, bytes, mime) for each image downloaded OK — used
                # to build the nested image embeddings once all are fetched.
                downloaded: list[tuple[str, str, bytes, str]] = []
                for index, url in enumerate(urls):
                    data = await fetch(url)
                    if data is None:
                        counters["failed"] += 1
                        logger.warning(
                            f"Image download failed for sku={sku!r} ({url})",
                            extra={"request_id": sku, "function_name": "download_all", "status": "error",
                                   "input_data": {"sku": sku, "url": url}},
                        )
                        continue

                    counters["downloaded"] += 1
                    filename = self._filename_for(sku, url, index)
                    downloaded.append((filename, url, data, self._mime_for(filename)))

                    if write_images:
                        path = self._output_dir / filename
                        try:
                            await asyncio.to_thread(path.write_bytes, data)
                            logger.debug(
                                f"Saved image for sku={sku!r} → {path}",
                                extra={"function_name": "download_all", "input_data": {"sku": sku, "path": str(path)}},
                            )
                        except Exception as exc:
                            counters["image_save_failed"] += 1
                            logger.error(
                                f"Failed to write image for sku={sku!r} to {path}: {exc}",
                                extra={"function_name": "download_all", "status": "error",
                                       "error_details": str(exc), "input_data": {"sku": sku, "path": str(path)}},
                            )

                    if upload_image is not None:
                        try:
                            await upload_image(sku, filename, data)
                            counters["image_uploaded"] += 1
                            logger.debug(
                                f"Uploaded image for sku={sku!r} ({filename})",
                                extra={"request_id": sku, "function_name": "download_all", "input_data": {"sku": sku, "filename": filename}},
                            )
                        except Exception as exc:
                            counters["upload_failed"] += 1
                            logger.error(
                                f"Failed to upload image for sku={sku!r} ({filename}): {exc}",
                                extra={"request_id": sku, "function_name": "download_all", "status": "error",
                                       "error_details": str(exc), "input_data": {"sku": sku, "filename": filename}},
                            )

                    if want_base64:
                        b64.append(base64.b64encode(data).decode("ascii"))

                # Parallel-array object: image_url[i], image_base64[i] and
                # image_vector[i] all describe the same image.
                images_obj: dict = {"image_url": [], "image_base64": [], "image_vector": []}
                if embed_image is not None and downloaded:
                    try:
                        vectors = await embed_image([(data, mime) for _, _, data, mime in downloaded])
                    except Exception as exc:
                        counters["embed_failed"] += len(downloaded)
                        logger.error(
                            f"Image embedding failed for sku={sku!r}: {exc}",
                            extra={"request_id": sku, "function_name": "download_all", "status": "error",
                                   "error_details": str(exc), "input_data": {"sku": sku, "images": len(downloaded)}},
                        )
                        vectors = []
                    for (_filename, url, data, _mime), vector in zip(downloaded, vectors):
                        images_obj["image_url"].append(url)
                        images_obj["image_base64"].append(base64.b64encode(data).decode("ascii"))
                        images_obj["image_vector"].append(vector)
                        counters["images_embedded"] += 1

                # Return None when nothing was embedded so the caller skips it.
                return b64, (images_obj if images_obj["image_vector"] else None)

            async def process(sku: str, row: dict) -> None:
                """Write a SKU's local JSON, download its images, then upload JSON.

                The whole per-SKU unit of work is bounded by the shared
                semaphore so memory and open upload clients stay capped. The
                storage JSON upload runs *after* the image downloads because it
                carries the base64 ``image_content`` built from the downloaded
                bytes; the local JSON file never gets that heavy field.
                """
                async with sem:
                    # Download/save/upload every image first; its base64 is needed
                    # when the storage upload — or the opt-in local embed — uses it,
                    # and its vectors feed the nested images field for indexing.
                    want_b64 = upload_json is not None or (write_image_content_local and write_json is not None)
                    image_b64, images_obj = await image_to_storage(sku, row, want_base64=want_b64)

                    # Local JSON. image_content is embedded only when explicitly
                    # enabled (write_image_content_local); otherwise the local file
                    # stays lean and the base64 lives only in the storage JSON. The
                    # heavy image vectors are likewise kept out of the lean local
                    # file (they go to storage JSON and the in-memory index row).
                    if write_json is not None:
                        local_row = {**row, "image_content": image_b64} if write_image_content_local else row
                        try:
                            await write_json(sku, local_row)
                        except Exception as exc:
                            # write_json (the exporter) already logged the error
                            # with sku context. Count it and keep going so one
                            # failed local write can't abort the whole run — the
                            # SKU's storage upload still proceeds below.
                            counters["json_write_failed"] += 1
                            logger.debug(
                                f"Local JSON write failed for sku={sku!r}; continuing: {exc}",
                                extra={"request_id": sku, "function_name": "download_all",
                                       "input_data": {"sku": sku}},
                            )

                    # Attach the parallel-array images object to the shared
                    # transformed row (done after the lean local write above) so the
                    # OpenSearch indexer — which runs later over this same dict —
                    # stores the images field. image_content (base64) is NOT added
                    # here: it is never indexed, only written to storage JSON.
                    if images_obj:
                        row["images"] = images_obj

                    # Storage JSON always carries the base64 image_content. The
                    # field is never indexed into OpenSearch (that uses the plain
                    # transformed dict, which never gets image_content).
                    if upload_json is not None:
                        await json_to_storage(sku, {**row, "image_content": image_b64})

            await asyncio.gather(*(process(sku, row) for sku, row in transformed.items()))

        logger.info(
            f"Processing complete — images: {counters['downloaded']} downloaded, "
            f"{counters['skipped']} skipped, {counters['failed']} failed "
            f"({len(url_cache)} unique URLs); "
            f"embeddings: {counters['images_embedded']} embedded, {counters['embed_failed']} failed; "
            f"storage: {counters['json_uploaded']} JSON + {counters['image_uploaded']} images uploaded, "
            f"{counters['upload_failed']} upload failures; "
            f"local writes: {counters['json_write_failed']} JSON + {counters['image_save_failed']} image write failures",
            extra={
                "function_name": "download_all",
                "status": "success",
                "input_data": {**counters, "unique_urls": len(url_cache)},
            },
        )
        return counters

    async def _get_bytes(
        self, session: aiohttp.ClientSession, url: str
    ) -> bytes | None:
        """Fetch a single URL's bytes, returning None on any failure.

        Concurrency is bounded by the caller (each SKU's work runs under the
        shared semaphore in :meth:`download_all`), so this method does not
        acquire the semaphore itself — doing so would deadlock when a bounded
        SKU coroutine awaits a fetch that also needs a slot.
        """
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30), ssl=SSL_CONTEXT) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as exc:
            logger.warning(
                f"Image download failed for {url}: {exc}",
                extra={
                    "function_name": "_get_bytes",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"url": url},
                },
            )
            return None
