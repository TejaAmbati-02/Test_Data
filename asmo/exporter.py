"""Persists fetched and transformed data to the local filesystem as JSON.

This module is the output stage of the pipeline. Given the raw products and
prices fetched from CommerceTools and the flattened SKU rows produced by the
transformer, it writes one JSON file per record into dedicated output folders
(``out_products``, ``out_prices`` and ``out_transformed``).

The blocking file I/O is executed on worker threads via
:func:`asyncio.to_thread` so that writing does not stall the event loop in the
otherwise asynchronous pipeline. The public entry point is
:meth:`Exporter.export`, which returns an :class:`ExportPaths` describing where
each category of data was written.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from .products import Product
from .prices import Price
from .logging_service import logger, log_function


@dataclass
class ExportPaths:
    """Locations of the folders that each category of export was written to.

    Attributes:
        products: Directory containing the raw product JSON files.
        prices: Directory containing the raw price JSON files.
        transformed: Directory containing the transformed SKU-row JSON files.
    """

    products: Path
    prices: Path
    transformed: Path


class Exporter:
    """Writes fetched and transformed data as JSON files into the configured output folders.

    Each exporter instance resolves its three output folders relative to a base
    directory supplied at construction time. Records are written one file per
    record, named after the record id (or SKU) with hyphens stripped.
    """

    PRODUCTS_DIR = Path("out_products")
    PRICES_DIR = Path("out_prices")
    TRANSFORMED_DIR = Path("out_transformed")

    def __init__(self, base_dir: Path = Path(".")) -> None:
        """Initialize the exporter and resolve its output folders.

        Args:
            base_dir: Base directory under which the ``out_products``,
                ``out_prices`` and ``out_transformed`` folders are placed.
                Defaults to the current working directory.
        """
        self._products_dir = base_dir / self.PRODUCTS_DIR
        self._prices_dir = base_dir / self.PRICES_DIR
        self._transformed_dir = base_dir / self.TRANSFORMED_DIR


    @log_function
    async def export(
        self,
        products: list[Product],
        prices: list[Price],
        transformed: dict[str, dict],
    ) -> ExportPaths:
        """Write all products, prices and transformed rows to disk (batch).

        Convenience entry point that writes the raw product and price documents
        via :meth:`export_raw` and then the transformed SKU rows in one batch.
        The pipeline itself no longer uses this method: it calls
        :meth:`export_raw` and streams the transformed rows out one SKU at a
        time via :meth:`write_transformed_row`, so each JSON file can be paired
        with its image download. This method is kept for standalone/batch use.

        Args:
            products: The fetched products whose raw documents are written.
            prices: The fetched prices whose raw documents are written.
            transformed: Mapping of SKU to transformed row to write.

        Returns:
            An :class:`ExportPaths` describing the folders written to.
        """
        paths = await self.export_raw(products, prices)
        await self._write_transformed(transformed)
        logger.info("Export complete", extra={"function_name": "export", "status": "success"})
        return paths

    @log_function
    async def export_raw(self, products: list[Product], prices: list[Price]) -> ExportPaths:
        """Write only the raw product and price documents to disk.

        Ensures all three output folders exist (including the transformed
        folder, so a concurrent per-SKU writer can drop files into it without
        racing on directory creation) and writes the raw product and price
        documents. The transformed rows are intentionally *not* written here —
        the pipeline streams those out per SKU via
        :meth:`write_transformed_row` so each one can be paired with its image
        download.

        Args:
            products: The fetched products whose raw documents are written.
            prices: The fetched prices whose raw documents are written.

        Returns:
            An :class:`ExportPaths` describing the folders written to.
        """
        logger.info(
            f"Exporting {len(products)} products and {len(prices)} prices (raw)",
            extra={"function_name": "export_raw"},
        )
        for folder in (self._products_dir, self._prices_dir, self._transformed_dir):
            folder.mkdir(parents=True, exist_ok=True)

        await self._write_each(self._products_dir, [p.raw for p in products])
        await self._write_each(self._prices_dir, [p.raw for p in prices])

        return ExportPaths(
            products=self._products_dir,
            prices=self._prices_dir,
            transformed=self._transformed_dir,
        )

    async def write_transformed_row(self, sku: str, row: dict) -> Path:
        """Write a single transformed SKU row to the transformed folder.

        Used by the pipeline to stream transformed JSON out one SKU at a time,
        so each file's creation can be immediately followed by that SKU's image
        download. The blocking write is offloaded to a worker thread. The file
        is named after the SKU with hyphens removed, matching the image naming.

        Args:
            sku: The SKU the row belongs to; also the file stem.
            row: The transformed row to serialise as JSON.

        Returns:
            The path the JSON file was written to.
        """
        self._transformed_dir.mkdir(parents=True, exist_ok=True)
        file_path = self._transformed_dir / f"{sku.replace('-', '')}.json"
        await asyncio.to_thread(self._write_row_sync, file_path, sku, row)
        return file_path

    async def _write_each(self, folder: Path, records: list[dict]) -> None:
        """Write a list of raw records to a folder, one JSON file per record.

        Offloads the blocking writes to a worker thread and logs the elapsed
        time once complete.

        Args:
            folder: Destination directory for the JSON files.
            records: Raw record dicts to write; each must contain an ``id``.
        """
        from datetime import datetime
        logger.info(
            f"Writing {len(records)} records to {folder}/",
            extra={"function_name": "_write_each", "input_data": {"count": len(records), "folder": str(folder)}},
        )
        t0 = datetime.now()
        await asyncio.to_thread(self._write_each_sync, folder, records)
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f"{len(records):>6} records → {folder}/ in {elapsed:.3f}s",
            extra={
                "function_name": "_write_each",
                "latency": f"{elapsed:.3f}s",
                "status": "success",
                "input_data": {"count": len(records), "folder": str(folder)},
            },
        )

    async def _write_transformed(self, transformed: dict[str, dict]) -> None:
        """Write the transformed SKU rows to the transformed folder.

        Offloads the blocking writes to a worker thread and logs the elapsed
        time once complete.

        Args:
            transformed: Mapping of SKU to transformed row; each SKU becomes a
                JSON file in the transformed output folder.
        """
        from datetime import datetime
        logger.info(
            f"Writing {len(transformed)} transformed records to {self._transformed_dir}/",
            extra={"function_name": "_write_transformed", "input_data": {"count": len(transformed)}},
        )
        t0 = datetime.now()
        await asyncio.to_thread(self._write_transformed_sync, transformed)
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f"{len(transformed):>6} records → {self._transformed_dir}/ in {elapsed:.3f}s",
            extra={
                "function_name": "_write_transformed",
                "latency": f"{elapsed:.3f}s",
                "status": "success",
                "input_data": {"count": len(transformed)},
            },
        )

    @staticmethod
    def _write_each_sync(folder: Path, records: list[dict]) -> None:
        """Synchronously write raw records to a folder, one JSON file per record.

        Runs on a worker thread (via :meth:`_write_each`). Each file is named
        after the record's ``id`` with hyphens removed and is written as
        indented, non-ASCII-escaped JSON.

        Args:
            folder: Destination directory for the JSON files.
            records: Raw record dicts to write; each must contain an ``id``.

        Raises:
            Exception: Re-raises any write error after logging it with the
                offending record id.
        """
        for record in records:
            try:
                file_path = folder / f"{record['id'].replace('-', '')}.json"
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2, ensure_ascii=False)
                logger.debug(
                    f"Wrote record id={record.get('id')!r} to {folder}/",
                    extra={"function_name": "_write_each_sync", "input_data": {"id": record.get("id"), "folder": str(folder)}},
                )
            except Exception as exc:
                logger.error(
                    f"Failed to write record id={record.get('id')!r} to {folder}/: {exc}",
                    extra={
                        "function_name": "_write_each_sync",
                        "status": "error",
                        "error_details": str(exc),
                        "input_data": {"id": record.get("id"), "folder": str(folder)},
                    },
                )
                raise

    def _write_transformed_sync(self, transformed: dict[str, dict]) -> None:
        """Synchronously write transformed rows, one JSON file per SKU.

        Runs on a worker thread (via :meth:`_write_transformed`). Delegates each
        row to :meth:`_write_row_sync` so the batch and per-SKU
        (:meth:`write_transformed_row`) write paths share identical behaviour.

        Args:
            transformed: Mapping of SKU to transformed row to write.

        Raises:
            Exception: Re-raises any write error after logging it with the
                offending SKU.
        """
        for sku, row in transformed.items():
            self._write_row_sync(self._transformed_dir / f"{sku.replace('-', '')}.json", sku, row)

    @staticmethod
    def _write_row_sync(file_path: Path, sku: str, row: dict) -> None:
        """Synchronously write one transformed SKU row to ``file_path``.

        Runs on a worker thread. Writes indented, non-ASCII-escaped JSON and
        logs success at debug level. Shared by both the batch writer
        (:meth:`_write_transformed_sync`) and the per-SKU writer
        (:meth:`write_transformed_row`).

        Args:
            file_path: Destination JSON file path.
            sku: The SKU the row belongs to (used for logging context).
            row: The transformed row to serialise.

        Raises:
            Exception: Re-raises any write error after logging it with the SKU.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2, ensure_ascii=False)
            logger.debug(
                f"Wrote transformed row sku={sku!r} to {file_path.parent}/",
                extra={
                    "request_id": sku,
                    "function_name": "_write_row_sync",
                    "input_data": {"sku": sku, "folder": str(file_path.parent)},
                },
            )
        except Exception as exc:
            logger.error(
                f"Failed to write transformed row sku={sku!r}: {exc}",
                extra={
                    "request_id": sku,
                    "function_name": "_write_row_sync",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"sku": sku, "folder": str(file_path.parent)},
                },
            )
            raise
