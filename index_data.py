"""Standalone indexer — embed transformed rows and push them to OpenSearch.

Reads the transformed SKU JSON files produced by the pipeline (default
``out_transformed/``), embeds each row's ``content`` with OCI GenAI, and
bulk-indexes every row (all fields + the ``content_vector`` embedding) into the
configured OpenSearch index. The index is created if it does not already exist,
otherwise the data is uploaded into the existing index.

Usage:
    uv run python index_data.py [TRANSFORMED_DIR]

Configuration (OpenSearch + OCI) is read from environment variables / .env;
see asmo/search_index.py and asmo/embeddings.py.
"""
import asyncio
import json
import sys
from pathlib import Path

import truststore
from dotenv import load_dotenv

# Use the OS trust store for TLS before any SSL context is created.
truststore.inject_into_ssl()
# override=True so values in .env take precedence over any stale OS environment
# variables (e.g. a previously `setx`-ed OPENSEARCH_USER on Windows).
load_dotenv(override=True)

from asmo.logging_service import setup_logging, logger  # noqa: E402
from asmo.search_index import OpenSearchIndexer  # noqa: E402

setup_logging()

DEFAULT_DIR = Path("out_transformed")


def load_transformed(folder: Path) -> dict[str, dict]:
    """Load transformed rows from a folder of per-SKU JSON files.

    Each ``*.json`` file is one transformed row. Rows are keyed by their ``sku``
    field, falling back to the file stem when a row has no ``sku``.

    Args:
        folder: Directory containing the transformed JSON files.

    Returns:
        Mapping of SKU to transformed row.
    """
    if not folder.is_dir():
        raise NotADirectoryError(f"Transformed folder not found: {folder}")

    rows: dict[str, dict] = {}
    files = sorted(folder.glob("*.json"))
    for path in files:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            logger.error(
                f"Skipping '{path.name}': {exc}",
                extra={"function_name": "load_transformed", "status": "error", "error_details": str(exc)},
            )
            continue
        key = row.get("sku") or path.stem
        rows[key] = row
    logger.info(
        f"Loaded {len(rows)} transformed row(s) from {folder}/",
        extra={"function_name": "load_transformed", "input_data": {"count": len(rows), "folder": str(folder)}},
    )
    return rows


async def main() -> None:
    """Load transformed rows and index them (creating the index if needed)."""
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    logger.info(f"Standalone indexer starting — source={folder}", extra={"function_name": "main"})

    transformed = load_transformed(folder)
    if not transformed:
        logger.warning("Nothing to index — no transformed rows found", extra={"function_name": "main"})
        return

    indexer = OpenSearchIndexer()
    try:
        summary = await indexer.index_rows(transformed)
        logger.info(
            f"Standalone indexer finished — {summary['indexed']} indexed, {summary['failed']} failed",
            extra={"function_name": "main", "status": "success", "input_data": summary},
        )
    finally:
        await indexer.close()


if __name__ == "__main__":
    asyncio.run(main())
