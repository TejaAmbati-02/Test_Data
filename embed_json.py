"""Embed transformed JSON rows and write them to ``final_json/`` (no OpenSearch push).

Reads every per-SKU JSON file in ``out_transformed/`` and, using the same OCI
``cohere.embed-v4.0`` model as ``ingest_json.py`` (so vectors match the search API):

  * embeds ``content``        -> ``content_vector``  (input_type=SEARCH_DOCUMENT)
  * embeds ``image_content``  -> nested ``images: [{image_url, image_vector}, ...]``
    (the raw base64 ``image_content`` is removed, it is not in the index mapping)

Outputs (in ``final_json/``):
  * ``<sku>.json``            one enriched document per SKU
  * ``index_mapping.json``    body for ``PUT <index>`` (create the index in Dev Tools)
  * ``bulk_NNN.ndjson``       bodies for ``POST _bulk`` (paste into Dev Tools), split
                              into chunks of ``--docs-per-bulk`` docs so each paste
                              stays small enough for the Dashboards console

Usage:
    python embed_json.py                                   # out_transformed -> final_json
    python embed_json.py IN_DIR --out-dir final_json --index asmo-search-api-index
    python embed_json.py --no-image-embeddings --limit 10

Requires the same .env / OCI config as ingest_json.py (OCI_COMPARTMENT_ID, ...).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Importing ingest_json also loads .env, injects the OS trust store and sets up logging.
from ingest_json import apply_frontend_schema, attach_image_embeddings, load_transformed  # noqa: E402
from asmo.embeddings import OCIEmbedder  # noqa: E402
from asmo.logging_service import logger  # noqa: E402
from asmo.search_index import build_index_body  # noqa: E402


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="embed_json.py", description=__doc__.splitlines()[0])
    p.add_argument("input_dir", nargs="?", default="out_transformed", help="Directory of per-SKU *.json files.")
    p.add_argument("--out-dir", default="final_json", help="Where to write the enriched JSON (default: final_json).")
    p.add_argument("--pattern", default="*.json", help="Glob for input files (default: *.json).")
    p.add_argument("--index", default=None, help="Index name used in the bulk files (default: $OPENSEARCH_INDEX).")
    p.add_argument("--docs-per-bulk", type=int, default=25, help="Docs per bulk_NNN.ndjson file (default: 25).")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N files.")
    p.add_argument("--no-image-embeddings", action="store_true", help="Skip image_content embedding (strip it only).")
    p.add_argument("--max-products-per-image", type=int, default=0, metavar="N",
                   help="Skip images whose URL is shared by more than N products (placeholders). 0 = embed all.")
    return p.parse_args(argv)


async def embed_content(rows: dict[str, dict], embedder: OCIEmbedder, vector_field: str) -> int:
    """Embed each row's ``content`` into ``vector_field``; returns the number embedded."""
    skus = [sku for sku, row in rows.items() if (row.get("content") or "").strip()]
    if not skus:
        return 0
    vectors = await embedder.embed([rows[s]["content"] for s in skus], input_type="SEARCH_DOCUMENT")
    for sku, vec in zip(skus, vectors):
        rows[sku][vector_field] = vec
    return len(vectors)


def write_outputs(rows: dict[str, dict], out_dir: Path, index: str, vector_field: str,
                  dim: int, image_dim: int, docs_per_bulk: int) -> int:
    """Write per-SKU JSON, the index mapping and chunked _bulk NDJSON files. Returns bulk file count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("bulk_*.ndjson"):  # avoid stale chunks from a previous, larger run
        old.unlink()

    for sku, row in rows.items():
        (out_dir / f"{sku}.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "index_mapping.json").write_text(
        json.dumps(build_index_body(vector_field, dim, image_dim), indent=2), encoding="utf-8")

    items = list(rows.items())
    step = max(1, docs_per_bulk)
    n_files = 0
    for n_files, start in enumerate(range(0, len(items), step), start=1):
        lines = []
        for sku, row in items[start:start + step]:
            lines.append(json.dumps({"index": {"_index": index, "_id": sku}}))
            lines.append(json.dumps(row, ensure_ascii=False))
        # _bulk requires a trailing newline after the last line.
        (out_dir / f"bulk_{n_files:03d}.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_files


async def run(args: argparse.Namespace) -> int:
    in_dir, out_dir = Path(args.input_dir), Path(args.out_dir)
    if not in_dir.is_dir():
        logger.error(f"Input directory not found: {in_dir}")
        return 2

    rows = load_transformed(in_dir, args.pattern, require_content=False, limit=args.limit)
    if not rows:
        logger.error(f"No JSON rows loaded from {in_dir}/")
        return 2
    apply_frontend_schema(rows)

    # Same env vars/defaults as asmo.search_index.OpenSearchConfig, without needing OpenSearch credentials.
    index = args.index or os.environ.get("OPENSEARCH_INDEX", "asmo-search-api-index")
    vector_field = os.environ.get("OPENSEARCH_VECTOR_FIELD", "content_vector")
    dim = int(os.environ.get("OPENSEARCH_VECTOR_DIM", "1536"))
    image_dim = int(os.environ.get("OPENSEARCH_IMAGE_VECTOR_DIM", str(dim)))
    embedder = OCIEmbedder()

    n_text = await embed_content(rows, embedder, vector_field)
    img = await attach_image_embeddings(
        rows, embedder, enabled=not args.no_image_embeddings,
        max_products_per_image=args.max_products_per_image or None,
    )

    n_bulk = write_outputs(rows, out_dir, index, vector_field,
                           dim, image_dim, args.docs_per_bulk)

    print(f"\nDone: {len(rows)} doc(s) -> {out_dir}/")
    print(f"  content vectors : {n_text}  (field '{vector_field}', dim {dim})")
    print(f"  image vectors   : {img['images_embedded']}  (nested 'images.image_vector')")
    print(f"  bulk files      : {n_bulk} x bulk_NNN.ndjson for index '{index}'")
    print(f"  index mapping   : {out_dir / 'index_mapping.json'}")
    return 0


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        sys.exit(asyncio.run(run(args)))
    except KeyError as exc:  # e.g. OCI_COMPARTMENT_ID missing
        logger.error(f"Missing required environment variable: {exc}")
        sys.exit(5)


if __name__ == "__main__":
    main()
