"""Embed transformed JSON rows and write them to ``final_json/`` (no OpenSearch push).

Reads every per-SKU JSON file in ``out_transformed/`` and, using the same OCI
``cohere.embed-v4.0`` model as ``ingest_json.py`` (so vectors match the search API):

  * embeds ``content``        -> ``content_vector``  (input_type=SEARCH_DOCUMENT)
  * embeds ``image_content``  -> nested ``images: [{image_url, image_vector}, ...]``
    (the raw base64 ``image_content`` is removed, it is not in the index mapping)

Output: ONE file, ``final_json/final.json`` — a JSON array of every document
(all original fields + ``content_vector`` + ``images``).

With ``--bulk`` it also writes, for pushing via OpenSearch Dashboards Dev Tools:
  * ``final_json/bulk.ndjson``          body for ``POST _bulk`` (all docs, one file)
  * ``final_json/index_mapping.json``   body for ``PUT <index>``

Usage:
    python embed_json.py                     # all files in out_transformed -> final_json/final.json
    python embed_json.py --bulk              # also write bulk.ndjson + index_mapping.json
    python embed_json.py --out-file other.json --limit 10

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
    p.add_argument("--out-file", default="final_json/final.json", help="Output JSON file (default: final_json/final.json).")
    p.add_argument("--bulk", action="store_true", help="Also write bulk.ndjson + index_mapping.json next to the output file.")
    p.add_argument("--pattern", default="*.json", help="Glob for input files (default: *.json).")
    p.add_argument("--index", default=None, help="Index name used in the bulk files (default: $OPENSEARCH_INDEX).")
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


def write_outputs(rows: dict[str, dict], out_file: Path, bulk: bool, index: str,
                  vector_field: str, dim: int, image_dim: int) -> None:
    """Write all docs to one JSON array file; optionally a single _bulk NDJSON + index mapping."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(list(rows.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    if not bulk:
        return
    lines = []
    for sku, row in rows.items():
        lines.append(json.dumps({"index": {"_index": index, "_id": sku}}))
        lines.append(json.dumps(row, ensure_ascii=False))
    # _bulk requires a trailing newline after the last line.
    (out_file.parent / "bulk.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_file.parent / "index_mapping.json").write_text(
        json.dumps(build_index_body(vector_field, dim, image_dim), indent=2), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    in_dir, out_file = Path(args.input_dir), Path(args.out_file)
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

    write_outputs(rows, out_file, args.bulk, index, vector_field, dim, image_dim)

    print(f"\nDone: {len(rows)} doc(s) -> {out_file}")
    print(f"  content vectors : {n_text}  (field '{vector_field}', dim {dim})")
    print(f"  image vectors   : {img['images_embedded']}  (nested 'images.image_vector')")
    if args.bulk:
        print(f"  bulk file       : {out_file.parent / 'bulk.ndjson'}  (index '{index}')")
        print(f"  index mapping   : {out_file.parent / 'index_mapping.json'}")
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
