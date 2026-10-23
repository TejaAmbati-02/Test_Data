"""Inspect and query the ASMO OpenSearch index (run this in AVD).

A small test/utility to confirm an ingest worked end-to-end:

  * with no query flag it prints the index health — exists?, document count,
    whether the nested ``images`` mapping is present, and one sample document;
  * ``--text "…"`` runs a semantic text search (kNN over ``content_vector``);
  * ``--image PATH`` runs an image search (nested kNN over
    ``images.image_vector`` with ``score_mode=max`` + ``inner_hits``, so each
    SKU is ranked by its best-matching image and you see which image matched).

Both query modes embed the query with the same OCI ``embed-v4.0`` model used at
ingest time, so results are directly comparable.

Usage:
    uv run python query_index.py                              # index health + sample doc
    uv run python query_index.py --text "water pump"          # text semantic search
    uv run python query_index.py --image out_images/001000001chemicals.jpg
    uv run python query_index.py --text "valve" --size 5

Configuration (OpenSearch + OCI) is read from environment variables / .env.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import truststore
from dotenv import load_dotenv

truststore.inject_into_ssl()
load_dotenv(override=True)

from asmo.embeddings import OCIEmbedder  # noqa: E402
from asmo.logging_service import setup_logging, logger  # noqa: E402
from asmo.search_index import OpenSearchIndexer  # noqa: E402

setup_logging()


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="query_index.py", description="Inspect and query the ASMO OpenSearch index.")
    p.add_argument("--text", metavar="QUERY", help="Run a semantic text search for this query.")
    p.add_argument("--image", metavar="PATH", help="Run an image search using this image file.")
    p.add_argument("--size", type=int, default=5, help="Number of hits to return (default: 5).")
    p.add_argument(
        "--emit-query", action="store_true",
        help="Embed the --image/--text and PRINT a ready-to-paste OpenSearch Dashboards "
             "Dev Tools query (with the vector inlined) instead of running it.",
    )
    return p.parse_args(argv)


def build_text_body(vector: list[float], vector_field: str, size: int) -> dict:
    """kNN search body over the content (text) vector."""
    return {
        "size": size,
        "_source": ["sku_id", "product_id", "product_title", "category_name", "sku_image_url"],
        "query": {"knn": {vector_field: {"vector": vector, "k": size}}},
    }


async def show_health(indexer: OpenSearchIndexer, client) -> None:
    """Print index existence, doc count, image-mapping presence and a sample doc."""
    cfg = indexer.config
    name = cfg.index_name

    if not await client.indices.exists(index=name):
        print(f"\n  Index '{name}' does NOT exist on {cfg.host}. Ingest first (ingest_json.py --recreate).\n")
        return

    count = (await client.count(index=name)).get("count", 0)
    mapping = await client.indices.get_mapping(index=name)
    props = mapping[name]["mappings"]["properties"]
    has_images = "images" in props
    txt_dim = props.get(cfg.vector_field, {}).get("dimension")

    print(f"\n  Index          : {name}")
    print(f"  Documents      : {count}")
    print(f"  Text vector    : {cfg.vector_field} (dim {txt_dim})")
    print(f"  Images field   : {'present (parallel arrays, store-only — not kNN-searchable)' if has_images else 'MISSING — recreate the index'}")

    sample = await client.search(index=name, body={"size": 1, "_source": {"excludes": ["content_vector", "images.image_vector", "images.image_base64", "content"]}})
    hits = sample.get("hits", {}).get("hits", [])
    if hits:
        src = hits[0]["_source"]
        n_imgs = len((src.get("images") or {}).get("image_url", []))
        print(f"  Sample images  : {n_imgs} image(s) in the sample document")
        print("\n  Sample document (vectors / base64 / content omitted):")
        print("   ", json.dumps(src, ensure_ascii=False)[:800])
    print()


async def search_text(embedder: OCIEmbedder, client, index: str, vector_field: str, query: str, size: int) -> None:
    """Embed the query text and run a kNN search over the content vector."""
    print(f"\n  Text search: {query!r}")
    vec = await embedder.embed_query(query)
    if not vec:
        print("  (failed to embed query)")
        return
    resp = await client.search(index=index, body=build_text_body(vec, vector_field, size))
    _print_hits(resp)


IMAGE_SEARCH_UNAVAILABLE = (
    "\n  Image (kNN) search is not available with the current images schema:\n"
    "  image vectors are stored as a parallel array (image_vector: [[...], ...]),\n"
    "  which OpenSearch cannot vector-search. Use --text for semantic search.\n"
)


def _print_hits(resp: dict) -> None:
    """Pretty-print search hits."""
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        print("  No matches.")
        return
    for rank, h in enumerate(hits, start=1):
        src = h.get("_source", {})
        print(f"  {rank}. score={h.get('_score'):.4f}  sku={src.get('sku_id')}  {src.get('product_title') or ''}")
    print()


async def emit_query(args: argparse.Namespace) -> int:
    """Embed the image/text and print a paste-ready Dashboards Dev Tools query."""
    if not args.image and not args.text:
        print("\n  --emit-query needs --image PATH or --text QUERY.\n")
        return 2
    if args.image:
        print(IMAGE_SEARCH_UNAVAILABLE)
        return 1
    cfg = OpenSearchIndexer().config
    embedder = OCIEmbedder()
    vec = await embedder.embed_query(args.text)
    if not vec:
        print("  (failed to embed query)")
        return 1
    body = build_text_body(vec, cfg.vector_field, args.size)
    print(f"\n# Text kNN for {args.text!r} — paste into OpenSearch Dashboards → Dev Tools:\n")
    print(f"POST {cfg.index_name}/_search\n{json.dumps(body)}\n")
    return 0


async def run(args: argparse.Namespace) -> int:
    if args.emit_query:
        return await emit_query(args)

    indexer = OpenSearchIndexer()
    cfg = indexer.config
    try:
        if not await indexer.ping():
            print(f"\n  OpenSearch not reachable at https://{cfg.host}:{cfg.port} — check the network / OPENSEARCH_* settings.\n")
            return 2
        client = indexer._connect()  # reuse the configured, authenticated client

        if not args.text and not args.image:
            await show_health(indexer, client)
            return 0

        if args.image:
            print(IMAGE_SEARCH_UNAVAILABLE)
        if args.text:
            embedder = OCIEmbedder()
            await search_text(embedder, client, cfg.index_name, cfg.vector_field, args.text, args.size)
        return 0
    except Exception as exc:
        logger.error(f"Query failed: {exc}", extra={"function_name": "run", "status": "error", "error_details": str(exc)})
        print(f"\n  Error: {exc}\n")
        return 1
    finally:
        await indexer.close()


def main() -> None:
    sys.exit(asyncio.run(run(parse_args(sys.argv[1:]))))


if __name__ == "__main__":
    main()
