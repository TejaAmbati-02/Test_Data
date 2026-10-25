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
from pathlib import Path

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
        "_source": ["sku_id", "product_key", "product_name", "category_name", "image_urls"],
        "query": {"knn": {vector_field: {"vector": vector, "k": size}}},
    }


def build_image_body(vector: list[float], size: int) -> dict:
    """Nested kNN body over every product image; each SKU scores by its best image."""
    return {
        "size": size,
        "_source": ["sku_id", "product_key", "product_name", "category_name", "image_urls"],
        "query": {
            "nested": {
                "path": "images",
                "score_mode": "max",
                "query": {"knn": {"images.image_vector": {"vector": vector, "k": size}}},
                "inner_hits": {"size": 1, "_source": ["images.image_url"]},
            }
        },
    }


def _image_mime(data: bytes) -> str | None:
    """Return the MIME type for PNG/JPEG/GIF/WebP bytes, else None."""
    if data[:8].startswith(b"\x89PNG"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def embed_image_file(embedder: OCIEmbedder, path: str) -> list[float] | None:
    """Read an image file and embed it with the same model used at ingest."""
    data = Path(path).read_bytes()
    mime = _image_mime(data)
    if mime is None:
        print(f"  Unsupported image format: {path} (use PNG, JPEG, GIF or WebP)")
        return None
    vectors = await embedder.embed_images([(data, mime)])
    return vectors[0] if vectors else None


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
    has_images = props.get("images", {}).get("type") == "nested"
    txt_dim = props.get(cfg.vector_field, {}).get("dimension")

    print(f"\n  Index          : {name}")
    print(f"  Documents      : {count}")
    print(f"  Text vector    : {cfg.vector_field} (dim {txt_dim})")
    print(f"  Images field   : {'nested kNN (images.image_vector)' if has_images else 'MISSING or old format — recreate the index'}")

    sample = await client.search(index=name, body={"size": 1, "_source": {"excludes": ["content_vector", "images.image_vector", "content"]}})
    hits = sample.get("hits", {}).get("hits", [])
    if hits:
        src = hits[0]["_source"]
        n_imgs = len(src.get("images") or [])
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


async def search_image(embedder: OCIEmbedder, client, index: str, path: str, size: int) -> None:
    """Embed the image file and run a nested kNN search over every product image."""
    print(f"\n  Image search: {path}")
    vec = await embed_image_file(embedder, path)
    if not vec:
        print("  (failed to embed image)")
        return
    resp = await client.search(index=index, body=build_image_body(vec, size))
    _print_hits(resp)


def _print_hits(resp: dict) -> None:
    """Pretty-print search hits."""
    hits = resp.get("hits", {}).get("hits", [])
    if not hits:
        print("  No matches.")
        return
    for rank, h in enumerate(hits, start=1):
        src = h.get("_source", {})
        print(f"  {rank}. score={h.get('_score'):.4f}  sku={src.get('sku_id')}  {src.get('product_name') or ''}")
        matched = h.get("inner_hits", {}).get("images", {}).get("hits", {}).get("hits", [])
        if matched:
            print(f"       matched image: {matched[0].get('_source', {}).get('image_url')}")
    print()


async def emit_query(args: argparse.Namespace) -> int:
    """Embed the image/text and print a paste-ready Dashboards Dev Tools query."""
    if not args.image and not args.text:
        print("\n  --emit-query needs --image PATH or --text QUERY.\n")
        return 2
    cfg = OpenSearchIndexer().config
    embedder = OCIEmbedder()
    if args.image:
        vec = await embed_image_file(embedder, args.image)
        if not vec:
            print("  (failed to embed image)")
            return 1
        body = build_image_body(vec, args.size)
        print(f"\n# Image kNN for {args.image!r} — paste into OpenSearch Dashboards → Dev Tools:\n")
        print(f"POST {cfg.index_name}/_search\n{json.dumps(body)}\n")
        return 0
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

        embedder = OCIEmbedder()
        if args.image:
            await search_image(embedder, client, cfg.index_name, args.image, args.size)
        if args.text:
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
