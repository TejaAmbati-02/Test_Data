"""Known-answer image search tests (run this in AVD).

Every image in ``test_images_known/`` is a byte-identical copy of a product image
that the ingest embedded, and ``expected.json`` names the SKU(s) it belongs to.
For each case this checks:

  * STORED — the expected SKU has a stored image vector whose exact cosine
    similarity to the test image is >= 0.98 (a copy should be ~1.0). Failing
    means the ingest dropped the image or stored another image's vector.
  * RANK   — a nested kNN (k=100, as the API uses) ranks an expected SKU #1.
    Failing while STORED passes means approximate search missed the match.

Usage:
    python check_image_search.py                      # all cases
    python check_image_search.py --only 172000-001    # cases whose file name contains this
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from query_index import OCIEmbedder, OpenSearchIndexer, _cosine, embed_image_file

CASES_DIR = Path(__file__).parent / "test_images_known"
MIN_COSINE = 0.98
K = 100


async def best_stored_cosine(client, index: str, skus: list[str], vec: list[float]) -> tuple[float | None, int]:
    """Highest cosine between `vec` and any image vector stored on the given SKUs, and the vector count."""
    resp = await client.search(index=index, body={
        "size": len(skus),
        "_source": ["sku_id", "images"],
        "query": {"terms": {"sku_id": skus}},
    })
    best, count = None, 0
    for h in resp["hits"]["hits"]:
        for img in h["_source"].get("images") or []:
            v = img.get("image_vector")
            if v:
                count += 1
                sim = _cosine(vec, v)
                best = sim if best is None else max(best, sim)
    return best, count


async def top_hit(client, index: str, vec: list[float]) -> tuple[str | None, float | None]:
    resp = await client.search(index=index, body={
        "size": 1,
        "_source": ["sku_id"],
        "query": {"nested": {
            "path": "images", "score_mode": "max",
            "query": {"knn": {"images.image_vector": {"vector": vec, "k": K}}},
        }},
    })
    hits = resp["hits"]["hits"]
    return (hits[0]["_source"].get("sku_id"), hits[0]["_score"]) if hits else (None, None)


async def run(only: str | None) -> int:
    cases = json.loads((CASES_DIR / "expected.json").read_text())["cases"]
    if only:
        cases = [c for c in cases if only in c["image"]]
    indexer = OpenSearchIndexer()
    index = indexer.config.index_name
    embedder = OCIEmbedder()
    failures = 0
    try:
        client = indexer._connect()
        print(f"\n  {'image':<30} {'stored vecs':>11} {'best cos':>9} {'top hit':<30} {'score':>7}  result")
        for c in cases:
            vec = await embed_image_file(embedder, str(CASES_DIR / c["image"]))
            if not vec:
                print(f"  {c['image']:<30} could not embed the test image  FAIL")
                failures += 1
                continue
            expected = c["expected_skus"]
            best, count = await best_stored_cosine(client, index, expected[:200], vec)
            top, score = await top_hit(client, index, vec)
            stored_ok = best is not None and best >= MIN_COSINE
            rank_ok = top in expected
            result = "PASS" if stored_ok and rank_ok else (
                "FAIL stored" if not stored_ok else "FAIL rank (vector OK, kNN missed it)")
            failures += result != "PASS"
            cos = f"{best:.4f}" if best is not None else "-"
            sc = f"{score:.4f}" if score is not None else "-"
            print(f"  {c['image']:<30} {count:>11} {cos:>9} {str(top):<30} {sc:>7}  {result}")
        print(f"\n  {len(cases) - failures}/{len(cases)} passed\n")
        return 0 if failures == 0 else 1
    finally:
        await indexer.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Known-answer image search tests.")
    p.add_argument("--only", help="Run only cases whose image file name contains this text.")
    sys.exit(asyncio.run(run(p.parse_args().only)))


if __name__ == "__main__":
    main()
