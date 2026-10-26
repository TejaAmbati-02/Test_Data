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

from query_index import OCIEmbedder, OpenSearchIndexer, embed_image_file

CASES_DIR = Path(__file__).parent / "test_images_known"
MIN_COSINE = 0.98
K = 100
PROBE_EF = (500, 2000, 10000)  # ef_search values tried when the k=100 search misses


def _is_vector(v) -> bool:
    return isinstance(v, list) and bool(v) and all(isinstance(x, (int, float)) for x in v)


async def best_stored_cosine(client, index: str, skus: list[str], vec: list[float]) -> tuple[float | None, int]:
    """Highest cosine between `vec` and any image vector stored on the given SKUs, and the vector count.

    Vectors can't be read back from ``_source`` (k-NN derived source leaves a
    placeholder ``1`` there), so this scores server-side with the same nested kNN
    the API uses, filtered to the expected SKUs. A filter that selective makes
    Lucene search exactly rather than approximately. Lucene ``cosinesimil``
    scores are ``(1 + cosine) / 2``.
    """
    filt = {"terms": {"sku_id": skus}}
    resp = await client.search(index=index, body={
        "size": len(skus),
        "_source": ["sku_id"],
        "query": {"nested": {
            "path": "images", "score_mode": "max",
            "query": {"knn": {"images.image_vector": {"vector": vec, "k": max(K, len(skus)), "filter": filt}}},
        }},
    })
    best = None
    for h in resp["hits"]["hits"]:
        sim = 2.0 * h["_score"] - 1.0
        best = sim if best is None else max(best, sim)

    # Stored image count per SKU (nested docs), independent of scoring.
    counts = await client.search(index=index, body={
        "size": len(skus),
        "_source": False,
        "query": {"bool": {"filter": [filt, {"nested": {
            "path": "images", "query": {"match_all": {}},
            "inner_hits": {"size": 0, "_source": False},
        }}]}},
    })
    count = sum(h.get("inner_hits", {}).get("images", {}).get("hits", {}).get("total", {}).get("value", 0)
                for h in counts["hits"]["hits"])
    return best, count


async def top_hit(client, index: str, vec: list[float], ef_search: int | None = None) -> tuple[str | None, float | None]:
    knn = {"vector": vec, "k": K}
    if ef_search:
        # Lucene ignores index.knn.algo_param.ef_search; the per-query override is the only knob.
        knn["method_parameters"] = {"ef_search": ef_search}
    resp = await client.search(index=index, body={
        "size": 1,
        "_source": ["sku_id"],
        "query": {"nested": {
            "path": "images", "score_mode": "max",
            "query": {"knn": {"images.image_vector": knn}},
        }},
    })
    hits = resp["hits"]["hits"]
    return (hits[0]["_source"].get("sku_id"), hits[0]["_score"]) if hits else (None, None)


async def probe_ef(client, index: str, vec: list[float], expected: list[str]) -> str:
    """Retry a missed search with wider ef_search to tell a recall problem from a graph problem."""
    for ef in PROBE_EF:
        try:
            top, _ = await top_hit(client, index, vec, ef_search=ef)
        except Exception as exc:
            return f"ef_search override rejected: {str(exc)[:80]}"
        if top in expected:
            return f"found at ef_search={ef}"
    return f"still missed at ef_search={PROBE_EF[-1]}"


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
            if not _is_vector(vec):
                print(f"  {c['image']:<30} could not embed the test image "
                      f"(got {type(vec).__name__}: {str(vec)[:60]})  FAIL")
                failures += 1
                continue
            expected = c["expected_skus"]
            best, count = await best_stored_cosine(client, index, expected[:200], vec)
            top, score = await top_hit(client, index, vec)
            stored_ok = best is not None and best >= MIN_COSINE
            rank_ok = top in expected
            result = "PASS" if stored_ok and rank_ok else (
                "FAIL stored" if not stored_ok else "FAIL rank (vector OK, kNN missed it)")
            if stored_ok and not rank_ok:
                result += "; " + await probe_ef(client, index, vec, expected)
            failures += not result.startswith("PASS")
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
