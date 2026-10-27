"""Image search quality on real photos, and calibration of IMAGE_MIN_SCORE (run in AVD).

``check_image_search.py`` only proves byte-identical catalogue copies are found.
This measures what users actually get: photos taken of real products (other
angles, lighting, backgrounds, crops). Put the photos in ``test_images_real/``
with a ``labels.json`` in the same format as ``test_images_known/expected.json``:

    {"cases": [
      {"image": "valve_phone_1.jpg",  "expected_skus": ["001300-001-instrumentation-1"]},
      {"image": "random_chair.jpg",   "expected_skus": []}
    ]}

An empty ``expected_skus`` marks a negative: a photo of something not in the
catalogue, which ideally returns nothing. Include some — they set the threshold.

It runs the API's image query (nested kNN, k=100, ef_search=500) and reports:
  * hit@1 / hit@5 / hit@10 and MRR over the positive cases;
  * per-case rank and score of the first correct SKU, and the top wrong score;
  * for candidate IMAGE_MIN_SCORE values, how many correct matches are kept and
    how many negatives are cut, plus a suggested value.

Scores are (1 + cosine) / 2, the same scale as the API's IMAGE_MIN_SCORE.

Usage:
    python eval_image_search.py                         # test_images_real/
    python eval_image_search.py --dir test_images_known # sanity check: all hit@1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from query_index import OCIEmbedder, OpenSearchIndexer, embed_image_file

K = 100
EF_SEARCH = 500  # must match the API's knn_ef_search
TOP_N = 10
THRESHOLDS = [round(0.60 + 0.025 * i, 3) for i in range(15)]  # 0.600 .. 0.950


async def search(client, index: str, vec: list[float]) -> list[tuple[str, float]]:
    """Top TOP_N (sku_id, score) for the API's image-only query."""
    resp = await client.search(index=index, body={
        "size": TOP_N,
        "_source": ["sku_id"],
        "query": {"nested": {
            "path": "images", "score_mode": "max",
            "query": {"knn": {"images.image_vector": {
                "vector": vec, "k": K, "method_parameters": {"ef_search": EF_SEARCH},
            }}},
        }},
    })
    return [(h["_source"].get("sku_id"), h["_score"]) for h in resp["hits"]["hits"]]


def suggest_threshold(pos_scores: list[float], neg_scores: list[float]) -> float | None:
    """Lowest threshold that cuts every negative while keeping every correct top-1 match.

    None when there are no positives or no negatives (nothing to separate), or
    when the two overlap — then a threshold would cost correct matches.
    """
    if not pos_scores or not neg_scores:
        return None
    cutting = [t for t in THRESHOLDS if t <= min(pos_scores) and all(s < t for s in neg_scores)]
    return cutting[0] if cutting else None


async def run(cases_dir: Path, labels: str) -> int:
    cases = json.loads((cases_dir / labels).read_text())["cases"]
    indexer = OpenSearchIndexer()
    index = indexer.config.index_name
    embedder = OCIEmbedder()
    try:
        client = indexer._connect()
        positives: list[dict] = []
        negatives: list[dict] = []
        print(f"\n  {'image':<34} {'rank':>4} {'correct':>8} {'top wrong':>9}  top hit")
        for c in cases:
            vec = await embed_image_file(embedder, str(cases_dir / c["image"]))
            if not vec:
                print(f"  {c['image']:<34} could not embed — skipped")
                continue
            hits = await search(client, index, vec)
            expected = set(c.get("expected_skus") or [])
            rank = next((i for i, (sku, _) in enumerate(hits, 1) if sku in expected), None)
            correct = hits[rank - 1][1] if rank else None
            wrong = next((s for sku, s in hits if sku not in expected), None)
            top = hits[0][0] if hits else None
            row = {"image": c["image"], "rank": rank, "correct": correct, "wrong": wrong,
                   "top_score": hits[0][1] if hits else None}
            (positives if expected else negatives).append(row)
            f = lambda x: f"{x:.4f}" if x is not None else "-"
            label = "negative" if not expected else (str(rank) if rank else f">{TOP_N}")
            print(f"  {c['image']:<34} {label:>4} {f(correct):>8} {f(wrong):>9}  {top}")

        if positives:
            n = len(positives)
            hit = lambda k: sum(1 for p in positives if p["rank"] and p["rank"] <= k) / n
            mrr = sum(1 / p["rank"] for p in positives if p["rank"]) / n
            print(f"\n  Positives: {n}   hit@1 {hit(1):.0%}   hit@5 {hit(5):.0%}   hit@10 {hit(10):.0%}   MRR {mrr:.3f}")
        if negatives:
            print(f"  Negatives: {len(negatives)}   top scores: "
                  + ", ".join(f"{p['top_score']:.3f}" for p in negatives if p["top_score"] is not None))

        top1 = [p["correct"] for p in positives if p["rank"] == 1]
        neg_top = [p["top_score"] for p in negatives if p["top_score"] is not None]
        if positives or negatives:
            print(f"\n  {'IMAGE_MIN_SCORE':>15} {'correct top-1 kept':>19} {'negatives cut':>14}")
            for t in THRESHOLDS:
                kept = sum(s >= t for s in top1)
                cut = sum(s < t for s in neg_top)
                print(f"  {t:>15.3f} {f'{kept}/{len(top1)}':>19} {f'{cut}/{len(neg_top)}':>14}")
            t = suggest_threshold(top1, neg_top)
            if t is not None:
                print(f"\n  Suggested IMAGE_MIN_SCORE={t} (keeps every correct top-1 match, cuts every negative). "
                      "More photos give a safer value.\n")
            elif not top1 or not neg_top:
                print("\n  No threshold suggested — need both correct top-1 matches and negatives.\n")
            else:
                print("\n  No threshold separates them — some negatives score above correct matches. "
                      "Leave IMAGE_MIN_SCORE=0 or accept losing matches (see table).\n")
        return 0
    finally:
        await indexer.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Image search quality on real photos + IMAGE_MIN_SCORE calibration.")
    p.add_argument("--dir", type=Path, default=Path(__file__).parent / "test_images_real")
    p.add_argument("--labels", default=None, help="Labels file in --dir (default: labels.json, else expected.json).")
    a = p.parse_args()
    labels = a.labels or ("labels.json" if (a.dir / "labels.json").exists() else "expected.json")
    if not (a.dir / labels).exists():
        sys.exit(f"No {labels} in {a.dir} — see the docstring at the top of this file for the format.")
    sys.exit(asyncio.run(run(a.dir, labels)))


if __name__ == "__main__":
    main()
