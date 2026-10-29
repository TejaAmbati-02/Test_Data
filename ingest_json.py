"""Standalone JSON → OpenSearch ingester (run this in AVD).

Reads the transformed per-SKU JSON files produced by a *local* pipeline run
(default ``out_transformed/``) and ingests them into the configured OpenSearch
index — embedding each row's ``content`` with OCI GenAI and bulk-indexing every
field (plus the ``content_vector`` and any nested ``images`` vectors the rows
already carry).

This is the counterpart to running ``main.py`` locally: produce the JSON on your
laptop, copy ``out_transformed/`` into AVD (where the cluster-internal
OpenSearch host is reachable), then run this script there.

It performs explicit pre-flight checks and fails fast with a clear message and a
non-zero exit code rather than erroring mid-run:

  1. the input directory exists and contains JSON files;
  2. each file parses and (optionally) carries a ``content`` field;
  3. the OpenSearch cluster is reachable (ping);
  4. the index exists / is created, and its vector dimension matches the
     embedder (``ensure_index`` — use ``--recreate`` to drop and rebuild).

Usage:
    python ingest_json.py [INPUT_DIR] [options]

    python ingest_json.py                      # index out_transformed/
    python ingest_json.py out_transformed      # explicit directory
    python ingest_json.py --recreate           # drop + rebuild the index
    python ingest_json.py --dry-run            # checks only, no indexing
    python ingest_json.py --limit 50           # index only the first 50

Exit codes:
    0 success (all rows indexed)        3 index dimension mismatch
    1 some documents failed to index    4 OpenSearch unreachable
    2 nothing to ingest / bad input     5 unexpected error

Configuration (OpenSearch + OCI) is read from environment variables / .env;
see asmo/search_index.py and asmo/embeddings.py.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
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

from asmo.embeddings import OCIEmbedder  # noqa: E402
from asmo.images import shared_image_urls  # noqa: E402
from asmo.logging_service import setup_logging, logger  # noqa: E402
from asmo.search_index import OpenSearchIndexer  # noqa: E402

setup_logging()

DEFAULT_DIR = Path("out_transformed")

# Exit codes (see module docstring).
EXIT_OK = 0
EXIT_DOC_FAILURES = 1
EXIT_BAD_INPUT = 2
EXIT_DIM_MISMATCH = 3
EXIT_UNREACHABLE = 4
EXIT_ERROR = 5


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="ingest_json.py",
        description="Read transformed per-SKU JSON files and ingest them into OpenSearch.",
    )
    parser.add_argument(
        "input_dir", nargs="?", default=str(DEFAULT_DIR),
        help=f"Directory of per-SKU *.json files (default: {DEFAULT_DIR}/).",
    )
    parser.add_argument(
        "--pattern", default="*.json",
        help="Glob for JSON files within the input directory (default: *.json).",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and recreate the index before indexing (DESTRUCTIVE — all existing docs are lost).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Index only the first N rows (after sorting by filename) — useful for a smoke test.",
    )
    parser.add_argument(
        "--require-content", action="store_true",
        help="Skip rows that have no (or empty) 'content' field instead of indexing them.",
    )
    parser.add_argument(
        "--no-image-embeddings", action="store_true",
        help="Do not embed the base64 'image_content'; strip it and index text/content only.",
    )
    parser.add_argument(
        "--max-products-per-image", type=int, default=0, metavar="N",
        help="Do not embed an image whose URL is shared by more than N distinct products "
             "(placeholder / stock photos); 1 is right for a real catalogue. 0 embeds every image (default).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run every check (load, ping, index presence) but do not embed or index anything.",
    )
    return parser.parse_args(argv)


def load_transformed(folder: Path, pattern: str, require_content: bool, limit: int | None) -> dict[str, dict]:
    """Load transformed rows from a folder of per-SKU JSON files.

    Each matched file is one transformed row, keyed by its ``sku_id`` field
    (falling back to the file stem). Unreadable/invalid files are skipped and logged, not
    fatal. Rows without ``content`` are skipped only when ``require_content`` is
    set (they would index with an empty text vector otherwise).

    Args:
        folder: Directory containing the transformed JSON files.
        pattern: Glob selecting the files within ``folder``.
        require_content: When True, drop rows whose ``content`` is missing/blank.
        limit: When set, stop after this many successfully loaded rows.

    Returns:
        Mapping of SKU to transformed row.
    """
    rows: dict[str, dict] = {}
    files = sorted(folder.glob(pattern))
    skipped_parse = skipped_no_content = 0
    with_images = 0

    for path in files:
        if limit is not None and len(rows) >= limit:
            break
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            skipped_parse += 1
            logger.error(
                f"Skipping '{path.name}': {exc}",
                extra={"function_name": "load_transformed", "status": "error", "error_details": str(exc)},
            )
            continue
        if not isinstance(row, dict):
            skipped_parse += 1
            logger.error(
                f"Skipping '{path.name}': top-level JSON is {type(row).__name__}, expected object",
                extra={"function_name": "load_transformed", "status": "error"},
            )
            continue
        if require_content and not (row.get("content") or "").strip():
            skipped_no_content += 1
            continue
        if row.get("images"):
            with_images += 1
        key = row.get("sku_id") or path.stem
        rows[key] = row

    logger.info(
        f"Loaded {len(rows)} row(s) from {folder}/ (pattern={pattern!r}) — "
        f"{with_images} with nested image vectors; "
        f"skipped {skipped_parse} unreadable, {skipped_no_content} without content",
        extra={
            "function_name": "load_transformed",
            "input_data": {
                "count": len(rows), "folder": str(folder), "with_images": with_images,
                "skipped_parse": skipped_parse, "skipped_no_content": skipped_no_content,
            },
        },
    )
    return rows


def apply_frontend_schema(rows: dict[str, dict]) -> None:
    """Add the frontend-contract fields to every row, in place.

    The transformer now writes these itself; this backfills JSON files produced
    before that change (it is a no-op on newer files). The search API returns
    these keys as-is, so they are indexed alongside the transformer's own fields:

      * ``sku_name`` — the product title (copy of ``product_name``);
      * ``temporary_unavailable`` — the primary seller's flag
        (``seller_details[0]``, the cheapest offer, like ``price`` / ``moq`` /
        ``sold_by``); ``None`` when the SKU has no seller offers;
      * ``contract_indicator`` — coerced to a real boolean (the transformer
        writes ``""`` as a placeholder until a source is agreed).
    """
    for row in rows.values():
        row["sku_name"] = row.get("product_name")
        sellers = row.get("seller_details") or []
        row["temporary_unavailable"] = sellers[0].get("temporary_unavailable") if sellers else None
        row["contract_indicator"] = bool(row.get("contract_indicator"))


def _supported_image_mime(data: bytes) -> str | None:
    """Return the MIME type if the bytes are an OCI-supported image, else None.

    OCI's multimodal embed only accepts PNG, JPEG, WebP and GIF. Anything else
    — BMP/TIFF/SVG, or a non-image payload such as an HTML error page a server
    returned with a 200 status — is detected by magic bytes and returned as
    ``None`` so the caller can skip it instead of sending it (which 400s the
    whole embed request).
    """
    if data[:8].startswith(b"\x89PNG"):
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


async def _embed_images_resilient(embedder: OCIEmbedder, images: list[tuple[bytes, str]]) -> tuple[list[list[float] | None], int]:
    """Embed images tolerantly: a single bad image never fails the whole set.

    Images are embedded in small batches; if a batch is rejected (e.g. OCI 400
    on one corrupt/unsupported image that slipped past the magic-byte check),
    that batch is retried one image at a time so the good images still get
    vectors and only the bad ones are dropped.

    Args:
        embedder: The OCI embedder.
        images: ``(bytes, mime)`` tuples to embed.

    Returns:
        ``(vectors, failed)`` — ``vectors[i]`` is the embedding for ``images[i]``
        or ``None`` when that image could not be embedded; ``failed`` counts the
        ``None`` entries.
    """
    chunk = embedder.image_batch
    vectors: list[list[float] | None] = [None] * len(images)
    failed = 0
    for start in range(0, len(images), chunk):
        batch = images[start:start + chunk]
        try:
            vecs = await embedder.embed_images(batch)
            for j, vec in enumerate(vecs):
                vectors[start + j] = vec
        except Exception as exc:
            logger.warning(
                f"Image embed batch of {len(batch)} failed ({exc}); retrying per-image",
                extra={"function_name": "_embed_images_resilient", "status": "error", "error_details": str(exc)},
            )
            for j, img in enumerate(batch):
                try:
                    one = await embedder.embed_images([img])
                    vectors[start + j] = one[0] if one else None
                    if not one:
                        failed += 1
                except Exception as exc2:
                    failed += 1
                    logger.warning(
                        f"Skipping unembeddable image: {exc2}",
                        extra={"function_name": "_embed_images_resilient", "status": "error", "error_details": str(exc2)},
                    )
    return vectors, failed


async def attach_image_embeddings(
    rows: dict[str, dict], embedder: OCIEmbedder, enabled: bool, max_products_per_image: int | None = None,
) -> dict:
    """Embed each row's base64 ``image_content`` and attach nested ``images``.

    For every row that carries an ``image_content`` list (base64 images written
    by a local pipeline run with ``IMAGE_CONTENT_LOCAL=true``) and does not
    already have pre-computed ``images`` vectors, every image is decoded and
    embedded via OCI multimodal ``embed-v4.0``. The resulting vectors are
    attached as the nested ``images`` list — one ``{image_url, image_vector}``
    per image — pairing each image with the matching entry in ``image_urls``
    by position.

    The raw base64 ``image_content`` is **always removed** from every row before
    returning, whether or not embedding ran, so it is never indexed into
    OpenSearch (it is large and not in the mapping).

    Args:
        rows: Mapping of SKU to row; mutated in place.
        embedder: The OCI embedder used to vectorise images.
        enabled: When False, skip embedding entirely (only strip
            ``image_content``) — e.g. for a text-only ingest.
        max_products_per_image: Images whose URL is shared by more than this
            many distinct products are placeholders: they are not embedded, and
            are removed from any pre-computed ``images`` list. ``None`` keeps all.

    Returns:
        A summary dict with ``rows_with_content``, ``images_embedded`` and
        ``decode_failed`` counts.
    """
    # (sku, bytes, mime, base64, url) for every image awaiting a vector.
    work: list[tuple[str, bytes, str, str, str | None]] = []
    decode_failed = 0
    unsupported = 0
    rows_with_content = 0
    placeholder_skipped = 0
    placeholder_urls = shared_image_urls(rows, max_products_per_image)
    if placeholder_urls:
        logger.info(
            f"Not embedding {len(placeholder_urls)} placeholder image URL(s) shared by more than "
            f"{max_products_per_image} product(s)",
            extra={"function_name": "attach_image_embeddings", "input_data": {"placeholder_urls": len(placeholder_urls)}},
        )

    if enabled:
        for sku, row in rows.items():
            if row.get("images"):  # already carries vectors — only drop placeholders
                kept = [img for img in row["images"] if img.get("image_url") not in placeholder_urls]
                placeholder_skipped += len(row["images"]) - len(kept)
                if kept:
                    row["images"] = kept
                else:
                    row.pop("images")
                continue
            b64_list = row.get("image_content") or []
            if not b64_list:
                continue
            rows_with_content += 1
            urls = row.get("image_urls") or []
            for i, b64 in enumerate(b64_list):
                try:
                    data = base64.b64decode(b64, validate=True)
                except (binascii.Error, ValueError):
                    decode_failed += 1
                    continue
                mime = _supported_image_mime(data)
                if mime is None:
                    # Not PNG/JPEG/WebP/GIF (e.g. BMP/TIFF/SVG or a non-image
                    # error body). OCI would 400 on it, so skip it entirely.
                    unsupported += 1
                    logger.warning(
                        f"Skipping unsupported image format for sku={sku!r} (image {i})",
                        extra={"request_id": sku, "function_name": "attach_image_embeddings",
                               "input_data": {"sku": sku, "index": i, "head": data[:8].hex()}},
                    )
                    continue
                url = urls[i] if i < len(urls) else None
                if url in placeholder_urls:
                    placeholder_skipped += 1
                    continue
                work.append((sku, data, mime, b64, url))

    images_embedded = 0
    embed_failed = 0
    if work:
        logger.info(
            f"Embedding {len(work)} supported image(s) from image_content across {rows_with_content} row(s)",
            extra={"function_name": "attach_image_embeddings", "input_data": {"images": len(work), "rows": rows_with_content}},
        )
        vectors, embed_failed = await _embed_images_resilient(
            embedder, [(data, mime) for _sku, data, mime, _id, _url in work]
        )
        images_by_sku: dict[str, list[dict]] = {}
        for (sku, _data, _mime, _b64, url), vector in zip(work, vectors):
            if vector is None:  # dropped by the resilient embedder — skip it
                continue
            images_by_sku.setdefault(sku, []).append({"image_url": url, "image_vector": vector})
            images_embedded += 1
        for sku, obj in images_by_sku.items():
            rows[sku]["images"] = obj

    # Never index the base64 blob — strip it from every row regardless.
    for row in rows.values():
        row.pop("image_content", None)

    logger.info(
        f"Image embedding done — {images_embedded} vector(s) from {rows_with_content} row(s); "
        f"{unsupported} unsupported, {decode_failed} decode failure(s), {embed_failed} embed failure(s), "
        f"{placeholder_skipped} placeholder image(s) skipped"
        + ("" if enabled else " (disabled — image_content stripped, not embedded)"),
        extra={
            "function_name": "attach_image_embeddings",
            "status": "success",
            "input_data": {"rows_with_content": rows_with_content, "images_embedded": images_embedded,
                           "unsupported": unsupported, "decode_failed": decode_failed, "embed_failed": embed_failed,
                           "placeholder_skipped": placeholder_skipped},
        },
    )
    return {"rows_with_content": rows_with_content, "images_embedded": images_embedded,
            "unsupported": unsupported, "decode_failed": decode_failed, "embed_failed": embed_failed,
            "placeholder_skipped": placeholder_skipped}


async def run(args: argparse.Namespace) -> int:
    """Execute the ingest with all pre-flight checks; return a process exit code."""
    folder = Path(args.input_dir)

    # 1) Input directory must exist and hold JSON files.
    if not folder.is_dir():
        logger.error(f"Input directory not found: {folder}", extra={"function_name": "run", "status": "error"})
        return EXIT_BAD_INPUT

    transformed = load_transformed(folder, args.pattern, args.require_content, args.limit)
    if not transformed:
        logger.warning(
            f"Nothing to ingest — no usable rows in {folder}/ (pattern={args.pattern!r})",
            extra={"function_name": "run"},
        )
        return EXIT_BAD_INPUT
    apply_frontend_schema(transformed)

    # One embedder shared by image embedding (below) and content embedding
    # (inside index_rows), so a single OCI client is reused.
    embedder = OCIEmbedder()
    indexer = OpenSearchIndexer(embedder=embedder)
    cfg = indexer.config
    try:
        # 2) Cluster reachability — fail fast with a clear message.
        logger.info(
            f"Checking OpenSearch at https://{cfg.host}:{cfg.port} (index={cfg.index_name})",
            extra={"function_name": "run", "input_data": {"host": cfg.host, "port": cfg.port, "index": cfg.index_name}},
        )
        if not await indexer.ping():
            logger.error(
                f"OpenSearch is not reachable at https://{cfg.host}:{cfg.port} — "
                f"check you are on the AVD/cluster network and OPENSEARCH_* settings.",
                extra={"function_name": "run", "status": "error"},
            )
            return EXIT_UNREACHABLE

        existed = await indexer.index_exists()
        logger.info(
            f"Index '{cfg.index_name}' {'exists' if existed else 'does not exist yet'}"
            + (" — will be recreated" if (existed and args.recreate) else ""),
            extra={"function_name": "run", "input_data": {"exists": existed, "recreate": args.recreate}},
        )

        # 3) Ensure the index (presence + vector-dimension check). ensure_index
        #    raises RuntimeError on a dimension mismatch when not recreating.
        try:
            created = await indexer.ensure_index(recreate=args.recreate)
        except RuntimeError as exc:
            logger.error(str(exc), extra={"function_name": "run", "status": "error", "error_details": str(exc)})
            return EXIT_DIM_MISMATCH
        if not created and not args.dry_run:
            # An index built before sku_name / contracted_scope joined the
            # mapping gets them added here, so no --recreate is needed.
            await indexer.add_missing_fields()
        logger.info(
            f"Index ready — {'created' if created else 'reused existing'} '{cfg.index_name}'",
            extra={"function_name": "run", "input_data": {"created": created}},
        )

        if args.dry_run:
            img_rows = sum(1 for r in transformed.values() if r.get("image_content"))
            img_count = sum(len(r.get("image_content") or []) for r in transformed.values())
            logger.info(
                f"Dry run — {len(transformed)} row(s) ready for '{cfg.index_name}'. "
                f"{img_rows} row(s) carry image_content ({img_count} image(s)) that "
                + ("would be embedded" if not args.no_image_embeddings else "would be stripped (image embeddings disabled)")
                + ". No embedding or indexing performed.",
                extra={"function_name": "run", "status": "success",
                       "input_data": {"rows": len(transformed), "image_rows": img_rows, "images": img_count}},
            )
            return EXIT_OK

        # 4) Embed each row's base64 image_content into nested images vectors
        #    (and always strip the base64 so it is never indexed).
        await attach_image_embeddings(
            transformed, embedder, enabled=not args.no_image_embeddings,
            max_products_per_image=args.max_products_per_image or None,
        )

        # 5) Embed content + bulk-index (nested image vectors ride along on rows).
        summary = await indexer.index_rows(transformed)
        failed = summary.get("failed", 0)
        logger.info(
            f"Ingest finished — {summary.get('indexed', 0)} indexed, {failed} failed "
            f"into '{cfg.index_name}'",
            extra={"function_name": "run", "status": "success" if not failed else "error", "input_data": summary},
        )
        return EXIT_OK if not failed else EXIT_DOC_FAILURES
    except Exception as exc:
        logger.error(
            f"Ingest failed: {exc}",
            extra={"function_name": "run", "status": "error", "error_details": str(exc)},
        )
        return EXIT_ERROR
    finally:
        await indexer.close()


def main() -> None:
    args = parse_args(sys.argv[1:])
    logger.info(
        f"JSON → OpenSearch ingester starting — source={args.input_dir}"
        + (" [dry-run]" if args.dry_run else "")
        + (" [recreate]" if args.recreate else ""),
        extra={"function_name": "main"},
    )
    code = asyncio.run(run(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
