"""Transforms raw CommerceTools product and price payloads into flat SKU rows.

This module takes the raw product documents fetched from CommerceTools together
with their matched prices and flattens them into a dictionary keyed by SKU. Each
SKU row combines product-level fields (title, description and the three-level
category hierarchy — category, subcategory, product type) with variant-level
fields (SKU key/id, primary image) and the selected seller/price (seller id and
name, price, currency, lead time, MOQ, unit type) so that downstream consumers
receive one self-contained record per sellable unit, matching the ingestion
field specification.

The public entry point is :func:`transform`. The remaining functions are private
helpers that parse individual products, variants, categories, attributes and
image URLs. All functions are coroutines so the transform step can be awaited as
part of the asynchronous pipeline, even though most of the work is CPU-bound dict
manipulation.
"""

from .products import Product
from .prices import Price
from .logging_service import logger, log_function


@log_function
async def transform(
    products: list[Product],
    prices: list[Price],
    agreements_by_product: dict[str, list[dict]] | None = None,
) -> dict[str, dict]:
    """Return a dict of sku → parsed row, one entry per SKU (masterVariant + all variants).

    Args:
        products: The products to flatten.
        prices: The standalone prices to match to each SKU.
        agreements_by_product: Optional mapping of product id → list of valid
            purchase agreements (``{contract_id, valid_until}``) to attach to
            every row of that product. When ``None``/absent, rows carry no
            ``purchase_agreements``.
    """
    logger.info(
        f"Transforming {len(products)} products and {len(prices)} prices",
        extra={"function_name": "transform", "input_data": {"products": len(products), "prices": len(prices)}},
    )
    prices_by_sku: dict[str, list[Price]] = {}
    for p in prices:
        prices_by_sku.setdefault(p.sku, []).append(p)

    logger.info(
        f"Price index built — {len(prices_by_sku)} unique SKUs across {len(prices)} prices",
        extra={"function_name": "transform", "input_data": {"unique_skus": len(prices_by_sku)}},
    )

    agreements_by_product = agreements_by_product or {}
    result = {}
    no_price: list[str] = []
    failed_products = 0
    for product in products:
        try:
            rows = await _parse_product(product.raw, prices_by_sku, agreements_by_product.get(product.id))
        except Exception:
            # _parse_product already logged the error with full product context.
            # Skip this product rather than aborting the whole transform so one
            # bad document can't take the entire run down.
            failed_products += 1
            continue
        for sku, row in rows.items():
            result[sku] = row
            if row.get("price") is None:
                no_price.append(sku)

    with_price = len(result) - len(no_price)
    coverage_pct = 100 * with_price / len(result) if result else 0
    logger.info(
        f"Transform complete — {len(result)} SKU rows | price coverage: {with_price}/{len(result)} ({coverage_pct:.1f}%)"
        + (f" | {failed_products} product(s) skipped (parse failed)" if failed_products else ""),
        extra={
            "function_name": "transform",
            "status": "success",
            "input_data": {
                "output_rows": len(result),
                "with_price": with_price,
                "without_price": len(no_price),
                "price_coverage_pct": round(coverage_pct, 1),
                "failed_products": failed_products,
            },
        },
    )
    if no_price:
        logger.warning(
            f"{len(no_price)} SKU(s) have no matched price",
            extra={"function_name": "transform", "input_data": {"skus": no_price[:20]}},
        )
    return result


async def _parse_product(
    raw: dict,
    prices_by_sku: dict[str, list[Price]],
    purchase_agreements: list[dict] | None = None,
) -> dict[str, dict]:
    """Flatten a single raw product document into one row per variant SKU.

    Extracts the product-level fields shared across all variants (title,
    description, the three-level category hierarchy — category, subcategory and
    product type — and any valid purchase agreements), then parses every variant
    (the master variant plus all additional variants) into its own SKU row via
    :func:`_parse_variant`.

    Args:
        raw: The raw product document as returned by CommerceTools.
        prices_by_sku: Mapping of SKU to its list of matched prices, used by the
            variant parser to select the lowest contracted price and its seller.
        purchase_agreements: Optional list of the product's valid purchase
            agreements (``{contract_id, valid_until}``) to attach to every row.

    Returns:
        A dict mapping each variant's SKU to its parsed row. Variants without a
        SKU are skipped.

    Raises:
        Exception: Re-raises any error encountered while parsing the product
            after logging it with the product key for context.
    """
    try:
        product_fields = {
            "product_id": raw.get("id"),
            "product_title": raw.get("name", {}).get("en"),
            "product_description": raw.get("description", {}).get("en"),
            **await _category_fields(raw),
            "purchase_agreements": purchase_agreements or None,
        }

        logger.info(
            f"Extracted product-level fields for id={raw.get('id')!r} ({len(product_fields)} fields)",
            extra={
                "function_name": "_parse_product",
                "input_data": {"id": raw.get("id"), "key": raw.get("key"), "variants": 1 + len(raw.get("variants", []))},
                "response_payload": product_fields,
            },
        )

        all_variants = [raw.get("masterVariant", {}), *raw.get("variants", [])]

        # Contract ids of the product's valid purchase agreements drive seller
        # selection: a SKU's price is matched to one of these contracts.
        contract_ids = {a["contract_id"] for a in (purchase_agreements or []) if a.get("contract_id")}

        rows: dict[str, dict] = {}
        failed_variants = 0
        for v in all_variants:
            try:
                row = await _parse_variant(product_fields, v, prices_by_sku, contract_ids)
            except Exception:
                # _parse_variant already logged the error with sku context. Skip
                # the single bad variant rather than failing the whole product.
                failed_variants += 1
                continue
            sku = row.get("sku_id")
            if sku:
                rows[sku] = row
        if failed_variants:
            logger.warning(
                f"{failed_variants} variant(s) skipped (parse failed) for product key={raw.get('key')!r}",
                extra={
                    "function_name": "_parse_product",
                    "input_data": {"key": raw.get("key"), "failed_variants": failed_variants},
                },
            )
        return rows
    except Exception as exc:
        logger.error(
            f"Failed to parse product key={raw.get('key')!r}: {exc}",
            extra={
                "function_name": "_parse_product",
                "status": "error",
                "error_details": str(exc),
                "input_data": {"id": raw.get("id"), "key": raw.get("key")},
            },
        )
        raise


def _attr(attributes: list, name: str):
    """Return the value of the named variant attribute, or ``None`` if absent.

    Args:
        attributes: A list of attribute dicts shaped like
            ``{"name": ..., "value": ...}``.
        name: The attribute name to look up.

    Returns:
        The matching attribute's ``value``, or ``None`` when not present.
    """
    for a in attributes or []:
        if a.get("name") == name:
            return a.get("value")
    return None


def _select_price(sku_prices: list[Price], contract_ids: set[str]) -> tuple[Price | None, bool]:
    """Select the standalone price that supplies a SKU's seller details.

    Mirrors the spec's standalone-price query
    (``contractId=… AND isBlockedForDeletion!="L" AND sku=…``):

      1. drop prices blocked for deletion (``is_blocked_for_deletion == "L"``);
      2. prefer a usable price whose ``contract_id`` is one of the product's valid
         purchase-agreement contracts — the lowest-amount such price;
      3. otherwise fall back to the lowest ``CONTRACTED`` price, or the lowest
         usable price of any scope.

    Args:
        sku_prices: All standalone prices matched to this SKU.
        contract_ids: Contract ids of the product's valid purchase agreements.

    Returns:
        A ``(price, matched_by_contract)`` tuple. ``price`` is ``None`` when no
        usable price exists; ``matched_by_contract`` is ``True`` when the price
        was chosen by a purchase-agreement contract match.
    """
    usable = [p for p in sku_prices if p.is_blocked_for_deletion != "L"]
    if contract_ids:
        matched = [p for p in usable if p.contract_id in contract_ids]
        if matched:
            return min(matched, key=lambda p: p.amount), True
    contracted = [p for p in usable if p.scope == "CONTRACTED"]
    pool = contracted if contracted else usable
    return min(pool, key=lambda p: p.amount, default=None), False


async def _parse_variant(
    product_fields: dict,
    variant: dict,
    prices_by_sku: dict[str, list[Price]],
    contract_ids: set[str],
) -> dict:
    """Build a single flat SKU row for one product variant.

    Merges the shared product-level fields (title, description, category
    hierarchy, purchase agreements) with the variant's master-variant details
    (SKU key/id, primary image, material numbers) and the selected seller/price.

    Seller selection (:func:`_select_price`) drops prices blocked for deletion
    (deletion code ``"L"``), then prefers a price whose ``contractId`` matches one
    of the product's valid purchase agreements, falling back to the lowest
    ``CONTRACTED`` (else any) price. All seller fields come from that one price.

    Args:
        product_fields: The product-level fields shared by every variant of the
            parent product (produced by :func:`_parse_product`).
        variant: The raw variant document (master variant or additional variant).
        prices_by_sku: Mapping of SKU to its list of matched prices.
        contract_ids: Contract ids of the product's valid purchase agreements.

    Returns:
        A dict representing the fully flattened SKU row for this variant.

    Raises:
        Exception: Re-raises any error encountered while parsing the variant
            after logging it with the SKU for context.
    """
    try:
        sku = variant.get("sku")
        v_attrs = variant.get("attributes", [])
        image_urls = await _image_urls(variant)

        sku_prices = prices_by_sku.get(sku, []) if sku else []
        min_price, matched_contract = _select_price(sku_prices, contract_ids)

        logger.info(
            f"SKU={sku!r} → prices={len(sku_prices)} "
            f"selected={'contract' if matched_contract else 'fallback'} "
            f"contract={min_price.contract_id if min_price else None!r} "
            f"price={f'{min_price.amount} {min_price.currency}' if min_price else 'none'} "
            f"seller={min_price.seller_name_en if min_price else 'none'}",
            extra={
                "request_id": sku,
                "function_name": "_parse_variant",
                "input_data": {
                    "sku": sku,
                    "total_prices": len(sku_prices),
                    "contract_matched": matched_contract,
                    "selected_price": min_price.amount if min_price else None,
                    "currency": min_price.currency if min_price else None,
                    "seller_id": min_price.seller_id if min_price else None,
                    "contract_id": min_price.contract_id if min_price else None,
                },
            },
        )

        row = {
            **product_fields,

            # --- Master-variant / SKU details ---
            "sku_key": variant.get("key"),
            "sku_id": sku,
            "sku_image_url": image_urls[0] if image_urls else None,
            # Full image-URL list is kept for the image-download step (images.py)
            # and downstream display; sku_image_url above is the spec's primary.
            "image_urls": image_urls or None,

            # --- Material numbers ---
            "asmo_material_number": _attr(v_attrs, "ownInventoryManagedProduct"),
            "aramco_material_number": _attr(v_attrs, "manufacturerNumber"),

            # --- Seller / price (all from the one selected standalone price) ---
            "seller_id": min_price.seller_id if min_price else None,
            "seller_name": min_price.seller_name_en if min_price else None,
            "price": min_price.amount if min_price else None,
            "currency_type": min_price.currency if min_price else None,
            "lead_time": min_price.lead_time_days if min_price else None,
            "moq": min_price.moq if min_price else None,
            "unit_type": min_price.unit_type if min_price else None,
            "temporary_unavailable": min_price.temporary_unavailable if min_price else None,
            "is_blocked_for_deletion": min_price.is_blocked_for_deletion if min_price else None,
        }
        # Combined free-text blob of every field value, pipe-separated — useful
        # as a single searchable/embeddable field downstream. Built from the row
        # as assembled above, so it reflects exactly the fields emitted.
        row["content"] = _combine_content(row)
        logger.debug(
            f"Built content field for sku={sku!r} ({len(row['content'])} chars)",
            extra={"request_id": sku, "function_name": "_parse_variant", "input_data": {"sku": sku}},
        )
        logger.info(
            f"Parsed row for sku={sku!r} ({len(row)} fields)",
            extra={
                "request_id": sku,
                "function_name": "_parse_variant",
                "input_data": {"sku": sku, "variantId": variant.get("id")},
                "response_payload": row,
            },
        )
        return row
    except Exception as exc:
        logger.error(
            f"Failed to parse variant sku={variant.get('sku')!r}: {exc}",
            extra={
                "request_id": variant.get("sku"),
                "function_name": "_parse_variant",
                "status": "error",
                "error_details": str(exc),
                "input_data": {"sku": variant.get("sku"), "variantId": variant.get("id")},
            },
        )
        raise


# The three category levels the spec maps, in hierarchy order, and the field
# prefix each is emitted under.
_CATEGORY_LEVELS = ("category", "sub_category", "product_type")

# Empty three-level block, returned when a product has no (expanded) categories.
_EMPTY_CATEGORY_FIELDS = {
    f"{prefix}_{suffix}": None
    for prefix in _CATEGORY_LEVELS
    for suffix in ("name", "id", "slug", "description", "key")
}


def _category_block(obj: dict, prefix: str) -> dict:
    """Build the five spec fields (name/id/slug/description/key) for one level.

    Args:
        obj: The expanded category ``obj`` for this hierarchy level.
        prefix: Field prefix — ``category``, ``sub_category`` or ``product_type``.

    Returns:
        A dict of ``{prefix}_name``, ``{prefix}_id``, ``{prefix}_slug``,
        ``{prefix}_description`` and ``{prefix}_key`` for this level.
    """
    return {
        f"{prefix}_name": (obj.get("name") or {}).get("en"),
        f"{prefix}_id": obj.get("id"),
        f"{prefix}_slug": (obj.get("slug") or {}).get("en"),
        f"{prefix}_description": (obj.get("description") or {}).get("en"),
        f"{prefix}_key": obj.get("key"),
    }


async def _category_fields(raw: dict) -> dict:
    """Derive the three-level category hierarchy from ``categories[0]``.

    The spec maps Category / Subcategory / Product type to successive ancestors
    of the product's first category. In the real data ``ancestors[0]`` is the
    CommerceTools internal "All" root (its own ``ancestors`` list is empty), so
    a literal ``ancestors[0..2]`` mapping would make Category="All". Instead we
    build the hierarchy from root to leaf — the expanded ancestor ``obj``s
    followed by the category itself — and drop the "All" root, then assign the
    first three surviving levels to:

        * ``category_*``      — top business category   (e.g. "Static Equipment")
        * ``sub_category_*``  — second level            (e.g. "Pipe")
        * ``product_type_*``  — third level / leaf      (e.g. "Pipe, Seamless CS")

    Each level contributes ``*_name``, ``*_id``, ``*_slug``, ``*_description``
    and ``*_key``. Missing levels yield ``None`` for all of their fields.

    Args:
        raw: The raw product document (with ``categories[*].ancestors[*]``
            expanded).

    Returns:
        A dict with the fifteen category fields. All values are ``None`` when the
        product has no (expanded) categories.
    """
    logger.debug(
        f"Deriving category fields for key={raw.get('key')!r}",
        extra={
            "function_name": "_category_fields",
            "input_data": {"key": raw.get("key"), "categories": len(raw.get("categories", []))},
        },
    )
    categories = raw.get("categories", [])
    leaf = categories[0].get("obj", {}) if categories else {}
    if not leaf:
        logger.debug(
            f"No expanded category on product key={raw.get('key')!r}; returning empty category fields",
            extra={"function_name": "_category_fields", "input_data": {"key": raw.get("key")}},
        )
        return dict(_EMPTY_CATEGORY_FIELDS)

    # Full hierarchy, root → leaf: each ancestor's expanded obj, then the leaf
    # category itself. Drop the CT internal "All" root (empty own ancestors); the
    # leaf keeps its ancestors so it always survives.
    chain = [a.get("obj", {}) for a in leaf.get("ancestors", [])] + [leaf]
    chain = [c for c in chain if c.get("ancestors")]

    fields = dict(_EMPTY_CATEGORY_FIELDS)
    for prefix, obj in zip(_CATEGORY_LEVELS, chain):
        fields.update(_category_block(obj, prefix))

    logger.debug(
        f"Resolved {min(len(chain), len(_CATEGORY_LEVELS))} category level(s) for key={raw.get('key')!r}",
        extra={
            "function_name": "_category_fields",
            "input_data": {"key": raw.get("key"), "levels": len(chain)},
        },
    )
    return fields


def _stringify(value) -> str | None:
    """Render a single field value as a plain string for the ``content`` blob.

    Flattens the value types that appear in a transformed row so they can be
    concatenated into one searchable string:

        * ``None`` and empty/blank strings -> ``None`` (omitted from the blob).
        * ``bool`` -> ``"True"``/``"False"``.
        * ``list``/``tuple`` -> its non-empty items joined with ``", "``.
        * ``dict`` -> its non-empty values joined with ``", "`` (e.g. a hazard
          rating ``{"code": ..., "value": ...}``).
        * everything else -> ``str(value)`` stripped of surrounding whitespace.

    Args:
        value: The field value to stringify.

    Returns:
        The string form, or ``None`` when the value carries no content.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = [s for s in (_stringify(v) for v in value) if s]
        return ", ".join(parts) if parts else None
    if isinstance(value, dict):
        parts = [s for s in (_stringify(v) for v in value.values()) if s]
        return ", ".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def _combine_content(row: dict) -> str:
    """Combine all of a row's fields into one pipe-separated ``key:value`` string.

    Iterates the row in insertion order and joins each field as ``key:value``
    with `` | `` between fields (e.g. ``productName_en:Acme | price:12.5``).
    ``None`` and blank values are skipped so the blob stays dense. Any
    pre-existing ``content`` key is excluded so the field never folds its own
    previous value back in.

    Args:
        row: The assembled transformed row (its own ``content`` key, if present,
            is ignored).

    Returns:
        A single ``" | "``-separated string of ``key:value`` pairs for every
        meaningful field.
    """
    parts = [
        f"{key}:{text}"
        for key, value in row.items()
        if key != "content"
        for text in (_stringify(value),)
        if text
    ]
    return " | ".join(parts)


async def _image_urls(variant: dict) -> list[str]:
    """Resolve every available image URL for a variant.

    Prefers the variant's ``images`` list, returning the URL of every entry in
    order. When there are no images, falls back to the source URIs of every
    asset tagged ``"image"``. The first element (when present) is the variant's
    primary image, matching the single-URL behaviour callers relied on.

    Args:
        variant: The raw variant document.

    Returns:
        A list of image URLs (possibly empty) in document order.
    """
    urls = [img.get("url") for img in variant.get("images", []) if img.get("url")]
    if urls:
        logger.debug(
            f"{len(urls)} image URL(s) resolved from images for sku={variant.get('sku')!r}",
            extra={"function_name": "_image_urls", "input_data": {"sku": variant.get("sku"), "source": "images", "count": len(urls)}},
        )
        return urls

    asset_urls = [
        uri
        for asset in variant.get("assets", [])
        if "image" in asset.get("tags", [])
        for src in asset.get("sources", [])
        for uri in (src.get("uri"),)
        if uri
    ]
    if asset_urls:
        logger.debug(
            f"{len(asset_urls)} image URL(s) resolved from asset sources for sku={variant.get('sku')!r}",
            extra={"function_name": "_image_urls", "input_data": {"sku": variant.get("sku"), "source": "assets", "count": len(asset_urls)}},
        )
        return asset_urls

    logger.debug(
        f"No image URL found for sku={variant.get('sku')!r}",
        extra={"function_name": "_image_urls", "input_data": {"sku": variant.get("sku")}},
    )
    return []
