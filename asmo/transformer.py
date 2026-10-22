"""Transforms raw CommerceTools product and price payloads into flat SKU rows.

This module takes the raw product documents fetched from CommerceTools together
with their matched prices and flattens them into a dictionary keyed by SKU. Each
SKU row combines product-level fields (category, name, hazard ratings, custom
attributes) with variant-level fields (availability, image, lowest contracted
price, seller, lead time) so that downstream consumers receive one self-contained
record per sellable unit.

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
async def transform(products: list[Product], prices: list[Price], channel_id: str) -> dict[str, dict]:
    """Return a dict of sku → parsed row, one entry per SKU (masterVariant + all variants)."""
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

    result = {}
    no_price: list[str] = []
    failed_products = 0
    for product in products:
        try:
            rows = await _parse_product(product.raw, prices_by_sku, channel_id)
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


async def _parse_product(raw: dict, prices_by_sku: dict[str, list[Price]], channel_id: str) -> dict[str, dict]:
    """Flatten a single raw product document into one row per variant SKU.

    Extracts the product-level fields that are shared across all variants
    (category hierarchy, localized name/description/slug, custom attributes and
    hazard ratings), then computes product-wide availability figures and parses
    every variant (the master variant plus all additional variants) into its own
    row via :func:`_parse_variant`.

    Args:
        raw: The raw product document as returned by CommerceTools.
        prices_by_sku: Mapping of SKU to its list of matched prices, used by the
            variant parser to select the lowest contracted price.
        channel_id: The supply/inventory channel whose availability should be
            reported for each variant.

    Returns:
        A dict mapping each variant's SKU to its parsed row. Variants without a
        SKU are skipped.

    Raises:
        Exception: Re-raises any error encountered while parsing the product
            after logging it with the product key for context.
    """
    try:
        product_fields = {
            **await _category_fields(raw),
            "product_key": raw.get("key"),
            "product_name_en": raw.get("name", {}).get("en"),
            "product_name_de": raw.get("name", {}).get("de"),
            "product_description_en": raw.get("description", {}).get("en"),
            "product_description_de": raw.get("description", {}).get("de"),
            "product_slug": raw.get("slug", {}).get("en"),
            "product_group": await _attr(raw.get("attributes", []), "productGroup"),
            "material_type": await _attr(raw.get("attributes", []), "materialType"),
            "customer_id": await _attr(raw.get("attributes", []), "ZCustId"),
            "customer_material_number": await _attr(raw.get("attributes", []), "zCustMAT_9M"),
            "chemical_hazard_bulletin": await _attr(raw.get("attributes", []), "chemicalHazardBulletin"),
            "hazard_health": await _hazard(raw.get("attributes", []), "hazardRatingHealthChar", "hazardRatingHealthValue"),
            "hazard_flame": await _hazard(raw.get("attributes", []), "hazardRatingFlameChar", "hazardRatingFlameValue"),
            "hazard_reactive": await _hazard(raw.get("attributes", []), "hazardRatingReactiveChar", "hazardRatingReactiveValue"),
        }

        logger.info(
            f"Extracted product-level fields for key={raw.get('key')!r} ({len(product_fields)} fields)",
            extra={
                "function_name": "_parse_product",
                "input_data": {"id": raw.get("id"), "key": raw.get("key"), "variants": 1 + len(raw.get("variants", []))},
                "response_payload": product_fields,
            },
        )

        all_variants = [raw.get("masterVariant", {}), *raw.get("variants", [])]

        total_qty = 0
        for v in all_variants:
            avail = await _channel_avail(v, channel_id)
            total_qty += avail.get("availableQuantity") or 0

        in_stock_any = False
        for v in all_variants:
            for ch in v.get("availability", {}).get("channels", {}).values():
                if ch.get("isOnStock"):
                    in_stock_any = True
                    break

        rows: dict[str, dict] = {}
        failed_variants = 0
        for v in all_variants:
            try:
                row = await _parse_variant(product_fields, v, prices_by_sku, channel_id, total_qty, in_stock_any)
            except Exception:
                # _parse_variant already logged the error with sku context. Skip
                # the single bad variant rather than failing the whole product.
                failed_variants += 1
                continue
            sku = row.get("sku")
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


async def _parse_variant(
    product_fields: dict,
    variant: dict,
    prices_by_sku: dict[str, list[Price]],
    channel_id: str,
    total_qty: int,
    in_stock_any: bool,
) -> dict:
    """Build a single flat row for one product variant.

    Merges the shared product-level fields with variant-specific values:
    identifiers (SKU, variant id/key, manufacturer numbers), image URL,
    channel availability, and the selected price. Price selection prefers
    ``CONTRACTED`` prices when any exist and otherwise falls back to the full
    price pool; the lowest-amount price in the chosen pool is used.

    Args:
        product_fields: The product-level fields shared by every variant of the
            parent product (produced by :func:`_parse_product`).
        variant: The raw variant document (master variant or additional variant).
        prices_by_sku: Mapping of SKU to its list of matched prices.
        channel_id: The supply/inventory channel whose availability is reported.
        total_qty: Total available quantity summed across all variants of the
            parent product.
        in_stock_any: Whether any channel of any variant of the parent product is
            currently in stock.

    Returns:
        A dict representing the fully flattened row for this variant.

    Raises:
        Exception: Re-raises any error encountered while parsing the variant
            after logging it with the SKU for context.
    """
    try:
        sku = variant.get("sku")
        avail = await _channel_avail(variant, channel_id)
        v_attrs = variant.get("attributes", [])
        image_urls = await _image_urls(variant)

        sku_prices = prices_by_sku.get(sku, []) if sku else []
        contracted = [p for p in sku_prices if p.scope == "CONTRACTED"]
        pool = contracted if contracted else sku_prices
        min_price = min(pool, key=lambda p: p.amount, default=None)

        logger.info(
            f"SKU={sku!r} → total_prices={len(sku_prices)} contracted={len(contracted)} "
            f"selected={'CONTRACTED' if contracted else 'pool'} "
            f"price={f'{min_price.amount} {min_price.currency}' if min_price else 'none'} "
            f"qty={total_qty} in_stock={in_stock_any}",
            extra={
                "request_id": sku,
                "function_name": "_parse_variant",
                "input_data": {
                    "sku": sku,
                    "total_prices": len(sku_prices),
                    "contracted_prices": len(contracted),
                    "selected_price": min_price.amount if min_price else None,
                    "currency": min_price.currency if min_price else None,
                    "total_qty": total_qty,
                    "in_stock": in_stock_any,
                },
            },
        )

        row = {
            **product_fields,
            "sku": sku,
            "variant_id": variant.get("id"),
            "variant_key": variant.get("key"),
            "manufacturer_number": await _attr(v_attrs, "manufacturerNumber"),
            "product_manufacturer_number": await _attr(v_attrs, "productManufacturerNumber"),
            "erp_material_number": await _attr(v_attrs, "ownInventoryManagedProduct"),
            "image_urls": image_urls or None,
            "available_quantity": avail.get("availableQuantity"),
            "restockable_in_days": avail.get("restockableInDays"),
            "in_stock_any_channel": in_stock_any,
            "total_available_quantity": total_qty,
            "currency_type": min_price.currency if min_price else None,
            "price": min_price.amount if min_price else None,
            "lowest_contracted_price": min_price.amount if min_price else None,
            "seller_name_en": min_price.seller_name_en if min_price else None,
            "seller_name_de": min_price.seller_name_de if min_price else None,
            "contract_ids": [p.contract_id for p in contracted if p.contract_id] or None,
            "leadTimeDays": min_price.lead_time_days if min_price else None,
            "unitOfMeasure": next((p.price_unit for p in pool if p.price_unit is not None), None),
            "moq": next((p.moq for p in pool if p.moq is not None), None),
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


async def _category_fields(raw: dict) -> dict:
    """Derive root and sub category fields from a product's category references.

    Each category reference carries an expanded ``obj`` describing the
    subcategory. The business root is the first ancestor that itself has
    ancestors, which skips the CommerceTools internal "All" root (whose
    ``ancestors`` list is empty). Roots and subcategories are deduplicated by
    key, and for each field a single value is returned when only one distinct
    value exists, otherwise a list of values is returned.

    Args:
        raw: The raw product document.

    Returns:
        A dict with localized name/key/slug (and order hint / description) for
        both the root and sub categories. All values are ``None`` when the
        product has no categories.
    """
    logger.debug(
        f"Deriving category fields for key={raw.get('key')!r}",
        extra={
            "function_name": "_category_fields",
            "input_data": {"key": raw.get("key"), "categories": len(raw.get("categories", []))},
        },
    )
    categories = raw.get("categories", [])
    if not categories:
        logger.debug(
            f"No categories on product key={raw.get('key')!r}; returning empty category fields",
            extra={"function_name": "_category_fields", "input_data": {"key": raw.get("key")}},
        )
        return {
            "category_name": None, "category_key": None,
            "category_slug": None, "category_order_hint": None,
            "sub_category_name": None, "sub_category_key": None,
            "sub_category_slug": None, "sub_category_description": None,
        }

    root_seen: set[str] = set()
    sub_seen: set[str] = set()
    roots: list[dict] = []
    subs: list[dict] = []

    for cat in categories:
        sub = cat.get("obj", {})
        sub_key = sub.get("key")

        # Collect subcategory (deduplicated by key)
        if sub_key and sub_key not in sub_seen:
            sub_seen.add(sub_key)
            subs.append(sub)

        # Skip CT internal "All" root (obj.ancestors is empty); first with ancestors is business root
        ancestors = sub.get("ancestors", [])
        business_ancestors = [a for a in ancestors if a.get("obj", {}).get("ancestors")]
        if business_ancestors:
            root = business_ancestors[0].get("obj", {})
            root_key = root.get("key")
            if root_key and root_key not in root_seen:
                root_seen.add(root_key)
                roots.append(root)

    def _pick(items: list[dict], field: str, nested: str | None = None):
        values = [
            (item.get(field, {}).get(nested) if nested else item.get(field))
            for item in items
        ]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return values[0] if len(values) == 1 else values

    logger.debug(
        f"Resolved {len(roots)} root and {len(subs)} sub categories for key={raw.get('key')!r}",
        extra={
            "function_name": "_category_fields",
            "input_data": {"key": raw.get("key"), "roots": len(roots), "subs": len(subs)},
        },
    )
    return {
        "category_name": _pick(roots, "name", "en"),
        "category_key": _pick(roots, "key"),
        "category_slug": _pick(roots, "slug", "en"),
        "category_order_hint": await _to_float(_pick(roots, "orderHint")),
        "sub_category_name": _pick(subs, "name", "en"),
        "sub_category_key": _pick(subs, "key"),
        "sub_category_slug": _pick(subs, "slug", "en"),
        "sub_category_description": _pick([{**s, "desc": (s.get("description") or {}).get("en")} for s in subs], "desc"),
    }


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


async def _attr(attributes: list, name: str):
    """Return the value of the named attribute, or ``None`` if absent.

    Args:
        attributes: A list of CommerceTools attribute dicts, each shaped like
            ``{"name": ..., "value": ...}``. A falsy value is treated as empty.
        name: The attribute name to look up.

    Returns:
        The attribute's ``value`` if a matching attribute is found, otherwise
        ``None``.
    """
    for a in attributes or []:
        if a.get("name") == name:
            logger.debug(
                f"Attribute {name!r} found",
                extra={"function_name": "_attr", "input_data": {"name": name}},
            )
            return a.get("value")
    logger.debug(
        f"Attribute {name!r} not found",
        extra={"function_name": "_attr", "input_data": {"name": name}},
    )
    return None


async def _hazard(attrs: list, char_key: str, value_key: str) -> dict | None:
    """Combine a hazard rating's code and value attributes into one dict.

    Args:
        attrs: The list of attribute dicts to search.
        char_key: Attribute name holding the hazard code/character.
        value_key: Attribute name holding the hazard value.

    Returns:
        A dict ``{"code": ..., "value": ...}`` when either attribute is present,
        or ``None`` when both are absent.
    """
    char = await _attr(attrs, char_key)
    val = await _attr(attrs, value_key)
    if char is None and val is None:
        logger.debug(
            f"No hazard rating for {char_key!r}/{value_key!r}",
            extra={"function_name": "_hazard", "input_data": {"char_key": char_key, "value_key": value_key}},
        )
        return None
    logger.debug(
        f"Hazard rating resolved for {char_key!r}/{value_key!r}",
        extra={"function_name": "_hazard", "input_data": {"char_key": char_key, "value_key": value_key}},
    )
    return {"code": char, "value": val}


async def _to_float(value) -> float | None:
    """Best-effort conversion of a value to ``float``.

    Args:
        value: The value to coerce. May be ``None`` or a non-numeric type.

    Returns:
        ``None`` when ``value`` is ``None``; the ``float`` conversion when it
        succeeds; or the original ``value`` unchanged when conversion raises
        ``ValueError``/``TypeError``.
    """
    if value is None:
        logger.debug(
            "Coercing None to float -> None",
            extra={"function_name": "_to_float", "input_data": {"value": None}},
        )
        return None
    try:
        result = float(value)
        logger.debug(
            f"Coerced value to float={result}",
            extra={"function_name": "_to_float", "input_data": {"value": value}},
        )
        return result
    except (ValueError, TypeError):
        logger.debug(
            f"Could not coerce value to float; returning unchanged value={value!r}",
            extra={"function_name": "_to_float", "input_data": {"value": value}},
        )
        return value


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


async def _channel_avail(variant: dict, channel_id: str) -> dict:
    """Return the availability block for a specific channel on a variant.

    Args:
        variant: The raw variant document.
        channel_id: The channel id whose availability should be returned.

    Returns:
        The availability dict for the requested channel (typically containing
        ``availableQuantity``, ``isOnStock`` and ``restockableInDays``), or an
        empty dict when the variant has no availability for that channel.
    """
    avail = variant.get("availability", {}).get("channels", {}).get(channel_id, {})
    logger.debug(
        f"Channel availability lookup for sku={variant.get('sku')!r} channel={channel_id!r} -> {'found' if avail else 'none'}",
        extra={"function_name": "_channel_avail", "input_data": {"sku": variant.get("sku"), "channel_id": channel_id}},
    )
    return avail
