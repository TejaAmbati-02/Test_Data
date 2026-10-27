"""Standalone-price retrieval from the CommerceTools API.

This module fetches standalone prices from CommerceTools and maps them to a
lightweight domain model. It exposes :class:`Price`, a dataclass flattening the
fields the pipeline needs (amount, currency, seller names, contract metadata,
…) alongside the raw payload, and :class:`PricesClient`, which retrieves the
full price list using parallel offset-based pagination. A module-level
:func:`_channel_name` helper resolves localised seller/channel names.

Network calls use :mod:`aiohttp` with the project's shared SSL context, require
a valid auth token (enforced via :func:`~asmo.auth.requires_token`), and are
logged through the structured logger.
"""

import asyncio
import aiohttp
from dataclasses import dataclass, field
from .auth import AuthClient, requires_token
from .config import Config
from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


@dataclass
class Price:
    """Lightweight representation of a CommerceTools standalone price.

    Flattens the parts of a standalone-price payload the pipeline consumes:
    identity (``id``, ``sku``), monetary value (``currency``, ``amount`` in
    major units), localised seller names, and custom-field metadata (``scope``,
    ``contract_id``, ``lead_time_days``, ``price_unit``, ``moq``). The original
    payload is retained in ``raw`` and excluded from the dataclass ``repr``.
    """

    id: str
    sku: str
    currency: str
    amount: float
    scope: str | None
    seller_id: str | None
    seller_name_en: str | None
    seller_name_de: str | None
    supplier_id: str | None
    contract_id: str | None
    lead_time_days: int | None
    price_unit: str | None
    unit_type: str | None
    moq: int | None
    temporary_unavailable: bool | None
    is_blocked_for_deletion: str | None
    raw: dict = field(repr=False)

    @classmethod
    def from_raw(cls, raw: dict) -> "Price":
        """Build a :class:`Price` from a raw standalone-price payload.

        Delegates the actual field extraction to :meth:`_from_raw`; this wrapper
        adds error handling, logging any parsing failure (tagged with the SKU as
        ``request_id``) before re-raising.

        Args:
            raw: A single standalone-price payload from the API.

        Returns:
            The populated :class:`Price` instance.

        Raises:
            Exception: Re-raised after logging if the payload cannot be parsed.
        """
        try:
            return cls._from_raw(raw)
        except Exception as exc:
            logger.error(
                f"Failed to parse price payload: {exc}",
                extra={
                    "request_id": raw.get("sku"),
                    "function_name": "Price.from_raw",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"id": raw.get("id"), "sku": raw.get("sku")},
                },
            )
            raise

    @classmethod
    def _from_raw(cls, raw: dict) -> "Price":
        """Extract and convert fields from a raw payload into a :class:`Price`.

        Reads the monetary ``value`` block and ``custom.fields``, converting the
        integer ``centAmount`` to a major-unit ``amount`` using
        ``fractionDigits`` (defaulting to 2), resolving English and German
        seller names via :func:`_channel_name`, and pulling metadata such as
        scope, contract id, lead time and minimum order quantity. The parsed
        price is logged (tagged with the SKU) before being returned.

        Args:
            raw: A single standalone-price payload from the API.

        Returns:
            The populated :class:`Price` instance.
        """
        # ------------------------------------------------------------------
        # STEP 1 — Pull the three nested JSON blocks we read fields out of.
        # A commercetools standalone-price payload looks like:
        #   {
        #     "id": "...", "sku": "...",
        #     "value":   { "currencyCode": "SAR", "centAmount": 10000, "fractionDigits": 2 },
        #     "channel": { "obj": { "id": "...", "name": {"en": "...", "de": "..."} } },  # expand=channel.id
        #     "custom":  { "fields": { "scope": "...", "contractId": "...", ... } }
        #   }
        # We dig those sub-objects out first (defaulting to {} so a missing block
        # never raises) and log exactly what we found.
        # ------------------------------------------------------------------
        value = raw.get("value", {})                       # JSON path: value
        custom_fields = raw.get("custom", {}).get("fields", {})  # JSON path: custom.fields
        channel_obj = raw.get("channel", {}).get("obj", {})      # JSON path: channel.obj (needs expand=channel.id)
        logger.debug(
            f"Price blocks for sku={raw.get('sku')!r}: value keys={list(value)}, "
            f"custom.fields keys={list(custom_fields)}, channel.obj present={bool(channel_obj)}",
            extra={
                "request_id": raw.get("sku"),
                "function_name": "Price._from_raw",
                "input_data": {"value": value, "custom_fields": custom_fields, "channel_obj_keys": list(channel_obj)},
            },
        )

        # ------------------------------------------------------------------
        # STEP 2 — Convert the money. commercetools stores the amount as an
        # integer in minor units (cents): value.centAmount=10000 with
        # value.fractionDigits=2 means 100.00. So amount = centAmount / 10**digits.
        # ------------------------------------------------------------------
        cent_amount = value.get("centAmount", 0)           # JSON path: value.centAmount (integer minor units)
        fraction_digits = value.get("fractionDigits", 2)   # JSON path: value.fractionDigits (how many minor digits)
        amount = cent_amount / (10 ** fraction_digits)     # major-unit price, e.g. 10000 / 100 = 100.0
        logger.debug(
            f"Money for sku={raw.get('sku')!r}: centAmount={cent_amount} / 10^{fraction_digits} "
            f"= {amount} {value.get('currencyCode')}",
            extra={"request_id": raw.get("sku"), "function_name": "Price._from_raw",
                   "input_data": {"centAmount": cent_amount, "fractionDigits": fraction_digits, "amount": amount}},
        )

        # ------------------------------------------------------------------
        # STEP 3 — Read each field we keep, right out of the blocks above.
        # The comment on every line is the exact commercetools JSON path.
        # ------------------------------------------------------------------
        price = cls(
            id=raw["id"],                                   # JSON path: id
            sku=raw.get("sku", "N/A"),                      # JSON path: sku
            currency=value.get("currencyCode", "N/A"),      # JSON path: value.currencyCode (e.g. "SAR")
            amount=amount,                                  # computed in STEP 2 from value.centAmount
            scope=custom_fields.get("scope"),               # JSON path: custom.fields.scope (e.g. "CONTRACTED")
            seller_id=channel_obj.get("id"),                # JSON path: channel.obj.id (the seller = supply channel)
            seller_name_en=_channel_name(raw, "en"),        # JSON path: channel.obj.name.en
            seller_name_de=_channel_name(raw, "de"),        # JSON path: channel.obj.name.de
            supplier_id=custom_fields.get("supplierId"),    # JSON path: custom.fields.supplierId
            contract_id=custom_fields.get("contractId"),    # JSON path: custom.fields.contractId (ties to a purchase agreement)
            lead_time_days=custom_fields.get("leadTimeDays"),   # JSON path: custom.fields.leadTimeDays
            price_unit=value.get("currencyCode"),           # JSON path: value.currencyCode (kept for back-compat)
            # Spec "unitOfMeasure" — JSON path: custom.fields.priceUnit.
            unit_type=custom_fields.get("priceUnit"),
            # Spec "MOQ" (Minimum Order Quantity) — JSON path: custom.fields.minimumQuantity.
            moq=custom_fields.get("minimumQuantity"),
            # Spec "temporary unavailable" — JSON path: custom.fields.temporarilyUnavailable
            # (the CommerceTools field name; temporaryUnavailable is also accepted), a
            # localized or plain "true"/"false" value (e.g. {"en": "false"}) coerced to a
            # bool by _localized_bool. None when absent.
            temporary_unavailable=_localized_bool(
                custom_fields.get("temporarilyUnavailable", custom_fields.get("temporaryUnavailable"))
            ),
            # Spec "blockingDeletionCode" — a deletion-code string ("L" == blocked).
            # JSON path: custom.fields.purchasingContractDeletionCode. None when absent.
            is_blocked_for_deletion=custom_fields.get("purchasingContractDeletionCode"),
            raw=raw,
        )

        # ------------------------------------------------------------------
        # STEP 4 — Log every field we extracted, next to the JSON path it came
        # from, so the parse is fully auditable from the logs alone.
        # ------------------------------------------------------------------
        logger.info(
            f"Parsed price id={price.id!r} sku={price.sku!r} amount={price.amount} {price.currency} "
            f"scope={price.scope!r} seller_id={price.seller_id!r} contract_id={price.contract_id!r}",
            extra={
                "request_id": price.sku,
                "function_name": "Price.from_raw",
                "input_data": {"value": value, "custom_fields": custom_fields},
                # field -> (commercetools JSON path, extracted value)
                "response_payload": {
                    "id": ("id", price.id),
                    "sku": ("sku", price.sku),
                    "currency": ("value.currencyCode", price.currency),
                    "amount": ("value.centAmount / 10^fractionDigits", price.amount),
                    "scope": ("custom.fields.scope", price.scope),
                    "seller_id": ("channel.obj.id", price.seller_id),
                    "seller_name_en": ("channel.obj.name.en", price.seller_name_en),
                    "supplier_id": ("custom.fields.supplierId", price.supplier_id),
                    "contract_id": ("custom.fields.contractId", price.contract_id),
                    "lead_time_days": ("custom.fields.leadTimeDays", price.lead_time_days),
                    "unit_type": ("custom.fields.priceUnit", price.unit_type),
                    "moq": ("custom.fields.minimumQuantity", price.moq),
                    "temporary_unavailable": ("custom.fields.temporarilyUnavailable", price.temporary_unavailable),
                    "is_blocked_for_deletion": ("custom.fields.purchasingContractDeletionCode", price.is_blocked_for_deletion),
                },
            },
        )
        return price


def _localized_bool(value) -> bool | None:
    """Coerce a localized/plain truthy value into a ``bool``, or ``None``.

    The ``temporarilyUnavailable`` custom field arrives as a localized string such
    as ``{"en": "false"}``; it may also be a plain ``"true"``/``"false"`` string
    or an actual bool. Returns the boolean meaning, or ``None`` when absent.

    Args:
        value: The raw field value (localized dict, string or bool).

    Returns:
        ``True``/``False`` when a value is present, else ``None``.
    """
    if isinstance(value, dict):
        value = value.get("en")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return None


def _channel_name(raw: dict, locale: str) -> str | None:
    """Resolve the expanded channel's name for a given locale.

    Reads the expanded ``channel.obj.name`` from the payload. When the name is a
    localised dict, returns the value for ``locale`` (or ``None`` if absent).
    When the name is a plain string, it is treated as the English name: returned
    for ``locale == "en"`` and ``None`` for any other locale.

    Args:
        raw: A single standalone-price payload with an expanded channel.
        locale: The locale key to resolve (e.g. ``"en"`` or ``"de"``).

    Returns:
        The seller/channel name for the locale, or ``None`` if unavailable.
    """
    channel_obj = raw.get("channel", {}).get("obj", {})
    name = channel_obj.get("name")
    if isinstance(name, dict):
        return name.get(locale) or None
    return name or None if locale == "en" else None


class PricesClient:
    """Fetches standalone prices from CommerceTools using offset-based pagination.

    Retrieves the first page to learn the total count, then fetches all
    remaining pages concurrently by offset, keeping large price lists fast to
    download. Page size comes from the configuration. Requires an
    :class:`~asmo.auth.AuthClient` for authorised requests.
    """

    def __init__(self, config: Config, auth: AuthClient) -> None:
        """Store the configuration and auth client used for price requests.

        Args:
            config: Configuration providing the API base URL and page size.
            auth: The authenticated client supplying request headers and token.
        """
        self._config = config
        self._auth = auth

    @log_function
    @requires_token
    async def get_all(self) -> list[Price]:
        """Fetch and return every standalone price.

        Fetches the first page (offset 0) to discover the ``total`` count, then,
        when more pages remain, issues all remaining offset requests
        concurrently with :func:`asyncio.gather` and concatenates their results.
        The combined raw results are mapped to :class:`Price` objects. Page
        counts and totals are logged.

        A valid token is guaranteed by :func:`~asmo.auth.requires_token`, and
        timing/entry/exit logging is added by
        :func:`~asmo.logging_service.log_function`.

        Returns:
            A list of :class:`Price` objects, one per raw price record.
        """
        logger.info("Starting price fetch", extra={"function_name": "get_all"})
        async with aiohttp.ClientSession() as session:
            first = await self._fetch_page(session, offset=0)
            total: int = first["total"]
            results: list[dict] = first["results"]

            if total > self._config.page_size:
                offsets = range(self._config.page_size, total, self._config.page_size)
                logger.info(
                    f"Fetching {len(offsets)} remaining price pages concurrently (total={total})",
                    extra={"function_name": "get_all", "input_data": {"pages": len(offsets) + 1}},
                )
                pages = await asyncio.gather(
                    *[self._fetch_page(session, offset=off) for off in offsets]
                )
                for page in pages:
                    results.extend(page["results"])

        logger.info(
            f"Fetched {len(results)} prices in {1 + (total - 1) // self._config.page_size} page(s)",
            extra={"function_name": "get_all", "status": "success", "input_data": {"total": len(results)}},
        )
        return [Price.from_raw(r) for r in results]

    async def _fetch_page(self, session: aiohttp.ClientSession, offset: int) -> dict:
        """Fetch a single page of standalone prices at a given offset.

        Issues a ``GET`` to the ``standalone-prices`` endpoint with the
        configured page-size limit, sorted by ``id`` ascending, starting at
        ``offset``, and expanding ``channel.id``. Because each page is addressed
        by an independent ``offset``, :meth:`get_all` can request many pages in
        parallel. The request, the HTTP response summary, and any failure (with
        elapsed time) are logged, and a non-2xx status raises via
        :meth:`aiohttp.ClientResponse.raise_for_status`.

        Args:
            session: An open aiohttp session used to issue the request.
            offset: The zero-based record offset for this page.

        Returns:
            The parsed JSON body of the response (including ``results`` and
            ``total``).

        Raises:
            Exception: Re-raised after logging on any request or HTTP error.
        """
        from datetime import datetime
        # ------------------------------------------------------------------
        # Build the request URL + query string, piece by piece.
        #
        #   base path  : {api_base}/standalone-prices
        #                api_base = https://api.{region}.commercetools.com/{project_key}
        #                (see Config.api_base), so the full path is e.g.
        #                https://api.eu.commercetools.com/easmo-dev/standalone-prices
        #   query args : limit  — page size (how many records per page)
        #                sort   — "id asc" so paging by offset is stable
        #                offset — where this page starts (0, page_size, 2*page_size, …)
        #                expand — "channel.id" so each price embeds its channel
        #                         (the seller) under channel.obj, which Price parses.
        # aiohttp URL-encodes `params` onto the URL, giving a final request of e.g.
        #   GET .../standalone-prices?limit=500&sort=id+asc&offset=0&expand=channel.id
        # ------------------------------------------------------------------
        endpoint = "standalone-prices"
        url = f"{self._config.api_base}/{endpoint}"
        params = {
            "limit": self._config.page_size,   # query: limit=<page_size>
            "sort": "id asc",                   # query: sort=id asc (stable offset paging)
            "offset": offset,                   # query: offset=<offset>
            "expand": "channel.id",             # query: expand=channel.id (embed seller channel)
        }
        logger.info(
            f"Built GET URL → {url} | query params → {params} "
            f"(api_base={self._config.api_base!r}, endpoint={endpoint!r}, offset={offset})",
            extra={"function_name": "_fetch_page", "input_data": {"url": url, "params": params, "offset": offset, "limit": self._config.page_size}},
        )
        t0 = datetime.now()
        try:
            async with session.get(
                url,
                headers=self._auth.headers,
                params=params,
                ssl=SSL_CONTEXT,
            ) as resp:
                full_url = str(resp.url)
                status = resp.status
                body = await resp.json()
                elapsed = (datetime.now() - t0).total_seconds()
                count = len(body.get("results", []))
                logger.info(
                    f"HTTP {status} ← {full_url} offset={offset} → {count} records (total={body.get('total')}) in {elapsed:.3f}s",
                    extra={
                        "function_name": "_fetch_page",
                        "latency": f"{elapsed:.3f}s",
                        "status": "success" if resp.ok else "error",
                        "input_data": {"full_url": full_url, "http_status": status, "offset": offset, "returned": count, "total": body.get("total")},
                        "response_payload": body,
                    },
                )
                resp.raise_for_status()
                return body
        except Exception as exc:
            elapsed = (datetime.now() - t0).total_seconds()
            logger.error(
                f"Request failed after {elapsed:.3f}s: GET {url} offset={offset}: {exc}",
                extra={
                    "function_name": "_fetch_page",
                    "latency": f"{elapsed:.3f}s",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"url": url, "params": params, "offset": offset},
                },
            )
            raise
