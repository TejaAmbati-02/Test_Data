"""Product-selection / purchase-agreement checks against the CommerceTools API.

Purchase agreements are modelled as CommerceTools product selections. They are
used for two things:

**Product scope** (the ingestion spec's contracted-scope validation):

  1. for each product id, call its product selections
     (``GET /products/{id}/product-selections``);
  2. each selection the product belongs to is a purchase agreement — it is
     valid when it is **active** and **not expired** (``validUntil`` today or
     later);
  3. if no valid agreement remains, the product is **not indexed**;
  4. every valid agreement's ``{contract_id, supplier_id, valid_from,
     valid_until}`` is attached to the product (its contract ids also steer
     which seller price is preferred).

**Seller availability** (each seller offer's ``temporary_unavailable``). Each
selection names its supplier in the ``supplier`` custom field — the same channel
a standalone price's ``channel`` points at (the price's ``seller_id``). From the
same per-product call above, a seller is **available** for the product when one
of the product's selections whose ``supplier`` is that seller is **active** and
has **started** (``validFrom`` today or earlier; no ``validFrom`` means no start
restriction) — its offer then has ``temporary_unavailable = false``; otherwise
(inactive / not yet started, or no agreement for the product)
``temporary_unavailable = true``.

Because ``active``/``validFrom``/``validUntil``/``contractId``/``supplier`` are
project-specific custom fields rather than standard CommerceTools attributes,
they are read defensively so a differently-shaped payload degrades gracefully
rather than crashing. Both checks always run as part of the pipeline.

Network calls use :mod:`aiohttp` with the project's shared SSL context, require a
valid auth token (via :func:`~asmo.auth.requires_token`), and are logged through
the structured logger.
"""

import asyncio
from datetime import date, datetime

import aiohttp

from .auth import AuthClient, requires_token
from .config import Config
from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


def _as_date(value) -> date | None:
    """Parse a ``validFrom`` / ``validUntil`` value into a :class:`date`, or ``None``.

    Accepts ISO date (``2026-10-23``) and ISO datetime (``2026-10-23T..Z``)
    strings. Returns ``None`` for missing or unparseable values so the caller can
    treat a missing/bad ``validUntil`` as "cannot prove it is unexpired".

    Args:
        value: The raw ``validFrom`` / ``validUntil`` value from the payload.

    Returns:
        The parsed date, or ``None`` when absent/unparseable.
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        # Tolerate a trailing Z and a time component; fall back to date-only.
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def _agreement_info(assignment: dict) -> dict:
    """Extract one purchase agreement's validity and capture fields.

    A product-selection *assignment* ties a product to a product selection (the
    purchase agreement); a selection fetched directly is passed wrapped as
    ``{"productSelection": {"id": ..., "obj": <selection>}}``. ``active``,
    ``validFrom``, ``validUntil``, ``contractId`` and ``supplier`` are read from
    the assignment's own ``custom.fields`` first, then from
    ``productSelection.obj.custom.fields``. The contract id falls back to the
    selection's ``key`` and then its ``id`` so a valid agreement always carries an
    identifier. ``active`` is treated as true when no flag is present (the project
    may not model one) and only an explicit falsy value rejects the agreement.

    Args:
        assignment: One product-selection assignment payload.

    Returns:
        A dict with ``active`` (bool), ``contract_id``, ``supplier_id``,
        ``valid_from`` / ``valid_until`` (raw strings) and ``valid_from_date`` /
        ``valid_until_date`` (parsed :class:`date` or ``None``).
    """
    # One assignment (one entry of /products/{id}/product-selections .results) looks like:
    #   {
    #     "productSelection": {                 # JSON path: productSelection (a reference)
    #        "typeId": "product-selection",
    #        "id": "a463a502-...",              # JSON path: productSelection.id
    #        "obj": {                           # JSON path: productSelection.obj (needs expand)
    #           "key": "assortment-supplier01",
    #           "custom": { "fields": {         # JSON path: productSelection.obj.custom.fields
    #               "active": true, "validFrom": "2026-01-01", "validUntil": "2026-12-31",
    #               "contractId": "CONTRACT-327",
    #               "supplier": {"typeId": "channel", "id": "3a7b23e0-..."}, ...
    #           } }
    #        }
    #     }
    #   }
    selection_ref = assignment.get("productSelection", {})   # JSON path: productSelection
    selection = selection_ref.get("obj", {}) or {}           # JSON path: productSelection.obj

    # validUntil / active / contractId are custom fields. We look in the
    # assignment's own custom.fields first, then the product selection's — so
    # wherever this project stores them, pick() finds them.
    sources = [
        assignment.get("custom", {}).get("fields", {}),      # JSON path: custom.fields (on the assignment)
        selection.get("custom", {}).get("fields", {}),       # JSON path: productSelection.obj.custom.fields
    ]

    def pick(key):
        # Return the first non-null value for `key` across the source blocks above.
        for fields in sources:
            if fields.get(key) is not None:
                return fields.get(key)
        return None

    active = pick("active")                                   # custom.fields.active
    valid_from = pick("validFrom")                           # custom.fields.validFrom
    valid_until = pick("validUntil")                         # custom.fields.validUntil
    supplier = pick("supplier")                              # custom.fields.supplier (channel reference)
    supplier_id = supplier.get("id") if isinstance(supplier, dict) else supplier
    # contractId is the key join to standalone prices. Fall back to the selection
    # key, then its id, so a valid agreement always carries an identifier.
    contract_id = pick("contractId") or selection.get("key") or selection_ref.get("id")

    info = {
        # No active flag anywhere → cannot disprove it; treat as active. An
        # explicit falsy value rejects the agreement.
        "active": True if active is None else bool(active),
        "contract_id": contract_id,                          # from custom.fields.contractId (or key/id)
        "supplier_id": supplier_id,                          # custom.fields.supplier.id (the seller's channel id)
        "valid_from": valid_from,                            # raw custom.fields.validFrom string
        "valid_from_date": _as_date(valid_from),             # parsed to a date for the start check
        "valid_until": valid_until,                          # raw custom.fields.validUntil string
        "valid_until_date": _as_date(valid_until),           # parsed to a date for the expiry check
    }
    logger.debug(
        f"Agreement fields: selection.id={selection_ref.get('id')!r} key={selection.get('key')!r} "
        f"→ active={info['active']} contract_id={info['contract_id']!r} supplier_id={info['supplier_id']!r} "
        f"valid_from={info['valid_from']!r} valid_until={info['valid_until']!r} (parsed {info['valid_until_date']})",
        extra={"function_name": "_agreement_info", "input_data": info | {
            "valid_from_date": str(info["valid_from_date"]), "valid_until_date": str(info["valid_until_date"])}},
    )
    return info


def _in_scope(info: dict, today: date) -> bool:
    """True when an agreement puts its product in scope: active and not expired.

    ``validUntil`` is required — without it the agreement cannot be proven unexpired.
    """
    return info["active"] and info["valid_until_date"] is not None and info["valid_until_date"] >= today


def _available(info: dict, today: date) -> bool:
    """True when an agreement makes its seller available: active and started.

    ``validFrom`` is optional — absent means no start restriction.
    """
    return info["active"] and (info["valid_from_date"] is None or info["valid_from_date"] <= today)


def _agreement(info: dict) -> dict:
    """The agreement fields attached to a product's rows."""
    return {
        "contract_id": info["contract_id"],
        "supplier_id": info["supplier_id"],
        "valid_from": info["valid_from"],
        "valid_until": info["valid_until"],
    }


class ProductSelectionsClient:
    """Resolves products' valid purchase agreements and sellers' availability.

    Requires an :class:`~asmo.auth.AuthClient` for authorised requests.
    """

    # Cap in-flight requests so a large catalogue does not open thousands of
    # simultaneous connections.
    MAX_CONCURRENCY = 20

    # For /products/{id}/product-selections: expand the referenced product
    # selection so its custom fields (active, validFrom/validUntil, contractId,
    # supplier) are inline. The documented "productSelection.id;obj.custom.supplier"
    # is rejected by the API as a malformed path selector (HTTP 400).
    EXPAND = "productSelection"

    def __init__(self, config: Config, auth: AuthClient) -> None:
        """Store the configuration and auth client used for selection requests.

        Args:
            config: Configuration providing the API base URL and page size.
            auth: The authenticated client supplying request headers and token.
        """
        self._config = config
        self._auth = auth

    @log_function
    @requires_token
    async def get_agreements_and_availability(
        self, product_ids: list[str],
    ) -> tuple[dict[str, list[dict]], dict[str, dict[str, bool]]]:
        """Return each product's valid agreements and each seller's availability for it.

        For every product id, calls ``GET /products/{id}/product-selections``
        (with ``expand=productSelection``) and reads each selection's custom
        fields once for both checks:

        * **scope** — the selection is active and ``validUntil`` >= today;
        * **availability** — the selection's ``supplier`` (a seller's channel id)
          is available for the product when the selection is active and
          ``validFrom`` <= today.

        Args:
            product_ids: The product ids to check.

        Returns:
            ``(agreements, availability)``: product id → non-empty list of
            ``{contract_id, supplier_id, valid_from, valid_until}`` (products with
            no valid agreement omitted), and product id → {supplier id →
            available}. A seller missing from a product's map has no agreement
            for it (treat as unavailable).
        """
        today = date.today()
        sem = asyncio.Semaphore(self.MAX_CONCURRENCY)
        logger.info(
            f"Checking product selections for {len(product_ids)} product(s) as of {today.isoformat()}",
            extra={"function_name": "get_agreements_and_availability", "input_data": {"products": len(product_ids)}},
        )

        async def for_product(session: aiohttp.ClientSession, pid: str) -> tuple[str, list[dict], dict[str, bool]]:
            async with sem:
                rows = await self._list_all(session, f"products/{pid}/product-selections", expand=self.EXPAND)
            infos = [_agreement_info(r) for r in rows]
            valid = [_agreement(i) for i in infos if _in_scope(i, today)]
            sellers: dict[str, bool] = {}
            for i in infos:
                if i["supplier_id"]:
                    sellers[i["supplier_id"]] = sellers.get(i["supplier_id"], False) or _available(i, today)
            logger.debug(
                f"Product {pid} → {len(valid)} valid of {len(infos)} agreement(s); sellers available={sellers}",
                extra={"request_id": pid, "function_name": "get_agreements_and_availability",
                       "input_data": {"product_id": pid, "valid": len(valid), "total": len(infos), "sellers": sellers}},
            )
            return pid, valid, sellers

        async with aiohttp.ClientSession() as session:
            triples = await asyncio.gather(*[for_product(session, pid) for pid in product_ids])

        agreements = {pid: valid for pid, valid, _ in triples if valid}
        availability = {pid: sellers for pid, _, sellers in triples if sellers}
        logger.info(
            f"Product-selection check complete — {len(agreements)}/{len(product_ids)} product(s) in scope, "
            f"{len(availability)} with seller availability",
            extra={"function_name": "get_agreements_and_availability", "status": "success",
                   "input_data": {"products": len(product_ids), "in_scope": len(agreements),
                                  "with_availability": len(availability)}},
        )
        return agreements, availability

    async def _list_all(
        self, session: aiohttp.ClientSession, path: str, where: str | None = None, expand: str | None = None,
    ) -> list[dict]:
        """Page through an offset-paginated endpoint and return all results.

        Fetches the first page to learn ``total``, then issues the remaining
        offset pages (usually none) concurrently and concatenates them. When the
        endpoint omits ``total``, pages sequentially until a page comes back short.

        Args:
            session: An open aiohttp session.
            path: The endpoint path relative to the API base (no leading slash).
            where: Optional query predicate (``where=`` parameter).
            expand: Optional ``expand=`` parameter.

        Returns:
            Every ``results`` row across all pages.
        """
        first = await self._fetch_page(session, path, offset=0, where=where, expand=expand)
        total: int | None = first.get("total")
        results: list[dict] = list(first.get("results", []))
        page_size = self._config.page_size
        if total is None:
            # Some endpoints (e.g. /product-selections/{id}/products) omit
            # `total` unless withTotal=true — page sequentially until a short page.
            page = first
            while len(page.get("results", [])) >= page_size:
                page = await self._fetch_page(session, path, offset=len(results), where=where, expand=expand)
                results.extend(page.get("results", []))
        elif total > page_size:
            offsets = range(page_size, total, page_size)
            pages = await asyncio.gather(
                *[self._fetch_page(session, path, offset=off, where=where, expand=expand) for off in offsets]
            )
            for page in pages:
                results.extend(page.get("results", []))
        return results

    async def _fetch_page(
        self, session: aiohttp.ClientSession, path: str, offset: int, where: str | None = None, expand: str | None = None,
    ) -> dict:
        """Fetch a single offset page from ``path``.

        Args:
            session: An open aiohttp session.
            path: The endpoint path relative to the API base (no leading slash).
            offset: The zero-based record offset for this page.
            where: Optional query predicate (``where=`` parameter).
            expand: Optional ``expand=`` parameter.

        Returns:
            The parsed JSON body (including ``results`` and ``total``).

        Raises:
            Exception: Re-raised after logging on any request or HTTP error.
        """
        # ------------------------------------------------------------------
        # Build the request URL + query string, piece by piece.
        #
        #   base path  : {api_base}/{path}
        #                `path` is passed in by the caller, e.g.
        #                  "products/{id}/product-selections" — the agreements a
        #                      product belongs to (scope + seller availability).
        #                api_base = https://api.{region}.commercetools.com/{project_key}
        #   query args : limit  — page size (records per page)
        #                offset — where this page starts (0, page_size, …)
        #                where  — optional query predicate
        #                expand — e.g. productSelection
        # ------------------------------------------------------------------
        url = f"{self._config.api_base}/{path}"
        params = {
            "limit": self._config.page_size,   # query: limit=<page_size>
            "offset": offset,                   # query: offset=<offset>
        }
        if where:
            params["where"] = where             # query: where=<predicate>
        if expand:
            params["expand"] = expand           # query: expand=productSelection
        logger.debug(
            f"Built GET URL → {url} | query params → {params} "
            f"(api_base={self._config.api_base!r}, path={path!r}, offset={offset})",
            extra={"function_name": "_fetch_page", "input_data": {"url": url, "params": params, "path": path, "offset": offset}},
        )
        t0 = datetime.now()
        try:
            async with session.get(url, headers=self._auth.headers, params=params, ssl=SSL_CONTEXT) as resp:
                body = await resp.json()
                elapsed = (datetime.now() - t0).total_seconds()
                logger.debug(
                    f"HTTP {resp.status} ← {resp.url} → {len(body.get('results', []))} result(s) in {elapsed:.3f}s",
                    extra={
                        "function_name": "_fetch_page",
                        "latency": f"{elapsed:.3f}s",
                        "status": "success" if resp.ok else "error",
                        "input_data": {"path": path, "http_status": resp.status, "offset": offset, "total": body.get("total")},
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
