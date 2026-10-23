"""Product-selection / purchase-agreement validation against the CommerceTools API.

The ingestion spec gates a product on its **purchase agreements** (modelled as
CommerceTools product selections):

  1. retrieve product ids (done by :class:`~asmo.products.ProductsClient`);
  2. for each product id, call its product selections
     (``GET /products/{id}/product-selections``);
  3. each selection the product belongs to is a purchase agreement — validate it:
       * it must be **active**;
       * it must **not be expired** (``validUntil`` today or later);
       * reject any agreement that fails either check;
  4. if no valid agreement remains, the product is **not indexed**;
  5. for every valid agreement capture its **contract id** and **valid-until**
     date, and attach the list of ``{contract_id, valid_until}`` objects to the
     product.

:class:`ProductSelectionsClient` returns, per product id, the list of valid
purchase agreements (empty/absent → the product is out of scope). The pipeline
drops out-of-scope products and threads the agreement list onto each row.

Because ``active``/``validUntil``/``contractId`` are project-specific custom
fields rather than standard CommerceTools attributes, they are read defensively
(the assignment's own custom fields first, then the expanded product selection's)
so a differently-shaped payload degrades gracefully rather than crashing. The
whole step is opt-in via ``Config.selection_validation_enabled`` (env
``SELECTION_VALIDATION_ENABLED``).

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
    """Parse a ``validUntil`` value into a :class:`date`, or ``None``.

    Accepts ISO date (``2026-10-23``) and ISO datetime (``2026-10-23T..Z``)
    strings. Returns ``None`` for missing or unparseable values so the caller can
    treat a missing/bad ``validUntil`` as "cannot prove it is unexpired".

    Args:
        value: The raw ``validUntil`` value from the payload.

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
    purchase agreement). ``active``, ``validUntil`` and ``contractId`` are read
    from the assignment's own ``custom.fields`` first, then from the expanded
    ``productSelection.obj.custom.fields``. The contract id falls back to the
    selection's ``key`` and then its ``id`` so a valid agreement always carries an
    identifier. ``active`` is treated as true when no flag is present (the project
    may not model one) and only an explicit falsy value rejects the agreement.

    Args:
        assignment: One product-selection assignment payload.

    Returns:
        A dict with ``active`` (bool), ``contract_id``, ``valid_until`` (raw
        string) and ``valid_until_date`` (parsed :class:`date` or ``None``).
    """
    # One assignment (one entry of /products/{id}/product-selections .results) looks like:
    #   {
    #     "productSelection": {                 # JSON path: productSelection (a reference)
    #        "typeId": "product-selection",
    #        "id": "a463a502-...",              # JSON path: productSelection.id
    #        "obj": {                           # JSON path: productSelection.obj (needs expand)
    #           "key": "assortment-supplier01",
    #           "custom": { "fields": {         # JSON path: productSelection.obj.custom.fields
    #               "active": true, "validUntil": "2026-12-31", "contractId": "CONTRACT-327", ...
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
    valid_until = pick("validUntil")                         # custom.fields.validUntil
    # contractId is the key join to standalone prices. Fall back to the selection
    # key, then its id, so a valid agreement always carries an identifier.
    contract_id = pick("contractId") or selection.get("key") or selection_ref.get("id")

    info = {
        # No active flag anywhere → cannot disprove it; treat as active. An
        # explicit falsy value rejects the agreement.
        "active": True if active is None else bool(active),
        "contract_id": contract_id,                          # from custom.fields.contractId (or key/id)
        "valid_until": valid_until,                          # raw custom.fields.validUntil string
        "valid_until_date": _as_date(valid_until),           # parsed to a date for the expiry check
    }
    logger.debug(
        f"Agreement fields: selection.id={selection_ref.get('id')!r} key={selection.get('key')!r} "
        f"→ active={info['active']} contract_id={info['contract_id']!r} "
        f"valid_until={info['valid_until']!r} (parsed {info['valid_until_date']})",
        extra={"function_name": "_agreement_info", "input_data": info | {"valid_until_date": str(info["valid_until_date"])}},
    )
    return info


class ProductSelectionsClient:
    """Resolves each product's valid purchase agreements via its product selections.

    Calls ``/products/{id}/product-selections`` per product (bounded
    concurrency), validates every agreement (active + not expired) and returns
    the captured ``{contract_id, valid_until}`` objects for the valid ones.
    Requires an :class:`~asmo.auth.AuthClient` for authorised requests.
    """

    # Cap in-flight per-product requests so a large catalogue does not open
    # thousands of simultaneous connections.
    MAX_CONCURRENCY = 20

    # Expand the referenced product selection (and its supplier custom field),
    # matching the documented request exactly. ``validUntil``/``active``/
    # ``contractId`` are inline custom fields on productSelection.obj.custom.fields
    # (so present regardless of expand); the expand resolves the supplier
    # reference. Per the API-playground docs: expand=productSelection.id;obj.custom.supplier.
    EXPAND = "productSelection.id;obj.custom.supplier"

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
    async def get_valid_agreements(self, product_ids: list[str]) -> dict[str, list[dict]]:
        """Return each product's valid purchase agreements, keyed by product id.

        For every id, fetches its product selections and keeps the agreements
        that are active and not expired, capturing each one's ``contract_id`` and
        ``valid_until``. Products with no valid agreement are omitted from the
        result (the caller excludes them from indexing). Requests are issued
        concurrently with a bounded semaphore.

        Args:
            product_ids: The product ids to validate.

        Returns:
            Mapping of product id → non-empty list of ``{contract_id,
            valid_until}`` objects. Out-of-scope products are absent.
        """
        today = date.today()
        sem = asyncio.Semaphore(self.MAX_CONCURRENCY)
        logger.info(
            f"Validating purchase agreements for {len(product_ids)} product(s) as of {today.isoformat()}",
            extra={"function_name": "get_valid_agreements", "input_data": {"products": len(product_ids)}},
        )
        async with aiohttp.ClientSession() as session:
            pairs = await asyncio.gather(
                *[self._for_product(session, pid, today, sem) for pid in product_ids]
            )

        valid = {pid: agreements for pid, agreements in pairs if agreements}
        logger.info(
            f"Purchase-agreement validation complete — {len(valid)}/{len(product_ids)} product(s) in scope",
            extra={
                "function_name": "get_valid_agreements",
                "status": "success",
                "input_data": {"products": len(product_ids), "in_scope": len(valid)},
            },
        )
        return valid

    async def _for_product(
        self, session: aiohttp.ClientSession, product_id: str, today: date, sem: asyncio.Semaphore
    ) -> tuple[str, list[dict]]:
        """Return ``(product_id, valid_agreements)`` for a single product.

        Args:
            session: An open aiohttp session.
            product_id: The product whose selections to validate.
            today: The date to compare each agreement's ``validUntil`` against.
            sem: Semaphore bounding concurrent per-product requests.

        Returns:
            The product id paired with its list of valid ``{contract_id,
            valid_until}`` objects (empty when none qualify).
        """
        async with sem:
            rows = await self._list_all(session, f"products/{product_id}/product-selections")

        agreements: list[dict] = []
        rejected = 0
        for row in rows:
            info = _agreement_info(row)
            unexpired = info["valid_until_date"] is not None and info["valid_until_date"] >= today
            if info["active"] and unexpired:
                agreements.append({"contract_id": info["contract_id"], "valid_until": info["valid_until"]})
            else:
                rejected += 1

        logger.debug(
            f"Product {product_id} → {len(agreements)} valid, {rejected} rejected of {len(rows)} agreement(s)",
            extra={
                "request_id": product_id,
                "function_name": "_for_product",
                "input_data": {"product_id": product_id, "valid": len(agreements), "rejected": rejected},
            },
        )
        return product_id, agreements

    async def _list_all(self, session: aiohttp.ClientSession, path: str) -> list[dict]:
        """Page through an offset-paginated endpoint and return all results.

        Fetches the first page to learn ``total``, then issues the remaining
        offset pages (usually none for a single product) and concatenates them.

        Args:
            session: An open aiohttp session.
            path: The endpoint path relative to the API base (no leading slash).

        Returns:
            Every ``results`` row across all pages.
        """
        first = await self._fetch_page(session, path, offset=0)
        total: int = first.get("total", 0)
        results: list[dict] = list(first.get("results", []))
        page_size = self._config.page_size
        if total > page_size:
            offsets = range(page_size, total, page_size)
            pages = await asyncio.gather(
                *[self._fetch_page(session, path, offset=off) for off in offsets]
            )
            for page in pages:
                results.extend(page.get("results", []))
        return results

    async def _fetch_page(self, session: aiohttp.ClientSession, path: str, offset: int) -> dict:
        """Fetch a single offset page from ``path`` with the selection expand.

        Args:
            session: An open aiohttp session.
            path: The endpoint path relative to the API base (no leading slash).
            offset: The zero-based record offset for this page.

        Returns:
            The parsed JSON body (including ``results`` and ``total``).

        Raises:
            Exception: Re-raised after logging on any request or HTTP error.
        """
        # ------------------------------------------------------------------
        # Build the request URL + query string, piece by piece.
        #
        #   base path  : {api_base}/{path}
        #                `path` is passed in by the caller and is one of:
        #                  "products/{product_id}/product-selections" — the agreements
        #                      a product belongs to (this is what validation uses), or
        #                  "product-selections" — the full selection list.
        #                api_base = https://api.{region}.commercetools.com/{project_key}
        #                → e.g. .../easmo-dev/products/<id>/product-selections
        #   query args : limit  — page size (records per page)
        #                offset — where this page starts (0, page_size, …)
        #                expand — EXPAND = "productSelection.id;obj.custom.supplier",
        #                         so each assignment embeds its productSelection.obj
        #                         (with custom.fields holding validUntil/active/contractId)
        #                         and resolves the supplier reference.
        # ------------------------------------------------------------------
        url = f"{self._config.api_base}/{path}"
        params = {
            "limit": self._config.page_size,   # query: limit=<page_size>
            "offset": offset,                   # query: offset=<offset>
            "expand": self.EXPAND,              # query: expand=productSelection.id;obj.custom.supplier
        }
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
                    f"HTTP {resp.status} ← {resp.url} → {len(body.get('results', []))} agreement(s) in {elapsed:.3f}s",
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
