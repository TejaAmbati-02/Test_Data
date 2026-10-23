"""Product-selection (contracted-scope) validation against the CommerceTools API.

The ingestion spec gates a product on its *product selection*: a product is in
"contracted scope" only when it belongs to a product selection that is **active**
and whose **validUntil** is today or later. The spec expresses this per product
(``GET /products/{id}/product-selections``), but to keep the bulk-fetch pipeline
fast this module resolves the same condition in bulk:

  1. list every product selection once (``GET /product-selections``);
  2. keep the selections that are active with ``validUntil >= today``;
  3. for each kept selection, list its assigned products
     (``GET /product-selections/{id}/products``) and union their ids.

The resulting set of in-scope product ids is what :class:`ProductSelectionsClient`
returns; the pipeline drops any product whose id is not in it.

Because ``active``/``validUntil`` are project-specific custom fields rather than
standard CommerceTools attributes, validity is read defensively (custom fields
first, then top-level) so a differently-shaped payload degrades to "unknown"
rather than crashing. This whole step is opt-in via
``Config.selection_validation_enabled`` (env ``SELECTION_VALIDATION_ENABLED``).

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
    treat "no/!bad validUntil" as "not provably valid".

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


def _is_selection_valid(selection: dict, today: date) -> bool:
    """Return True when a product selection is active and not expired.

    Reads ``active`` and ``validUntil`` from the selection's ``custom.fields``
    (where this project keeps them), falling back to top-level keys. A selection
    counts as valid only when ``active`` is truthy **and** ``validUntil`` is today
    or later. A missing/unparseable ``validUntil`` is treated as *not* valid, so
    the gate never admits a product on incomplete data.

    Args:
        selection: A single product-selection payload.
        today: The date to compare ``validUntil`` against.

    Returns:
        Whether the selection is active and valid as of ``today``.
    """
    fields = selection.get("custom", {}).get("fields", {})
    active = fields.get("active", selection.get("active"))
    valid_until = _as_date(fields.get("validUntil", selection.get("validUntil")))
    return bool(active) and valid_until is not None and valid_until >= today


class ProductSelectionsClient:
    """Resolves the set of product ids that are in a valid, active selection.

    Pages through all product selections, filters them by validity, then collects
    the assigned product ids of each valid selection. Requires an
    :class:`~asmo.auth.AuthClient` for authorised requests.
    """

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
    async def get_valid_product_ids(self) -> set[str]:
        """Return every product id that belongs to a valid, active selection.

        Lists all product selections, keeps those that are active with
        ``validUntil >= today`` (per :func:`_is_selection_valid`), and unions the
        assigned product ids of the kept selections — fetching each selection's
        products concurrently.

        Returns:
            A set of in-scope product ids. May be empty when no selection is
            currently valid.
        """
        today = date.today()
        async with aiohttp.ClientSession() as session:
            selections = await self._list_all(session, "product-selections")
            valid = [s for s in selections if _is_selection_valid(s, today)]
            logger.info(
                f"Product selections — {len(valid)}/{len(selections)} active and valid as of {today.isoformat()}",
                extra={
                    "function_name": "get_valid_product_ids",
                    "input_data": {"total": len(selections), "valid": len(valid)},
                },
            )

            product_id_lists = await asyncio.gather(
                *[self._products_of(session, s["id"]) for s in valid if s.get("id")]
            )

        ids: set[str] = set()
        for id_list in product_id_lists:
            ids.update(id_list)
        logger.info(
            f"Resolved {len(ids)} in-scope product id(s) across {len(valid)} valid selection(s)",
            extra={
                "function_name": "get_valid_product_ids",
                "status": "success",
                "input_data": {"product_ids": len(ids), "valid_selections": len(valid)},
            },
        )
        return ids

    async def _products_of(self, session: aiohttp.ClientSession, selection_id: str) -> list[str]:
        """Return the product ids assigned to one product selection.

        Args:
            session: An open aiohttp session.
            selection_id: The product selection's id.

        Returns:
            The list of assigned product ids (possibly empty).
        """
        rows = await self._list_all(session, f"product-selections/{selection_id}/products")
        return [
            pid
            for row in rows
            for pid in (row.get("product", {}).get("id") or row.get("id"),)
            if pid
        ]

    async def _list_all(self, session: aiohttp.ClientSession, path: str) -> list[dict]:
        """Page through an offset-paginated endpoint and return all results.

        Fetches the first page to learn ``total``, then issues the remaining
        offset pages concurrently and concatenates their results.

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
        """Fetch a single offset page from ``path``.

        Args:
            session: An open aiohttp session.
            path: The endpoint path relative to the API base (no leading slash).
            offset: The zero-based record offset for this page.

        Returns:
            The parsed JSON body (including ``results`` and ``total``).

        Raises:
            Exception: Re-raised after logging on any request or HTTP error.
        """
        params = {"limit": self._config.page_size, "offset": offset}
        url = f"{self._config.api_base}/{path}"
        t0 = datetime.now()
        try:
            async with session.get(url, headers=self._auth.headers, params=params, ssl=SSL_CONTEXT) as resp:
                body = await resp.json()
                elapsed = (datetime.now() - t0).total_seconds()
                logger.info(
                    f"HTTP {resp.status} ← {resp.url} → {len(body.get('results', []))} records "
                    f"(total={body.get('total')}) in {elapsed:.3f}s",
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
