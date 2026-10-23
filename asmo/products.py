"""Product retrieval from the CommerceTools API.

This module fetches product projections from CommerceTools and maps them to a
lightweight domain model. It exposes :class:`Product`, a small dataclass holding
the fields the pipeline needs plus the raw payload, and :class:`ProductsClient`,
which pages through the full catalogue using cursor-based pagination.

Network calls use :mod:`aiohttp` with the project's shared SSL context, require
a valid auth token (enforced via :func:`~asmo.auth.requires_token`), and are
logged through the structured logger.
"""

import aiohttp
from dataclasses import dataclass, field
from .auth import AuthClient, requires_token
from .config import Config
from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


@dataclass
class Product:
    """Lightweight representation of a CommerceTools product projection.

    Attributes:
        id: The product's CommerceTools id.
        name: A display name resolved from the localised name field.
        raw: The full original payload, retained for downstream use and
            excluded from the dataclass ``repr``.
    """

    id: str
    name: str
    raw: dict = field(repr=False)

    @classmethod
    def from_raw(cls, raw: dict) -> "Product":
        """Build a :class:`Product` from a raw CommerceTools projection.

        Resolves the display name from the localised ``name`` field, preferring
        English (``en``) then German (``de``) and falling back to ``"N/A"``. The
        parsed result is logged, and any parsing failure is logged with the
        offending id before being re-raised.

        Args:
            raw: A single product-projection payload from the API.

        Returns:
            The populated :class:`Product` instance.

        Raises:
            Exception: Re-raised after logging if the payload cannot be parsed
                (for example, a missing ``id``).
        """
        try:
            name = raw.get("name", {})
            display_name = name.get("en") or name.get("de") or "N/A"
            logger.info(
                f"Parsed product id={raw.get('id')!r} name={display_name!r}",
                extra={
                    "function_name": "Product.from_raw",
                    "input_data": {"id": raw.get("id"), "name_field": name},
                    "response_payload": {"id": raw["id"], "name": display_name},
                },
            )
            return cls(id=raw["id"], name=display_name, raw=raw)
        except Exception as exc:
            logger.error(
                f"Failed to parse product payload: {exc}",
                extra={
                    "function_name": "Product.from_raw",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"id": raw.get("id")},
                },
            )
            raise


class ProductsClient:
    """Fetches product projections from CommerceTools using cursor-based pagination.

    Walks the catalogue in fixed-size pages of :data:`PAGE_SIZE`, advancing with
    an ``id``-based cursor rather than an offset so the traversal stays correct
    and efficient across very large result sets. Requires an
    :class:`~asmo.auth.AuthClient` for authorised requests.
    """

    PAGE_SIZE = 500

    def __init__(self, config: Config, auth: AuthClient) -> None:
        """Store the configuration and auth client used for product requests.

        Args:
            config: Configuration providing the API base URL and related
                settings.
            auth: The authenticated client supplying request headers and token.
        """
        self._config = config
        self._auth = auth

    @log_function
    @requires_token
    async def get_all(self) -> list[Product]:
        """Fetch and return every product projection in the catalogue.

        Pages through the API with cursor-based pagination: each page is sorted
        by ``id`` ascending and the next request filters on ``id > last_id``
        using the last id of the previous batch. Paging stops when a batch comes
        back empty or smaller than :data:`PAGE_SIZE`. Progress and totals are
        logged as pages arrive.

        A valid token is guaranteed by :func:`~asmo.auth.requires_token`, and
        timing/entry/exit logging is added by
        :func:`~asmo.logging_service.log_function`.

        Returns:
            A list of :class:`Product` objects, one per raw projection, in id
            order.
        """
        results: list[dict] = []
        last_id: str | None = None
        page_num = 0

        logger.info("Starting product fetch", extra={"function_name": "get_all"})
        async with aiohttp.ClientSession() as session:
            while True:
                page = await self._fetch_page(session, last_id=last_id)
                batch: list[dict] = page["results"]
                page_num += 1
                if not batch:
                    break
                results.extend(batch)
                logger.info(
                    f"Products page {page_num}: {len(batch)} records (total so far: {len(results)})",
                    extra={"function_name": "get_all", "input_data": {"page": page_num, "batch_size": len(batch)}},
                )
                if len(batch) < self.PAGE_SIZE:
                    break
                last_id = batch[-1]["id"]

        logger.info(
            f"Fetched {len(results)} products in {page_num} page(s)",
            extra={"function_name": "get_all", "status": "success", "input_data": {"total": len(results)}},
        )
        return [Product.from_raw(r) for r in results]

    async def _fetch_page(self, session: aiohttp.ClientSession, last_id: str | None) -> dict:
        """Fetch a single page of product projections via cursor pagination.

        Issues a ``GET`` to the ``product-projections`` endpoint with a page
        limit of :data:`PAGE_SIZE`, sorted by ``id`` ascending, and expanding
        category ancestors. When ``last_id`` is provided it adds a
        ``where=id > "<last_id>"`` predicate so the request returns only
        products *after* the previous batch — the cursor that drives
        :meth:`get_all`. The request, the HTTP response summary, and any failure
        (with elapsed time) are logged, and a non-2xx status raises via
        :meth:`aiohttp.ClientResponse.raise_for_status`.

        Args:
            session: An open aiohttp session used to issue the request.
            last_id: The id of the last product from the previous page, or
                ``None`` for the first page.

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
        #   base path  : {api_base}/product-projections
        #                api_base = https://api.{region}.commercetools.com/{project_key}
        #                → e.g. https://api.eu.commercetools.com/easmo-dev/product-projections
        #                (A "product projection" is the product's current data already
        #                 flattened — equivalent to masterData.current on /products/{id}.)
        #   query args : limit  — page size (records per page)
        #                sort   — "id asc" so the id cursor below is monotonic
        #                expand — "categories[*].ancestors[*]" so every category
        #                         reference embeds its full ancestor chain under
        #                         categories[i].obj.ancestors[j].obj, which the
        #                         transformer walks to build the category levels.
        #                where  — the CURSOR: only added after page 1. "id > <last_id>"
        #                         returns the records AFTER the previous page's last id
        #                         (more reliable than offset for very large catalogues).
        # ------------------------------------------------------------------
        endpoint = "product-projections"
        url = f"{self._config.api_base}/{endpoint}"
        params = {
            "limit": self.PAGE_SIZE,                      # query: limit=<page_size>
            "sort": "id asc",                              # query: sort=id asc (stable id cursor)
            "expand": "categories[*].ancestors[*]",        # query: expand category ancestor chain
        }
        if last_id:
            # Cursor predicate: fetch products whose id sorts after the last one we saw.
            params["where"] = f'id > "{last_id}"'          # query: where=id > "<last_id>"
        logger.info(
            f"Built GET URL → {url} | query params → {params} "
            f"(api_base={self._config.api_base!r}, endpoint={endpoint!r}, cursor last_id={last_id!r})",
            extra={"function_name": "_fetch_page", "input_data": {"url": url, "params": params, "last_id": last_id, "limit": self.PAGE_SIZE}},
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
                    f"HTTP {status} ← {full_url} → {count} records (total={body.get('total')}) in {elapsed:.3f}s",
                    extra={
                        "function_name": "_fetch_page",
                        "latency": f"{elapsed:.3f}s",
                        "status": "success" if resp.ok else "error",
                        "input_data": {"full_url": full_url, "http_status": status, "returned": count, "total": body.get("total")},
                        "response_payload": body,
                    },
                )
                resp.raise_for_status()
                return body
        except Exception as exc:
            elapsed = (datetime.now() - t0).total_seconds()
            logger.error(
                f"Request failed after {elapsed:.3f}s: GET {url} last_id={last_id!r}: {exc}",
                extra={
                    "function_name": "_fetch_page",
                    "latency": f"{elapsed:.3f}s",
                    "status": "error",
                    "error_details": str(exc),
                    "input_data": {"url": url, "params": params, "last_id": last_id},
                },
            )
            raise
