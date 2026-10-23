"""OpenSearch kNN indexing for the ASMO Data pipeline.

Takes the transformed SKU rows (each carrying a ``content`` blob), embeds that
content with :class:`~asmo.embeddings.OCIEmbedder`, and bulk-indexes every row —
all of its fields plus the ``content_vector`` embedding — into an OpenSearch
index configured for vector (kNN) search.

Behaviour:
    * The index is created from :func:`build_index_body` only when it does not
      already exist; an existing index is left untouched.
    * Each document is indexed with ``_id`` set to the SKU, so re-runs upsert
      (overwrite) rather than duplicate.
    * Every field of the transformed row is mapped explicitly (see
      :func:`build_index_body`) so types are correct for search and filtering.

Connection and index settings are read from the environment (see
:class:`OpenSearchConfig`). Credentials are never logged. This module can be
used as a pipeline step (via :class:`OpenSearchIndexer`) or run standalone (see
``index_data.py`` at the project root).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .embeddings import OCIEmbedder
from .logging_service import logger
from opensearchpy import AsyncOpenSearch


@dataclass
class OpenSearchConfig:
    """Connection and index settings for the OpenSearch cluster.

    All values come from environment variables. ``OPENSEARCH_USER`` and
    ``OPENSEARCH_PASSWORD`` are required; the rest have defaults matching the
    project's dev cluster.
    """

    host: str = field(default_factory=lambda: os.environ.get("OPENSEARCH_HOST", "opensearch.opensearch-dev.svc.cluster.local"))
    port: int = field(default_factory=lambda: int(os.environ.get("OPENSEARCH_PORT", "443")))
    user: str = field(default_factory=lambda: os.environ["OPENSEARCH_USER"])
    password: str = field(default_factory=lambda: os.environ["OPENSEARCH_PASSWORD"])
    index_name: str = field(default_factory=lambda: os.environ.get("OPENSEARCH_INDEX", "asmo-search-api-index"))
    vector_field: str = field(default_factory=lambda: os.environ.get("OPENSEARCH_VECTOR_FIELD", "content_vector"))
    vector_dim: int = field(default_factory=lambda: int(os.environ.get("OPENSEARCH_VECTOR_DIM", "1536")))
    image_vector_dim: int = field(default_factory=lambda: int(os.environ.get("OPENSEARCH_IMAGE_VECTOR_DIM", "1536")))
    timeout: int = field(default_factory=lambda: int(os.environ.get("OPENSEARCH_TIMEOUT", "30")))
    verify_certs: bool = field(default_factory=lambda: os.environ.get("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true")
    bulk_chunk_size: int = field(default_factory=lambda: int(os.environ.get("OPENSEARCH_BULK_CHUNK", "500")))
    recreate_index: bool = field(default_factory=lambda: os.environ.get("OPENSEARCH_RECREATE_INDEX", "false").lower() == "true")


def build_index_body(vector_field: str, dim: int) -> dict:
    """Build the index settings + mapping body for the search index.

    Enables kNN with an HNSW (Lucene, cosine) index on ``vector_field`` and maps
    **every** field produced by the transformer so each has the correct type for
    search, filtering and aggregation: the product title/description and category
    name/description fields use text analyzers; ids, keys, slugs and the seller
    id/unit use keyword (with a searchable text sub-field); price/lead time/MOQ
    are numeric; availability flags are boolean.

    Images are stored in a single ``images`` object holding parallel arrays
    (``image_url``, ``image_base64``, ``image_vector``), kept verbatim in
    ``_source`` but not indexed (``enabled: false``). The image vectors are a
    list-of-vectors and are therefore not kNN-searchable; only ``content_vector``
    powers vector (text) search.

    Args:
        vector_field: Name of the kNN vector field (e.g. ``content_vector``).
        dim: Dimension of the content embedding vectors (1536 for Cohere
            embed-v4.0).

    Returns:
        A dict suitable for passing as the ``body`` of ``indices.create``.
    """

    def text_field(analyzer: str = "standard") -> dict:
        return {
            "type": "text",
            "analyzer": analyzer,
            "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
        }

    def keyword_field() -> dict:
        return {
            "type": "keyword",
            "fields": {"text": {"type": "text", "analyzer": "standard"}},
        }

    def numeric_field(num_type: str) -> dict:
        return {"type": num_type}

    def knn_vector(dimension: int) -> dict:
        return {
            "type": "knn_vector",
            "dimension": dimension,
            "method": {
                "name": "hnsw",
                "space_type": "cosinesimil",
                "engine": "lucene",
                "parameters": {"ef_construction": 128, "m": 24},
            },
        }

    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            }
        },
        "mappings": {
            "_source": {"enabled": True},
            "properties": {
                "content": text_field(),
                vector_field: knn_vector(dim),

                # --- Product ---
                "product_id":          keyword_field(),
                "product_title":       text_field("english"),
                "product_description": text_field("english"),

                # --- Category (top business category) ---
                "category_name":        text_field(),
                "category_id":          keyword_field(),
                "category_slug":        keyword_field(),
                "category_description": text_field(),
                "category_key":         keyword_field(),

                # --- Subcategory ---
                "sub_category_name":        text_field(),
                "sub_category_id":          keyword_field(),
                "sub_category_slug":        keyword_field(),
                "sub_category_description": text_field(),
                "sub_category_key":         keyword_field(),

                # --- Product type (leaf category level) ---
                "product_type_name":        text_field(),
                "product_type_id":          keyword_field(),
                "product_type_slug":        keyword_field(),
                "product_type_description": text_field(),
                "product_type_key":         keyword_field(),

                # --- Master-variant / SKU ---
                "sku_key":       keyword_field(),
                "sku_id":        keyword_field(),
                "sku_image_url": keyword_field(),
                "image_urls":    keyword_field(),

                # --- Images (parallel arrays, store-only) ---
                # One object holding three positional arrays, one slot per image:
                #   image_url:    ["url1", "url2", ...]
                #   image_base64: ["b64_1", "b64_2", ...]
                #   image_vector: [[float, ...], [float, ...], ...]
                # enabled:false keeps the whole object verbatim in _source but
                # indexes nothing. NOTE: image_vector is a list-of-vectors, so it
                # is NOT kNN-searchable — image/visual search is unavailable with
                # this schema (chosen deliberately). content_vector still powers
                # text search.
                "images": {"type": "object", "enabled": False},

                # --- Seller / price (from the selected standalone price) ---
                "seller_id":               keyword_field(),
                "seller_name":             text_field("english"),
                "price":                   numeric_field("float"),
                "currency_type":           keyword_field(),
                "lead_time":               numeric_field("integer"),
                "moq":                     numeric_field("integer"),
                "unit_type":               keyword_field(),
                "temporary_unavailable":   {"type": "boolean"},
                "is_blocked_for_deletion": {"type": "boolean"},
            },
        },
    }


class OpenSearchIndexer:
    """Creates the search index (if needed) and bulk-indexes transformed rows.

    Embeddings are produced by the injected :class:`~asmo.embeddings.OCIEmbedder`
    (a default one, reading the environment, is created when none is supplied).
    The OpenSearch client is created lazily and must be released with
    :meth:`close` when done.
    """

    def __init__(self, embedder: OCIEmbedder | None = None, config: OpenSearchConfig | None = None) -> None:
        """Initialise the indexer.

        Args:
            embedder: Embedding provider; defaults to a new :class:`OCIEmbedder`.
            config: OpenSearch settings; defaults to reading the environment.
        """
        self._config = config or OpenSearchConfig()
        self._embedder = embedder or OCIEmbedder()
        self._client = None

    def _connect(self):
        """Build (once) and return the async OpenSearch client.

        ``opensearchpy`` is imported here so the rest of the package imports even
        when it is not installed.
        """
        if self._client is None:
            self._client = AsyncOpenSearch(
                hosts=[{"host": self._config.host, "port": self._config.port}],
                http_auth=(self._config.user, self._config.password),
                use_ssl=True,
                verify_certs=self._config.verify_certs,
                ssl_show_warn=False,
                http_compress=True,
                timeout=self._config.timeout,
            )
            logger.info(
                f"OpenSearch client ready — https://{self._config.host}:{self._config.port} index={self._config.index_name}",
                extra={"function_name": "OpenSearchIndexer._connect",
                       "input_data": {"host": self._config.host, "port": self._config.port, "index": self._config.index_name}},
            )
        return self._client

    @property
    def config(self) -> OpenSearchConfig:
        """The resolved OpenSearch configuration (host, index name, dims, …)."""
        return self._config

    async def ping(self) -> bool:
        """Return True if the OpenSearch cluster is reachable, False otherwise.

        Used by standalone tools to fail fast with a clear message before
        attempting to create an index or bulk-index, rather than surfacing a
        low-level connection error mid-run.
        """
        client = self._connect()
        try:
            return bool(await client.ping())
        except Exception as exc:
            logger.error(
                f"OpenSearch ping failed for https://{self._config.host}:{self._config.port}: {exc}",
                extra={"function_name": "OpenSearchIndexer.ping", "status": "error", "error_details": str(exc)},
            )
            return False

    async def index_exists(self) -> bool:
        """Return True if the configured index already exists on the cluster."""
        client = self._connect()
        return bool(await client.indices.exists(index=self._config.index_name))

    async def _existing_vector_dim(self) -> int | None:
        """Return the mapped dimension of the vector field on the live index.

        Returns ``None`` when the index or the vector field mapping is absent (or
        cannot be read), so callers can distinguish "no mapping" from a concrete
        dimension.
        """
        client = self._connect()
        try:
            mapping = await client.indices.get_mapping(index=self._config.index_name)
            props = mapping[self._config.index_name]["mappings"]["properties"]
            return props[self._config.vector_field].get("dimension")
        except Exception:
            return None

    async def ensure_index(self, recreate: bool | None = None) -> bool:
        """Create the index from :func:`build_index_body`, if needed.

        If the index already exists, its ``content_vector`` dimension is checked
        against the configured :attr:`OpenSearchConfig.vector_dim`. A mismatch
        (e.g. an index built at 384 dims while the embedder now returns 1536)
        would make every document fail to index, so this raises a clear error
        telling you to drop and rebuild — unless ``recreate=True``, in which case
        the index is deleted and recreated at the configured dimension.

        Args:
            recreate: When ``True``, drop any existing index and recreate it
                (destructive — all previously indexed documents are lost). When
                ``None`` (default), falls back to
                :attr:`OpenSearchConfig.recreate_index` (env
                ``OPENSEARCH_RECREATE_INDEX``).

        Returns:
            ``True`` if the index was created, ``False`` if a compatible index
            already existed and was reused.

        Raises:
            RuntimeError: If the existing index's vector dimension does not match
                the configured dimension and recreate is not requested.
        """
        if recreate is None:
            recreate = self._config.recreate_index
        client = self._connect()
        exists = await client.indices.exists(index=self._config.index_name)

        if exists and recreate:
            logger.warning(
                f"Recreating index '{self._config.index_name}' — existing data will be dropped",
                extra={"function_name": "ensure_index", "input_data": {"index": self._config.index_name}},
            )
            await client.indices.delete(index=self._config.index_name)
            exists = False

        if exists:
            existing_dim = await self._existing_vector_dim()
            if existing_dim is not None and existing_dim != self._config.vector_dim:
                msg = (
                    f"Index '{self._config.index_name}' exists with {self._config.vector_field} "
                    f"dimension {existing_dim}, but embeddings are {self._config.vector_dim}-dim. "
                    f"Every document would be rejected. Drop and rebuild the index "
                    f"(DELETE {self._config.index_name}) or re-run with recreate=True."
                )
                logger.error(
                    msg,
                    extra={"function_name": "ensure_index", "status": "error",
                           "input_data": {"existing_dim": existing_dim, "configured_dim": self._config.vector_dim}},
                )
                raise RuntimeError(msg)
            logger.info(
                f"Index '{self._config.index_name}' already exists (dim={existing_dim}) — skipping creation",
                extra={"function_name": "ensure_index", "input_data": {"index": self._config.index_name, "dim": existing_dim}},
            )
            return False

        body = build_index_body(self._config.vector_field, self._config.vector_dim)
        response = await client.indices.create(index=self._config.index_name, body=body)
        logger.info(
            f"Index '{self._config.index_name}' created at dim={self._config.vector_dim} — acknowledged={response.get('acknowledged')}",
            extra={"function_name": "ensure_index", "status": "success",
                   "input_data": {"index": self._config.index_name, "dim": self._config.vector_dim, "acknowledged": response.get("acknowledged")}},
        )
        return True

    async def index_rows(self, transformed: dict[str, dict]) -> dict:
        """Embed each row's ``content`` and bulk-index all rows into OpenSearch.

        Ensures the index exists, embeds the ``content`` blob of every row, then
        bulk-indexes each row (all fields plus the ``content_vector`` embedding)
        using the SKU as the document ``_id``. Failures for individual documents
        are collected and reported, not raised.

        Args:
            transformed: Mapping of SKU to transformed row (as produced by the
                transformer, including the ``content`` field).

        Returns:
            A summary dict with ``indexed`` and ``failed`` counts.
        """
        if not transformed:
            logger.warning("No transformed rows to index", extra={"function_name": "index_rows"})
            return {"indexed": 0, "failed": 0}

        client = self._connect()
        await self.ensure_index()

        skus = list(transformed.keys())
        rows = [transformed[sku] for sku in skus]
        texts = [(row.get("content") or "") for row in rows]

        logger.info(
            f"Indexing {len(rows)} rows into '{self._config.index_name}' — embedding content first",
            extra={"function_name": "index_rows", "input_data": {"rows": len(rows), "index": self._config.index_name}},
        )
        vectors = await self._embedder.embed(texts, input_type="SEARCH_DOCUMENT")
        if len(vectors) != len(rows):
            raise RuntimeError(f"Embedding count {len(vectors)} does not match row count {len(rows)}")

        actions = []
        for sku, row, vector in zip(skus, rows, vectors):
            source = dict(row)
            source[self._config.vector_field] = vector
            actions.append({"_index": self._config.index_name, "_id": sku, "_source": source})

        from opensearchpy.helpers import async_bulk

        indexed, errors = await async_bulk(
            client,
            actions,
            chunk_size=self._config.bulk_chunk_size,
            raise_on_error=False,
            stats_only=False,
        )
        failed = len(errors) if isinstance(errors, list) else int(errors)
        if failed:
            logger.error(
                f"Bulk index completed with {failed} failure(s) out of {len(actions)}",
                extra={"function_name": "index_rows", "status": "error",
                       "input_data": {"indexed": indexed, "failed": failed}, "error_details": str(errors[:3]) if isinstance(errors, list) else str(errors)},
            )
        logger.info(
            f"Index complete — {indexed} indexed, {failed} failed",
            extra={"function_name": "index_rows", "status": "success", "input_data": {"indexed": indexed, "failed": failed}},
        )
        return {"indexed": indexed, "failed": failed}

    async def close(self) -> None:
        """Close the underlying OpenSearch client if one was opened."""
        if self._client is not None:
            await self._client.close()
            self._client = None
