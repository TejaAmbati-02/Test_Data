"""OCI Generative AI text embeddings for the ASMO Data pipeline.

Wraps OCI Generative AI's Cohere *embed* models to turn text — specifically the
transformed row ``content`` blob — into dense vectors for OpenSearch kNN search.
The default model ``cohere.embed-v4.0`` is requested at 1536 dimensions (via
``output_dimensions``), which matches the search-index mapping and suits the
mixed English/German product text.

Note: the OCI chat model ``cohere.command-a-03-2025`` cannot produce
embeddings; embeddings come from the ``cohere.embed-*`` family via the
``embed_text`` API. ``cohere.embed-v4.0`` supports configurable output
dimensions (256/512/1024/1536); the ``*-light-v3.0`` variants are fixed at 384.

Configuration is read from environment variables (see
:class:`OCIEmbedderConfig`). The OCI Python SDK is synchronous, so blocking
calls are offloaded to worker threads via :func:`asyncio.to_thread` to stay
friendly to the otherwise asynchronous pipeline. Credentials live in the OCI
config file and are never logged.
"""
from __future__ import annotations

import asyncio
import os
import oci
from dataclasses import dataclass, field

from .logging_service import logger

# Cohere embed on OCI accepts at most 96 inputs per request.
_MAX_BATCH = 96


@dataclass
class OCIEmbedderConfig:
    """Connection and model settings for OCI GenAI embeddings.

    All values come from environment variables so deployment-specific details
    stay out of the code. ``OCI_COMPARTMENT_ID`` is required; the rest have
    sensible defaults.

    Attributes:
        config_file: Path to the OCI SDK config file (``OCI_CONFIG_FILE``).
        profile: Profile name within that config file (``OCI_CONFIG_PROFILE``).
        region: OCI region hosting the GenAI inference endpoint (``OCI_REGION``).
        compartment_id: Compartment OCID the model runs in (``OCI_COMPARTMENT_ID``).
        model_id: Embed model id (``OCI_EMBED_MODEL_ID``).
        dimension: Expected embedding dimension (``OCI_EMBED_DIM``); used only as
            a sanity check against the vectors returned.
    """

    config_file: str = field(default_factory=lambda: os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"))
    profile: str = field(default_factory=lambda: os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT"))
    region: str = field(default_factory=lambda: os.environ.get("OCI_REGION", "me-riyadh-1"))
    compartment_id: str = field(default_factory=lambda: os.environ["OCI_COMPARTMENT_ID"])
    model_id: str = field(default_factory=lambda: os.environ.get("OCI_EMBED_MODEL_ID", "cohere.embed-v4.0"))
    dimension: int = field(default_factory=lambda: int(os.environ.get("OCI_EMBED_DIM", "1536")))

    @property
    def endpoint(self) -> str:
        """Return the GenAI inference service endpoint URL for the region."""
        return f"https://inference.generativeai.{self.region}.oci.oraclecloud.com"


class OCIEmbedder:
    """Generates text embeddings via OCI Generative AI (Cohere embed models).

    The underlying OCI client is created lazily on first use and reused for the
    lifetime of the embedder. Inputs are embedded in batches of at most
    :data:`_MAX_BATCH` (96), and each blocking SDK call runs on a worker thread.
    """

    def __init__(self, config: OCIEmbedderConfig | None = None, batch_size: int = _MAX_BATCH) -> None:
        """Initialise the embedder.

        Args:
            config: Embedder configuration; a default-constructed
                :class:`OCIEmbedderConfig` (reading the environment) is used
                when omitted.
            batch_size: Maximum inputs per OCI request; capped at 96.
        """
        self._config = config or OCIEmbedderConfig()
        self._batch_size = max(1, min(batch_size, _MAX_BATCH))
        self._client = None

    def _client_sync(self):
        """Build (once) and return the synchronous OCI GenAI inference client.

        The ``oci`` SDK is imported here so the rest of the package imports even
        when OCI is not installed. Credentials are read from the configured OCI
        config file and profile; the region is forced to the configured value.
        """
        if self._client is None:
            oci_config = oci.config.from_file(
                file_location=os.path.expanduser(self._config.config_file),
                profile_name=self._config.profile,
            )
            oci_config["region"] = self._config.region
            self._client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=self._config.endpoint,
            )
            logger.info(
                f"OCI GenAI embedder ready — model={self._config.model_id} region={self._config.region}",
                extra={
                    "function_name": "OCIEmbedder._client_sync",
                    "input_data": {"model_id": self._config.model_id, "region": self._config.region, "endpoint": self._config.endpoint},
                },
            )
        return self._client

    def _embed_batch_sync(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Embed one batch (<= batch size) of texts with a blocking OCI call.

        Args:
            texts: The input strings to embed.
            input_type: Cohere input type — ``"SEARCH_DOCUMENT"`` when indexing
                documents, ``"SEARCH_QUERY"`` when embedding a search query.

        Returns:
            One embedding vector (list of floats) per input, in input order.
        """
        import oci

        client = self._client_sync()
        details = oci.generative_ai_inference.models.EmbedTextDetails(
            inputs=texts,
            serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                serving_type="ON_DEMAND",
                model_id=self._config.model_id,
            ),
            compartment_id=self._config.compartment_id,
            input_type=input_type,
            # END (not NONE) so an over-length content blob is truncated rather
            # than failing the whole batch; Cohere embed-v4.0's context is large
            # so this rarely triggers.
            truncate="END",
            embedding_types=["float"],
            output_dimensions=self._config.dimension,
            is_echo=False,
        )
        response = client.embed_text(details)
        return self._extract_vectors(response.data)

    @staticmethod
    def _extract_vectors(result) -> list[list[float]]:
        """Pull float vectors from an EmbedTextResult.

        When ``embedding_types`` is requested, Cohere embed-v4.0 may return the
        vectors under ``embeddings`` (the legacy field) and/or keyed by type in
        ``embeddings_by_type`` (e.g. ``{"float": [[...], ...]}``). Prefer the
        plain ``embeddings`` list and fall back to the ``float`` entry of
        ``embeddings_by_type``.
        """
        vectors = getattr(result, "embeddings", None)
        if vectors:
            return vectors
        by_type = getattr(result, "embeddings_by_type", None) or {}
        if isinstance(by_type, dict):
            return by_type.get("float") or by_type.get("FLOAT") or []
        return getattr(by_type, "float", None) or []

    async def embed(self, texts: list[str], input_type: str = "SEARCH_DOCUMENT") -> list[list[float]]:
        """Embed a list of texts, batching requests and awaiting off-thread.

        Args:
            texts: The input strings to embed.
            input_type: Cohere input type (see :meth:`_embed_batch_sync`).

        Returns:
            One embedding vector per input, in the same order as ``texts``.
            Returns an empty list when ``texts`` is empty.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        total_batches = (len(texts) + self._batch_size - 1) // self._batch_size
        for batch_num, start in enumerate(range(0, len(texts), self._batch_size), start=1):
            batch = texts[start:start + self._batch_size]
            logger.info(
                f"Embedding batch {batch_num}/{total_batches} ({len(batch)} inputs)",
                extra={"function_name": "OCIEmbedder.embed", "input_data": {"batch": batch_num, "size": len(batch), "input_type": input_type}},
            )
            batch_vectors = await asyncio.to_thread(self._embed_batch_sync, batch, input_type)
            vectors.extend(batch_vectors)

        if vectors and len(vectors[0]) != self._config.dimension:
            logger.warning(
                f"Embedding dimension mismatch — got {len(vectors[0])}, expected {self._config.dimension}",
                extra={"function_name": "OCIEmbedder.embed", "status": "error",
                       "input_data": {"got": len(vectors[0]), "expected": self._config.dimension}},
            )
        logger.info(
            f"Embedded {len(vectors)} texts in {total_batches} batch(es)",
            extra={"function_name": "OCIEmbedder.embed", "status": "success", "input_data": {"count": len(vectors)}},
        )
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single search query string and return its vector."""
        vectors = await self.embed([text], input_type="SEARCH_QUERY")
        return vectors[0] if vectors else []
