"""Centralized configuration for CommerceTools connectivity.

This module defines :class:`Config`, a single dataclass that gathers every
setting needed to authenticate against and query a CommerceTools project.
Credentials and project coordinates are read from environment variables when a
``Config`` is instantiated, so :func:`dotenv.load_dotenv` must be called first
(done in ``main.py``) to make ``.env`` values available. Derived endpoints and
OAuth scopes are exposed as read-only properties computed from those settings.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """All CommerceTools connection settings in one place.

    Reads credentials from environment variables; call load_dotenv() before
    instantiating (done in main.py) so .env values are present.

    Required environment variables (``CT_CLIENT_ID``, ``CT_CLIENT_SECRET`` and
    ``CT_CHANNEL_ID``) are read eagerly via ``default_factory`` and will raise
    ``KeyError`` on instantiation if missing. ``CT_PROJECT_KEY`` and
    ``CT_REGION`` are optional, and ``UPLOAD_ENABLED`` defaults to ``False``.

    Attributes:
        client_id: OAuth client id (from ``CT_CLIENT_ID``).
        client_secret: OAuth client secret (from ``CT_CLIENT_SECRET``).
        project_key: CommerceTools project key (from ``CT_PROJECT_KEY``).
        region: CommerceTools region/host segment (from ``CT_REGION``).
        channel_id: Supply/inventory channel id (from ``CT_CHANNEL_ID``).
        upload_enabled: Whether uploads are enabled (from ``UPLOAD_ENABLED``).
        index_enabled: Whether OpenSearch indexing is enabled (from ``INDEX_ENABLED``).
        selection_validation_enabled: Whether to gate products on a valid, active
            product selection (contracted scope) before transforming them (from
            ``SELECTION_VALIDATION_ENABLED``). Defaults to ``False`` so the
            bulk-only run keeps every product; when ``True``, products with no
            active selection whose ``validUntil`` is in the future are excluded.
        save_local: Whether to also write products/prices/transformed JSON and
            images to local disk (from ``SAVE_LOCAL``). Defaults to ``False`` so
            data streams straight to storage/index without filling the disk.
        image_content_local: Whether the base64 ``image_content`` is also embedded
            in the *local* transformed JSON files (from ``IMAGE_CONTENT_LOCAL``).
            Only takes effect when ``save_local`` is on. Defaults to ``False`` so
            the base64 blobs go only to the uploaded storage JSON and local files
            stay lean.
        page_size: Page size used when paging through API results.
    """

    client_id: str = field(default_factory=lambda: os.environ["CT_CLIENT_ID"])
    client_secret: str = field(default_factory=lambda: os.environ["CT_CLIENT_SECRET"])
    project_key: str = field(default_factory=lambda: os.environ.get("CT_PROJECT_KEY"))
    region: str = field(default_factory=lambda: os.environ.get("CT_REGION"))
    channel_id: str = field(default_factory=lambda: os.environ["CT_CHANNEL_ID"])
    upload_enabled: bool = field(default_factory=lambda: os.environ.get("UPLOAD_ENABLED", "false").lower() == "true")
    index_enabled: bool = field(default_factory=lambda: os.environ.get("INDEX_ENABLED", "false").lower() == "true")
    selection_validation_enabled: bool = field(default_factory=lambda: os.environ.get("SELECTION_VALIDATION_ENABLED", "false").lower() == "true")
    save_local: bool = field(default_factory=lambda: os.environ.get("SAVE_LOCAL", "false").lower() == "true")
    image_content_local: bool = field(default_factory=lambda: os.environ.get("IMAGE_CONTENT_LOCAL", "false").lower() == "true")
    page_size: int = 500

    @property
    def auth_url(self) -> str:
        """The OAuth token endpoint for the configured region."""
        return f"https://auth.{self.region}.commercetools.com/oauth/token"

    @property
    def api_base(self) -> str:
        """The base API URL for the configured region and project."""
        return f"https://api.{self.region}.commercetools.com/{self.project_key}"

    @property
    def scope(self) -> str:
        """The space-separated OAuth scope string for the configured project.

        Each required scope is suffixed with the project key, as expected by the
        CommerceTools OAuth server.
        """
        scopes = [
            "manage_project",
            "view_audit_log",
        ]
        return " ".join(f"{s}:{self.project_key}" for s in scopes)
