"""ASMO Data package.

This package implements the ASMO data pipeline: fetching products and prices
from the upstream commercetools-style APIs, transforming them into the shapes
the downstream search index expects, exporting the results to local JSON files,
and uploading those artifacts to OpenShift ODF (S3-compatible) object storage.

Public surface (re-exported for convenience):
    - ``Pipeline`` / ``PipelineResult``: the orchestration entry point and its
      structured result object.
    - ``Config``: environment-backed runtime configuration.
    - ``Product`` / ``Price``: the core domain models.
    - ``Exporter`` / ``ExportPaths``: local export helpers.
    - ``transform``: the product/price transformation function.
    - ``setup_logging`` / ``logger``: the shared logging service entry points.

Importing from ``asmo`` directly (e.g. ``from asmo import Pipeline``) is the
supported way to consume these symbols; submodule layout is an implementation
detail and may change.
"""
from .pipeline import Pipeline, PipelineResult
from .config import Config
from .products import Product
from .prices import Price
from .exporter import Exporter, ExportPaths
from .transformer import transform
from .logging_service import setup_logging, logger

__all__ = ["Pipeline", "PipelineResult", "Config", "Product", "Price", "Exporter", "ExportPaths", "transform", "setup_logging", "logger"]
