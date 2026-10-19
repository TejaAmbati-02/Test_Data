"""Shared TLS setup for all outbound HTTP calls.

Behind the corporate proxy/MITM CA, TLS must validate against the OS trust
store (where the corporate root CA lives), not certifi's bundle.

``truststore.inject_into_ssl()`` works for ``requests`` because it reads the
default SSL context, but ``aiohttp`` builds and caches its own SSL contexts
internally and does not reliably pick up that global patch. So we build one
explicit OS-trust-store-backed context here and pass it to every aiohttp call
via ``ssl=SSL_CONTEXT``.
"""
import ssl

import truststore

# One shared context, reusable across every aiohttp session/request.
SSL_CONTEXT: ssl.SSLContext = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
