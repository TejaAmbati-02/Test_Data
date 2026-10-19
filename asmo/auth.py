"""OAuth2 authentication for the CommerceTools API.

This module encapsulates the CommerceTools client-credentials flow. It exposes
:class:`AuthClient`, which acquires and caches a bearer access token and hands
out ready-to-use ``Authorization`` headers, and :func:`requires_token`, a
decorator that transparently authenticates before a protected coroutine runs.

All network interaction goes through :mod:`aiohttp` using the project's shared
SSL context, and every step is logged through the structured logger. Secrets
(the client secret and the access token) are never written to the logs.
"""

import aiohttp
from functools import wraps
from .config import Config
from .ssl_support import SSL_CONTEXT
from .logging_service import logger, log_function


def requires_token(method):
    """Decorator ensuring a valid access token exists before ``method`` runs.

    Wraps an async method so that, on invocation, it locates the relevant
    :class:`AuthClient` — ``self`` itself when the decorated object *is* an
    ``AuthClient``, otherwise ``self._auth`` on a client that holds one — and
    calls :meth:`AuthClient.authenticate` if no token has been acquired yet.
    Once a token is guaranteed to be present, the original method is awaited and
    its result returned unchanged.

    Args:
        method: The coroutine method to protect.

    Returns:
        An async wrapper that lazily authenticates, then delegates to
        ``method``.
    """
    @wraps(method)
    async def wrapper(self, *args, **kwargs):
        auth: "AuthClient" = getattr(self, "_auth", self)
        if not auth._token:
            await auth.authenticate()
        return await method(self, *args, **kwargs)
    return wrapper


class AuthClient:
    """Acquires and holds a CommerceTools OAuth2 client-credentials token.

    A single instance is shared by the API clients. It stores the configuration
    needed to reach the auth server and caches the most recently fetched bearer
    token so repeated requests reuse it rather than re-authenticating. Use
    :meth:`authenticate` to obtain a token and the :attr:`headers` property to
    build authorised request headers.
    """

    def __init__(self, config: Config) -> None:
        """Initialise the client with its configuration and no token yet.

        Args:
            config: Configuration providing the auth URL, client credentials and
                scope used when authenticating.
        """
        self._config = config
        self._token: str | None = None


    @log_function
    async def authenticate(self) -> None:
        """Fetch a fresh access token from CommerceTools and cache it.

        Opens a short-lived :class:`aiohttp.ClientSession`, delegates the actual
        token request to :meth:`_fetch_token`, and stores the returned token on
        the instance for later reuse via :attr:`token` and :attr:`headers`. Any
        failure raised by :meth:`_fetch_token` propagates to the caller. The
        token value itself is never logged.
        """
        logger.info("Authenticating with CommerceTools", extra={"function_name": "authenticate"})
        async with aiohttp.ClientSession() as session:
            self._token = await self._fetch_token(session)
        logger.info("Token acquired successfully", extra={"function_name": "authenticate", "status": "success"})

    @property
    def token(self) -> str:
        """Return the cached access token.

        Returns:
            The bearer access token acquired by :meth:`authenticate`.

        Raises:
            RuntimeError: If :meth:`authenticate` has not been called yet, so no
                token is available.
        """
        if not self._token:
            raise RuntimeError("Call authenticate() before accessing the token.")
        return self._token

    @property
    def headers(self) -> dict:
        """Return the HTTP headers carrying the bearer token.

        Returns:
            A dict with a single ``Authorization`` header of the form
            ``"Bearer <token>"``, suitable for authorised API requests.

        Raises:
            RuntimeError: Propagated from :attr:`token` if no token has been
                acquired yet.
        """
        return {"Authorization": f"Bearer {self.token}"}


    async def _fetch_token(self, session: aiohttp.ClientSession) -> str:
        """Perform the OAuth2 client-credentials token request.

        Sends a form-encoded ``POST`` to the configured auth URL with
        ``grant_type=client_credentials`` and the configured scope, using HTTP
        Basic auth built from the client id and secret and the shared SSL
        context. The JSON response is parsed and logged, but the
        ``access_token`` field is masked (replaced with ``"***"``) before being
        recorded so the token never reaches the logs.

        Error handling distinguishes three cases, each logged and re-raised as a
        ``RuntimeError`` (except the generic branch, which re-raises as-is):

        * :class:`aiohttp.ClientResponseError` — a non-2xx status; the server's
          ``error_description``/``message`` is surfaced in the raised message.
        * :class:`aiohttp.ClientConnectionError` — the auth server was
          unreachable.
        * any other :class:`Exception` — an unexpected failure.

        Args:
            session: An open aiohttp session used to issue the request.

        Returns:
            The ``access_token`` string from a successful response.

        Raises:
            RuntimeError: On an HTTP error response or a connection failure.
            Exception: Any unexpected error is logged and re-raised unchanged.
        """
        from datetime import datetime
        request_data = {"grant_type": "client_credentials", "scope": self._config.scope}
        logger.info(
            f"POST {self._config.auth_url} client_id={self._config.client_id} data={request_data}",
            extra={
                "function_name": "_fetch_token",
                "input_data": {"url": self._config.auth_url, "client_id": self._config.client_id, "data": request_data},
            },
        )
        t0 = datetime.now()
        try:
            async with session.post(
                self._config.auth_url,
                auth=aiohttp.BasicAuth(self._config.client_id, self._config.client_secret),
                data=request_data,
                ssl=SSL_CONTEXT,
            ) as resp:
                full_url = str(resp.url)
                status = resp.status
                body = await resp.json(content_type=None)
                elapsed = (datetime.now() - t0).total_seconds()
                safe_body = {k: ("***" if k == "access_token" else v) for k, v in body.items()}
                logger.info(
                    f"HTTP {status} ← {full_url} in {elapsed:.3f}s (token_type={body.get('token_type')}, expires_in={body.get('expires_in')})",
                    extra={
                        "function_name": "_fetch_token",
                        "latency": f"{elapsed:.3f}s",
                        "status": "success" if resp.ok else "error",
                        "input_data": {"full_url": full_url, "http_status": status, "token_type": body.get("token_type"), "expires_in": body.get("expires_in")},
                        "response_payload": safe_body,
                    },
                )
                resp.raise_for_status()
                return body["access_token"]
        except aiohttp.ClientResponseError as e:
            error_msg = body.get("error_description") or body.get("message") or str(e)
            logger.error(
                f"Authentication failed ({e.status}): {error_msg}",
                extra={"function_name": "_fetch_token", "status": "error", "error_details": error_msg},
            )
            raise RuntimeError(f"Authentication failed ({e.status}): {error_msg}") from e
        except aiohttp.ClientConnectionError as e:
            logger.error(
                f"Could not reach auth server: {self._config.auth_url}",
                extra={"function_name": "_fetch_token", "status": "error", "error_details": str(e)},
            )
            raise RuntimeError(f"Could not reach auth server: {self._config.auth_url}") from e
        except Exception as e:
            logger.error(
                f"Unexpected error during authentication: {e}",
                extra={"function_name": "_fetch_token", "status": "error", "error_details": str(e)},
            )
            raise
        finally:
            logger.info("Token fetch attempt complete")
