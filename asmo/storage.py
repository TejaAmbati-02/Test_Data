"""Object storage access layer for OpenShift ODF (S3-compatible) buckets.

This module encapsulates everything the pipeline needs to move JSON artifacts
between the local filesystem and the project's S3-compatible object store:

    - :class:`Folder`: the fixed set of logical "folders" (key prefixes) used to
      namespace objects inside the single bucket.
    - :class:`ObjectStorageConfig`: environment-backed connection configuration.
    - :class:`LocalFolderReader`: a helper that reads (and optionally validates)
      JSON files from a local directory prior to upload.
    - :class:`ObjectStorageClient`: the async S3 client that performs bucket,
      upload, download, read, delete, existence, metadata and presigned-URL
      operations.

All network operations are implemented with ``aioboto3`` and are fully
asynchronous. A single S3 client is opened lazily on first use and reused by
every method (and every concurrent task) for the lifetime of the instance; it
is closed by :meth:`ObjectStorageClient.aclose` or by using the client as an
``async with`` context manager.

Logging follows the project-wide convention: a human-readable message plus an
``extra`` dict carrying structured fields (``function_name``, ``input_data``,
``response_payload``, ``status``, ``error_details``, ``latency``). Credentials
(access key, secret key) are never logged.
"""
import asyncio
import aioboto3
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from botocore.client import Config
from typing import Optional
from enum import Enum
from .logging_service import logger


class Folder(str, Enum):
    """Logical folders (S3 key prefixes) used to namespace objects in the bucket.

    The project stores everything inside a single bucket and uses a leading key
    prefix to separate concerns. Each member's value is the prefix string that
    is prepended to object keys (e.g. ``products-data/<filename>``).

    Being a :class:`str` subclass, members compare equal to and serialize as
    their string value, so they can be used directly wherever a prefix string
    is expected.

    Members:
        PRODUCTS: Prefix for transformed product documents.
        CATEGORY: Prefix for category documents.
        IMAGES: Prefix for product image files.
    """

    PRODUCTS = "products-data"
    CATEGORY = "category-data"
    IMAGES = "products-images"


class ObjectStorageConfig:
    """Connection configuration for OpenShift ODF (S3-compatible) object storage.

    All settings are read from environment variables at construction time, so a
    missing required variable surfaces immediately as a :class:`KeyError`.

    Attributes:
        endpoint_url: Fully-qualified HTTPS endpoint built from ``S3_HOST`` and
            ``S3_PORT`` (defaults to port ``443``).
        bucket_name: Target bucket name (``S3_BUCKET_NAME``).
        access_key: S3 access key id (``S3_ACCESS_KEY``). Sensitive — never log.
        secret_key: S3 secret access key (``S3_SECRET_KEY``). Sensitive — never
            log.
        verify_ssl: Whether to verify TLS certificates (``S3_VERIFY_SSL``,
            defaults to ``False``).

    Environment variables:
        Required: ``S3_HOST``, ``S3_BUCKET_NAME``, ``S3_ACCESS_KEY``,
        ``S3_SECRET_KEY``.
        Optional: ``S3_PORT`` (default ``443``), ``S3_VERIFY_SSL`` (default
        ``false``).

    Raises:
        KeyError: If any required environment variable is not set.
    """

    def __init__(self):
        """Read S3 connection settings from the environment.

        Credentials are stored on the instance but are intentionally excluded
        from any logging. A log line is emitted recording only the non-sensitive
        connection parameters (endpoint, bucket, TLS verification flag).
        """
        logger.debug(
            "Loading ObjectStorageConfig from environment",
            extra={"function_name": "ObjectStorageConfig.__init__"},
        )
        self.endpoint_url = f"https://{os.environ['S3_HOST']}:{os.environ.get('S3_PORT', '443')}"
        self.bucket_name = os.environ["S3_BUCKET_NAME"]
        self.access_key = os.environ["S3_ACCESS_KEY"]
        self.secret_key = os.environ["S3_SECRET_KEY"]
        self.verify_ssl = os.environ.get("S3_VERIFY_SSL", "false").lower() == "true"
        logger.info(
            f"Object storage configured for bucket '{self.bucket_name}' at '{self.endpoint_url}'",
            extra={
                "function_name": "ObjectStorageConfig.__init__",
                "input_data": {
                    "endpoint_url": self.endpoint_url,
                    "bucket_name": self.bucket_name,
                    "verify_ssl": self.verify_ssl,
                    "access_key": "***",
                    "secret_key": "***",
                },
                "status": "success",
            },
        )


class LocalFolderReader:
    """Reads (and optionally validates) JSON files from a local directory.

    Typically used as the source side of an upload: a directory of exported
    JSON files is read into memory as ``(filename, bytes)`` pairs that are then
    pushed to object storage.

    Attributes:
        folder_path: The :class:`~pathlib.Path` of the directory being read.
    """

    def __init__(self, folder_path: str):
        """Initialise the reader and validate that the directory exists.

        Args:
            folder_path: Path to the local directory to read JSON files from.

        Raises:
            FileNotFoundError: If the path does not exist.
            NotADirectoryError: If the path exists but is not a directory.
        """
        logger.debug(
            f"Initialising LocalFolderReader for '{folder_path}'",
            extra={"function_name": "LocalFolderReader.__init__", "input_data": {"folder_path": folder_path}},
        )
        self.folder_path = Path(folder_path)
        self._validate()

    def _validate(self):
        """Ensure :attr:`folder_path` exists and is a directory.

        Raises:
            FileNotFoundError: If the path does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        if not self.folder_path.exists():
            logger.error(
                f"Local folder not found: {self.folder_path}",
                extra={"function_name": "LocalFolderReader._validate", "status": "error", "input_data": {"folder_path": str(self.folder_path)}},
            )
            raise FileNotFoundError(f"Local folder not found: {self.folder_path}")
        if not self.folder_path.is_dir():
            logger.error(
                f"Path is not a directory: {self.folder_path}",
                extra={"function_name": "LocalFolderReader._validate", "status": "error", "input_data": {"folder_path": str(self.folder_path)}},
            )
            raise NotADirectoryError(f"Path is not a directory: {self.folder_path}")
        logger.debug(
            f"Validated local folder '{self.folder_path}'",
            extra={"function_name": "LocalFolderReader._validate", "status": "success"},
        )

    def get_json_files(self) -> list[Path]:
        """List the ``*.json`` files in the directory, sorted by name.

        Returns:
            A sorted list of :class:`~pathlib.Path` objects for every top-level
            ``.json`` file in the directory (non-recursive). An empty list is
            returned (with a warning logged) when no JSON files are present.
        """
        logger.debug(
            f"Scanning '{self.folder_path}' for JSON files",
            extra={"function_name": "get_json_files", "input_data": {"folder_path": str(self.folder_path)}},
        )
        files = sorted(self.folder_path.glob("*.json"))
        if not files:
            logger.warning(f"No JSON files found in '{self.folder_path}'")
        else:
            logger.info(f"Found {len(files)} JSON file(s) in '{self.folder_path}'")
        return files

    def read_file(self, file_path: Path) -> bytes:
        """Read a single file's raw bytes without validation.

        Args:
            file_path: Path to the file to read.

        Returns:
            The file's contents as ``bytes``.

        Raises:
            OSError: If the file cannot be read.
        """
        logger.debug(
            f"Reading file '{file_path.name}' (no validation)",
            extra={"function_name": "read_file", "input_data": {"file_path": str(file_path)}},
        )
        return file_path.read_bytes()

    def read_and_validate_file(self, file_path: Path) -> bytes:
        """Read a file's raw bytes and verify they are valid JSON.

        The bytes are returned unmodified; parsing is only used to validate the
        payload, so the original formatting is preserved on the way to storage.

        Args:
            file_path: Path to the JSON file to read.

        Returns:
            The file's raw contents as ``bytes``.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the contents are not valid JSON.
        """
        logger.debug(
            f"Reading and validating JSON file '{file_path.name}'",
            extra={"function_name": "read_and_validate_file", "input_data": {"file_path": str(file_path)}},
        )
        raw = file_path.read_bytes()
        try:
            json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(
                f"Invalid JSON in '{file_path.name}': {e}",
                extra={"function_name": "read_and_validate_file", "status": "error", "error_details": str(e), "input_data": {"file_path": str(file_path)}},
            )
            raise ValueError(f"Invalid JSON in '{file_path.name}': {e}")
        return raw

    def load_all(self, validate: bool = True) -> list[tuple[str, bytes]]:
        """Read every JSON file in the directory into memory.

        Files that fail to read or validate are skipped (an error is logged)
        rather than aborting the whole batch.

        Args:
            validate: When ``True`` (default), each file is parsed to confirm it
                is valid JSON before being included; when ``False``, files are
                read as-is without validation.

        Returns:
            A list of ``(filename, raw_bytes)`` tuples for every file that was
            read successfully, in sorted filename order.
        """
        logger.info(
            f"Loading all JSON files from '{self.folder_path}' (validate={validate})",
            extra={"function_name": "load_all", "input_data": {"folder_path": str(self.folder_path), "validate": validate}},
        )
        files = self.get_json_files()
        result = []
        for file_path in files:
            try:
                raw = self.read_and_validate_file(file_path) if validate else self.read_file(file_path)
                result.append((file_path.name, raw))
                logger.info(f"Read '{file_path.name}' ({len(raw)} bytes)")
            except (ValueError, OSError) as e:
                logger.error(f"Skipping '{file_path.name}': {e}")
        logger.info(
            f"Loaded {len(result)} of {len(files)} file(s) from '{self.folder_path}'",
            extra={"function_name": "load_all", "status": "success", "response_payload": {"loaded": len(result), "found": len(files)}},
        )
        return result


class ObjectStorageClient:
    """Async S3-compatible client for OpenShift ODF Object Storage.

    Wraps an ``aioboto3.Session`` and exposes high-level, folder-aware
    operations (upload, download, read, delete, existence checks, metadata and
    presigned URLs). A single S3 client is opened on first use and reused by
    every operation and concurrent task; call :meth:`aclose` (or use the client
    as an ``async with`` context manager) to close it. Use :meth:`instance` to
    share one client process-wide.

    Attributes:
        config: The :class:`ObjectStorageConfig` providing endpoint, bucket and
            credentials.

    Args:
        config: Connection configuration for the target bucket.
    """

    _instance: "ObjectStorageClient | None" = None

    def __init__(self, config: ObjectStorageConfig | None = None):
        """Store the configuration and create the shared aioboto3 session.

        Args:
            config: Connection configuration for the target bucket. When omitted,
                a fresh :class:`ObjectStorageConfig` is read from the environment.
        """
        config = config or ObjectStorageConfig()
        logger.debug(
            f"Initialising ObjectStorageClient for bucket '{config.bucket_name}'",
            extra={"function_name": "ObjectStorageClient.__init__", "input_data": {"bucket_name": config.bucket_name, "endpoint_url": config.endpoint_url}},
        )
        self.config = config
        self._session = aioboto3.Session()
        self._client_cm = None              # the open aioboto3 client context manager
        self._s3 = None                     # the shared S3 client reused by every method
        self._client_lock = asyncio.Lock()  # guards lazy client creation

    @classmethod
    def instance(cls, config: ObjectStorageConfig | None = None) -> "ObjectStorageClient":
        """Return the shared storage client, creating it on the first call.

        Later calls reuse the object built on the first call and ignore their
        argument, so the whole application shares one client (and one underlying
        S3 connection).
        """
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    async def _ensure_client(self):
        """Open the shared S3 client once (race-safe) and return it.

        All methods reuse this single client instead of opening a new one per
        call/file. It stays open until :meth:`aclose` (or ``async with`` exit).
        The lock makes creation atomic so concurrent callers share one client.
        """
        if self._s3 is None:
            async with self._client_lock:
                if self._s3 is None:   # re-check now that we hold the lock
                    cm = self._session.client(**self._client_kwargs())
                    self._s3 = await cm.__aenter__()
                    self._client_cm = cm
        return self._s3

    @asynccontextmanager
    async def _client(self):
        """Yield the shared S3 client without closing it on block exit.

        Lets every method keep its original ``async with ... as s3:`` shape while
        reusing one long-lived client. The client is closed only by
        :meth:`aclose` / ``async with self`` exit, not here.
        """
        client = await self._ensure_client()
        yield client

    async def aclose(self) -> None:
        """Close the shared S3 client if it was opened (safe no-op otherwise)."""
        if self._client_cm is not None:
            await self._client_cm.__aexit__(None, None, None)
            self._client_cm = None
            self._s3 = None

    async def __aenter__(self) -> "ObjectStorageClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _client_kwargs(self) -> dict:
        """Build the keyword arguments for ``session.client("s3", ...)``.

        Returns:
            A dict of parameters (service name, endpoint, credentials, signature
            config and TLS verification flag) ready to be splatted into
            ``aioboto3.Session.client``.

        Note:
            The returned dict contains the access and secret keys and must never
            be logged.
        """
        return {
            "service_name": "s3",
            "endpoint_url": self.config.endpoint_url,
            "aws_access_key_id": self.config.access_key,
            "aws_secret_access_key": self.config.secret_key,
            "config": Config(signature_version="s3v4"),
            "verify": self.config.verify_ssl,
        }

    @staticmethod
    def _build_key(folder: Folder, filename: str) -> str:
        """Compose a fully-qualified S3 object key from a folder and filename.

        Args:
            folder: The logical :class:`Folder` (key prefix).
            filename: The object's filename within the folder.

        Returns:
            The S3 key in the form ``"<folder.value>/<filename>"``.
        """
        return f"{folder.value}/{filename}"

    # Bucket

    async def bucket_exists(self) -> bool:
        """Check whether the configured bucket exists and is reachable.

        Issues a ``HeadBucket`` request. Any error (missing bucket, auth failure,
        connectivity) is treated as "does not exist / not reachable" and logged
        as a warning rather than raised.

        Returns:
            ``True`` if the bucket responds to ``HeadBucket``; ``False``
            otherwise.
        """
        logger.debug(
            f"Checking existence of bucket '{self.config.bucket_name}'",
            extra={"function_name": "bucket_exists", "input_data": {"bucket_name": self.config.bucket_name}},
        )
        async with self._client() as s3:
            try:
                await s3.head_bucket(Bucket=self.config.bucket_name)
                logger.info(
                    f"Bucket '{self.config.bucket_name}' exists",
                    extra={"function_name": "bucket_exists", "status": "success"},
                )
                return True
            except Exception as exc:
                logger.warning(
                    f"bucket_exists check failed for '{self.config.bucket_name}': {exc}",
                    extra={"function_name": "bucket_exists", "status": "error", "error_details": str(exc)},
                )
                return False

    async def list_folder(self, folder: Folder) -> list[dict]:
        """List the objects stored under a single folder prefix.

        Args:
            folder: The logical :class:`Folder` to enumerate.

        Returns:
            The raw ``Contents`` list from the ``ListObjectsV2`` response (each
            entry a dict with keys such as ``Key``, ``Size``, ``LastModified``).
            An empty list is returned when the folder has no objects.

        Note:
            Results are not paginated; only the first page returned by the S3
            API is included.
        """
        logger.debug(
            f"[{folder.value}] Listing objects",
            extra={"function_name": "list_folder", "input_data": {"folder": folder.value}},
        )
        async with self._client() as s3:
            response = await s3.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=f"{folder.value}/",
            )
            objects = response.get("Contents", [])
            logger.info(f"[{folder.value}] Found {len(objects)} object(s)")
            return objects

    async def list_all_folders(self) -> dict[str, list[dict]]:
        """List the objects in every :class:`Folder` concurrently.

        Returns:
            A dict mapping each folder's prefix value to its list of object
            descriptors (as returned by :meth:`list_folder`).
        """
        logger.info(
            "Listing objects for all folders",
            extra={"function_name": "list_all_folders", "input_data": {"folders": [f.value for f in Folder]}},
        )
        results = await asyncio.gather(*[self.list_folder(folder) for folder in Folder])
        return {folder.value: result for folder, result in zip(Folder, results)}

    # Folder existence

    async def folder_exists(self, folder: Folder) -> bool:
        """Return ``True`` if any object already lives under the folder's prefix.

        Args:
            folder: The logical :class:`Folder` to check.

        Returns:
            ``True`` if at least one object has the ``<folder>/`` prefix.
        """
        logger.debug(
            f"[{folder.value}] Checking whether folder exists",
            extra={"function_name": "folder_exists", "input_data": {"folder": folder.value}},
        )
        async with self._client() as s3:
            response = await s3.list_objects_v2(
                Bucket=self.config.bucket_name,
                Prefix=f"{folder.value}/",
                MaxKeys=1,
            )
            return response.get("KeyCount", 0) > 0

    async def ensure_folder(self, folder: Folder) -> None:
        """Create the folder (a zero-byte prefix marker) if it doesn't exist yet.

        Object storage has no real folders — a prefix only "exists" once an
        object carries it. If the folder is empty, an empty ``<folder>/`` marker
        object is created so the folder is present even before the first file
        lands. If it already exists, this is a no-op.

        Args:
            folder: The logical :class:`Folder` to ensure.
        """
        if await self.folder_exists(folder):
            logger.info(
                f"[{folder.value}] Folder already exists",
                extra={"function_name": "ensure_folder", "status": "success"},
            )
            return
        async with self._client() as s3:
            await s3.put_object(Bucket=self.config.bucket_name, Key=f"{folder.value}/", Body=b"")
            logger.info(
                f"[{folder.value}] Folder did not exist — created",
                extra={"function_name": "ensure_folder", "status": "success"},
            )

    # Upload

    async def upload_file(self, folder: Folder, local_path: str, filename: str) -> None:
        """Upload a file from the local filesystem to the folder.

        Args:
            folder: The target :class:`Folder` (key prefix).
            local_path: Path to the local file to upload.
            filename: The object filename to create within the folder.

        Returns:
            None.

        Raises:
            Exception: Any error raised by the underlying ``upload_file`` S3
                call (e.g. missing local file, auth or connectivity errors) is
                propagated unchanged.
        """
        s3_key = self._build_key(folder, filename)
        logger.debug(
            f"[{folder.value}] Uploading local file '{local_path}' -> '{s3_key}'",
            extra={"function_name": "upload_file", "input_data": {"local_path": local_path, "key": s3_key}},
        )
        async with self._client() as s3:
            await s3.upload_file(local_path, self.config.bucket_name, s3_key)
            logger.info(
                f"[{folder.value}] Uploaded '{local_path}' -> '{s3_key}'",
                extra={"function_name": "upload_file", "input_data": {"local_path": local_path, "key": s3_key}, "status": "success"},
            )

    async def upload_bytes(
        self,
        folder: Folder,
        filename: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Upload an in-memory bytes payload as an object.

        Args:
            folder: The target :class:`Folder` (key prefix).
            filename: The object filename to create within the folder.
            data: The raw bytes to store as the object body.
            content_type: The MIME type to record on the object (defaults to
                ``application/octet-stream``).

        Returns:
            None.

        Raises:
            Exception: Any error raised by the underlying ``put_object`` S3 call
                is propagated unchanged.
        """
        s3_key = self._build_key(folder, filename)
        logger.debug(
            f"[{folder.value}] Putting {len(data)} bytes -> '{s3_key}' (content_type={content_type})",
            extra={"function_name": "upload_bytes", "input_data": {"key": s3_key, "size": len(data), "content_type": content_type}},
        )
        async with self._client() as s3:
            await s3.put_object(
                Bucket=self.config.bucket_name,
                Key=s3_key,
                Body=data,
                ContentType=content_type,
            )
            logger.info(
                f"[{folder.value}] Uploaded {len(data)} bytes -> '{s3_key}'",
                extra={"function_name": "upload_bytes", "input_data": {"key": s3_key, "size": len(data)}, "status": "success"},
            )

    async def upload_many(self, folder: Folder, files: list[tuple[str, bytes, str]]) -> None:
        """Upload several in-memory payloads concurrently.

        Args:
            folder: The target :class:`Folder` for every object.
            files: A list of ``(filename, data, content_type)`` tuples, one per
                object to upload.

        Returns:
            None.

        Raises:
            Exception: If any individual :meth:`upload_bytes` call fails, the
                error is propagated by :func:`asyncio.gather`.
        """
        logger.info(
            f"[{folder.value}] Uploading {len(files)} object(s) concurrently",
            extra={"function_name": "upload_many", "input_data": {"folder": folder.value, "count": len(files)}},
        )
        await asyncio.gather(
            *[self.upload_bytes(folder, filename, data, content_type)
              for filename, data, content_type in files]
        )
        logger.info(
            f"[{folder.value}] Finished uploading {len(files)} object(s)",
            extra={"function_name": "upload_many", "status": "success", "response_payload": {"count": len(files)}},
        )

    async def upload_from_local_folder(
        self,
        local_folder: str,
        target_folder: Folder,
        validate_json: bool = True,
        concurrency: int = 10,
    ) -> dict[str, int]:
        """Read a local folder of JSON files and upload them to a target folder.

        Files are read (and optionally JSON-validated) via
        :class:`LocalFolderReader`, then uploaded concurrently with a bounded
        number of in-flight uploads. Individual upload failures are counted and
        logged but do not abort the batch.

        Args:
            local_folder: Path to the local directory containing JSON files.
            target_folder: The destination :class:`Folder` in object storage.
            validate_json: When ``True`` (default), each file is validated as
                JSON before upload.
            concurrency: Maximum number of uploads allowed in flight at once.

        Returns:
            A summary dict ``{"uploaded": <int>, "failed": <int>}``. When there
            are no files to upload, returns ``{"uploaded": 0, "failed": 0}``.
        """
        logger.info(
            f"[{target_folder.value}] Uploading from local folder '{local_folder}' "
            f"(validate_json={validate_json}, concurrency={concurrency})",
            extra={"function_name": "upload_from_local_folder", "input_data": {"local_folder": local_folder, "target_folder": target_folder.value, "validate_json": validate_json, "concurrency": concurrency}},
        )
        reader = LocalFolderReader(local_folder)
        files = reader.load_all(validate=validate_json)

        if not files:
            logger.warning("No files to upload. Aborting.")
            return {"uploaded": 0, "failed": 0}

        semaphore = asyncio.Semaphore(concurrency)
        uploaded, failed = 0, 0

        async def _upload_one(filename: str, data: bytes):
            """Upload a single file under the concurrency semaphore.

            Updates the enclosing ``uploaded``/``failed`` counters. Any upload
            error is caught and logged so the overall batch can continue.
            """
            nonlocal uploaded, failed
            async with semaphore:
                try:
                    await self.upload_bytes(
                        folder=target_folder,
                        filename=filename,
                        data=data,
                        content_type="application/json",
                    )
                    uploaded += 1
                except Exception as e:
                    logger.error(f"Failed to upload '{filename}': {e}")
                    failed += 1

        await asyncio.gather(*[_upload_one(name, data) for name, data in files])

        logger.info(
            f"[{target_folder.value}] Upload complete — {uploaded} succeeded, {failed} failed"
        )
        return {"uploaded": uploaded, "failed": failed}


    async def download_file(self, folder: Folder, filename: str, local_path: str) -> None:
        """Download an object to a local file path.

        Args:
            folder: The source :class:`Folder` (key prefix).
            filename: The object filename within the folder.
            local_path: Destination path on the local filesystem.

        Returns:
            None.

        Raises:
            Exception: Any error raised by the underlying ``download_file`` S3
                call (e.g. missing key, auth errors) is propagated unchanged.
        """
        s3_key = self._build_key(folder, filename)
        logger.debug(
            f"[{folder.value}] Downloading '{s3_key}' -> '{local_path}'",
            extra={"function_name": "download_file", "input_data": {"key": s3_key, "local_path": local_path}},
        )
        async with self._client() as s3:
            await s3.download_file(self.config.bucket_name, s3_key, local_path)
            logger.info(
                f"[{folder.value}] Downloaded '{s3_key}' -> '{local_path}'",
                extra={"function_name": "download_file", "input_data": {"key": s3_key, "local_path": local_path}, "status": "success"},
            )

    async def read_object(self, folder: Folder, filename: str) -> bytes:
        """Read an object's contents into memory.

        Args:
            folder: The source :class:`Folder` (key prefix).
            filename: The object filename within the folder.

        Returns:
            The object body as ``bytes``.

        Raises:
            Exception: Any error raised by the underlying ``get_object`` S3 call
                is propagated unchanged.
        """
        s3_key = self._build_key(folder, filename)
        logger.debug(
            f"[{folder.value}] Getting object '{s3_key}'",
            extra={"function_name": "read_object", "input_data": {"key": s3_key}},
        )
        async with self._client() as s3:
            response = await s3.get_object(Bucket=self.config.bucket_name, Key=s3_key)
            content = await response["Body"].read()
            logger.info(
                f"[{folder.value}] Read {len(content)} bytes from '{s3_key}'",
                extra={"function_name": "read_object", "input_data": {"key": s3_key}, "response_payload": {"size": len(content)}, "status": "success"},
            )
            return content

    async def read_many(self, folder: Folder, filenames: list[str]) -> dict[str, bytes]:
        """Read several objects concurrently.

        Args:
            folder: The source :class:`Folder` for every object.
            filenames: The object filenames to read.

        Returns:
            A dict mapping each filename to its contents as ``bytes``.

        Raises:
            Exception: If any individual :meth:`read_object` call fails, the
                error is propagated by :func:`asyncio.gather`.
        """
        logger.info(
            f"[{folder.value}] Reading {len(filenames)} object(s) concurrently",
            extra={"function_name": "read_many", "input_data": {"folder": folder.value, "count": len(filenames)}},
        )
        contents = await asyncio.gather(
            *[self.read_object(folder, filename) for filename in filenames]
        )
        return dict(zip(filenames, contents))

    # Delete 

    async def delete_object(self, folder: Folder, filename: str) -> None:
        """Delete a single object.

        Args:
            folder: The :class:`Folder` (key prefix) the object lives in.
            filename: The object filename to delete.

        Returns:
            None. Deleting a non-existent key is not treated as an error by S3.

        Raises:
            Exception: Any error raised by the underlying ``delete_object`` S3
                call is propagated unchanged.
        """
        s3_key = self._build_key(folder, filename)
        logger.debug(
            f"[{folder.value}] Deleting '{s3_key}'",
            extra={"function_name": "delete_object", "input_data": {"key": s3_key}},
        )
        async with self._client() as s3:
            await s3.delete_object(Bucket=self.config.bucket_name, Key=s3_key)
            logger.info(
                f"[{folder.value}] Deleted '{s3_key}'",
                extra={"function_name": "delete_object", "input_data": {"key": s3_key}, "status": "success"},
            )

    async def delete_many(self, folder: Folder, filenames: list[str]) -> None:
        """Delete several objects in a single batch request.

        Args:
            folder: The :class:`Folder` (key prefix) the objects live in.
            filenames: The object filenames to delete. An empty list is a no-op.

        Returns:
            None. Per-object errors reported by S3 are logged (as an error) but
            not raised.

        Raises:
            Exception: Any error raised by the underlying ``delete_objects`` S3
                call itself is propagated unchanged.
        """
        if not filenames:
            logger.debug(
                f"[{folder.value}] delete_many called with no filenames; nothing to do",
                extra={"function_name": "delete_many", "input_data": {"folder": folder.value}},
            )
            return
        logger.debug(
            f"[{folder.value}] Batch deleting {len(filenames)} object(s)",
            extra={"function_name": "delete_many", "input_data": {"folder": folder.value, "count": len(filenames)}},
        )
        objects = [{"Key": self._build_key(folder, f)} for f in filenames]
        async with self._client() as s3:
            response = await s3.delete_objects(
                Bucket=self.config.bucket_name,
                Delete={"Objects": objects, "Quiet": True},
            )
            errors = response.get("Errors", [])
            if errors:
                logger.error(
                    f"[{folder.value}] Batch delete errors: {errors}",
                    extra={"function_name": "delete_many", "status": "error", "error_details": str(errors)},
                )
            else:
                logger.info(
                    f"[{folder.value}] Batch deleted {len(filenames)} object(s)",
                    extra={"function_name": "delete_many", "status": "success", "response_payload": {"count": len(filenames)}},
                )

    async def delete_folder(self, folder: Folder) -> None:
        """Delete every object stored under a folder prefix.

        Lists the folder's current objects and batch-deletes them.

        Args:
            folder: The :class:`Folder` to clear.

        Returns:
            None.

        Raises:
            Exception: Any error raised by the underlying list/delete S3 calls
                is propagated unchanged.
        """
        logger.info(
            f"[{folder.value}] Clearing folder",
            extra={"function_name": "delete_folder", "input_data": {"folder": folder.value}},
        )
        objects = await self.list_folder(folder)
        filenames = [obj["Key"].removeprefix(f"{folder.value}/") for obj in objects]
        await self.delete_many(folder, filenames)
        logger.info(
            f"[{folder.value}] Folder cleared",
            extra={"function_name": "delete_folder", "status": "success", "response_payload": {"count": len(filenames)}},
        )


    async def object_exists(self, folder: Folder, filename: str) -> bool:
        """Check whether a specific object exists.

        Issues a ``HeadObject`` request; any error (including a missing key) is
        interpreted as "does not exist" and logged rather than raised.

        Args:
            folder: The :class:`Folder` (key prefix) to look in.
            filename: The object filename to check.

        Returns:
            ``True`` if the object exists; ``False`` otherwise.
        """
        s3_key = self._build_key(folder, filename)
        logger.debug(
            f"[{folder.value}] Checking existence of '{s3_key}'",
            extra={"function_name": "object_exists", "input_data": {"key": s3_key}},
        )
        async with self._client() as s3:
            try:
                await s3.head_object(Bucket=self.config.bucket_name, Key=s3_key)
                logger.info(
                    f"[{folder.value}] Object '{s3_key}' exists",
                    extra={"function_name": "object_exists", "input_data": {"key": s3_key}, "status": "success"},
                )
                return True
            except Exception as exc:
                logger.info(
                    f"object_exists check failed for '{s3_key}': {exc}",
                    extra={"function_name": "object_exists", "status": "error", "error_details": str(exc), "input_data": {"key": s3_key}},
                )
                return False

    async def object_metadata(self, folder: Folder, filename: str) -> Optional[dict]:
        """Fetch an object's metadata via ``HeadObject``.

        Args:
            folder: The :class:`Folder` (key prefix) to look in.
            filename: The object filename to inspect.

        Returns:
            The full ``HeadObject`` response dict (content length, content type,
            last-modified, etc.) if the object exists; ``None`` if the lookup
            fails for any reason (the failure is logged, not raised).
        """
        s3_key = self._build_key(folder, filename)
        logger.debug(
            f"[{folder.value}] Fetching metadata for '{s3_key}'",
            extra={"function_name": "object_metadata", "input_data": {"key": s3_key}},
        )
        async with self._client() as s3:
            try:
                return await s3.head_object(Bucket=self.config.bucket_name, Key=s3_key)
            except Exception as exc:
                logger.info(
                    f"object_metadata lookup failed for '{s3_key}': {exc}",
                    extra={"function_name": "object_metadata", "status": "error", "error_details": str(exc), "input_data": {"key": s3_key}},
                )
                return None

    async def generate_presigned_url(
        self, folder: Folder, filename: str, expires_in: int = 3600
    ) -> str:
        """Generate a time-limited presigned GET URL for an object.

        Args:
            folder: The :class:`Folder` (key prefix) the object lives in.
            filename: The object filename to generate a URL for.
            expires_in: Lifetime of the URL in seconds (default ``3600``).

        Returns:
            A presigned HTTPS URL that grants temporary read access to the
            object without requiring credentials.

        Raises:
            Exception: Any error raised by the underlying
                ``generate_presigned_url`` S3 call is propagated unchanged.
        """
        s3_key = self._build_key(folder, filename)
        logger.debug(
            f"[{folder.value}] Generating presigned URL for '{s3_key}' (expires_in={expires_in}s)",
            extra={"function_name": "generate_presigned_url", "input_data": {"key": s3_key, "expires_in": expires_in}},
        )
        async with self._client() as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket_name, "Key": s3_key},
                ExpiresIn=expires_in,
            )
            logger.info(
                f"[{folder.value}] Generated presigned URL for '{s3_key}'",
                extra={"function_name": "generate_presigned_url", "input_data": {"key": s3_key, "expires_in": expires_in}, "status": "success"},
            )
            return url
