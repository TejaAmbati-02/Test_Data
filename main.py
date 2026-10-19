import asyncio
import logging
import threading
import truststore
from dotenv import load_dotenv
from asmo import Pipeline
from asmo.storage import ObjectStorageClient
from asmo.logging_service import setup_logging, logger

# Use the OS trust store for TLS so requests to CommerceTools succeed behind a
# corporate proxy that injects its own root CA. Must run before any SSL context
# (i.e. any aiohttp session) is created.
truststore.inject_into_ssl()

# override=True so values in .env take precedence over any stale OS environment
# variables (e.g. a previously `setx`-ed OPENSEARCH_USER on Windows).
load_dotenv(override=True)
setup_logging()


async def main() -> None:
    logger.info("Pipeline starting")
    try:
        result = await Pipeline().run()
        logger.info(f"Pipeline finished — {result.summary()}")
    finally:
        # Close the shared S3 client (no-op if it was never opened).
        await ObjectStorageClient.instance().aclose()


if __name__ == "__main__":
    asyncio.run(main())
    # Diagnostic: anything other than MainThread here is what keeps the process
    # alive. If this logs a thread name, that library/thread is the culprit.
    alive = [t.name for t in threading.enumerate() if t is not threading.main_thread()]
    if alive:
        logger.warning(f"Non-main threads still alive after run: {alive}")
    logging.shutdown()
