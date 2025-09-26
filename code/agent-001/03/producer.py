import json
import time
import datetime
import logging
import signal
import threading
import urllib.request
import urllib.error
from pathlib import Path

DEFAULT_URL = "https://official-joke-api.appspot.com/jokes/programming/random"
DEFAULT_OUTPUT_DIR = "/tmp/agent-001/"
DEFAULT_INTERVAL_SECONDS = 1 * 60  # 1 minute


def ensure_output_dir(path: str) -> Path:
    """Ensure the output directory exists and return a Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_text(url: str, timeout: int = 10) -> str:
    """Fetch text content from a URL and return it as a string.

    Uses the standard library so there are no extra dependencies.
   """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return raw.decode("utf-8")
            except Exception:
                return raw.decode(errors="replace")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to fetch {url}: {e}")


def run_loop(url: str = DEFAULT_URL, output_dir: str = DEFAULT_OUTPUT_DIR, interval_seconds: int = DEFAULT_INTERVAL_SECONDS, stop_event: threading.Event = None) -> None:
    """Run the fetch->write loop until stop_event is set.

    The loop is responsive to shutdown requests via the provided stop_event.
   """
    if stop_event is None:
        stop_event = threading.Event()

    out_dir = ensure_output_dir(output_dir)
    logging.info("Starting producer loop: url=%s interval=%s output=%s", url, interval_seconds, out_dir)

    while not stop_event.is_set():
        start = time.time()
        try:
            json_text = json.loads(fetch_text(url))[0]
            target_file = out_dir / f"{json_text['id']}.txt"
            with target_file.open("w", encoding="utf-8") as fh:
                fh.write(json_text["setup"] + "\n" + json_text["punchline"] + "\n")
            logging.info("Wrote file %s", target_file)
        except Exception as exc:
            logging.exception("Error during fetch/write: %s", exc)

        # Wait for next cycle but be responsive to stop_event
        elapsed = time.time() - start
        wait_for = max(0, interval_seconds - elapsed)
        logging.debug("Sleeping for %.1f seconds", wait_for)
        # wait returns True if the event is set while waiting
        stop_event.wait(wait_for)

    logging.info("Producer loop exiting cleanly.")


def _setup_signal_handlers(stop_event: threading.Event) -> None:
    """Attach SIGINT and SIGTERM handlers to set the stop_event for graceful shutdown."""

    def _handler(signum, frame):
        logging.info("Received signal %s, shutting down...", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stop_event = threading.Event()
    _setup_signal_handlers(stop_event)

    try:
        run_loop(stop_event=stop_event)
    except Exception:
        logging.exception("Unexpected error in main loop")
    finally:
        logging.info("Shutdown complete")


if __name__ == "__main__":
    main()
