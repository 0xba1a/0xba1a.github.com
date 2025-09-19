import os
import sys
import json
import time
import logging
import datetime
import glob
import signal
import re
from pathlib import Path
from typing import Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

import requests

OUTPUT_DIR = Path("/tmp/agent-001/")
STATE_FILE = OUTPUT_DIR / "state.json"

# Azure OpenAI settings - must be provided as environment variables
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

shutdown_requested = False


def _signal_handler(signum, frame):
    global shutdown_requested
    logger.info("Signal %s received, will shut down gracefully", signum)
    shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to load state file, starting fresh")
    # default state
    return {"processed": {}, "last_sent": {}}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


SYSTEM_PROMPT = (
    "You are an helpful assistant that explains a programmer joke and identify whether it is culturally appropriate to be shared in a professional office environment.\n"
    "Goals:\n"
    "(1) Decide whether the joke is funny or not (funny: true/false).\n"
    "(2) Categorize the joke into one of these categories: 'Safe for work', 'Offensive', 'Dark humor'.\n"
    "(3) And briefly explain the joke in 1 paragraph.\n"
    "Your response must be a single JSON object with keys: funny (bool), category (string), explanation (string).\n"
)


def _extract_json(text: str) -> Optional[dict]:
    """Try to extract the first JSON object from a text blob."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def chat_completion(messages, tools=None, temperature=0.0, max_tokens=800) -> Dict[str, Any]:
    """Call Azure OpenAI chat completion returning the full JSON, supporting tool (function) calls."""
    # Random jitter 3-5s to reduce rate spikes
    time.sleep(3 + (2 * os.urandom(1)[0] / 255.0))

    if not AZURE_ENDPOINT or not AZURE_KEY:
        raise RuntimeError("Azure OpenAI credentials (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY) not set")

    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_KEY,
    }
    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    return resp.json()


def _parse_final_json(content: str) -> Optional[Dict[str, Any]]:
    obj = _extract_json(content)
    if not obj:
        return None
    # Minimal validation
    if {"safe", "category", "explanation"}.issubset(obj.keys()):
        return obj
    return obj  # return anyway; caller can decide


def process_joke_file(path: Path, state: Dict[str, Any]) -> None:
    logger.info("Processing joke file: %s", path)
    joke = path.read_text(encoding="utf-8").strip()
    file_id = path.name

    if file_id in state.get("processed", {}):
        logger.info("Already processed %s, skipping", file_id)
        return

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"joke: `{joke}`"},
        ]
        response = chat_completion(messages)["choices"][0]["message"]["content"]
        result = _parse_final_json(response)
    except Exception as e:
        logger.exception("LLM tool-driven processing failed for %s\nException: %s", file_id, e)
        sys.exit(1)

    # Mark processed
    state.setdefault("processed", {})[file_id] = {"agent": "002", "joke": joke, "processed_at": datetime.datetime.utcnow().isoformat(), "funny": result["funny"], "explanation": result["explanation"], "category": result["category"]}
    save_state(state)


def main_loop(poll_interval: int = 60):
    state = load_state()
    logger.info("Agent started, watching %s", OUTPUT_DIR)

    while not shutdown_requested:
        txt_files = sorted(glob.glob(str(OUTPUT_DIR / "*.txt")))
        for f in txt_files:
            if shutdown_requested:
                break
            process_joke_file(Path(f), state)
        # Sleep and be responsive to shutdown
        for _ in range(int(poll_interval)):
            if shutdown_requested:
                break
            time.sleep(1)

    logger.info("Agent shutting down")


if __name__ == "__main__":
    main_loop()