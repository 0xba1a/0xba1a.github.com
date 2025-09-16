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
QUOTE_JSON_DIR = OUTPUT_DIR / "quotes_json"

# Azure OpenAI settings - must be provided as environment variables
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUOTE_JSON_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")

shutdown_requested = False


def _signal_handler(signum, frame):
    global shutdown_requested
    logger.info("Signal %s received, will shut down gracefully", signum)
    shutdown_requested = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

### State Management ###

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

### State Management ###


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


def _assistant_message(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return data["choices"][0]["message"]
    except Exception:
        raise RuntimeError(f"Unexpected response format: {data}")


def _parse_final_json(content: str) -> Optional[Dict[str, Any]]:
    obj = _extract_json(content)
    if not obj:
        return None
    # Minimal validation
    if {"safe", "category", "explanation"}.issubset(obj.keys()) and obj.get("category") in ALLOWED_CATEGORIES:
        return obj
    return obj  # return anyway; caller can decide


def process_quote_file(path: Path, state: Dict[str, Any]) -> None:
    logger.info("Processing quote file: %s", path)
    quote = path.read_text(encoding="utf-8").strip()
    file_id = path.name

    if file_id in state.get("processed", {}):
        logger.info("Already processed %s, skipping", file_id)
        return

    try:
        message = {
            "role": "user",
            "content": f"""
                    Explain me the quote: {quote} in a short paragraph.
                    Your response should be strictly in JSON format with the following fields:
                    - funny: boolean, true if you find it funny. False otherwise.
                    - explanation: string, a short paragraph explaining the quote.
                    """,
        }
        response = chat_completion([message])["choices"][0]["message"]["content"]
        result = _parse_final_json(response)
    except Exception:
        logger.exception("LLM tool-driven processing failed for %s", file_id)
        sys.exit(1)
        # result = {"safe": False, "category": "Other", "explanation": "LLM error"}

    # Mark processed
    state.setdefault("processed", {})[file_id] = {"agent": "002", "quote": quote, "processed_at": datetime.datetime.utcnow().isoformat(), "funny": result["funny"], "explanation": result["explanation"]}
    save_state(state)


def main_loop(poll_interval: int = 60):
    state = load_state()
    logger.info("Agent started, watching %s", OUTPUT_DIR)

    while not shutdown_requested:
        txt_files = sorted(glob.glob(str(OUTPUT_DIR / "*.txt")))
        for f in txt_files:
            if shutdown_requested:
                break
            process_quote_file(Path(f), state)
        # Sleep and be responsive to shutdown
        for _ in range(int(poll_interval)):
            if shutdown_requested:
                break
            time.sleep(1)

    logger.info("Agent shutting down")


if __name__ == "__main__":
    main_loop()