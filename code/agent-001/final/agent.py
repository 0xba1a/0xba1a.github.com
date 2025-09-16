import os
import sys
import json
import time
import logging
import subprocess
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

# Groups mapping (labels expected from the model)
GROUPS = {
    "Database, filesystem": "db-team@example.com",
    "operating systems, Linux": "os-team@example.com",
    "Web technology, front-end, back-end, react, angular, javascript, css": "web-team@example.com",
    "Other": "general@example.com",
}

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


def call_azure_chat(messages, temperature=0.0, max_tokens=800) -> str:
    # Random wait between 3 to 5 seconds
    time.sleep(3 + (2 * os.urandom(1)[0] / 255.0))

    if not AZURE_ENDPOINT or not AZURE_KEY:
        raise RuntimeError("Azure OpenAI credentials (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY) not set")

    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_KEY,
    }
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    # Azure returns choices[0].message.content
    return data["choices"][0]["message"]["content"]


PROMPT_TEMPLATE = {
    "system": (
        "You classify short engineering quotes and optionally use tools.\n"
        "Goals: (1) Decide if the quote is safe to share in an office (safe: true/false). "
        "(2) Decide whether it is funny or not (funny: true/false). "
        "(3) Pick a category from the EXACT enum: [Database, filesystem | operating systems, Linux | Web technology, front-end, back-end, react, angular, javascript, css | Other]. "
        "(4) Provide a short explanation (1 paragraph).\n"
        "(5) If you find it funny, and safe, send an email to the relevant team.\n"
        "Use the browse tool ONLY to disambiguate unclear or unknown technical terms.\n"
        "When you are confident and find the quote funny, safe == true and funny == true, use send_email tool forward it to the relevant team.\n"
        "Don't mention whether the quote is safe or funny in the explanation. Those information should be part of the json response only.\n"
        "When finished (after any tool calls), output ONLY a single JSON object: {\"safe\": <bool>, \"funny\": <bool>, \"category\": <enum>, \"explanation\": <string>} with no prose before or after.\n"
        "Never invent new categories. Never wrap JSON in markdown fences."
    )
}

ALLOWED_CATEGORIES = [
    "Database, filesystem",
    "operating systems, Linux",
    "Web technology, front-end, back-end, react, angular, javascript, css",
    "Other",
]

# Define tool (function) schemas for GPT-4.1 function calling
FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Look up a technical term within the context of the quote to disambiguate meaning before classification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "The technical term or phrase to research."},
                    "quote": {"type": "string", "description": "The original quote being classified."}
                },
                "required": ["term", "quote"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send the quote via email to the relevant team group once you are confident it's safe and correctly categorized.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_label": {"type": "string", "enum": ALLOWED_CATEGORIES, "description": "Category/team to notify."},
                    "quote": {"type": "string", "description": "The original quote."},
                    "explanation": {"type": "string", "description": "Reason the quote is relevant and safe."}
                },
                "required": ["group_label", "quote", "explanation"],
            },
        },
    },
]


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


def classify_and_act_on_quote(quote: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Tool (function) calling loop with GPT-4.1 until final JSON classification."""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": PROMPT_TEMPLATE["system"]},
        {"role": "user", "content": f"Quote: {quote}"},
    ]

    max_cycles = 10
    for cycle in range(max_cycles):
        try:
            data = chat_completion(messages, tools=FUNCTION_TOOLS)
        except Exception:
            logger.exception("chat_completion failed")
            # retry after a pause
            time.sleep(10)
            continue
        msg = _assistant_message(data)
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        # If the model produced tool calls, execute them sequentially
        if tool_calls:
            for tc in tool_calls:
                if tc.get("type") != "function":
                    continue
                fn = tc["function"]["name"]
                # arguments sometimes come as JSON string
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except Exception:
                    args = {}
                tool_result: Any = {}

                if fn == "browse":
                    term = args.get("term", "")
                    # ensure quote passed (model might omit or alter) - override with original
                    tool_result = run_browse(term, quote)
                elif fn == "send_email":
                    group_label = args.get("group_label") or "Other"
                    explanation = args.get("explanation", "")
                    today = datetime.date.today().isoformat()
                    last_sent = state.setdefault("last_sent", {})
                    if last_sent.get(group_label) == today:
                        tool_result = {"sent": False, "reason": "already_sent_today"}
                    else:
                        sent = send_email(group_label, quote, explanation)
                        if sent:
                            last_sent[group_label] = today
                            save_state(state)
                        tool_result = {"sent": bool(sent), "reason": "ok" if sent else "failed"}
                else:
                    tool_result = {"error": f"Unknown tool {fn}"}

                # Append tool output message referencing tool_call_id
                messages.append(msg) if cycle == 0 and len(tool_calls) == 1 else messages.append({"role": "assistant", "content": content, "tool_calls": []}) if content and cycle == 0 else None
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": fn,
                    "content": tool_result if isinstance(tool_result, str) else json.dumps(tool_result)
                })

        # No tool calls: attempt to parse final JSON classification
        if content:
            parsed = _parse_final_json(content)
            if parsed:
                logging.info(f"Quote:**{quote}**")
                time.sleep(10)  # slight pause before next iteration
                return parsed
            else:
                # Provide feedback to model to output correct JSON
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": "Please output ONLY the final JSON object now."})
                continue

    logging.warning("Exceeded max tool cycles without valid final JSON; returning fallback")
    return {"safe": False, "category": "Other", "explanation": "Model failed to return final JSON in time"}


def run_browse(term: str, quote: str) -> str:
    """Invoke the browse.py tool with the search term in the context of the quote and return its stdout."""
    browser_arg = f"Define the term '{term}' in the context of this quote: '{quote}'"
    cmd = [sys.executable, "browser.py", browser_arg]
    logger.info("Running browse tool for term: %s", term)
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=600)
        logger.debug("browse output: %s", out)
        return out
    except subprocess.CalledProcessError as e:
        logger.error("browse.py failed: %s", e.output)
        return ""
    except Exception:
        logger.exception("Error running browse.py")
        return ""


def send_email(group_label: str, quote: str, explanation: str) -> bool:
    """Call send_email.py tool. group_label must be one of GROUPS keys."""
    to_group = GROUPS.get(group_label, GROUPS["Other"])
    cmd = [sys.executable, "send_email.py", to_group, quote, explanation]
    logger.info("Sending email to %s for group %s", to_group, group_label)
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        logger.exception("send_email.py returned non-zero")
        return False
    except Exception:
        logger.exception("Error running send_email.py")
        return False


def process_quote_file(path: Path, state: Dict[str, Any]) -> None:
    logger.info("Processing quote file: %s", path)
    quote = path.read_text(encoding="utf-8").strip()
    file_id = path.name

    if file_id in state.get("processed", {}):
        logger.info("Already processed %s, skipping", file_id)
        return

    try:
        result = classify_and_act_on_quote(quote, state)
    except Exception:
        logger.exception("LLM tool-driven processing failed for %s", file_id)
        sys.exit(1)
        # result = {"safe": False, "category": "Other", "explanation": "LLM error"}

    # Save per-quote json
    quote_json_path = QUOTE_JSON_DIR / (file_id + ".json")
    record = {"file": file_id, "quote": quote, "result": result, "timestamp": datetime.datetime.utcnow().isoformat()}
    quote_json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    # Mark processed
    state.setdefault("processed", {})[file_id] = {"agent": "004", "quote": quote, "processed_at": datetime.datetime.utcnow().isoformat(), "result": result}
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