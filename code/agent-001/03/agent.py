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

from datetime import datetime, timezone

OUTPUT_DIR = Path("/tmp/agent-001/")
STATE_FILE = OUTPUT_DIR / "state.json"
DARK_FILE = OUTPUT_DIR / "dark.json"

# Azure OpenAI settings - must be provided as environment variables
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Groups mapping (labels expected from the model)
GROUPS = {
    "system": {
        "email": "system@example.com",
        "description": "OS and Platform developers, System administrators and DevOps team",
        "keywords": [
            "operating systems",
            "Linux",
            "Unix",
            "Windows",
            "macOS",
            "DevOps",
            "SysAdmin",
            "infrastructure",
            "cloud",
            "virtualization",
            "containers",
            "Kubernetes",
            "networking",
        ],
    },
    "oops": {
        "email": "oops@example.com",
        "description": "Application and services developers",
        "keywords": [
            "application",
            "services",
            "java",
            "python",
            "c#",
            "go",
            "ruby",
            "php",
            "node.js",
            "dotnet",
            "API",
            "microservices",
            "REST",
            "SOAP",
        ],
    },
    "web": {
        "email": "web-team@example.com",
        "description": "Web technology, front-end, back-end, react, angular, javascript, css developers",
        "keywords": [
            "Web technology",
            "front-end",
            "back-end",
            "react",
            "angular",
            "javascript",
            "css",
            "HTML",
            "web development",
            "UX",
            "UI",
            "web design",
            "web frameworks",
        ],
    },
    "Other": {
        "email": "all@example.com",
        "description": "Everything else, general audience",
        "keywords": [],
    },
}
ALLOWED_CATEGORIES = list(GROUPS.keys())

SYSTEM_PROMPT = f"""
    You are an helpful assistant that helps me to send a funny morning email to my colleagues.
    You will be provided with a programmer joke.
    Your task is to:
    (1) Decide the safe of the joke (safe: safe/dark/offensive).
    (2) Identify to which group the joke to be sent ({GROUPS.keys()}).
    (3) And briefly explain the joke in 1 paragraph.
    You have multiple steps to complete your task.
    IMPORTANT:
      - If there is ANY technical term you are not 100% certain about, FIRST call the `browse` tool before final JSON.
      - If safe == "safe" you MUST attempt the `send_email` tool once before giving the final JSON.
      - Final JSON ONLY after required tool usage (or explicit determination no browse needed AND email attempted when safe).
    Your final response must be a single JSON object with keys: safe (string), category (string), explanation (string) and is_email_sent (boolean).

    The category must be one of these values: system, oops, web, Other.

    Below you can find relevant keywords for each group to help you decide the correct category:
    {json.dumps({k: v["keywords"] for k, v in GROUPS.items()}, indent=4)}

    The safe value must be one of these values: safe, dark, offensive.
    The explanation must be a brief explanation of the joke.

    You have two tools in your toolbox:
    1) A `browse` tool to look up technical terms you don't understand in the context of the joke. You can use this tool to disambiguate the meaning of the joke before classifying it or deciding whether it is safe for work.
    2) An `send_email` tool to send the joke to the relevant team group once you are confident it's safe and correctly categorized.
    Use the `browse` tool first if you need to look up any terms.
    Only use the `send_email` tool once you are confident in your classification and explanation.

    If the Joke is classified as dark, store that in dark.json in the {OUTPUT_DIR} directory. This is for me to forward to my friends later in the day.
"""

# Define tool (function) schemas for GPT-4.1 function calling
FUNCTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "Look up a technical term within the context of the joke to disambiguate meaning before classification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {
                        "type": "string",
                        "description": "The technical term or phrase to research.",
                    },
                    "joke": {
                        "type": "string",
                        "description": "(Optional) The original joke for extra context.",
                    },
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send the joke via email to the relevant team group once you are confident it's safe and correctly categorized.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_label": {
                        "type": "string",
                        "enum": ALLOWED_CATEGORIES,
                        "description": "Category/team to notify.",
                    },
                    "joke": {
                        "type": "string",
                        "description": "The original joke.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Reason the joke is relevant and safe.",
                    },
                },
                "required": ["group_label", "joke", "explanation"],
            },
        },
    },
]

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agent")


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


def chat_completion(
    messages, tools=None, temperature=0.0, max_tokens=800
) -> Dict[str, Any]:
    """Call Azure OpenAI chat completion returning the full JSON, supporting tool (function) calls."""
    time.sleep(3 + (2 * os.urandom(1)[0] / 255.0))  # jitter
    if not AZURE_ENDPOINT or not AZURE_KEY:
        raise RuntimeError(
            "Azure OpenAI credentials (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY) not set"
        )

    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_KEY}
    payload: Dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = requests.post(url, headers=headers, json=payload, timeout=90)
    if resp.status_code >= 400:
        logging.error(
            "Azure OpenAI 4xx/5xx response %s: %s", resp.status_code, resp.text
        )
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
    # Minimal validation (is_email_sent may be absent; we'll add later)
    required = {"safe", "category", "explanation"}
    if not required.issubset(obj.keys()):
        return None
    if obj.get("category") not in GROUPS.keys():
        return None
    if obj.get("safe") not in {"safe", "dark", "offensive"}:
        return None
    return obj


def _append_dark_joke(joke: str, parsed: Dict[str, Any]) -> None:
    """Persist dark jokes to DARK_FILE as an array of entries."""
    try:
        if DARK_FILE.exists():
            arr = json.loads(DARK_FILE.read_text(encoding="utf-8"))
            if not isinstance(arr, list):  # recover if corrupted
                arr = []
        else:
            arr = []
        arr.append(
            {
                "joke": joke,
                "ts": datetime.now(timezone.utc).isoformat(),
                "explanation": parsed.get("explanation", ""),
            }
        )
        DARK_FILE.write_text(json.dumps(arr, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to append dark joke to %s", DARK_FILE)


def classify_and_act_on_joke(joke: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Tool (function) calling loop with GPT-4.1 until final JSON classification.

    Guarantees:
      * If classification is safe, an email attempt is performed (tool call or forced local send) before returning.
      * If classification is dark, joke is stored in dark.json.
      * Adds is_email_sent boolean to final JSON.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}"},
        {"role": "user", "content": f"joke: {joke}"},
    ]

    max_cycles = 10
    email_sent_flag: bool = False
    last_email_attempt_reason: str = ""
    for cycle in range(max_cycles):
        try:
            data = chat_completion(messages, tools=FUNCTION_TOOLS)
        except Exception:
            logger.exception("chat_completion failed")
            time.sleep(5)
            continue
        msg = _assistant_message(data)
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""

        # ALWAYS append assistant message so tool_call references remain valid
        messages.append(
            {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")}
        )

        if tool_calls:
            for tc in tool_calls:
                if tc.get("type") != "function":
                    continue
                fn = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                try:
                    args = (
                        json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    )
                except Exception:
                    args = {}
                if fn == "browse":
                    term = args.get("term", "")
                    logger.info(f" 🌐  Browsing for term: {term}")
                    tool_result = run_browse(term, joke)
                elif fn == "send_email":
                    group_label = args.get("group_label") or "Other"
                    explanation = args.get("explanation", "")
                    logger.info(f" ✉️  Sending email to group: {group_label}")
                    sent = send_email(group_label, joke, explanation)
                    tool_result = {
                        "sent": bool(sent),
                        "reason": "ok" if sent else "failed",
                    }
                    email_sent_flag = email_sent_flag or bool(tool_result.get("sent"))
                    last_email_attempt_reason = tool_result.get("reason", "")
                else:
                    tool_result = {"error": f"Unknown tool {fn}"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "name": fn,
                        "content": tool_result
                        if isinstance(tool_result, str)
                        else json.dumps(tool_result),
                    }
                )
            continue  # next cycle after tools

        if content:
            parsed = _parse_final_json(content)
            if parsed:
                # Enforce side-effects BEFORE returning.
                if parsed["safe"] == "safe" and not email_sent_flag:
                    # Model skipped tool call; perform mandatory send_email now.
                    group_label = parsed.get("category", "Other")
                    explanation = parsed.get("explanation", "")
                    sent = send_email(group_label, joke, explanation)

                if parsed["safe"] == "dark":
                    _append_dark_joke(joke, parsed)

                parsed["is_email_sent"] = bool(email_sent_flag)
                if email_sent_flag and not parsed["explanation"]:
                    parsed["explanation"] = parsed.get(
                        "explanation", "Sent without explanation provided"
                    )
                logging.info(" ✅  Task complete")
                logging.info(f"joke: {joke}")
                logging.info(f"safe: {parsed['safe']}")
                logging.info(f"category: {parsed['category']}")
                if parsed["safe"] == "safe":
                    logging.info(
                        "email_sent=%s reason=%s",
                        parsed["is_email_sent"],
                        last_email_attempt_reason,
                    )
                time.sleep(1)
                return parsed
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": "Return only the final JSON object now.",
                    }
                )
                continue

    logging.warning(
        "Exceeded max tool cycles without valid final JSON; returning fallback"
    )
    return {
        "safe": "dark",
        "category": "Other",
        "explanation": "Model failed to return final JSON in time",
    }


def run_browse(term: str, joke: str) -> str:
    """Invoke the browse.py tool with the search term in the context of the joke and return its stdout."""
    browser_arg = f"Define the term '{term}' in the context of this joke: '{joke}'"
    cmd = ["python", "./browser.py", browser_arg]
    logger.info("Running browse tool for term: %s", term)
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, timeout=600
        )
        logger.debug("browse output: %s", out)
        return out
    except subprocess.CalledProcessError as e:
        logger.error("browse.py failed: %s", e.output)
        return ""
    except Exception:
        logger.exception("Error running browse.py")
        return ""


def send_email(group_label: str, joke: str, explanation: str) -> bool:
    """Call send_email.py tool. group_label must be one of GROUPS keys."""
    group_email = GROUPS.get(group_label, GROUPS["Other"])["email"]
    # Use current interpreter for portability (virtualenv compatibility)
    cmd = [sys.executable, "send_email.py", group_email, joke, explanation]
    logger.info("Sending email to %s for group %s", group_email, group_label)
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        logger.exception("send_email.py returned non-zero")
        return False
    except Exception:
        logger.exception("Error running send_email.py")
        return False


def process_joke_file(path: Path, state: Dict[str, Any]) -> None:
    logger.info("\n\n*** ***")
    logger.info("Processing joke file: %s", path)
    joke = path.read_text(encoding="utf-8").strip()
    file_id = path.name

    if file_id in state.get("processed", {}):
        logger.info("Already processed %s, skipping", file_id)
        return

    try:
        result = classify_and_act_on_joke(joke, state)
    except Exception:
        logger.exception("LLM tool-driven processing failed for %s", file_id)
        sys.exit(1)
        # result = {"safe": False, "category": "Other", "explanation": "LLM error"}

    # Mark processed
    state.setdefault("processed", {})[file_id] = {
        "agent": "003",
        "joke": joke,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    save_state(state)


def main_loop(poll_interval: int = 60):
    state = load_state()
    logger.info("Agent started, watching %s", OUTPUT_DIR)

    while True:
        txt_files = sorted(glob.glob(str(OUTPUT_DIR / "*.txt")))
        for f in txt_files:
            process_joke_file(Path(f), state)
            # return
        # Sleep and be responsive to shutdown
        for _ in range(int(poll_interval)):
            time.sleep(1)


if __name__ == "__main__":
    main_loop()
