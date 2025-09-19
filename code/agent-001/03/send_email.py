#!/usr/bin/env python3
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("send_email")

OUTBOX = Path("/tmp/agent-001/outbox.json")
OUTBOX.parent.mkdir(parents=True, exist_ok=True)


def main():
    if len(sys.argv) < 4:
        print("Usage: send_email.py <to_group> <quote> <explanation>")
        sys.exit(2)
    to_group = sys.argv[1]
    quote = sys.argv[2]
    explanation = sys.argv[3]

    # Append to outbox file as a record
    record = {"to": to_group, "quote": quote, "explanation": explanation, "ts": datetime.now(timezone.utc).isoformat()}
    if OUTBOX.exists():
        arr = json.loads(OUTBOX.read_text(encoding="utf-8"))
    else:
        arr = []
    arr.append(record)
    OUTBOX.write_text(json.dumps(arr, indent=2), encoding="utf-8")
    logger.info("Queued email to %s", to_group)


if __name__ == "__main__":
    main()
