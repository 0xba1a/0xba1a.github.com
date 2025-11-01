import json
from dotenv import load_dotenv
import requests
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("jokes_filter")

load_dotenv()

AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = "gpt-5"
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")

JOKES_API = "https://official-joke-api.appspot.com/jokes/programming/random"

SYSTEM_PROMPT = """
You are a helpful AI assistant that analyzes jokes.
1. Explain the meaning of the joke in a concise manner.
2. Determine if the joke is safe for work (safe/unsafe).
Respond strictly in the following JSON format:
{{
  "meaning": "<explanation>",
  "safety": "<safe/unsafe>"
}}
"""

def get_joke() -> str:
    response = requests.get(JOKES_API)
    response.raise_for_status()
    joke_data = response.json()[0]
    return f"{joke_data['setup']} {joke_data['punchline']}"


def ask_ai(joke: str) -> (str, str):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_KEY,
    }
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": joke}
        ],
        "temperature": 1.0
    }
    response = requests.post(url, headers=headers, json=payload)
    # response.raise_for_status()
    if response.status_code != 200:
        logger.error(f"Error from AI service: {response.status_code} - {response.text}")
        raise RuntimeError("Failed to get response from AI service")
    result = response.json()
    content = result['choices'][0]['message']['content']
    result_json = json.loads(content)
    return result_json['meaning'], result_json['safety']


def send_email(joke: str, meaning: str) -> None:
    logger.info("Sending email with joke and its meaning...")
    pass


if __name__ == "__main__":
    joke = get_joke()
    logger.info(f"Fetched joke: {joke}")

    meaning, safety = ask_ai(joke)
    logger.info(f"Joke meaning: {meaning}")
    logger.info(f"Joke safety: {safety}")

    if safety.lower() != "safe":
        logger.warning("Joke deemed unsafe, not proceeding further.")
    else:
        send_email(joke, meaning)
