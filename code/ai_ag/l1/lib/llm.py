"""
LLM Library - Simple interface for interacting with Ollama
"""

import requests
import json
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


def ask_llm(prompt: str, model: str = "qwen3-coder") -> str:
    """
    Send a prompt to the locally running Ollama LLM and get a response.
    
    Args:
        prompt: The question or prompt to send to the LLM
        model: The Ollama model to use (default: llama3.2)
    
    Returns:
        The LLM's response as a string
    """
    url = "http://localhost:11434/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    logger.debug(f"Sending request to Ollama model: {model}")
    logger.debug(f"Prompt: {json.dumps(prompt, indent=4)}")
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.debug(f"Ollama response: {json.dumps(result, indent=4)}")
        return result.get("response", "")
    
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {e}"
