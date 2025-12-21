"""
LLM Library - Simple interface for interacting with Ollama and Azure OpenAI
"""

import requests
import json
import logging
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

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

def ask_llm_azure_openai(prompt: str) -> str:
    """
    Send a prompt to Azure OpenAI and get a response.
    
    Args:
        prompt: The question or prompt to send to the LLM
    
    Returns:
        The LLM's response as a string
    """
    # Load environment variables from the .env file in the parent directory (l1/)
    # This file is in l1/lib/llm.py, so .env is in ../.env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(env_path)
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    # Using a standard recent API version
    api_version = "2024-08-01-preview" 
    
    if not all([api_key, azure_endpoint, deployment_name]):
        logger.error("Missing Azure OpenAI environment variables")
        return "Error: Missing Azure OpenAI environment variables. Please check .env file."

    logger.debug(f"Sending request to Azure OpenAI deployment: {deployment_name}")

    try:
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )

        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content
        logger.debug("Azure OpenAI response received")
        return result
        
    except Exception as e:
        logger.error(f"Azure OpenAI Error: {e}")
        return f"Error connecting to Azure OpenAI: {e}"
