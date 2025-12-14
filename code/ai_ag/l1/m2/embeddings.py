"""
Embeddings - Generate embeddings using Ollama's embedding models
"""

import sys
sys.path.append('..')

from lib import ask_llm
import requests
import json
import logging
from typing import List

# Get logger for this module
logger = logging.getLogger(__name__)


def generate_embedding(text: str, model: str = "nomic-embed-text") -> List[float]:
    """
    Generate embedding vector for a given text using Ollama.
    
    Args:
        text: Text to embed
        model: Ollama embedding model to use
    
    Returns:
        List of floats representing the embedding vector
    """
    logger.debug(f"Generating embedding with model: {model}")
    url = "http://localhost:11434/api/embeddings"
    
    payload = {
        "model": model,
        "prompt": text
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        embedding = result.get("embedding", [])
        logger.debug(f"Generated embedding of dimension: {len(embedding)}")
        return embedding
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generating embedding: {e}")
        print(f"Error generating embedding: {e}")
        return []


def generate_embeddings_batch(texts: List[str], model: str = "nomic-embed-text") -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    
    Args:
        texts: List of texts to embed
        model: Ollama embedding model to use
    
    Returns:
        List of embedding vectors
    """
    logger.info(f"Generating embeddings for {len(texts)} text chunks")
    embeddings = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0:
            print(f"    Processing chunk {i + 1}/{len(texts)}...")
            logger.debug(f"Processed {i + 1}/{len(texts)} embeddings")
        
        embedding = generate_embedding(text, model)
        embeddings.append(embedding)
    
    logger.info(f"Successfully generated {len(embeddings)} embeddings")
    return embeddings
