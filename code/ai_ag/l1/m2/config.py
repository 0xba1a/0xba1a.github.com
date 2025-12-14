"""
Configuration settings for the Alter-Ego Chatbot
"""

import os

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# Data files
LINKEDIN_FILE = os.path.join(DATA_DIR, "Profile.csv")
GITHUB_FILE = os.path.join(DATA_DIR, "github_profile.json")
RESUME_FILE = os.path.join(DATA_DIR, "resume.pdf")

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3-coder"

# ChromaDB settings
COLLECTION_NAME = "personal_knowledge"

# Chunking settings
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 200  # characters

# RAG settings
TOP_K_RESULTS = 5  # Number of relevant chunks to retrieve
