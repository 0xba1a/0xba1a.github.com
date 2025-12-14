#!/usr/bin/env python3
"""
Alter-Ego Chatbot - Main Application

This is the main entry point for the Alter-Ego chatbot.
It automatically builds the vector database if needed, then starts the chat interface.
"""

import os
import sys
import logging
from indexer import build_vector_database, check_database_exists
from chatbot import AlterEgoChatbot


# Get logger for this module
logger = logging.getLogger(__name__)


def main():
    """
    Main function - orchestrates database creation and chatbot startup.
    """
    logger.info("=== Alter-Ego Chatbot Starting ===")
    print("\n" + "=" * 60)
    print("          WELCOME TO ALTER-EGO CHATBOT")
    print("=" * 60)
    
    # Check if vector database exists
    logger.debug("Checking if vector database exists")
    db_exists = check_database_exists()
    logger.info(f"Vector database exists: {db_exists}")
    
    if not db_exists:
        logger.info("Vector database not found. Starting build process")
        print("\n📊 Vector database not found. Building it now...")
        print("This is a one-time process and may take a few minutes.\n")
        
        # Build the database
        success = build_vector_database()
        
        if not success:
            logger.error("Failed to build vector database")
            print("\n❌ Failed to build vector database.")
            print("Please make sure you have added your data files:")
            print("  - data/Profile.csv (LinkedIn profile)")
            print("  - data/github_profile.json")
            print("  - data/resume.pdf")
            sys.exit(1)
        
        logger.info("Vector database built successfully")
    else:
        logger.info("Using existing vector database")
        print("\n✓ Vector database found!")
    
    # Initialize and start chatbot
    logger.info("Initializing chatbot")
    print("\n🤖 Starting chatbot...\n")
    
    try:
        chatbot = AlterEgoChatbot()
        logger.info("Chatbot initialized, starting interactive session")
        chatbot.run_interactive()
        logger.info("Interactive session ended normally")
    
    except Exception as e:
        logger.exception(f"Error starting chatbot: {e}")
        print(f"\n❌ Error starting chatbot: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print(f"  2. Make sure you have the required models:")
        print(f"     - ollama pull {config.EMBEDDING_MODEL}")
        print(f"     - ollama pull {config.CHAT_MODEL}")
        sys.exit(1)


if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.append('..')
    import config
    from lib import setup_logging
    
    # Setup logging before starting
    setup_logging('alter_ego')
    logger.info("Logging initialized")
    
    main()
