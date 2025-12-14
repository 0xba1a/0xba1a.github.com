"""
Chatbot - RAG-based chatbot for answering questions
"""

import sys
sys.path.append('..')

from lib import ask_llm
from indexer import VectorDatabase
import config
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


class AlterEgoChatbot:
    """
    RAG-based chatbot that answers questions about a person using their data.
    """
    
    def __init__(self):
        """
        Initialize the chatbot.
        """
        logger.info("Initializing AlterEgoChatbot")
        self.db = VectorDatabase()
        
        # Load existing collection
        if not self.db.get_collection():
            logger.error("Vector database not found during chatbot initialization")
            raise ValueError("Vector database not found. Please run indexing first.")
        
        logger.info("Chatbot initialized successfully")
        print("✓ Chatbot initialized and ready!")
    
    def retrieve_context(self, query: str, n_results: int = None) -> str:
        """
        Retrieve relevant context from the vector database.
        
        Args:
            query: User's question
            n_results: Number of results to retrieve
        
        Returns:
            Concatenated context string
        """
        n_results = n_results or config.TOP_K_RESULTS
        logger.debug(f"Retrieving context for query: '{query[:50]}...'")
        # Query the database
        results = self.db.query(query, n_results)
        logger.debug(f"Retrieved {len(results.get('documents', [[]])[0])} documents from database")
        
        # Extract and format documents
        if results and 'documents' in results and len(results['documents']) > 0:
            documents = results['documents'][0]  # First query result
            context = "\n\n---\n\n".join(documents)
            logger.info(f"Retrieved {len(documents)} context chunks ({len(context)} chars)")
            return context
        
        logger.warning("No context found for query")
        return ""
    
    def build_prompt(self, query: str, context: str) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            query: User's question
            context: Retrieved context
        
        Returns:
            Complete prompt string
        """
        prompt = f"""You are an AI assistant answering questions about a person based on their LinkedIn profile, GitHub profile, and resume.

IMPORTANT INSTRUCTIONS:
- Answer as if you ARE the person (use "I", "my", "me")
- Be friendly and conversational
- Only use information from the context provided below
- If the context doesn't contain enough information to answer, say so honestly
- Keep answers concise but informative

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def chat(self, query: str) -> str:
        """
        Process a user query and return an answer.
        
        Args:
            query: User's question
        
        Returns:
            Answer string
        """
        logger.info(f"Processing chat query: '{query}'")
        
        # Step 1: Retrieve relevant context
        context = self.retrieve_context(query)
        
        if not context:
            logger.warning("No context available for query")
            return "I don't have enough information to answer that question based on my available data."
        
        # Step 2: Build prompt with context
        prompt = self.build_prompt(query, context)
        logger.debug(f"Built prompt ({len(prompt)} chars)")
        
        # Step 3: Get response from LLM
        logger.debug(f"Sending prompt to LLM model: {config.CHAT_MODEL}")
        response = ask_llm(prompt, model=config.CHAT_MODEL)
        logger.info(f"Received response from LLM ({len(response)} chars)")
        
        return response
    
    def run_interactive(self):
        """
        Run an interactive chat session.
        """
        logger.info("Starting interactive chat session")
        print("\n" + "=" * 60)
        print("Alter-Ego Chatbot")
        print("=" * 60)
        print("\nAsk me anything! (Type 'quit' or 'exit' to stop)\n")
        
        query_count = 0
        while True:
            try:
                # Get user input
                query = input("You: ").strip()
                
                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    logger.info(f"User initiated exit. Total queries: {query_count}")
                    print("\nAlter-Ego: Goodbye! Have a great day!")
                    break
                
                # Skip empty queries
                if not query:
                    continue
                
                query_count += 1
                logger.debug(f"Query #{query_count}: {query}")
                
                # Get response
                print("\nAlter-Ego: ", end="", flush=True)
                response = self.chat(query)
                print(response)
                print()
            
            except KeyboardInterrupt:
                logger.info(f"Session interrupted by user. Total queries: {query_count}")
                print("\n\nAlter-Ego: Goodbye! Have a great day!")
                break
            except Exception as e:
                logger.exception(f"Error during chat: {e}")
                print(f"\n❌ Error: {e}")
                print("Please try again.\n")

