"""
Chatbot CoT - Chain of Thought RAG-based chatbot
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
    RAG-based chatbot that uses Chain of Thought reasoning to answer questions.
    """
    
    def __init__(self):
        """
        Initialize the chatbot.
        """
        logger.info("Initializing AlterEgoChatbot (CoT)")
        self.db = VectorDatabase()
        
        # Load existing collection
        if not self.db.get_collection():
            logger.error("Vector database not found during chatbot initialization")
            raise ValueError("Vector database not found. Please run indexing first.")
        
        logger.info("AlterEgoChatbot (CoT) initialized successfully")
        print("✓ AlterEgoChatbot (CoT) initialized and ready!")
    
    def _format_context_for_log(self, context) -> str:
        """
        Helper to format context for logging.
        Handles string, list of strings, or list of integers (tokens).
        """
        if isinstance(context, list):
            if not context:
                return "[]"
            # Check if it's a list of integers (tokens)
            if isinstance(context[0], int):
                return f"Tokens: {context}"
            # Assume list of strings
            return "\n---\n".join(str(item) for item in context)
        return str(context)

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
        
        # Extract and format documents
        if results and 'documents' in results and len(results['documents']) > 0:
            documents = results['documents'][0]  # First query result
            context = "\n\n---\n\n".join(documents)
            logger.info(f"Retrieved {len(documents)} context chunks")
            return context
        
        logger.warning("No context found for query")
        return ""
    
    def build_prompt(self, query: str, context: str) -> str:
        """
        Build the Chain of Thought prompt for the LLM.
        
        Args:
            query: User's question
            context: Retrieved context
        
        Returns:
            Complete prompt string with CoT instructions
        """
        prompt = f"""You are an AI assistant answering questions about a person based on their LinkedIn profile, GitHub profile, and resume.

IMPORTANT INSTRUCTIONS:
1. You must use Chain of Thought reasoning.
2. First, analyze the provided context step-by-step to find relevant information.
3. Second, synthesize that information to form an answer.
4. Answer as if you ARE the person (use "I", "my", "me").
5. If the context doesn't contain enough information, state that clearly in your reasoning.

CONTEXT:
{context}

QUESTION: {query}

RESPONSE FORMAT:
Reasoning:
[Your step-by-step analysis of the context and how it relates to the question]

Answer:
[Your final response to the user, based on the reasoning]
"""
        return prompt
    
    def generate_search_query(self, user_query: str) -> str:
        """
        Ask LLM to generate a better search query for the RAG system.
        
        Args:
            user_query: The original user question
            
        Returns:
            Optimized search query string
        """
        prompt = f"""You are an intelligent assistant with access to a RAG system containing a person's professional background (LinkedIn, GitHub, Resume).
        
The user has asked: "{user_query}"

To answer this accurately, what specific information should I search for in the database? 
Provide ONLY the search query that will retrieve the most relevant documents. Do not explain.
"""
        logger.debug(f"Generating search query for: '{user_query}'")
        search_query = ask_llm(prompt).strip().strip('"')
        logger.info(f"LLM generated search query: '{search_query}'")
        return search_query

    def chat(self, query: str) -> str:
        """
        Process a user query using 2-step reasoning:
        1. Generate search query
        2. Retrieve context
        3. Generate answer with CoT
        
        Args:
            query: User's question
        
        Returns:
            LLM response containing reasoning and answer
        """
        logger.info(f"Processing query: '{query}'")
        
        # 1. Generate optimized search query
        search_query = self.generate_search_query(query)
        print(f"(Internal Search Query: {search_query})")
        
        # 2. Retrieve context using the optimized query
        context = self.retrieve_context(search_query)
        
        logger.debug(f"Context content: {self._format_context_for_log(context)}")
        
        if not context:
            # Fallback: try original query if generated one failed
            logger.info("No context with generated query, trying original query")
            context = self.retrieve_context(query)
            
        if not context:
            logger.info("No context found, returning fallback response")
            return "I don't have enough information in my documents to answer that question."
        
        # 3. Build CoT prompt
        prompt = self.build_prompt(query, context)
        
        # 4. Ask LLM
        logger.debug("Sending prompt to LLM")
        response = ask_llm(prompt)
        logger.info("Received response from LLM")
        
        return response

    def run_interactive(self):
        """
        Run the chatbot in interactive mode.
        """
        print("\n" + "="*50)
        print("🤖 Alter-Ego Chatbot (Chain of Thought Mode)")
        print("Type 'exit', 'quit', or 'bye' to end the session.")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye! 👋")
                    break
                
                if not user_input:
                    continue
                
                print("\nThinking... 🤔")
                response = self.chat(user_input)
                print(f"\nBot:\n{response}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in interactive loop: {str(e)}")
                print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    # Setup logging if run directly
    from lib import setup_logging
    setup_logging("chatbot_cot")
    
    try:
        bot = AlterEgoChatbot()
        bot.run_interactive()
    except Exception as e:
        logger.critical(f"Failed to start chatbot: {str(e)}")
        print(f"Error: {str(e)}")
