"""
Module 1 - Joke Explainer
Fetches a programming joke and uses an LLM to explain why it's funny.
"""

import sys
sys.path.append('..')

from lib import ask_llm
from joke_api import fetch_programming_joke


def explain_joke(joke):
    """
    Use the LLM to explain why a joke is funny.
    
    Args:
        joke: A dictionary containing 'setup' and 'punchline' keys
    
    Returns:
        The LLM's explanation of the joke
    """
    prompt = f"""Here's a programming joke:

Setup: {joke['setup']}
Punchline: {joke['punchline']}

Please explain why this joke is funny in simple terms."""
    
    return ask_llm(prompt)


def main():
    """
    Main function - orchestrates fetching a joke and getting its explanation.
    """
    print("=" * 60)
    print("Programming Joke Explainer")
    print("=" * 60)
    print()
    
    # Step 1: Fetch a joke from the API
    print("Fetching a programming joke...")
    joke = fetch_programming_joke()
    
    if not joke:
        print("Failed to fetch a joke. Please try again.")
        return
    
    # Step 2: Display the joke
    print()
    print("JOKE:")
    print("-" * 60)
    print(f"Setup: {joke['setup']}")
    print(f"Punchline: {joke['punchline']}")
    print("-" * 60)
    print()
    
    # Step 3: Get explanation from LLM
    print("Getting explanation from LLM...")
    explanation = explain_joke(joke)
    
    # Step 4: Display the explanation
    print()
    print("EXPLANATION:")
    print("-" * 60)
    print(explanation)
    print("-" * 60)
    print()


if __name__ == "__main__":
    main()
