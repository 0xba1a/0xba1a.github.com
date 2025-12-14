"""
Module 1 - Joke Sentiment Analysis
Fetches a programming joke and analyzes if it's safe to share at work.
"""

import sys
import json
import re
sys.path.append('..')

from lib import ask_llm
from joke_api import fetch_programming_joke


def extract_json_from_response(response):
    """
    Extract JSON object from a response that may contain mixed text and JSON.
    
    Args:
        response: String that may contain JSON along with natural language
    
    Returns:
        Parsed JSON object or None if not found
    """
    # First try to parse the entire response
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object using regex
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def analyze_joke_sentiment(joke, max_retries=3):
    """
    Use the LLM to analyze if a joke is appropriate for office environment.
    Retries up to max_retries times if LLM doesn't return valid JSON.
    
    Args:
        joke: A dictionary containing 'setup' and 'punchline' keys
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        A dictionary with sentiment analysis results
    """
    initial_prompt = f"""Analyze this programming joke and determine if it's appropriate to share with office colleagues.

Joke Setup: {joke['setup']}
Joke Punchline: {joke['punchline']}

Respond ONLY with valid JSON in this exact format (no additional text):
{{
    "is_office_safe": true or false,
    "sentiment": "positive/neutral/negative",
    "reason": "brief explanation why it is or isn't office-safe"
}}"""
    
    retry_prompt = """Your previous response was not in valid JSON format. 
Please respond ONLY with valid JSON, nothing else. Use this exact format:
{
    "is_office_safe": true or false,
    "sentiment": "positive/neutral/negative",
    "reason": "brief explanation"
}"""
    
    prompt = initial_prompt
    
    for attempt in range(max_retries):
        print(f"  Attempt {attempt + 1}/{max_retries}...")
        response = ask_llm(prompt)
        
        # Try to extract JSON from response
        analysis = extract_json_from_response(response)
        
        if analysis:
            # Validate that we have the expected fields
            if all(key in analysis for key in ["is_office_safe", "sentiment", "reason"]):
                print(f"  ✓ Valid JSON received")
                return analysis
            else:
                print(f"  ✗ JSON missing required fields")
        else:
            print(f"  ✗ No valid JSON found in response")
        
        # If this isn't the last attempt, switch to retry prompt
        if attempt < max_retries - 1:
            prompt = retry_prompt
    
    # If all retries failed, return a default safe response
    print("  ⚠ All retries exhausted, using default safe response")
    return {
        "is_office_safe": True,
        "sentiment": "neutral",
        "reason": "Unable to analyze - LLM did not return valid JSON (assuming safe)"
    }


def main():
    """
    Main function - fetches a joke and analyzes if it's office-safe.
    """
    print("=" * 60)
    print("Programming Joke - Office Safety Analyzer")
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
    
    # Step 3: Analyze joke with LLM
    print("Analyzing joke sentiment...")
    analysis = analyze_joke_sentiment(joke)
    
    # Step 4: Display the analysis results
    print()
    print("ANALYSIS:")
    print("-" * 60)
    print(f"Sentiment: {analysis['sentiment'].upper()}")
    print(f"Reason: {analysis['reason']}")
    print()
    
    # Step 5: Make the recommendation
    if analysis['is_office_safe']:
        print("✓ SAFE TO SHARE with office colleagues!")
    else:
        print("✗ NOT RECOMMENDED to share at work")
    
    print("-" * 60)
    print()


if __name__ == "__main__":
    main()
