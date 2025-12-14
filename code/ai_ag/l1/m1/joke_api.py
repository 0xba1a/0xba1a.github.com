"""
Joke API - Fetch jokes from the official jokes API
"""

import requests


def fetch_programming_joke():
    """
    Fetch a random programming joke from the official jokes API.
    
    Returns:
        A dictionary containing the joke details (type, setup, punchline, id)
        or None if there was an error
    """
    url = "https://official-joke-api.appspot.com/jokes/programming/random"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # API returns a list with one joke
        jokes = response.json()
        
        if jokes and len(jokes) > 0:
            return jokes[0]
        
        return None
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching joke: {e}")
        return None
