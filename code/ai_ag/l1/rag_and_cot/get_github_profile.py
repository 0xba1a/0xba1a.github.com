#!/usr/bin/env python3
"""
GitHub Profile Downloader

Download a user's public GitHub profile and repository information.
Saves the data to data/github_profile.json
"""

import requests
import json
import os
import sys
from typing import Dict, List


def get_github_profile(username: str) -> Dict:
    """
    Fetch a GitHub user's public profile.
    
    Args:
        username: GitHub username
    
    Returns:
        Dictionary with profile data
    """
    url = f"https://api.github.com/users/{username}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching profile: {e}")
        return None


def get_github_repos(username: str, max_repos: int = 100) -> List[Dict]:
    """
    Fetch a GitHub user's public repositories.
    
    Args:
        username: GitHub username
        max_repos: Maximum number of repositories to fetch
    
    Returns:
        List of repository data
    """
    url = f"https://api.github.com/users/{username}/repos"
    params = {
        "per_page": min(max_repos, 100),
        "sort": "updated",
        "direction": "desc"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching repositories: {e}")
        return []


def create_github_summary(username: str) -> Dict:
    """
    Create a comprehensive GitHub profile summary.
    
    Args:
        username: GitHub username
    
    Returns:
        Dictionary with formatted profile data
    """
    print(f"\n📥 Fetching GitHub profile for: {username}")
    
    # Get profile
    print("  - Fetching user profile...")
    profile = get_github_profile(username)
    
    if not profile:
        return None
    
    # Get repositories
    print("  - Fetching repositories...")
    repos = get_github_repos(username)
    
    # Build summary
    summary = {
        "username": profile.get("login"),
        "name": profile.get("name"),
        "bio": profile.get("bio"),
        "location": profile.get("location"),
        "company": profile.get("company"),
        "blog": profile.get("blog"),
        "email": profile.get("email"),
        "twitter_username": profile.get("twitter_username"),
        "public_repos": profile.get("public_repos"),
        "public_gists": profile.get("public_gists"),
        "followers": profile.get("followers"),
        "following": profile.get("following"),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
        "profile_url": profile.get("html_url"),
        "repositories": []
    }
    
    # Process repositories
    for repo in repos:
        # Skip forks if you want only original repos
        if repo.get("fork"):
            continue
        
        repo_data = {
            "name": repo.get("name"),
            "description": repo.get("description"),
            "language": repo.get("language"),
            "stars": repo.get("stargazers_count"),
            "forks": repo.get("forks_count"),
            "url": repo.get("html_url"),
            "topics": repo.get("topics", []),
            "created_at": repo.get("created_at"),
            "updated_at": repo.get("updated_at")
        }
        
        summary["repositories"].append(repo_data)
    
    print(f"  ✓ Found {len(summary['repositories'])} repositories")
    
    return summary


def save_to_file(data: Dict, filepath: str):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Profile saved to: {filepath}")


def main():
    """
    Main function - download GitHub profile and save to file.
    """
    print("\n" + "=" * 60)
    print("          GitHub Profile Downloader")
    print("=" * 60)
    
    # Get username from command line or prompt
    if len(sys.argv) > 1:
        username = sys.argv[1]
    else:
        username = input("\nEnter GitHub username: ").strip()
    
    if not username:
        print("❌ No username provided!")
        sys.exit(1)
    
    # Create summary
    summary = create_github_summary(username)
    
    if not summary:
        print("\n❌ Failed to fetch GitHub profile.")
        print("Make sure the username is correct and you have internet connection.")
        sys.exit(1)
    
    # Save to file
    output_file = os.path.join(
        os.path.dirname(__file__),
        "data",
        "github_profile.json"
    )
    
    save_to_file(summary, output_file)
    
    # Display summary
    print("\n" + "=" * 60)
    print("Profile Summary")
    print("=" * 60)
    print(f"Name: {summary.get('name') or 'N/A'}")
    print(f"Username: {summary.get('username')}")
    print(f"Bio: {summary.get('bio') or 'N/A'}")
    print(f"Location: {summary.get('location') or 'N/A'}")
    print(f"Public Repos: {summary.get('public_repos')}")
    print(f"Followers: {summary.get('followers')}")
    print(f"Following: {summary.get('following')}")
    print(f"Repositories Saved: {len(summary.get('repositories', []))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
