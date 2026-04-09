#!/usr/bin/env python3
"""
YouTube Shorts Uploader CLI

Uploads a YouTube Short, appends predefined hashtags to the description,
schedules it based on the last video in a configured playlist, and adds
the new video to that playlist.

Usage:
    python yt_uploader.py --file video.mp4 --title "My Short" --description "Check this out"
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import yaml
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

SCOPES = ["https://www.googleapis.com/auth/youtube"]
API_SERVICE_NAME = "youtube"
API_VERSION = "v3"

MAX_RETRIES = 10
RETRIABLE_STATUS_CODES = [500, 502, 503, 504]


def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        sys.exit(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def authenticate(config, config_dir):
    """Authenticate with YouTube API using OAuth 2.0 local flow."""
    secrets_file = config.get("client_secrets_file", "client_secrets.json")
    if not os.path.isabs(secrets_file):
        secrets_file = os.path.join(config_dir, secrets_file)

    token_file = os.path.join(config_dir, "token.json")

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not os.path.exists(secrets_file):
                sys.exit(
                    f"Client secrets file not found: {secrets_file}\n"
                    "Download it from the Google Cloud Console:\n"
                    "  https://console.cloud.google.com/apis/credentials"
                )
            flow = InstalledAppFlow.from_client_secrets_file(secrets_file, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(token_file, "w") as f:
            f.write(creds.to_json())
        print(f"Token saved to {token_file}")

    return build(API_SERVICE_NAME, API_VERSION, credentials=creds)


def get_last_video_publish_time(youtube, playlist_id):
    """Get the publish/schedule time of the last video in a playlist.

    Returns a timezone-aware datetime or None if the playlist is empty.
    """
    # Paginate to the last page of the playlist
    last_item = None
    page_token = None

    while True:
        request = youtube.playlistItems().list(
            playlistId=playlist_id,
            part="contentDetails",
            maxResults=50,
            pageToken=page_token,
        )
        response = request.execute()

        items = response.get("items", [])
        if items:
            last_item = items[-1]

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    if not last_item:
        return None

    video_id = last_item["contentDetails"]["videoId"]

    # Get the video's status to check for scheduled publishAt
    video_response = youtube.videos().list(
        id=video_id,
        part="status,snippet",
    ).execute()

    if not video_response.get("items"):
        return None

    video = video_response["items"][0]

    # Prefer publishAt (scheduled time) over publishedAt (actual publish time)
    publish_at = video.get("status", {}).get("publishAt")
    if publish_at:
        return datetime.fromisoformat(publish_at.replace("Z", "+00:00"))

    published_at = video.get("snippet", {}).get("publishedAt")
    if published_at:
        return datetime.fromisoformat(published_at.replace("Z", "+00:00"))

    return None


def prompt_schedule_time():
    """Prompt the user for a schedule datetime when playlist is empty."""
    print("\nNo videos found in the playlist. Cannot auto-determine schedule time.")
    print("Enter the publish date and time for this video.")
    print("Format: YYYY-MM-DD HH:MM (in your local timezone)")
    print("Example: 2026-04-01 09:00\n")

    while True:
        user_input = input("Schedule time: ").strip()
        try:
            local_dt = datetime.strptime(user_input, "%Y-%m-%d %H:%M")
            # Attach local timezone
            local_dt = local_dt.astimezone()
            if local_dt <= datetime.now(timezone.utc):
                print("That time is in the past. Please enter a future time.")
                continue
            return local_dt
        except ValueError:
            print("Invalid format. Please use: YYYY-MM-DD HH:MM")


def compute_schedule_time(youtube, playlist_id):
    """Determine when to schedule the video.

    Looks at the last video in the playlist and schedules for the next day
    at the same time. If the playlist is empty, prompts the user.
    """
    last_time = get_last_video_publish_time(youtube, playlist_id)

    if last_time is None:
        return prompt_schedule_time()

    next_time = last_time + timedelta(days=1)
    print(f"Last video in playlist scheduled/published at: {last_time.isoformat()}")
    print(f"New video will be scheduled for: {next_time.isoformat()}")

    # If computed time is in the past, keep adding days until it's in the future
    now = datetime.now(timezone.utc)
    while next_time <= now:
        next_time += timedelta(days=1)
        print(f"Adjusted to future time: {next_time.isoformat()}")

    return next_time


def build_description(user_description, hashtags):
    """Append predefined hashtags to the user's description."""
    if not hashtags:
        return user_description
    hashtag_block = "\n".join(hashtags)
    return f"{user_description}\n\n{hashtag_block}"


def upload_video(youtube, file_path, body):
    """Upload video with resumable upload and exponential backoff retry."""
    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)

    insert_request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    response = None
    retry = 0

    while response is None:
        try:
            print("Uploading...")
            status, response = insert_request.next_chunk()
            if status:
                print(f"Upload progress: {int(status.progress() * 100)}%")
        except HttpError as e:
            if e.resp.status in RETRIABLE_STATUS_CODES:
                error = f"Retriable HTTP error {e.resp.status}: {e.content.decode()}"
            else:
                raise
        except (IOError, OSError) as e:
            error = f"Retriable error: {e}"
        else:
            if response and "id" in response:
                return response
            error = f"Unexpected response: {response}"

        print(error)
        retry += 1
        if retry > MAX_RETRIES:
            sys.exit("Max retries exceeded. Upload failed.")

        sleep_seconds = random.random() * (2 ** retry)
        print(f"Retrying in {sleep_seconds:.1f}s...")
        time.sleep(sleep_seconds)


def add_to_playlist(youtube, playlist_id, video_id):
    """Add the uploaded video to the target playlist."""
    youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {
                    "kind": "youtube#video",
                    "videoId": video_id,
                },
            }
        },
    ).execute()
    print(f"Video added to playlist {playlist_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a YouTube Short and schedule it automatically."
    )
    parser.add_argument("--file", required=True, help="Path to the video file")
    parser.add_argument("--title", required=True, help="Video title")
    parser.add_argument("--description", required=True, help="Video description")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "yt_upload_config.yaml"),
        help="Path to YAML config file (default: yt_upload_config.yaml)",
    )
    args = parser.parse_args()

    # Validate video file
    if not os.path.exists(args.file):
        sys.exit(f"Video file not found: {args.file}")

    # Load config
    config = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    playlist_id = config.get("playlist_id", "")
    if not playlist_id or playlist_id == "YOUR_PLAYLIST_ID_HERE":
        sys.exit("Please set a valid 'playlist_id' in the config file.")

    # Authenticate
    youtube = authenticate(config, config_dir)

    # Compute schedule time
    schedule_time = compute_schedule_time(youtube, playlist_id)
    publish_at = schedule_time.strftime("%Y-%m-%dT%H:%M:%S.000Z") if schedule_time.tzinfo == timezone.utc else schedule_time.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # Build description with hashtags
    hashtags = config.get("hashtags", [])
    description = build_description(args.description, hashtags)

    # Build video metadata
    body = {
        "snippet": {
            "title": args.title,
            "description": description,
            "tags": config.get("tags", []),
            "categoryId": config.get("category_id", "22"),
            "defaultLanguage": config.get("default_language", "en"),
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publish_at,
            "selfDeclaredMadeForKids": config.get("made_for_kids", False),
            "embeddable": config.get("embeddable", True),
            "license": config.get("license", "youtube"),
        },
    }

    print(f"\n{'='*50}")
    print(f"Title:       {args.title}")
    print(f"Description: {args.description}")
    print(f"Hashtags:    {' '.join(hashtags)}")
    print(f"Tags:        {', '.join(body['snippet']['tags'])}")
    print(f"Scheduled:   {publish_at}")
    print(f"File:        {args.file}")
    print(f"{'='*50}\n")

    # Upload
    response = upload_video(youtube, args.file, body)
    video_id = response["id"]
    print(f"\nVideo uploaded successfully!")
    print(f"Video ID: {video_id}")

    # Add to playlist
    add_to_playlist(youtube, playlist_id, video_id)

    # Summary
    print(f"\n{'='*50}")
    print(f"Video:    https://youtu.be/{video_id}")
    print(f"Studio:   https://studio.youtube.com/video/{video_id}/edit")
    print(f"Schedule: {publish_at}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
