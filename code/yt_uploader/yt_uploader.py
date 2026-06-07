#!/usr/bin/env python3
"""
YouTube Batch Uploader CLI

Recursively uploads all videos from a folder to YouTube, appends predefined
hashtags to the description, schedules them based on the last video in a
configured playlist, and adds each video to that playlist. Uploaded files
are moved to an ARCHIVE sub-folder.

Usage:
    python yt_uploader.py --folder /path/to/videos --config yt_upload_config.yaml
"""

import argparse
import os
import random
import shutil
import sys
import time

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

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


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


def get_playlist_count(youtube, playlist_id):
    """Get the number of items in a playlist."""
    count = 0
    page_token = None

    while True:
        request = youtube.playlistItems().list(
            playlistId=playlist_id,
            part="contentDetails",
            maxResults=50,
            pageToken=page_token,
        )
        response = request.execute()
        count += len(response.get("items", []))

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return count


def build_description(user_description, hashtags):
    """Append predefined hashtags to the user's description."""
    parts = []

    description_text = (user_description or "").strip()
    if description_text:
        parts.append(description_text)

    if hashtags:
        parts.append("\n".join(hashtags))

    return "\n\n".join(parts)


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


def collect_video_files(folder):
    """Recursively collect all video files in *folder*, sorted by name.

    Skips files inside any ARCHIVE directory.
    """
    videos = []
    for root, dirs, files in os.walk(folder):
        # Skip ARCHIVE directories
        dirs[:] = [d for d in dirs if d != "ARCHIVE"]
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, fname))
    return videos


def archive_video(file_path, folder):
    """Move *file_path* into an ARCHIVE sub-folder under *folder*."""
    archive_dir = os.path.join(folder, "ARCHIVE")
    os.makedirs(archive_dir, exist_ok=True)
    dest = os.path.join(archive_dir, os.path.basename(file_path))
    shutil.move(file_path, dest)
    print(f"Moved {file_path} -> {dest}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch-upload videos from a folder to YouTube."
    )
    parser.add_argument("--folder", required=True, help="Folder containing videos to upload")
    parser.add_argument("--description", default="", help="Video description")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "yt_upload_config.yaml"),
        help="Path to YAML config file (default: yt_upload_config.yaml)",
    )
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        sys.exit(f"Folder not found: {folder}")

    # Collect video files
    video_files = collect_video_files(folder)
    if not video_files:
        sys.exit(f"No video files found in {folder}")

    print(f"Found {len(video_files)} video(s) to upload:")
    for vf in video_files:
        print(f"  {vf}")

    # Load config
    config = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    playlist_id = config.get("playlist_id", "")
    if not playlist_id or playlist_id == "YOUR_PLAYLIST_ID_HERE":
        sys.exit("Please set a valid 'playlist_id' in the config file.")

    base_title = config.get("title", "")
    if not base_title:
        sys.exit("Please set a 'title' in the config file.")

    # Authenticate
    youtube = authenticate(config, config_dir)

    # Get current playlist count
    playlist_count = get_playlist_count(youtube, playlist_id)
    print(f"\nCurrent playlist has {playlist_count} video(s).")

    # Build description with hashtags
    hashtags = config.get("hashtags", [])
    description = build_description(args.description, hashtags)

    for idx, file_path in enumerate(video_files):
        video_number = playlist_count + idx + 1
        title = f"{base_title} {video_number}"

        snippet = {
            "title": title,
            "description": description,
            "tags": config.get("tags", []),
            "categoryId": config.get("category_id", "22"),
            "defaultLanguage": config.get("default_language", "en"),
        }

        game_title = config.get("game_title")
        if game_title:
            snippet["gameTitle"] = game_title

        body = {
            "snippet": snippet,
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": config.get("made_for_kids", False),
                "embeddable": config.get("embeddable", True),
                "license": config.get("license", "youtube"),
            },
        }

        print(f"\n{'='*50}")
        print(f"[{idx + 1}/{len(video_files)}]")
        print(f"Title:       {title}")
        print(f"File:        {file_path}")
        print(f"{'='*50}\n")

        # Upload
        response = upload_video(youtube, file_path, body)
        video_id = response["id"]
        print(f"\nVideo uploaded successfully!")
        print(f"Video ID: {video_id}")

        # Add to playlist
        add_to_playlist(youtube, playlist_id, video_id)

        # Archive the uploaded file
        archive_video(file_path, folder)

        # Summary
        print(f"\n{'='*50}")
        print(f"Video:    https://youtu.be/{video_id}")
        print(f"Studio:   https://studio.youtube.com/video/{video_id}/edit")
        print(f"{'='*50}")

    print(f"\nAll {len(video_files)} video(s) uploaded and archived successfully!")


if __name__ == "__main__":
    main()
