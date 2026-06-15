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
import datetime
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
            oauth_port = int(config.get("oauth_port", 8080))
            creds = flow.run_local_server(
                host="localhost",
                port=oauth_port,
                open_browser=False,
                authorization_prompt_message=(
                    "\nOpen this URL in your browser to authorize "
                    "(make sure port {} is forwarded):\n{{url}}\n".format(oauth_port)
                ),
            )

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


def get_uploads_playlist_id(youtube):
    """Return the authenticated channel's 'uploads' playlist id."""
    response = youtube.channels().list(part="contentDetails", mine=True).execute()
    items = response.get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def get_my_channel(youtube):
    """Return (channel_id, channel_title) for the authenticated channel."""
    response = youtube.channels().list(part="snippet", mine=True).execute()
    items = response.get("items", [])
    if not items:
        return None, None
    item = items[0]
    return item["id"], item["snippet"]["title"]


def verify_channel(youtube, expected_name):
    """Abort unless the authenticated channel title matches *expected_name*.

    With multiple channels on one Google account, the OAuth token determines
    which channel receives uploads. This guard prevents uploading to the wrong
    channel: delete token.json and re-run to pick the right channel at sign-in.
    """
    channel_id, channel_title = get_my_channel(youtube)
    if channel_title is None:
        sys.exit("Could not determine the authenticated YouTube channel.")

    print(f"Authenticated channel: {channel_title} ({channel_id})")
    if not expected_name:
        return

    if channel_title.strip().lower() != expected_name.strip().lower():
        sys.exit(
            f"\nWRONG CHANNEL: authenticated as '{channel_title}', but the config "
            f"expects '{expected_name}'.\n"
            "Delete the cached token and re-run, then pick the correct channel "
            "at Google's sign-in screen:\n"
            "  rm code/yt_uploader/token.json\n"
        )
    print(f"Channel verified: {channel_title}\n")


def get_latest_scheduled_datetime(youtube):
    """Find the furthest-future scheduled publishAt across the channel's uploads.

    Returns a timezone-aware UTC datetime, or None if nothing is scheduled.
    """
    uploads_id = get_uploads_playlist_id(youtube)
    if not uploads_id:
        return None

    video_ids = []
    page_token = None
    while True:
        response = youtube.playlistItems().list(
            playlistId=uploads_id,
            part="contentDetails",
            maxResults=50,
            pageToken=page_token,
        ).execute()
        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
        page_token = response.get("nextPageToken")
        if not page_token:
            break

    latest = None
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        response = youtube.videos().list(
            part="status", id=",".join(chunk)
        ).execute()
        for video in response.get("items", []):
            publish_at = video.get("status", {}).get("publishAt")
            if not publish_at:
                continue
            dt = datetime.datetime.fromisoformat(publish_at.replace("Z", "+00:00"))
            if latest is None or dt > latest:
                latest = dt
    return latest


def compute_first_slot(youtube, hour, minute):
    """Return the next free upload slot (local tz) at *hour:minute*.

    The next free day is the day after the latest already-scheduled upload, or
    tomorrow if nothing is scheduled. The result is always in the future.
    """
    now = datetime.datetime.now().astimezone()
    tz = now.tzinfo

    latest = get_latest_scheduled_datetime(youtube)
    if latest is not None:
        base_date = (latest.astimezone(tz) + datetime.timedelta(days=1)).date()
    else:
        base_date = now.date() + datetime.timedelta(days=1)

    slot = datetime.datetime(
        base_date.year, base_date.month, base_date.day, hour, minute, tzinfo=tz
    )
    while slot <= now:
        slot += datetime.timedelta(days=1)
    return slot


def to_publish_at(dt):
    """Convert a tz-aware datetime to an RFC3339 UTC string for publishAt."""
    return (
        dt.astimezone(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
        description="Batch-upload videos to YouTube, scheduled one per day."
    )
    parser.add_argument(
        "--folder",
        help="Folder containing videos to upload (overridden by config 'videos')",
    )
    parser.add_argument("--description", default="", help="Fallback video description")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "yt_upload_config.yaml"),
        help="Path to YAML config file (default: yt_upload_config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Authenticate, verify channel, and print the schedule without uploading",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))

    hashtags = config.get("hashtags", [])
    tags = config.get("tags", [])
    category_id = config.get("category_id", "22")
    default_language = config.get("default_language", "en")
    default_audio_language = config.get("default_audio_language", default_language)
    privacy_status = config.get("privacy_status", "private")
    hour = int(config.get("schedule_hour", 8))
    minute = int(config.get("schedule_minute", 15))
    playlist_id = config.get("playlist_id", "") or ""
    if playlist_id == "YOUR_PLAYLIST_ID_HERE":
        playlist_id = ""

    # Build the ordered list of (file_path, title, description, publish_at) jobs.
    # publish_at is an explicit tz-aware datetime override, or None for auto.
    tz = datetime.datetime.now().astimezone().tzinfo
    jobs = []
    manifest = config.get("videos")
    if manifest:
        for entry in manifest:
            file_path = entry["file"]
            if not os.path.isabs(file_path):
                file_path = os.path.normpath(os.path.join(config_dir, file_path))
            if not os.path.isfile(file_path):
                sys.exit(f"Video file not found: {file_path}")
            description = build_description(entry.get("description", ""), hashtags)
            explicit = entry.get("publish_at")
            when = None
            if explicit:
                try:
                    when = datetime.datetime.fromisoformat(str(explicit))
                except ValueError:
                    sys.exit(
                        f"Bad publish_at '{explicit}' (use 'YYYY-MM-DD HH:MM')."
                    )
                if when.tzinfo is None:
                    when = when.replace(tzinfo=tz)
            jobs.append((file_path, entry["title"], description, when))
    else:
        if not args.folder:
            sys.exit("Provide --folder or a 'videos' manifest in the config.")
        folder = os.path.abspath(args.folder)
        if not os.path.isdir(folder):
            sys.exit(f"Folder not found: {folder}")
        base_title = config.get("title", "")
        if not base_title:
            sys.exit("Set a 'title' in the config or use a 'videos' manifest.")
        description = build_description(args.description, hashtags)
        for idx, file_path in enumerate(collect_video_files(folder)):
            jobs.append((file_path, f"{base_title} {idx + 1}", description, None))

    if not jobs:
        sys.exit("No videos to upload.")

    print(f"Found {len(jobs)} video(s) to upload:")
    for fp, title, _, _ in jobs:
        print(f"  {title}\n    {fp}")

    # Authenticate and make sure we are on the intended channel.
    youtube = authenticate(config, config_dir)
    verify_channel(youtube, config.get("channel_name", ""))

    # Assign a slot per job: explicit publish_at wins; the rest auto-fill the
    # next free days (08:15), skipping any day already taken by an explicit slot.
    auto_needed = sum(1 for j in jobs if j[3] is None)
    auto_slots = []
    if auto_needed:
        first_slot = compute_first_slot(youtube, hour, minute)
        taken = {j[3].date() for j in jobs if j[3] is not None}
        cur = first_slot
        while len(auto_slots) < auto_needed:
            if cur.date() not in taken:
                auto_slots.append(cur)
            cur += datetime.timedelta(days=1)

    slots = []
    ai = 0
    for job in jobs:
        if job[3] is not None:
            slots.append(job[3])
        else:
            slots.append(auto_slots[ai])
            ai += 1

    print("Planned schedule:")
    for (_, title, _, _), slot in zip(jobs, slots):
        print(f"  {slot.strftime('%Y-%m-%d %H:%M %Z')}  ->  {title}")
    print()

    if args.dry_run:
        print("Dry run complete. No videos were uploaded.")
        return

    for idx, ((file_path, title, description, _), slot) in enumerate(zip(jobs, slots)):
        publish_at = to_publish_at(slot)

        snippet = {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": default_language,
            "defaultAudioLanguage": default_audio_language,
        }

        game_title = config.get("game_title")
        if game_title:
            snippet["gameTitle"] = game_title

        body = {
            "snippet": snippet,
            "status": {
                "privacyStatus": privacy_status,
                "publishAt": publish_at,
                "selfDeclaredMadeForKids": config.get("made_for_kids", False),
                "embeddable": config.get("embeddable", True),
                "license": config.get("license", "youtube"),
            },
        }

        print(f"\n{'='*50}")
        print(f"[{idx + 1}/{len(jobs)}]")
        print(f"Title:       {title}")
        print(f"File:        {file_path}")
        print(f"Publish at:  {slot.strftime('%Y-%m-%d %H:%M %Z')}  ({publish_at})")
        print(f"{'='*50}\n")

        # Upload
        response = upload_video(youtube, file_path, body)
        video_id = response["id"]
        print("\nVideo uploaded successfully!")
        print(f"Video ID: {video_id}")

        # Add to playlist (only if one is configured)
        if playlist_id:
            add_to_playlist(youtube, playlist_id, video_id)

        # Archive the uploaded file next to itself
        archive_video(file_path, os.path.dirname(file_path))

        # Summary
        print(f"\n{'='*50}")
        print(f"Video:    https://youtu.be/{video_id}")
        print(f"Studio:   https://studio.youtube.com/video/{video_id}/edit")
        print(f"{'='*50}")

    print(f"\nAll {len(jobs)} video(s) uploaded and scheduled successfully!")


if __name__ == "__main__":
    main()
