# YouTube Shorts Uploader CLI

A CLI tool to upload YouTube Shorts, auto-schedule them, and add them to a playlist.

## Features

- Uploads a video to YouTube as a Short
- Appends predefined hashtags and tags from config
- Auto-schedules the video based on the latest Short in a playlist (next day, 7:30 PM local time)
- Adds the uploaded video to the configured playlist
- Caches OAuth token for subsequent runs

## Prerequisites

- Python 3.9+
- A Google Cloud project with YouTube Data API v3 enabled

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Google Cloud OAuth credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the **YouTube Data API v3**:
   - Navigate to **APIs & Services** > **Library**
   - Search for "YouTube Data API v3" and click **Enable**
4. Configure the **OAuth consent screen**:
   - Go to **APIs & Services** > **OAuth consent screen**
   - Select **External** and click **Create**
   - Fill in the app name and your email
   - Under **Scopes**, add `https://www.googleapis.com/auth/youtube`
   - Under **Test users**, add the email of your YouTube account
   - Save
5. Create **OAuth client ID**:
   - Go to **APIs & Services** > **Credentials**
   - Click **Create Credentials** > **OAuth client ID**
   - Select **Desktop app** as the application type
   - Click **Create**
   - Click **Download JSON**
6. Rename the downloaded file to `client_secrets.json` and place it in this directory (alongside `yt_uploader.py`)

### 3. Configure `yt_upload_config.yaml`

Edit the config file to set your playlist ID, tags, hashtags, and other defaults:

```yaml
playlist_id: "YOUR_PLAYLIST_ID_HERE"   # Replace with your actual playlist ID
category_id: "27"                       # 27 = Education
hashtags:
  - "#Shorts"
  - "#YouTubeShorts"
```

You can find your playlist ID from the YouTube playlist URL: `https://www.youtube.com/playlist?list=PLAYLIST_ID`

## Usage

```bash
python3 yt_uploader.py \
  --file /path/to/video.mp4 \
  --title "Day 19 of Leetcode75 - Find Pivot Index - Part 1" \
  --description "How to solve the leetcode problem Find Pivot Index"
```

### Arguments

| Argument        | Required | Description                                      |
|-----------------|----------|--------------------------------------------------|
| `--file`        | Yes      | Path to the video file                           |
| `--title`       | Yes      | Video title                                      |
| `--description` | No       | Video description (defaults to empty text)       |
| `--config`      | No       | Path to YAML config (default: `yt_upload_config.yaml`) |

### First run

On the first run, the script will:
1. Open your browser to Google's sign-in page
2. Ask you to select the YouTube account to upload to
3. Request permission to manage your YouTube account
4. Save the OAuth token to `token.json` for future runs

Subsequent runs will reuse the cached token automatically.

### Scheduling

- The script finds the latest Short in the configured playlist and schedules the new video for **the next day at 7:30 PM (local timezone)**
- If no Shorts are found in the playlist, it will prompt you to enter a schedule time
- If the computed slot is in the past, it auto-adjusts forward to the next future **7:30 PM** slot
- Videos are uploaded as **private** with a `publishAt` time — YouTube auto-publishes them at the scheduled time

## Files

| File                     | Description                              |
|--------------------------|------------------------------------------|
| `yt_uploader.py`         | Main CLI script                          |
| `yt_upload_config.yaml`  | Default config (tags, hashtags, playlist) |
| `requirements.txt`       | Python dependencies                      |
| `client_secrets.json`    | OAuth credentials (not committed to git) |
| `token.json`             | Cached OAuth token (not committed to git)|

> **Note:** `client_secrets.json` and `token.json` are in `.gitignore` and should never be committed.
