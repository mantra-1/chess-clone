"""Chess.com API client for downloading game archives."""

import json
import time
from pathlib import Path

import requests

BASE_URL = "https://api.chess.com/pub"
HEADERS = {
    "User-Agent": "JebBot/1.0 (Chess analysis project; https://github.com/jebbot)"
}
REQUEST_DELAY = 1.0  # seconds between requests


def get_game_archives(username: str) -> list[str]:
    """Fetch list of monthly archive URLs for a player.

    Args:
        username: Chess.com username

    Returns:
        List of monthly archive URLs

    Raises:
        requests.HTTPError: If the API request fails
    """
    url = f"{BASE_URL}/player/{username}/games/archives"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json().get("archives", [])


def download_monthly_archive(archive_url: str) -> dict:
    """Download a single monthly archive.

    Args:
        archive_url: Full URL to the monthly archive

    Returns:
        JSON response containing "games" list

    Raises:
        requests.HTTPError: If the API request fails
    """
    response = requests.get(archive_url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def download_all_games(username: str, output_dir: Path) -> int:
    """Download all games for a player and save to JSON files.

    Args:
        username: Chess.com username
        output_dir: Directory to save JSON files (will create raw/ subdirectory)

    Returns:
        Total number of games downloaded

    Raises:
        requests.HTTPError: If any API request fails
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching game archives for {username}...")
    archives = get_game_archives(username)
    print(f"Found {len(archives)} monthly archives")

    total_games = 0

    for i, archive_url in enumerate(archives, 1):
        # Extract year and month from URL
        # URL format: https://api.chess.com/pub/player/{username}/games/{year}/{month}
        parts = archive_url.rstrip("/").split("/")
        year, month = parts[-2], parts[-1]

        print(f"[{i}/{len(archives)}] Downloading {year}-{month}...", end=" ")

        time.sleep(REQUEST_DELAY)  # Rate limiting

        try:
            data = download_monthly_archive(archive_url)
            games = data.get("games", [])
            total_games += len(games)

            # Save to file
            filename = f"{username}_{year}_{month}.json"
            filepath = raw_dir / filename
            with open(filepath, "w") as f:
                json.dump(data, f)

            print(f"{len(games)} games")

        except requests.HTTPError as e:
            print(f"ERROR: {e}")
            raise

    print(f"Downloaded {total_games} total games")
    return total_games
