"""Fetch official song lyrics from the web (lrclib.net)"""

from dataclasses import dataclass
from difflib import SequenceMatcher

import requests

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Speech_Recognition.synced_lyrics import synced_lyrics_to_plain

LRCLIB_API_BASE_URL = "https://lrclib.net/api"
REQUEST_TIMEOUT_SECONDS = 10
# Minimum similarity between the requested and the found artist/title
# before a search result is accepted as the correct song
MIN_SEARCH_RESULT_SIMILARITY = 0.6

_REQUEST_HEADERS = {
    "User-Agent": "UltraSinger (https://github.com/rakuri255/UltraSinger)"
}


@dataclass
class FetchedLyrics:
    """Official lyrics found online"""

    plain_lyrics: str
    # LRC formatted lyrics with line timestamps, when available
    synced_lyrics: str | None = None


def fetch_lyrics_from_web(artist: str, title: str) -> FetchedLyrics | None:
    """Fetch official lyrics for a song from lrclib.net.

    Returns the plain lyrics and, when available, time-synced lyrics in LRC
    format. Returns None if nothing suitable was found.
    """
    if not title:
        return None

    print(
        f"{ULTRASINGER_HEAD} Fetching official lyrics from {blue_highlighted('lrclib.net')} "
        f"for {blue_highlighted(f'{artist} - {title}')}"
    )

    fetched_lyrics = __fetch_exact_match(artist, title)
    if fetched_lyrics is None:
        fetched_lyrics = __fetch_from_search(artist, title)

    if fetched_lyrics is None:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('No official lyrics found online. Keeping transcribed lyrics.')}"
        )
    elif fetched_lyrics.synced_lyrics is not None:
        print(f"{ULTRASINGER_HEAD} Found official lyrics online (with synced line timings)")
    else:
        print(f"{ULTRASINGER_HEAD} Found official lyrics online")

    return fetched_lyrics


def __request_json(endpoint: str, params: dict) -> dict | list | None:
    try:
        response = requests.get(
            f"{LRCLIB_API_BASE_URL}/{endpoint}",
            params=params,
            headers=_REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return None
        return response.json()
    except (requests.RequestException, ValueError):
        return None


def __fetch_exact_match(artist: str, title: str) -> FetchedLyrics | None:
    if not artist or artist == "Unknown Artist":
        return None

    result = __request_json(
        "get", {"artist_name": artist, "track_name": title}
    )
    if isinstance(result, dict):
        return __get_lyrics(result)
    return None


def __fetch_from_search(artist: str, title: str) -> FetchedLyrics | None:
    query = f"{artist} {title}" if artist and artist != "Unknown Artist" else title
    results = __request_json("search", {"q": query})
    if not isinstance(results, list):
        return None

    best_lyrics = None
    best_similarity = MIN_SEARCH_RESULT_SIMILARITY
    for result in results:
        if not isinstance(result, dict):
            continue
        lyrics = __get_lyrics(result)
        if lyrics is None:
            continue
        similarity = __song_similarity(
            artist, title, result.get("artistName", ""), result.get("trackName", "")
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_lyrics = lyrics
    return best_lyrics


def __get_lyrics(result: dict) -> FetchedLyrics | None:
    if result.get("instrumental"):
        return None

    plain_lyrics = result.get("plainLyrics")
    if not isinstance(plain_lyrics, str) or not plain_lyrics.strip():
        plain_lyrics = None

    synced_lyrics = result.get("syncedLyrics")
    if not isinstance(synced_lyrics, str) or not synced_lyrics.strip():
        synced_lyrics = None

    if plain_lyrics is None and synced_lyrics is not None:
        plain_lyrics = synced_lyrics_to_plain(synced_lyrics)

    if not plain_lyrics:
        return None
    return FetchedLyrics(plain_lyrics=plain_lyrics, synced_lyrics=synced_lyrics)


def __song_similarity(artist: str, title: str, found_artist: str, found_title: str) -> float:
    title_similarity = SequenceMatcher(
        None, title.lower(), found_title.lower()
    ).ratio()
    if not artist or artist == "Unknown Artist":
        return title_similarity
    artist_similarity = SequenceMatcher(
        None, artist.lower(), found_artist.lower()
    ).ratio()
    return (title_similarity + artist_similarity) / 2
