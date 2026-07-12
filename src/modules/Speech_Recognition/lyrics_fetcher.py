"""Fetch official song lyrics from the web (lrclib.net)"""

from difflib import SequenceMatcher

import requests

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)

LRCLIB_API_BASE_URL = "https://lrclib.net/api"
REQUEST_TIMEOUT_SECONDS = 10
# Minimum similarity between the requested and the found artist/title
# before a search result is accepted as the correct song
MIN_SEARCH_RESULT_SIMILARITY = 0.6

_REQUEST_HEADERS = {
    "User-Agent": "UltraSinger (https://github.com/rakuri255/UltraSinger)"
}


def fetch_lyrics_from_web(artist: str, title: str) -> str | None:
    """Fetch official plain lyrics for a song from lrclib.net.

    Returns the lyrics text or None if nothing suitable was found.
    """
    if not title:
        return None

    print(
        f"{ULTRASINGER_HEAD} Fetching official lyrics from {blue_highlighted('lrclib.net')} "
        f"for {blue_highlighted(f'{artist} - {title}')}"
    )

    lyrics = __fetch_exact_match(artist, title)
    if lyrics is None:
        lyrics = __fetch_from_search(artist, title)

    if lyrics is None:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('No official lyrics found online. Keeping transcribed lyrics.')}"
        )
    else:
        print(f"{ULTRASINGER_HEAD} Found official lyrics online")

    return lyrics


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


def __fetch_exact_match(artist: str, title: str) -> str | None:
    if not artist or artist == "Unknown Artist":
        return None

    result = __request_json(
        "get", {"artist_name": artist, "track_name": title}
    )
    if isinstance(result, dict):
        return __get_plain_lyrics(result)
    return None


def __fetch_from_search(artist: str, title: str) -> str | None:
    query = f"{artist} {title}" if artist and artist != "Unknown Artist" else title
    results = __request_json("search", {"q": query})
    if not isinstance(results, list):
        return None

    best_lyrics = None
    best_similarity = MIN_SEARCH_RESULT_SIMILARITY
    for result in results:
        if not isinstance(result, dict):
            continue
        lyrics = __get_plain_lyrics(result)
        if lyrics is None:
            continue
        similarity = __song_similarity(
            artist, title, result.get("artistName", ""), result.get("trackName", "")
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_lyrics = lyrics
    return best_lyrics


def __get_plain_lyrics(result: dict) -> str | None:
    if result.get("instrumental"):
        return None
    lyrics = result.get("plainLyrics")
    if isinstance(lyrics, str) and lyrics.strip():
        return lyrics
    return None


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
