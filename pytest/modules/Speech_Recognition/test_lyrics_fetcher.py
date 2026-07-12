"""Tests for the lyrics fetcher module"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src"))

import requests

from modules.Speech_Recognition import lyrics_fetcher
from modules.Speech_Recognition.lyrics_fetcher import fetch_lyrics_from_web


def _response(status_code=200, json_data=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_data
    return response


class TestFetchLyricsFromWeb(unittest.TestCase):
    @patch.object(lyrics_fetcher.requests, "get")
    def test_exact_match_is_used(self, mock_get):
        mock_get.return_value = _response(
            json_data={"plainLyrics": "Is this the real life", "instrumental": False}
        )

        lyrics = fetch_lyrics_from_web("Queen", "Bohemian Rhapsody")

        self.assertEqual(lyrics.plain_lyrics, "Is this the real life")
        self.assertIsNone(lyrics.synced_lyrics)
        endpoint = mock_get.call_args[0][0]
        self.assertTrue(endpoint.endswith("/get"))

    @patch.object(lyrics_fetcher.requests, "get")
    def test_synced_lyrics_are_returned_when_available(self, mock_get):
        mock_get.return_value = _response(
            json_data={
                "plainLyrics": "Is this the real life",
                "syncedLyrics": "[00:01.00] Is this the real life",
                "instrumental": False,
            }
        )

        lyrics = fetch_lyrics_from_web("Queen", "Bohemian Rhapsody")

        self.assertEqual(lyrics.plain_lyrics, "Is this the real life")
        self.assertEqual(lyrics.synced_lyrics, "[00:01.00] Is this the real life")

    @patch.object(lyrics_fetcher.requests, "get")
    def test_plain_lyrics_are_derived_from_synced_lyrics(self, mock_get):
        mock_get.return_value = _response(
            json_data={
                "plainLyrics": None,
                "syncedLyrics": "[00:01.00] Is this the real life\n[00:03.00] Is this just fantasy",
                "instrumental": False,
            }
        )

        lyrics = fetch_lyrics_from_web("Queen", "Bohemian Rhapsody")

        self.assertEqual(
            lyrics.plain_lyrics, "Is this the real life\nIs this just fantasy"
        )
        self.assertIsNotNone(lyrics.synced_lyrics)

    @patch.object(lyrics_fetcher.requests, "get")
    def test_search_fallback_picks_best_match(self, mock_get):
        search_results = [
            {
                "artistName": "Somebody Else",
                "trackName": "Other Song",
                "plainLyrics": "wrong lyrics",
            },
            {
                "artistName": "Queen",
                "trackName": "Bohemian Rhapsody",
                "plainLyrics": "Is this the real life",
            },
        ]
        mock_get.side_effect = [
            _response(status_code=404),
            _response(json_data=search_results),
        ]

        lyrics = fetch_lyrics_from_web("Queen", "Bohemian Rhapsody")

        self.assertEqual(lyrics.plain_lyrics, "Is this the real life")

    @patch.object(lyrics_fetcher.requests, "get")
    def test_dissimilar_search_results_are_rejected(self, mock_get):
        search_results = [
            {
                "artistName": "Somebody Else",
                "trackName": "A Completely Different Song",
                "plainLyrics": "wrong lyrics",
            }
        ]
        mock_get.side_effect = [
            _response(status_code=404),
            _response(json_data=search_results),
        ]

        self.assertIsNone(fetch_lyrics_from_web("Queen", "Bohemian Rhapsody"))

    @patch.object(lyrics_fetcher.requests, "get")
    def test_instrumental_tracks_return_none(self, mock_get):
        mock_get.side_effect = [
            _response(json_data={"instrumental": True, "plainLyrics": None}),
            _response(json_data=[]),
        ]

        self.assertIsNone(fetch_lyrics_from_web("Artist", "Instrumental Track"))

    @patch.object(lyrics_fetcher.requests, "get")
    def test_network_error_returns_none(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("no network")

        self.assertIsNone(fetch_lyrics_from_web("Queen", "Bohemian Rhapsody"))

    @patch.object(lyrics_fetcher.requests, "get")
    def test_unknown_artist_skips_exact_match_and_searches_by_title(self, mock_get):
        mock_get.return_value = _response(
            json_data=[
                {
                    "artistName": "Queen",
                    "trackName": "Bohemian Rhapsody",
                    "plainLyrics": "Is this the real life",
                }
            ]
        )

        lyrics = fetch_lyrics_from_web("Unknown Artist", "Bohemian Rhapsody")

        self.assertEqual(lyrics.plain_lyrics, "Is this the real life")
        # Only the search endpoint may be called for unknown artists
        self.assertEqual(mock_get.call_count, 1)
        self.assertTrue(mock_get.call_args[0][0].endswith("/search"))

    def test_empty_title_returns_none(self):
        self.assertIsNone(fetch_lyrics_from_web("Queen", ""))


if __name__ == "__main__":
    unittest.main()
