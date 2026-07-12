"""Tests for the synced lyrics module"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src"))

from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.synced_lyrics import (
    align_synced_lyrics_to_transcription,
    parse_synced_lyrics,
    synced_lyrics_to_plain,
)


def _transcription(timed_words: list[tuple[str, float, float]]) -> list[TranscribedData]:
    # Transcribed words carry a trailing space as word-end marker
    return [
        TranscribedData(word=f"{word} ", start=start, end=end)
        for word, start, end in timed_words
    ]


class TestParseSyncedLyrics(unittest.TestCase):
    def test_parses_timestamps_and_words(self):
        lyrics = "[00:01.50] Hello world\n[01:02.25] Sing along"
        lines = parse_synced_lyrics(lyrics)

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0].start, 1.5)
        self.assertEqual(lines[0].words, ["Hello", "world"])
        self.assertEqual(lines[1].start, 62.25)
        self.assertEqual(lines[1].words, ["Sing", "along"])

    def test_skips_metadata_and_empty_lines(self):
        lyrics = "[ar:Queen]\n[00:01.00] Hello\n[00:05.00]\nno timestamp here"
        lines = parse_synced_lyrics(lyrics)

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].words, ["Hello"])

    def test_multiple_timestamps_and_sorting(self):
        lyrics = "[00:10.00][00:02.00] Repeated line\n[00:05.00] Middle"
        lines = parse_synced_lyrics(lyrics)

        self.assertEqual([line.start for line in lines], [2.0, 5.0, 10.0])
        self.assertEqual(lines[0].words, ["Repeated", "line"])

    def test_strips_enhanced_word_timestamps(self):
        lyrics = "[00:01.00] <00:01.00>Hello <00:01.50>world"
        lines = parse_synced_lyrics(lyrics)

        self.assertEqual(lines[0].words, ["Hello", "world"])

    def test_synced_lyrics_to_plain(self):
        lyrics = "[00:03.00] Second line\n[00:01.00] First line"
        self.assertEqual(synced_lyrics_to_plain(lyrics), "First line\nSecond line")


class TestAlignSyncedLyricsToTranscription(unittest.TestCase):
    def test_line_ends_are_marked(self):
        transcribed = _transcription(
            [("hello", 1.0, 1.5), ("world", 1.6, 2.0), ("sing", 5.0, 5.4), ("along", 5.5, 6.0)]
        )
        synced = "[00:01.00] Hello world\n[00:05.00] Sing along"

        new_data = align_synced_lyrics_to_transcription(transcribed, synced)

        self.assertIsNotNone(new_data)
        self.assertEqual([data.is_line_end for data in new_data], [False, True, False, False])

    def test_misheard_word_is_corrected(self):
        transcribed = _transcription([("hello", 1.0, 1.5), ("word", 1.6, 2.0)])
        synced = "[00:01.00] Hello world"

        new_data = align_synced_lyrics_to_transcription(transcribed, synced)

        self.assertEqual([data.word for data in new_data], ["Hello ", "world "])

    def test_missing_word_is_inserted_with_timing(self):
        transcribed = _transcription(
            [("hello", 1.0, 1.5), ("world", 1.6, 2.0), ("along", 6.0, 6.5)]
        )
        synced = "[00:01.00] Hello world\n[00:05.00] Sing along"

        new_data = align_synced_lyrics_to_transcription(transcribed, synced)

        self.assertEqual(
            [data.word for data in new_data], ["Hello ", "world ", "Sing ", "along "]
        )
        inserted = new_data[2]
        # The inserted word starts at the LRC line start and ends before the next word
        self.assertGreaterEqual(inserted.start, 5.0)
        self.assertLessEqual(inserted.end, 6.0)
        self.assertTrue(inserted.is_word_end)
        # The line break moved to the end of the first line
        self.assertTrue(new_data[1].is_line_end)

    def test_missing_line_is_inserted_at_line_start(self):
        transcribed = _transcription([("hello", 1.0, 1.5), ("world", 1.6, 2.0)])
        synced = "[00:01.00] Hello world\n[00:05.00] Sing along"

        new_data = align_synced_lyrics_to_transcription(transcribed, synced)

        self.assertEqual(len(new_data), 4)
        self.assertEqual([data.word for data in new_data[2:]], ["Sing ", "along "])
        self.assertGreaterEqual(new_data[2].start, 5.0)
        self.assertGreater(new_data[3].start, new_data[2].start)

    def test_missing_word_without_gap_is_skipped(self):
        transcribed = _transcription(
            [("hello", 1.0, 1.5), ("world", 1.6, 2.0), ("along", 2.05, 2.5)]
        )
        # "Sing" has no room between "world" and "along"
        synced = "[00:01.00] Hello world Sing along"

        new_data = align_synced_lyrics_to_transcription(transcribed, synced)

        self.assertEqual(
            [data.word for data in new_data], ["Hello ", "world ", "along "]
        )

    def test_extra_transcribed_words_keep_their_timing_and_line(self):
        transcribed = _transcription(
            [("hello", 1.0, 1.5), ("uh", 1.6, 1.8), ("world", 1.9, 2.3), ("sing", 5.0, 5.5)]
        )
        synced = "[00:01.00] Hello world\n[00:05.00] Sing"

        new_data = align_synced_lyrics_to_transcription(transcribed, synced)

        self.assertEqual(
            [data.word for data in new_data], ["Hello ", "uh ", "world ", "Sing "]
        )
        self.assertEqual([data.is_line_end for data in new_data], [False, False, True, False])

    def test_completely_different_lyrics_are_rejected(self):
        transcribed = _transcription(
            [("totally", 1.0, 1.5), ("different", 1.6, 2.0), ("words", 2.1, 2.5), ("here", 2.6, 3.0)]
        )
        synced = "[00:01.00] some other song lyrics entirely unrelated"

        self.assertIsNone(align_synced_lyrics_to_transcription(transcribed, synced))

    def test_unparsable_lyrics_are_rejected(self):
        transcribed = _transcription([("hello", 1.0, 1.5)])
        self.assertIsNone(align_synced_lyrics_to_transcription(transcribed, "no timestamps"))


if __name__ == "__main__":
    unittest.main()
