"""Tests for the lyrics aligner module"""

import os
import sys
import unittest
from dataclasses import dataclass

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src"))

from modules.Speech_Recognition.lyrics_aligner import (
    align_lyrics_to_transcription,
    match_word_sequences,
    normalize_word,
    split_lyrics_into_words,
)


@dataclass
class _Word:
    word: str


def _transcription(words: list[str]) -> list[_Word]:
    # Transcribed words carry a trailing space as word-end marker
    return [_Word(word=f"{word} ") for word in words]


class TestNormalizeWord(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_word("Hello, "), "hello")
        self.assertEqual(normalize_word("don’t"), "don't")
        self.assertEqual(normalize_word('"World!"'), "world")
        self.assertEqual(normalize_word("  Foo  "), "foo")


class TestSplitLyricsIntoWords(unittest.TestCase):
    def test_splits_lines_and_skips_section_markers(self):
        lyrics = "Hello world\n\n[Chorus]\nSing along"
        self.assertEqual(
            split_lyrics_into_words(lyrics), ["Hello", "world", "Sing", "along"]
        )

    def test_strips_enclosing_quotes(self):
        self.assertEqual(split_lyrics_into_words('"Hello" (world)'), ["Hello", "world"])


class TestMatchWordSequences(unittest.TestCase):
    def test_backends_return_the_same_format(self):
        transcribed = ["hello", "word", "how", "are", "you"]
        reference = ["hello", "world", "how", "are", "you"]

        for use_levenshtein in (False, True):
            ratio, opcodes = match_word_sequences(
                transcribed, reference, use_levenshtein
            )
            self.assertEqual(ratio, 0.8)
            self.assertIn(
                ("replace", 1, 2, 1, 2), [tuple(opcode) for opcode in opcodes]
            )


class TestAlignLyricsToTranscription(unittest.TestCase):
    use_levenshtein = False

    def _align(self, transcribed, reference):
        return align_lyrics_to_transcription(
            transcribed, reference, use_levenshtein=self.use_levenshtein
        )

    def test_misheard_word_is_corrected(self):
        transcribed = _transcription(["hello", "word", "how", "are", "you"])
        reference = "Hello world\nhow are you"

        corrections = self._align(transcribed, reference)

        self.assertEqual(corrections, 1)
        self.assertEqual(transcribed[1].word, "world ")

    def test_trailing_space_is_preserved(self):
        transcribed = _transcription(["hello", "word"])
        transcribed[0].word = "hello"  # no word-end marker
        self._align(transcribed, "hello world")

        self.assertEqual(transcribed[0].word, "hello")
        self.assertEqual(transcribed[1].word, "world ")

    def test_extra_transcribed_words_keep_their_timing(self):
        transcribed = _transcription(["hello", "uh", "world"])
        corrections = self._align(transcribed, "hello world")

        self.assertEqual(corrections, 0)
        self.assertEqual(
            [data.word for data in transcribed], ["hello ", "uh ", "world "]
        )

    def test_completely_different_lyrics_are_rejected(self):
        transcribed = _transcription(["totally", "different", "words", "here"])
        corrections = self._align(
            transcribed, "some other song lyrics entirely unrelated"
        )

        self.assertEqual(corrections, -1)
        self.assertEqual(transcribed[0].word, "totally ")

    def test_official_casing_is_applied(self):
        transcribed = _transcription(["i", "love", "paris"])
        corrections = self._align(transcribed, "I love Paris")

        # Casing fixes are applied but not counted as corrections
        self.assertEqual(corrections, 0)
        self.assertEqual(
            [data.word for data in transcribed], ["I ", "love ", "Paris "]
        )

    def test_empty_reference_is_rejected(self):
        transcribed = _transcription(["hello"])
        self.assertEqual(self._align(transcribed, "[Chorus]\n"), -1)


class TestAlignLyricsToTranscriptionWithLevenshtein(TestAlignLyricsToTranscription):
    """Rerun all alignment scenarios with the Levenshtein matcher"""

    use_levenshtein = True


if __name__ == "__main__":
    unittest.main()
