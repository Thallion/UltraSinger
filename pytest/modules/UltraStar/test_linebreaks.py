"""Tests for linebreaks.py"""

import unittest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Ultrastar.linebreaks import compute_linebreaks, MAX_LINE_CHARS


def notes(words: list[str], length: float = 0.3, rest: float = 0.05, rests: dict[int, float] = None):
    """Notes in a row; ``rests`` gives the rest after single notes"""
    segments = []
    time = 0.0
    for i, word in enumerate(words):
        segments.append(MidiSegment("C4", time, time + length, word))
        time += length + (rests or {}).get(i, rest)
    return segments


class ComputeLinebreaksTest(unittest.TestCase):
    def test_a_rest_after_a_word_ends_the_line(self):
        segments = notes(["one ", "two ", "three ", "four ", "five ", "six "], rests={2: 0.5})
        self.assertEqual(compute_linebreaks(segments), {2})

    def test_no_break_inside_a_word(self):
        segments = notes(["one ", "two ", "syl", "la", "ble ", "four ", "five "], rests={2: 0.5})
        self.assertEqual(compute_linebreaks(segments), set())

    def test_a_long_line_is_split_at_a_phrase_end(self):
        words = ["word "] * 12
        segments = notes(words)
        phrase_end = segments[4].end
        breaks = compute_linebreaks(segments, [phrase_end])
        self.assertIn(4, breaks)
        self.assert_lines_fit(segments, breaks)

    def test_a_long_line_without_hints_is_split_near_the_middle(self):
        segments = notes(["word "] * 12)
        breaks = compute_linebreaks(segments)
        self.assertEqual(breaks, {5})

    def test_a_short_line_that_flashes_by_joins_its_neighbour(self):
        segments = notes(["one ", "two ", "three ", "oh ", "four ", "five ", "six "], rests={2: 0.4, 3: 0.5})
        # "oh" alone is on screen for 0.3 s; the rest before it is the shorter one
        self.assertEqual(compute_linebreaks(segments), {3})

    def test_lines_of_reference_lyrics_end_the_line(self):
        segments = notes(["one ", "two ", "three ", "four ", "five ", "six ", "seven "])
        line_end = segments[3].end
        self.assertEqual(compute_linebreaks(segments, [line_end]), set())
        self.assertEqual(compute_linebreaks(segments, [line_end], has_lyric_lines=True), {3})

    def test_inside_a_line_of_reference_lyrics_only_a_long_rest_ends_the_line(self):
        words = ["one ", "two ", "three ", "four ", "five ", "six "]
        short_rest = notes(words, rests={2: 0.5})
        long_rest = notes(words, rests={2: 1.5})
        line_end = short_rest[5].end
        self.assertEqual(compute_linebreaks(short_rest, [line_end], has_lyric_lines=True), set())
        self.assertEqual(compute_linebreaks(long_rest, [long_rest[5].end], has_lyric_lines=True), {2})

    def test_few_notes(self):
        self.assertEqual(compute_linebreaks([]), set())
        self.assertEqual(compute_linebreaks(notes(["one "])), set())

    def assert_lines_fit(self, segments, breaks):
        start = 0
        for end in sorted(breaks) + [len(segments) - 1]:
            text = "".join(segment.word for segment in segments[start:end + 1]).strip()
            self.assertLessEqual(len(text), MAX_LINE_CHARS)
            start = end + 1


if __name__ == "__main__":
    unittest.main()
