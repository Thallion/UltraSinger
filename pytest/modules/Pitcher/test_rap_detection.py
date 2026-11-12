"""Tests for rap_detection.py"""

import os
import tempfile
import unittest

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.rap_detection import mark_rap_notes
from src.modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from src.modules.Ultrastar.ultrastar_writer import create_ultrastar_txt

FRAME = 0.016
LINE_SECONDS = 2.0
SYLLABLE_SECONDS = 0.2


def pitched(rapped_lines: set[int], line_count: int) -> PitchedData:
    """A held tone in sung lines; a pitch that wanders by 0.4 semitones a frame in rapped lines"""
    times = [i * FRAME for i in range(int(line_count * LINE_SECONDS / FRAME))]
    frequencies = []
    for i, time in enumerate(times):
        semitones = 0.0
        if int(time / LINE_SECONDS) in rapped_lines:
            step = i % 16
            semitones = 0.4 * (step if step < 8 else 16 - step)
        frequencies.append(220.0 * 2 ** (semitones / 12))
    return PitchedData(times, frequencies, [0.9] * len(times))


def lines_of_notes(line_count: int) -> tuple[list[MidiSegment], set[int]]:
    """Lines of eight syllables of 0.2 s; the last 0.4 s of a line are a rest"""
    segments = []
    linebreaks = set()
    for line in range(line_count):
        for k in range(8):
            start = line * LINE_SECONDS + k * SYLLABLE_SECONDS
            segments.append(MidiSegment("A3", start, start + SYLLABLE_SECONDS, "la "))
        linebreaks.add(len(segments) - 1)
    linebreaks.discard(len(segments) - 1)
    return segments, linebreaks


def rap_lines(segments: list[MidiSegment]) -> list[bool]:
    return [segments[i].rap for i in range(0, len(segments), 8)]


class MarkRapNotesTest(unittest.TestCase):
    def test_sung_lines_stay_sung(self):
        segments, linebreaks = lines_of_notes(4)
        mark_rap_notes(segments, pitched(set(), 4), linebreaks)
        self.assertEqual(rap_lines(segments), [False] * 4)

    def test_a_rapped_passage_is_marked(self):
        segments, linebreaks = lines_of_notes(5)
        mark_rap_notes(segments, pitched({1, 2, 3}, 5), linebreaks)
        self.assertEqual(rap_lines(segments), [False, True, True, True, False])

    def test_a_single_rapped_line_stays_sung(self):
        segments, linebreaks = lines_of_notes(5)
        mark_rap_notes(segments, pitched({2}, 5), linebreaks)
        self.assertEqual(rap_lines(segments), [False] * 5)

    def test_a_single_line_between_rapped_lines_goes_with_them(self):
        segments, linebreaks = lines_of_notes(5)
        mark_rap_notes(segments, pitched({1, 3}, 5), linebreaks)
        self.assertEqual(rap_lines(segments), [False, True, True, True, False])

    def test_slow_lines_on_a_wandering_pitch_stay_sung(self):
        # every line has three syllables of half a second
        slow = []
        slow_breaks = set()
        for line in range(4):
            for k in range(3):
                start = line * LINE_SECONDS + k * 0.5
                slow.append(MidiSegment("A3", start, start + 0.5, "la "))
            slow_breaks.add(len(slow) - 1)
        slow_breaks.discard(len(slow) - 1)
        mark_rap_notes(slow, pitched({0, 1, 2, 3}, 4), slow_breaks)
        self.assertFalse(any(segment.rap for segment in slow))

    def test_rap_notes_are_written_as_rap(self):
        segments, linebreaks = lines_of_notes(4)
        mark_rap_notes(segments, pitched({1, 2}, 4), linebreaks)
        txt = UltrastarTxtValue()
        txt.version = "1.2.0"
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "song.txt")
            create_ultrastar_txt(segments, path, txt, 120.0)
            with open(path, encoding="utf-8") as file:
                notes = [line for line in file.read().splitlines() if line[:2] in (": ", "R ")]
        self.assertEqual(len(notes), 32)
        self.assertEqual([line[0] for line in notes], [":"] * 8 + ["R"] * 16 + [":"] * 8)


if __name__ == "__main__":
    unittest.main()
