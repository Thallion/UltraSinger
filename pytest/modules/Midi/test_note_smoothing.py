"""Tests for the note smoothing module"""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../src"))

from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.note_smoothing import (
    midi_to_note,
    note_to_midi,
    smooth_midi_segments,
)


def _segments_from_notes(notes: list[str]) -> list[MidiSegment]:
    return [
        MidiSegment(note=note, start=float(i), end=float(i) + 0.5, word=f"w{i} ")
        for i, note in enumerate(notes)
    ]


class TestNoteConversion(unittest.TestCase):
    def test_note_to_midi(self):
        self.assertEqual(note_to_midi("C4"), 60)
        self.assertEqual(note_to_midi("A4"), 69)
        self.assertEqual(note_to_midi("C#4"), 61)
        self.assertEqual(note_to_midi("C♯4"), 61)
        self.assertEqual(note_to_midi("Db4"), 61)
        self.assertEqual(note_to_midi("D♭4"), 61)
        self.assertEqual(note_to_midi("C-1"), 0)
        self.assertIsNone(note_to_midi("invalid"))

    def test_midi_to_note(self):
        self.assertEqual(midi_to_note(60), "C4")
        self.assertEqual(midi_to_note(61), "C#4")
        self.assertEqual(midi_to_note(69), "A4")

    def test_roundtrip(self):
        for midi in range(21, 109):
            self.assertEqual(note_to_midi(midi_to_note(midi)), midi)


class TestSmoothMidiSegments(unittest.TestCase):
    def test_octave_error_is_corrected(self):
        # A5 is an octave error in an A4 context
        notes = ["A4", "A4", "A5", "A4", "A4"]
        segments = smooth_midi_segments(_segments_from_notes(notes))
        self.assertEqual([s.note for s in segments], ["A4", "A4", "A4", "A4", "A4"])

    def test_single_outlier_is_replaced_by_neighbours(self):
        notes = ["C4", "C4", "G#4", "C4", "C4"]
        segments = smooth_midi_segments(_segments_from_notes(notes))
        # Outlier G#4 (midi 68) between two C4 (60) becomes C4
        self.assertEqual(segments[2].note, "C4")

    def test_out_of_vocal_range_note_is_shifted_by_octaves(self):
        notes = ["C4", "C4", "C4", "C4", "C7", "C4", "C4", "C4", "C4"]
        segments = smooth_midi_segments(_segments_from_notes(notes))
        for segment in segments:
            self.assertEqual(segment.note, "C4")

    def test_melody_is_kept_intact(self):
        notes = ["C4", "D4", "E4", "F4", "G4", "A4", "G4", "E4", "C4"]
        segments = smooth_midi_segments(_segments_from_notes(notes))
        self.assertEqual([s.note for s in segments], notes)

    def test_corrected_notes_are_quantized_to_key(self):
        # D major: outlier correction result C4 is out of key and must be quantized to C#4
        notes = ["C4", "C4", "G#4", "C4", "C4"]
        allowed = {"D", "E", "F#", "G", "A", "B", "C#"}
        segments = smooth_midi_segments(_segments_from_notes(notes), allowed)
        self.assertEqual(segments[2].note, "C#4")

    def test_short_input_is_untouched(self):
        notes = ["C4", "C7"]
        segments = smooth_midi_segments(_segments_from_notes(notes))
        self.assertEqual([s.note for s in segments], notes)

    def test_unparseable_notes_are_ignored(self):
        notes = ["C4", "invalid", "C4", "C5", "C4"]
        segments = smooth_midi_segments(_segments_from_notes(notes))
        self.assertEqual(segments[1].note, "invalid")


if __name__ == "__main__":
    unittest.main()
