"""Tests for note_postprocessing.py"""

import unittest

import librosa

from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Midi.note_postprocessing import correct_octaves, fill_unvoiced_notes


def melody(midi_notes: list[int], rests: dict[int, float] = None) -> list[MidiSegment]:
    segments = []
    time = 0.0
    for i, midi in enumerate(midi_notes):
        segments.append(MidiSegment(librosa.midi_to_note(midi, unicode=False), time, time + 0.3, "la "))
        time += 0.3 + (rests or {}).get(i, 0.05)
    return segments


def midis(segments: list[MidiSegment]) -> list[int]:
    return [int(librosa.note_to_midi(segment.note)) for segment in segments]


class CorrectOctavesTest(unittest.TestCase):
    def test_a_single_note_an_octave_too_high_comes_back(self):
        result = correct_octaves(melody([60, 62, 64, 74, 64, 62, 60]))
        self.assertEqual(midis(result), [60, 62, 64, 62, 64, 62, 60])

    def test_a_short_run_an_octave_too_low_comes_back(self):
        result = correct_octaves(melody([60, 62, 64, 50, 52, 64, 62, 60]))
        self.assertEqual(midis(result), [60, 62, 64, 62, 64, 64, 62, 60])

    def test_a_long_passage_sung_an_octave_higher_stays(self):
        notes = [60, 62, 64, 62, 60] + [72, 74, 76, 74, 72, 74, 76, 74, 72] + [60, 62, 64, 62, 60]
        self.assertEqual(midis(correct_octaves(melody(notes))), notes)

    def test_a_leap_of_a_sixth_stays(self):
        notes = [60, 62, 69, 62, 60, 62, 64]
        self.assertEqual(midis(correct_octaves(melody(notes))), notes)

    def test_only_the_octave_changes(self):
        notes = [60, 62, 64, 77, 64, 50, 60, 85, 62]
        result = midis(correct_octaves(melody(notes)))
        self.assertTrue(all((a - b) % 12 == 0 for a, b in zip(notes, result)))

    def test_few_notes(self):
        self.assertEqual(midis(correct_octaves(melody([60, 80]))), [60, 80])


class FillUnvoicedNotesTest(unittest.TestCase):
    def test_takes_the_pitch_of_the_nearer_neighbour(self):
        segments = melody([60, 48, 67], rests={0: 0.05, 1: 0.5})
        segments[1].voiced = False
        self.assertEqual(midis(fill_unvoiced_notes(segments)), [60, 60, 67])

    def test_unvoiced_at_the_start_takes_the_next_one(self):
        segments = melody([48, 64, 67])
        segments[0].voiced = False
        self.assertEqual(midis(fill_unvoiced_notes(segments)), [64, 64, 67])

    def test_nothing_voiced_changes_nothing(self):
        segments = melody([48, 50])
        for segment in segments:
            segment.voiced = False
        self.assertEqual(midis(fill_unvoiced_notes(segments)), [48, 50])


if __name__ == "__main__":
    unittest.main()
