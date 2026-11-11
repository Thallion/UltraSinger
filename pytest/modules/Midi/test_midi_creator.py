"""Tests for midi_creator.py"""

import unittest

import librosa

from src.modules.Midi.midi_creator import (
    create_midi_note_from_pitched_data,
    create_midi_segments_from_transcribed_data,
    estimate_tuning,
    find_nearest_index,
    pitch_to_note,
    simplify_melismas,
    weighted_median_note,
    weighted_median_pitch,
)
from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Pitcher.pitched_data import PitchedData


def hz(midi: float) -> float:
    return float(librosa.midi_to_hz(midi))


class WeightedMedianTest(unittest.TestCase):
    def test_held_tone_wins_over_a_slide(self):
        # three frames of a slide up, then the held tone
        frequencies = [hz(57), hz(58), hz(59), hz(60), hz(60.1), hz(59.9), hz(60)]
        self.assertEqual(weighted_median_note(frequencies, [0.9] * 7), "C4")

    def test_tone_on_the_border_of_two_notes(self):
        # wobbles around 60.5: counting note names would let two outlier frames decide
        frequencies = [hz(60.45), hz(60.55), hz(60.4), hz(60.6), hz(60.45), hz(64), hz(64)]
        self.assertAlmostEqual(weighted_median_pitch(frequencies, [0.9] * 7), 60.55, places=2)

    def test_frames_without_confidence_do_not_count(self):
        frequencies = [hz(72), hz(72), hz(72), hz(60), hz(60)]
        confidences = [0.1, 0.1, 0.1, 0.9, 0.9]
        self.assertEqual(weighted_median_note(frequencies, confidences), "C4")

    def test_no_voiced_frame(self):
        self.assertIsNone(weighted_median_note([hz(60), hz(62)], [0.1, 0.2]))
        self.assertIsNone(weighted_median_pitch([], []))


class PitchToNoteTest(unittest.TestCase):
    def test_rounds_to_the_nearest_note(self):
        self.assertEqual(pitch_to_note(60.4), "C4")
        self.assertEqual(pitch_to_note(60.6), "C#4")

    def test_prefers_the_note_of_the_key_when_close(self):
        c_major = {"C", "D", "E", "F", "G", "A", "B"}
        self.assertEqual(pitch_to_note(60.55, c_major), "C4")
        # clearly the note outside the key
        self.assertEqual(pitch_to_note(60.9, c_major), "C#4")

    def test_sharps_are_written_in_ascii(self):
        self.assertEqual(pitch_to_note(61.0), "C#4")


class EstimateTuningTest(unittest.TestCase):
    def test_offset_of_a_detuned_recording(self):
        frequencies = [hz(60.3), hz(62.3), hz(64.3), hz(65.3)] * 50
        pitched_data = PitchedData([i * 0.016 for i in range(200)], frequencies, [0.95] * 200)
        self.assertAlmostEqual(estimate_tuning(pitched_data), 0.3, places=2)

    def test_too_little_singing(self):
        pitched_data = PitchedData([0.0, 0.016], [hz(60.3), hz(60.3)], [0.95, 0.95])
        self.assertEqual(estimate_tuning(pitched_data), 0.0)


class CreateMidiNoteTest(unittest.TestCase):
    def setUp(self):
        times = [i * 0.016 for i in range(100)]
        self.pitched_data = PitchedData(times, [hz(60)] * 50 + [hz(67)] * 50, [0.9] * 50 + [0.1] * 50)

    def test_voiced_note(self):
        segment = create_midi_note_from_pitched_data(0.1, 0.5, self.pitched_data, "la ")
        self.assertEqual((segment.note, segment.voiced, segment.word), ("C4", True, "la "))

    def test_unvoiced_note_is_marked(self):
        segment = create_midi_note_from_pitched_data(1.0, 1.4, self.pitched_data, "la ")
        self.assertFalse(segment.voiced)

    def test_note_after_the_end_of_the_pitch_data(self):
        segment = create_midi_note_from_pitched_data(5.0, 5.0, self.pitched_data, "la ")
        self.assertFalse(segment.voiced)

    def test_empty_transcription_gives_no_notes(self):
        self.assertEqual(create_midi_segments_from_transcribed_data([], self.pitched_data), [])


class SimplifyMelismasTest(unittest.TestCase):
    FRAME = 0.016

    def pitched(self, tones: list[tuple[float, float, float]]) -> PitchedData:
        """Pitch data of sung tones given as (start, end, midi pitch)"""
        times = [i * self.FRAME for i in range(int(4 / self.FRAME))]
        frequencies, confidence = [], []
        for time in times:
            tone = next((midi for start, end, midi in tones if start <= time < end), None)
            frequencies.append(hz(tone) if tone is not None else 100.0)
            confidence.append(0.9 if tone is not None else 0.0)
        return PitchedData(times, frequencies, confidence)

    def test_a_short_ornament_joins_the_note_before(self):
        pitched = self.pitched([(1.0, 1.6, 60), (1.6, 1.7, 62), (1.7, 2.0, 60)])
        segments = [MidiSegment("C4", 1.0, 1.6, "you", True, 60.0), MidiSegment("D4", 1.6, 1.7, "~", True, 62.0),
                    MidiSegment("C4", 1.7, 2.0, "~ ", True, 60.0)]

        result = simplify_melismas(segments, pitched)

        self.assertEqual([(s.word, s.note, s.start, s.end) for s in result], [("you ", "C4", 1.0, 2.0)])

    def test_a_wobble_around_the_border_of_two_notes_is_one_note(self):
        pitched = self.pitched([(1.0, 1.5, 60.4), (1.5, 2.0, 60.6)])
        segments = [MidiSegment("C4", 1.0, 1.5, "you", True, 60.4), MidiSegment("C#4", 1.5, 2.0, "~ ", True, 60.6)]

        result = simplify_melismas(segments, pitched)

        self.assertEqual([(s.word, s.start, s.end) for s in result], [("you ", 1.0, 2.0)])

    def test_two_held_tones_stay_two_notes(self):
        pitched = self.pitched([(1.0, 1.5, 60), (1.5, 2.0, 64)])
        segments = [MidiSegment("C4", 1.0, 1.5, "you", True, 60.0), MidiSegment("E4", 1.5, 2.0, "~ ", True, 64.0)]

        result = simplify_melismas(segments, pitched)

        self.assertEqual([s.note for s in result], ["C4", "E4"])

    def test_a_short_first_note_joins_the_tone_it_slides_into(self):
        pitched = self.pitched([(1.0, 1.1, 57), (1.1, 2.0, 60)])
        segments = [MidiSegment("A3", 1.0, 1.1, "you", True, 57.0), MidiSegment("C4", 1.1, 2.0, "~ ", True, 60.0)]

        result = simplify_melismas(segments, pitched)

        self.assertEqual([(s.word, s.note, s.start) for s in result], [("you ", "C4", 1.0)])

    def test_short_syllables_stay_notes_of_their_own(self):
        pitched = self.pitched([(1.0, 1.1, 60), (1.1, 1.2, 62)])
        segments = [MidiSegment("C4", 1.0, 1.1, "bo", True, 60.0), MidiSegment("D4", 1.1, 1.2, "dy ", True, 62.0)]

        result = simplify_melismas(segments, pitched)

        self.assertEqual([s.word for s in result], ["bo", "dy "])

    def test_a_tilde_after_a_rest_is_left_alone(self):
        pitched = self.pitched([(1.0, 1.5, 60), (2.0, 2.1, 60)])
        segments = [MidiSegment("C4", 1.0, 1.5, "you", True, 60.0), MidiSegment("C4", 2.0, 2.1, "~ ", True, 60.0)]

        self.assertEqual(len(simplify_melismas(segments, pitched)), 2)


class FindNearestIndexTest(unittest.TestCase):
    def test_nearest(self):
        times = [0.0, 1.0, 2.0, 3.0]
        self.assertEqual(find_nearest_index(times, 1.4), 1)
        self.assertEqual(find_nearest_index(times, 1.6), 2)
        self.assertEqual(find_nearest_index(times, -1.0), 0)

    def test_value_after_the_end_stays_inside(self):
        self.assertEqual(find_nearest_index([0.0, 1.0], 9.0), 1)


if __name__ == "__main__":
    unittest.main()
