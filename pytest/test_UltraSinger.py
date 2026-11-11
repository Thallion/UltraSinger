"""Tests for UltraSinger.py"""

import unittest

from src.UltraSinger import (
    add_hyphen_to_data,
    split_syllables_into_segments,
    merge_syllable_segments,
    remove_unecessary_punctuations,
    syllable_start_times,
)
from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Speech_Recognition.TranscribedData import TranscribedData


class SyllableTimesTest(unittest.TestCase):
    def test_syllables_start_where_their_first_character_is_sung(self):
        data = TranscribedData(word="around ", start=1.0, end=2.0,
                               char_starts=[1.0, 1.1, 1.15, 1.4, 1.6, 1.8])
        self.assertEqual(syllable_start_times(data, ["a", "round "]), [1.0, 1.1])

    def test_without_character_times_the_syllables_share_the_time(self):
        data = TranscribedData(word="around ", start=1.0, end=2.0)
        self.assertEqual(syllable_start_times(data, ["a", "round "]), [1.0, 1.5])

    def test_a_character_without_time_takes_the_next_one_of_the_syllable(self):
        data = TranscribedData(word="it's ", start=1.0, end=2.0, char_starts=[1.0, 1.2, None, 1.6])
        self.assertEqual(syllable_start_times(data, ["it", "'s "]), [1.0, 1.6])

    def test_syllables_too_short_for_a_note_share_the_time(self):
        data = TranscribedData(word="around ", start=1.0, end=2.0,
                               char_starts=[1.0, 1.01, 1.15, 1.4, 1.6, 1.8])
        self.assertEqual(syllable_start_times(data, ["a", "round "]), [1.0, 1.5])

    def test_add_hyphen_to_data(self):
        data = [TranscribedData(word="around ", start=1.0, end=2.0, char_starts=[1.0, 1.1, 1.15, 1.4, 1.6, 1.8]),
                TranscribedData(word="you ", start=2.0, end=2.5)]

        result = add_hyphen_to_data(data, [["a", "round "], None])

        self.assertEqual([(d.word, d.start, d.end, d.is_word_end) for d in result],
                         [("a", 1.0, 1.1, False), ("round ", 1.1, 2.0, True), ("you ", 2.0, 2.5, True)])

    def test_punctuation_is_removed_together_with_its_time(self):
        data = [TranscribedData(word="Hi, ", start=1.0, end=2.0, char_starts=[1.0, 1.5, 1.9])]
        remove_unecessary_punctuations(data)
        self.assertEqual((data[0].word, data[0].char_starts), ("Hi ", [1.0, 1.5]))


class PhraseEndTest(unittest.TestCase):
    def test_only_the_last_syllable_ends_the_phrase(self):
        data = [TranscribedData(word="around ", start=1.0, end=2.0, is_segment_end=True)]
        result = add_hyphen_to_data(data, [["a", "round "]])
        self.assertEqual([d.is_segment_end for d in result], [False, True])

    def test_the_last_segment_of_a_long_syllable_ends_the_phrase(self):
        data = [TranscribedData(word="you ", start=1.0, end=2.0, is_segment_end=True)]
        result = split_syllables_into_segments(data, 120)
        self.assertGreater(len(result), 1)
        self.assertEqual([d.is_segment_end for d in result], [False] * (len(result) - 1) + [True])


class MergeSyllableSegmentsTest(unittest.TestCase):
    def test_syllables_on_the_same_pitch_stay_separate_notes(self):
        data = [TranscribedData(word="bo", start=1.0, end=1.2, is_hyphen=True, is_word_end=False),
                TranscribedData(word="dy ", start=1.2, end=1.4, is_hyphen=True)]
        segments = [MidiSegment("D4", 1.0, 1.2, "bo"), MidiSegment("D4", 1.2, 1.4, "dy ")]

        merged_segments, merged_data = merge_syllable_segments(segments, data, 120)

        self.assertEqual([segment.word for segment in merged_segments], ["bo", "dy "])
        self.assertEqual(len(merged_data), 2)

    def test_a_held_syllable_becomes_one_note(self):
        data = [TranscribedData(word="you", start=1.0, end=1.125, is_hyphen=True, is_word_end=False),
                TranscribedData(word="~", start=1.125, end=1.25, is_hyphen=True, is_word_end=False),
                TranscribedData(word="~ ", start=1.25, end=1.375, is_hyphen=True)]
        segments = [MidiSegment("D4", 1.0, 1.125, "you"), MidiSegment("D4", 1.125, 1.25, "~"),
                    MidiSegment("D4", 1.25, 1.375, "~ ")]

        merged_segments, _ = merge_syllable_segments(segments, data, 120)

        self.assertEqual([(s.word, s.start, s.end) for s in merged_segments], [("you ", 1.0, 1.375)])

    def test_a_pitch_change_inside_a_syllable_stays(self):
        data = [TranscribedData(word="you", start=1.0, end=1.5, is_hyphen=True, is_word_end=False),
                TranscribedData(word="~ ", start=1.5, end=2.0, is_hyphen=True)]
        segments = [MidiSegment("D4", 1.0, 1.5, "you"), MidiSegment("A4", 1.5, 2.0, "~ ")]

        merged_segments, _ = merge_syllable_segments(segments, data, 120)

        self.assertEqual([segment.note for segment in merged_segments], ["D4", "A4"])


if __name__ == "__main__":
    unittest.main()
