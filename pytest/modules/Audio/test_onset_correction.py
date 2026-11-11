"""Tests for onset_correction.py"""

import unittest

from src.modules.Audio.onset_correction import snap_to_onsets
from src.modules.Speech_Recognition.TranscribedData import TranscribedData


def word(start: float, end: float) -> TranscribedData:
    return TranscribedData(word="la ", start=start, end=end)


class SnapToOnsetsTest(unittest.TestCase):
    def test_a_late_start_moves_back_and_the_note_before_ends_there(self):
        data = [word(1.0, 1.5), word(1.5, 2.0)]
        snap_to_onsets(data, [1.42])
        self.assertEqual((data[0].end, data[1].start, data[1].end), (1.42, 1.42, 2.0))

    def test_an_onset_too_far_away_is_ignored(self):
        data = [word(1.0, 1.5), word(1.5, 2.0)]
        snap_to_onsets(data, [1.25, 1.7])
        self.assertEqual((data[1].start, data[0].end), (1.5, 1.5))

    def test_the_nearest_onset_wins(self):
        data = [word(1.5, 2.0)]
        snap_to_onsets(data, [1.35, 1.48, 1.54])
        self.assertEqual(data[0].start, 1.48)

    def test_a_start_never_moves_before_the_note_before(self):
        data = [word(1.40, 1.5), word(1.5, 2.0)]
        snap_to_onsets(data, [1.41])
        self.assertEqual(data[1].start, 1.5)

    def test_a_rest_before_the_note_stays_a_rest(self):
        data = [word(1.0, 1.2), word(1.5, 2.0)]
        snap_to_onsets(data, [1.45])
        self.assertEqual((data[0].end, data[1].start), (1.2, 1.45))

    def test_no_onsets(self):
        data = [word(1.0, 1.5)]
        snap_to_onsets(data, [])
        self.assertEqual(data[0].start, 1.0)


if __name__ == "__main__":
    unittest.main()
