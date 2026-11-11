"""Tests for voicing.py"""

import unittest

from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Pitcher.voicing import trim_to_voicing
from src.modules.Speech_Recognition.TranscribedData import TranscribedData

FRAME = 0.016


def pitched(voiced_ranges: list[tuple[float, float]], seconds: float = 4.0) -> PitchedData:
    times = [i * FRAME for i in range(int(seconds / FRAME))]
    confidence = [0.9 if any(a <= t < b for a, b in voiced_ranges) else 0.1 for t in times]
    return PitchedData(times, [220.0] * len(times), confidence)


class TrimToVoicingTest(unittest.TestCase):
    def test_a_note_ends_where_its_singing_ends(self):
        data = [TranscribedData(word="la ", start=1.0, end=3.0)]
        trim_to_voicing(data, pitched([(1.0, 2.0)]))
        self.assertAlmostEqual(data[0].end, 2.0, delta=2 * FRAME)
        self.assertEqual(data[0].start, 1.0)

    def test_a_note_that_starts_long_before_its_singing_moves(self):
        data = [TranscribedData(word="la ", start=1.0, end=3.0)]
        trim_to_voicing(data, pitched([(2.0, 3.0)]))
        self.assertAlmostEqual(data[0].start, 1.96, delta=2 * FRAME)
        self.assertEqual(data[0].end, 3.0)

    def test_consonants_at_the_start_stay(self):
        data = [TranscribedData(word="sta ", start=1.0, end=2.0)]
        trim_to_voicing(data, pitched([(1.08, 2.0)]))
        self.assertEqual((data[0].start, data[0].end), (1.0, 2.0))

    def test_a_note_without_singing_stays(self):
        data = [TranscribedData(word="la ", start=1.0, end=2.0)]
        trim_to_voicing(data, pitched([(3.0, 3.5)]))
        self.assertEqual((data[0].start, data[0].end), (1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
