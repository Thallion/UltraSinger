"""Tests for reference_lyrics.py"""

import os
import tempfile
import unittest

from src.modules.Speech_Recognition.reference_lyrics import (
    choose_lyrics,
    clean_name,
    correct_segments,
    find_lyrics_file,
    match_ratio,
    parse_lyrics,
    plan_missed_lines,
)


def lines_of(segments, line_ends):
    """The corrected words as lines of text"""
    words = [word for segment in segments for word in segment["text"].split()]
    lines, line = [], []
    for word, line_end in zip(words, line_ends):
        line.append(word)
        if line_end:
            lines.append(" ".join(line))
            line = []
    return lines + ([" ".join(line)] if line else [])


class ParseLyricsTest(unittest.TestCase):
    def test_words_and_line_ends(self):
        words = parse_lyrics("Hello world\n\nIt's me\n")
        self.assertEqual([word.text for word in words], ["Hello", "world", "It's", "me"])
        self.assertEqual([word.line_end for word in words], [False, True, False, True])

    def test_section_labels_and_timestamps_are_left_out(self):
        words = parse_lyrics("[Chorus]\n[00:12.50] Hello world\n(Verse 2)\n[00:15.00]It's me")
        self.assertEqual([word.text for word in words], ["Hello", "world", "It's", "me"])

    def test_punctuation_alone_is_no_word(self):
        self.assertEqual([word.text for word in parse_lyrics("Hello - world ...")], ["Hello", "world"])


SYNCED = """[00:10.00] one two three
[00:12.00] four five six
[00:14.00] seven eight nine
[00:16.00] ten eleven twelve
[00:18.00] thirteen fourteen fifteen
[00:20.00] sixteen seventeen
[00:22.00]
[00:40.00] the last line
[00:43.00]
"""


class SyncedLyricsTest(unittest.TestCase):
    def test_lines_keep_their_times(self):
        words = parse_lyrics("[00:10.00] Hello world\n[00:12.50] It's me\n[00:15.00]\n[00:20.00] again")
        self.assertEqual([(w.text, w.line_start, w.line_stop) for w in words],
                         [("Hello", 10.0, 12.5), ("world", 10.0, 12.5), ("It's", 12.5, 15.0), ("me", 12.5, 15.0),
                          ("again", 20.0, None)])

    def test_plain_lyrics_have_no_times(self):
        self.assertTrue(all(w.line_start is None for w in parse_lyrics("Hello world\nIt's me")))

    def test_a_whole_missed_line_is_left_for_its_time(self):
        lyrics = parse_lyrics(SYNCED)
        segments = [{"text": "one two three four five six seven eight nine ten eleven twelve "
                             "thirteen fourteen fifteen sixteen seventeen", "start": 10.0, "end": 22.0}]
        corrected, _, lyric_indexes = correct_segments(segments, lyrics)
        self.assertEqual(lyric_indexes, list(range(17)))
        self.assertNotIn("last", corrected[0]["text"])

    def placed(self, shift: float = 0.5, skip=()):
        """Words timed as in SYNCED, moved by ``shift``"""
        lyrics = parse_lyrics(SYNCED)
        return lyrics, {i: (w.line_start + shift, w.line_start + shift + 0.3)
                        for i, w in enumerate(lyrics) if i not in skip}

    def test_a_missed_line_is_searched_at_its_time(self):
        lyrics, placed = self.placed(skip=(17, 18, 19))
        plan, misplaced = plan_missed_lines(lyrics, placed, 60.0)
        self.assertEqual(misplaced, set())
        self.assertEqual(len(plan), 1)
        segment, indexes = plan[0]
        self.assertEqual((segment["text"], indexes), ("the last line", [17, 18, 19]))
        self.assertAlmostEqual(segment["start"], 40.2)   # 40 s + the offset of 0.5 s - the lead of 0.3 s
        self.assertAlmostEqual(segment["end"], 43.5)     # where the synced lyrics end the line

    def test_a_line_timed_far_from_its_time_is_misplaced(self):
        lyrics, placed = self.placed()
        for index in (17, 18, 19):  # squeezed in right after the line before
            placed[index] = (21.0 + index * 0.1, 21.05 + index * 0.1)
        plan, misplaced = plan_missed_lines(lyrics, placed, 60.0)
        self.assertEqual(misplaced, {17, 18, 19})
        self.assertEqual([indexes for _, indexes in plan], [[17, 18, 19]])

    def test_a_missed_line_never_lies_over_timed_words(self):
        lyrics, placed = self.placed(skip=(3, 4, 5))
        plan, _ = plan_missed_lines(lyrics, placed, 60.0)
        segment, _ = plan[0]
        self.assertGreaterEqual(segment["start"], placed[2][1])
        self.assertLessEqual(segment["end"], placed[6][0])

    def test_times_of_another_recording_are_not_used(self):
        lyrics, placed = self.placed(skip=(17, 18, 19))
        for index in placed:  # every line off by another amount
            placed[index] = (placed[index][0] + (index // 3) * 1.5, placed[index][1] + (index // 3) * 1.5)
        self.assertEqual(plan_missed_lines(lyrics, placed, 60.0), ([], set()))

    def test_nothing_missed(self):
        lyrics, placed = self.placed()
        self.assertEqual(plan_missed_lines(lyrics, placed, 60.0), ([], set()))


class CorrectSegmentsTest(unittest.TestCase):
    def test_misheard_words_are_replaced(self):
        segments = [{"text": " I took a walk around the word", "start": 1.0, "end": 4.0},
                    {"text": " to ease my troubled mine", "start": 4.0, "end": 7.0}]
        lyrics = parse_lyrics("I took a walk around the world\nTo ease my troubled mind")
        corrected, line_ends, _ = correct_segments(segments, lyrics)
        self.assertEqual([segment["text"] for segment in corrected],
                         ["I took a walk around the world", "To ease my troubled mind"])
        self.assertEqual((corrected[0]["start"], corrected[1]["end"]), (1.0, 7.0))
        self.assertEqual(lines_of(corrected, line_ends),
                         ["I took a walk around the world", "To ease my troubled mind"])

    def test_one_word_heard_as_two(self):
        segments = [{"text": "we are the champions my friend and we will keep on fighting", "start": 0, "end": 5}]
        lyrics = parse_lyrics("We are the champions my friend\nAnd we'll keep on fighting")
        corrected, _, _ = correct_segments(segments, lyrics)
        self.assertEqual(corrected[0]["text"], "We are the champions my friend And we'll keep on fighting")

    def test_a_missing_word_is_added_to_the_segment_before(self):
        segments = [{"text": "one two three four", "start": 0, "end": 2},
                    {"text": "six seven eight nine", "start": 2, "end": 4}]
        lyrics = parse_lyrics("one two three four five\nsix seven eight nine")
        corrected, line_ends, _ = correct_segments(segments, lyrics)
        self.assertEqual(corrected[0]["text"], "one two three four five")
        self.assertEqual(lines_of(corrected, line_ends), ["one two three four five", "six seven eight nine"])

    def test_made_up_words_are_dropped_but_a_passage_the_lyrics_lack_stays(self):
        passage = "this whole spoken passage is not part of the lyrics at all"
        segments = [{"text": "one two thank you three four", "start": 0, "end": 2},
                    {"text": passage, "start": 2, "end": 6},
                    {"text": "five six seven eight nine ten eleven twelve", "start": 6, "end": 9}]
        lyrics = parse_lyrics("one two three four\nfive six seven eight nine ten eleven twelve")
        corrected, _, _ = correct_segments(segments, lyrics)
        self.assertEqual([segment["text"] for segment in corrected],
                         ["one two three four", passage, "five six seven eight nine ten eleven twelve"])

    def test_nothing_to_correct(self):
        self.assertIsNone(correct_segments([], parse_lyrics("la la")))
        self.assertIsNone(correct_segments([{"text": "la la", "start": 0, "end": 1}], []))


class ChooseLyricsTest(unittest.TestCase):
    SEGMENTS = [{"text": "never gonna give you up never gonna let you down", "start": 0, "end": 5}]

    def test_the_lyrics_that_fit_best_are_chosen(self):
        other = parse_lyrics("Completely different words of another song")
        cover = parse_lyrics("Never gonna give you up\nI will let you down")
        right = parse_lyrics("Never gonna give you up\nNever gonna let you down")
        self.assertIs(choose_lyrics(self.SEGMENTS, [other, cover, right]), right)

    def test_lyrics_of_a_part_of_the_song_lose_against_the_whole_lyrics(self):
        part = parse_lyrics("Never gonna give you up")
        whole = parse_lyrics("Never gonna give you up\nNever gonna let you town")
        self.assertIs(choose_lyrics(self.SEGMENTS, [part, whole]), whole)

    def test_given_lyrics_may_fit_less(self):
        lyrics = parse_lyrics("Never ever give it up\nI will let him drown")
        self.assertIsNone(choose_lyrics(self.SEGMENTS, [lyrics]))
        self.assertIs(choose_lyrics(self.SEGMENTS, [lyrics], 0.2), lyrics)

    def test_no_lyrics_fit(self):
        other = parse_lyrics("Completely different words of another song")
        self.assertIsNone(choose_lyrics(self.SEGMENTS, [other]))
        self.assertIsNone(choose_lyrics(self.SEGMENTS, []))


class CleanNameTest(unittest.TestCase):
    def test_additions_in_brackets_are_removed(self):
        self.assertEqual(clean_name("Bara bada bastu (ESC 2025 Sweden)"), "Bara bada bastu")
        self.assertEqual(clean_name("Ich bin bereit [Album Version]"), "Ich bin bereit")
        self.assertEqual(clean_name("Rihanna feat. Jay-Z"), "Rihanna")
        self.assertEqual(clean_name("(What's the Story) Morning Glory"), "(What's the Story) Morning Glory")


class MatchRatioTest(unittest.TestCase):
    def test_case_accents_and_punctuation_do_not_count(self):
        self.assertEqual(match_ratio(["Hallo,", "MÄDCHEN", "strasse"], parse_lyrics("hallo Mädchen Straße!")), 1.0)


class FindLyricsFileTest(unittest.TestCase):
    def test_file_next_to_the_input(self):
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "Artist - Title.mp3")
            lyrics = os.path.join(folder, "Artist - Title (Lyrics).txt")
            self.assertIsNone(find_lyrics_file(audio))
            with open(lyrics, "w", encoding="utf-8") as file:
                file.write("la la")
            self.assertEqual(find_lyrics_file(audio), lyrics)

    def test_no_file_for_a_link(self):
        self.assertIsNone(find_lyrics_file("https://example.com/watch?v=1"))


if __name__ == "__main__":
    unittest.main()
