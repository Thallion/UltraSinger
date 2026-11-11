"""Tests for the ultrastar_writer.py module."""

import unittest

from packaging import version
from unittest.mock import patch, mock_open
from src.modules.Ultrastar.ultrastar_writer import create_ultrastar_txt, format_separated_string
from src.modules.Midi.MidiSegment import MidiSegment
from src.modules.Ultrastar.ultrastar_txt import UltrastarTxtValue, UltrastarTxtTag


class TestCreateUltrastarTxt(unittest.TestCase):
    def test_create_ultrastar_txt_from_automation_default_values(self):
        # Arrange
        bpm, midi_segments, ultrastar_file_output = self.arrange()
        class_under_test = UltrastarTxtValue()

        # Act and Assert default values
        class_under_test.version = "0.2.0"
        expected_calls = self.default_values(class_under_test, class_under_test.version)
        self.act_and_assert(bpm, class_under_test, expected_calls, midi_segments,
                            ultrastar_file_output)

        class_under_test.version = "1.0.0"
        expected_calls = self.default_values(class_under_test, class_under_test.version)
        self.act_and_assert(bpm, class_under_test, expected_calls, midi_segments,
                            ultrastar_file_output)

        class_under_test.version = "1.1.0"
        expected_calls = self.default_values(class_under_test, class_under_test.version)
        self.act_and_assert(bpm, class_under_test, expected_calls, midi_segments,
                            ultrastar_file_output)

        class_under_test.version = "1.2.0"
        expected_calls = self.default_values(class_under_test, class_under_test.version)
        self.act_and_assert(bpm, class_under_test, expected_calls, midi_segments,
                            ultrastar_file_output)

    def test_create_ultrastar_txt_from_automation_full_values(self):
        # Arrange
        bpm, midi_segments, ultrastar_file_output = self.arrange()

        class_under_test = UltrastarTxtValue()
        class_under_test.artist = "artist"
        class_under_test.title = "title"
        class_under_test.year = "2023"
        class_under_test.language = "de"
        class_under_test.genre = "pop, rock"
        class_under_test.tags = "pop, rock"
        class_under_test.cover = "cover [CO].jpg"
        class_under_test.video = "video.mp4"
        class_under_test.videoUrl = "https://xyz.com/video"
        class_under_test.mp3 = "music.mp3"
        class_under_test.audio = "music.mp3"
        class_under_test.vocals = "vocals.mp3"
        class_under_test.instrumental = "instrumental.mp3"

        # Act and Assert full values

        class_under_test.version = "0.2.0"
        expected_calls = self.full_values(class_under_test, class_under_test.version)
        self.act_and_assert(bpm, class_under_test, expected_calls, midi_segments,
                            ultrastar_file_output)

        class_under_test.version = "1.0.0"
        expected_calls = self.full_values(class_under_test, class_under_test.version)
        self.act_and_assert(bpm, class_under_test, expected_calls, midi_segments,
                            ultrastar_file_output)

        class_under_test.version = "1.1.0"
        expected_calls = self.full_values(class_under_test, class_under_test.version)
        self.act_and_assert(bpm, class_under_test, expected_calls, midi_segments,
                            ultrastar_file_output)

        class_under_test.version = "1.2.0"
        expected_calls = self.full_values(class_under_test, class_under_test.version)
        self.act_and_assert(bpm, class_under_test, expected_calls, midi_segments,
                            ultrastar_file_output)

    def arrange(self) -> tuple[int, list[MidiSegment], str]:
        midi_segments = [
            MidiSegment(note="C#3", start=0.5, end=2.5, word="UltraSinger "),
            MidiSegment(note="D3", start=3.0, end=4.5, word="is "),
            MidiSegment(note="D#3", start=5.5, end=7.5, word="cool! ")]

        ultrastar_file_output = "output.txt"
        bpm = 120
        return bpm, midi_segments, ultrastar_file_output

    def act_and_assert(self, bpm, default_ultrastar_class, expected_calls_default_values, midi_segments,
                       ultrastar_file_output):
        # Act
        mock_file = self.act(bpm, default_ultrastar_class, midi_segments, ultrastar_file_output)

        # Assert the file was opened and is utf-8
        mock_file.assert_called_once_with(ultrastar_file_output, "w", encoding='utf-8')

        # Assert that expected_calls_default_values were written to the file
        mock_file_handle = mock_file.return_value.__enter__.return_value.write
        write_calls = [args[0][0] for args in mock_file_handle.call_args_list]
        self.assertEqual(write_calls, expected_calls_default_values)

    @staticmethod
    def default_values(default_ultrastar_class, ver):
        expected_calls = []
        if version.parse(ver) >= version.parse("1.0.0"):
            expected_calls.append(f"#{UltrastarTxtTag.VERSION.value}:{default_ultrastar_class.version}\n")
        expected_calls += [
            f"#{UltrastarTxtTag.ARTIST.value}:{default_ultrastar_class.artist}\n",
            f"#{UltrastarTxtTag.TITLE.value}:{default_ultrastar_class.title}\n",
            f"#{UltrastarTxtTag.MP3.value}:{default_ultrastar_class.mp3}\n"
        ]
        if version.parse(ver) >= version.parse("1.1.0"):
            expected_calls += [f"#{UltrastarTxtTag.AUDIO.value}:{default_ultrastar_class.audio}\n"]
        if default_ultrastar_class.video is not None:
            expected_calls += [
                f"#{UltrastarTxtTag.VIDEO.value}:{default_ultrastar_class.video}\n"
            ]
        if version.parse(ver) >= version.parse("1.2.0"):
            if default_ultrastar_class.videoUrl is not None:
                expected_calls += [f"#{UltrastarTxtTag.VIDEOURL.value}:{default_ultrastar_class.videoUrl}\n"]
        expected_calls += [
            f"#{UltrastarTxtTag.BPM.value}:390.0\n",
            f"#{UltrastarTxtTag.GAP.value}:500\n",
            f"#{UltrastarTxtTag.CREATOR.value}:{default_ultrastar_class.creator}\n",
            f"#{UltrastarTxtTag.COMMENT.value}:{default_ultrastar_class.comment}\n",
            ": 0 52 1 UltraSinger \n",
            "- 52\n",
            ": 65 39 2 is \n",
            "- 104\n",
            ": 130 52 3 cool! \n",
            "E"
        ]

        return expected_calls

    @staticmethod
    def full_values(default_ultrastar_class, ver):
        expected_calls = []
        if version.parse(ver) >= version.parse("1.0.0"):
            expected_calls.append(f"#{UltrastarTxtTag.VERSION.value}:{default_ultrastar_class.version}\n")
        expected_calls.append(f"#{UltrastarTxtTag.ARTIST.value}:{default_ultrastar_class.artist}\n")
        expected_calls.append(f"#{UltrastarTxtTag.TITLE.value}:{default_ultrastar_class.title}\n")
        expected_calls.append(f"#{UltrastarTxtTag.YEAR.value}:{default_ultrastar_class.year}\n")
        expected_calls.append(f"#{UltrastarTxtTag.LANGUAGE.value}:German\n")
        expected_calls.append(f"#{UltrastarTxtTag.GENRE.value}:{default_ultrastar_class.genre}\n")
        expected_calls.append(f"#{UltrastarTxtTag.COVER.value}:{default_ultrastar_class.cover}\n")
        expected_calls.append(f"#{UltrastarTxtTag.MP3.value}:{default_ultrastar_class.mp3}\n")
        if version.parse(ver) >= version.parse("1.1.0"):
            expected_calls.append(f"#{UltrastarTxtTag.AUDIO.value}:{default_ultrastar_class.audio}\n")
            expected_calls.append(f"#{UltrastarTxtTag.VOCALS.value}:{default_ultrastar_class.vocals}\n")
            expected_calls.append(f"#{UltrastarTxtTag.INSTRUMENTAL.value}:{default_ultrastar_class.instrumental}\n")
        expected_calls.append(f"#{UltrastarTxtTag.VIDEO.value}:{default_ultrastar_class.video}\n")
        if version.parse(ver) >= version.parse("1.2.0"):
            expected_calls.append(f"#{UltrastarTxtTag.VIDEOURL.value}:{default_ultrastar_class.videoUrl}\n")
        expected_calls.append(f"#{UltrastarTxtTag.BPM.value}:390.0\n")
        expected_calls.append(f"#{UltrastarTxtTag.GAP.value}:500\n")
        if version.parse(ver) >= version.parse("1.1.0"):
            expected_calls.append(f"#{UltrastarTxtTag.TAGS.value}:{default_ultrastar_class.tags}\n")
        expected_calls.append(f"#{UltrastarTxtTag.CREATOR.value}:{default_ultrastar_class.creator}\n")
        expected_calls.append(f"#{UltrastarTxtTag.COMMENT.value}:{default_ultrastar_class.comment}\n")
        expected_calls.append(": 0 52 1 UltraSinger \n")
        expected_calls.append("- 52\n")
        expected_calls.append(": 65 39 2 is \n")
        expected_calls.append("- 104\n")
        expected_calls.append(": 130 52 3 cool! \n")
        expected_calls.append("E")

        return expected_calls

    @staticmethod
    def act(bpm, default_ultrastar_class, midi_segments, ultrastar_file_output):
        with patch("builtins.open", mock_open()) as mock_file:
            create_ultrastar_txt(
                midi_segments, ultrastar_file_output,
                default_ultrastar_class, bpm
            )
        return mock_file

    def test_format_separated_string(self):
        self.assertEqual(format_separated_string('rock,pop,rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock;pop;rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock/pop/rock-pop,'), 'Rock, Pop, Rock-Pop')
        self.assertEqual(format_separated_string('rock,pop/rock-pop;80s,'), 'Rock, Pop, Rock-Pop, 80s')
        self.assertEqual(format_separated_string('rock, pop, rock-pop, '), 'Rock, Pop, Rock-Pop')


if __name__ == "__main__":
    unittest.main()
