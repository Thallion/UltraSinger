"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import getopt
import hashlib
import os
import sys
import Levenshtein
import librosa

from packaging import version

from modules import os_helper
from modules.init_interactive_mode import init_settings_interactive
from modules.Audio.denoise import denoise_vocal_audio
from modules.Audio.separation import separate_vocal_from_audio
from modules.Audio.vocal_chunks import (
    create_audio_chunks_from_transcribed_data,
    create_audio_chunks_from_ultrastar_data,
)
from modules.Audio.key_detector import detect_key_from_audio, get_allowed_notes_for_key
from modules.Audio.silence_processing import remove_silence_from_transcription_data, mute_no_singing_parts
from modules.Audio.separation import DemucsModel
from modules.Audio.convert_audio import convert_audio_to_mono_wav, convert_audio_format
from modules.Audio.youtube import (
    download_from_youtube,
)
from modules.Audio.bpm import get_bpm_from_file
from modules.Audio.onset_correction import detect_vocal_onsets, snap_to_onsets

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    red_highlighted,
    green_highlighted,
    cyan_highlighted,
    bright_green_highlighted,
)
from modules.Midi.midi_creator import (
    create_midi_segments_from_transcribed_data,
    simplify_melismas,
    create_repitched_midi_segments_from_ultrastar_txt,
    create_midi_file,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.note_postprocessing import fill_unvoiced_notes, correct_octaves
from modules.Midi.note_length_calculator import get_thirtytwo_note_second, get_sixteenth_note_second
from modules.Pitcher.pitcher import (
    get_pitch_with_file,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.voicing import trim_to_voicing
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Speech_Recognition.hyphenation import (
    hyphenate_each_word,
)
from modules.Speech_Recognition.Whisper import transcribe_with_whisper
from modules.Speech_Recognition.reference_lyrics import (
    MIN_MATCH_RATIO,
    MIN_MATCH_RATIO_GIVEN_LYRICS,
    fetch_lyrics_from_lrclib,
    find_lyrics_file,
    read_lyrics_file,
)
from modules.Ultrastar import (
    ultrastar_writer,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Ultrastar.ultrastar_score_calculator import Score, calculate_score_points
from modules.Ultrastar.ultrastar_txt import FILE_ENCODING, FormatVersion
from modules.Ultrastar.coverter.ultrastar_txt_converter import from_ultrastar_txt, \
    create_ultrastar_txt_from_midi_segments, create_ultrastar_txt_from_automation
from modules.Ultrastar.ultrastar_parser import parse_ultrastar_txt
from modules.common_print import print_support, print_help, print_version
from modules.os_helper import check_file_exists, get_unused_song_output_dir
from modules.plot import create_plots
from modules.musicbrainz_client import search_musicbrainz
from modules.sheet import create_sheet
from modules.ProcessData import ProcessData, ProcessDataPaths, MediaInfo
from modules.DeviceDetection.device_detection import check_gpu_support
from modules.Image.image_helper import save_image
from modules.ffmpeg_helper import (
    is_ffmpeg_available,
    get_ffmpeg_and_ffprobe_paths,
    is_video_file,
    separate_audio_video,
)

from Settings import Settings

settings = Settings()
PHRASE_END_PUNCTUATION = (".", ",", "!", "?", ";", ":")


def syllable_start_times(data: TranscribedData, syllables: list[str]) -> list[float]:
    """Start time of each syllable of a word.

    The forced alignment knows when each character of the word is sung. Where
    it does not, the syllables share the time of the word by their length.
    """
    duration = data.end - data.start
    count = len(syllables)
    even = [data.start + duration * i / count for i in range(count)]

    char_starts = data.char_starts
    word = data.word.rstrip(" ")
    if not char_starts or len(char_starts) != len(word) or "".join(syllables).rstrip(" ") != word:
        return even

    starts = [data.start]
    offset = len(syllables[0])
    for syllable in syllables[1:]:
        start = next((t for t in char_starts[offset:offset + len(syllable)] if t is not None), None)
        offset += len(syllable)
        if start is None or not data.start < start < data.end:
            return even
        starts.append(start)

    # Every syllable has to be long enough to be a note
    min_duration = min(0.05, duration / count / 2)
    for i in range(1, count):
        if starts[i] - starts[i - 1] < min_duration:
            return even
    if data.end - starts[-1] < min_duration:
        return even
    return starts


def add_hyphen_to_data(
        transcribed_data: list[TranscribedData], hyphen_words: list[list[str]]
):
    """Add hyphen to transcribed data return new data list"""
    new_data = []

    for i, data in enumerate(transcribed_data):
        if not hyphen_words[i]:
            new_data.append(data)
        else:
            syllables = hyphen_words[i]
            starts = syllable_start_times(data, syllables) + [data.end]
            for j, syllable in enumerate(syllables):
                dup = copy.copy(data)
                dup.char_starts = None
                dup.start = starts[j]
                dup.end = starts[j + 1]
                dup.word = syllable
                dup.is_hyphen = True
                dup.is_word_end = j == len(syllables) - 1
                dup.is_segment_end = data.is_segment_end and dup.is_word_end
                new_data.append(dup)

    return new_data


# Todo: Unused
def correct_words(recognized_words, word_list_file):
    """Docstring"""
    with open(word_list_file, "r", encoding="utf-8") as file:
        text = file.read()
    word_list = text.split()

    for i, rec_word in enumerate(recognized_words):
        if rec_word.word in word_list:
            continue

        closest_word = min(
            word_list, key=lambda x: Levenshtein.distance(rec_word.word, x)
        )
        print(recognized_words[i].word + " - " + closest_word)
        recognized_words[i].word = closest_word
    return recognized_words


def remove_unecessary_punctuations(transcribed_data: list[TranscribedData]) -> None:
    """Remove unecessary punctuations from transcribed data"""
    punctuation = ".,"
    for data in transcribed_data:
        word = data.word.rstrip(" ")
        if data.char_starts is not None and len(data.char_starts) == len(word):
            data.char_starts = [t for char, t in zip(word, data.char_starts) if char not in punctuation]
        data.word = data.word.translate({ord(i): None for i in punctuation})


def run() -> tuple[str, Score, Score]:
    """The processing function of this program"""
    #List selected options (can add more later)
    if settings.keep_numbers:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Numbers will be transcribed as numerics (i.e. 1, 2, 3, etc.)')}")
    if settings.create_plot:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Plot will be created')}")
    if settings.keep_cache:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Cache folder will not be deleted')}")
    if settings.create_audio_chunks:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Audio chunks will be created')}")
    if not settings.create_karaoke:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Karaoke txt will not be created')}")
    if not settings.use_separated_vocal:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Vocals will not be separated')}")
    if not settings.hyphenation:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Hyphenation will not be applied')}")
    if settings.quantize_to_key:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Notes will be quantized to the detected musical key')}")

    process_data = InitProcessData()

    process_data.process_data_paths.cache_folder_path = (
        os.path.join(settings.output_folder_path, "cache")
        if settings.cache_override_path is None
        else settings.cache_override_path
    )

    # Create process audio
    process_data.process_data_paths.processing_audio_path = CreateProcessAudio(process_data)

    # Get BPM from wav file
    if not settings.input_file_is_ultrastar_txt:
        process_data.media_info.bpm = get_bpm_from_file(process_data.process_data_paths.processing_audio_path)

    # Detect key
    detected_key, detected_mode = detect_key_from_audio(process_data.process_data_paths.processing_audio_path)
    if process_data.media_info.music_key is None:
        process_data.media_info.music_key = f"{detected_key} {detected_mode}"

    # Audio transcription
    process_data.media_info.language = settings.language
    if not settings.ignore_audio:
        TranscribeAudio(process_data)

    # Pitch audio
    process_data.pitched_data = pitch_audio(process_data.process_data_paths)

    # Fit the notes to the singing and split syllables into segments
    if not settings.ignore_audio:
        process_data.transcribed_data = trim_to_voicing(process_data.transcribed_data, process_data.pitched_data)
        if settings.onset_correction:
            onsets = detect_vocal_onsets(process_data.process_data_paths.processing_audio_path)
            process_data.transcribed_data = snap_to_onsets(process_data.transcribed_data, onsets)
        process_data.transcribed_data = split_syllables_into_segments(process_data.transcribed_data,
                                                                  process_data.media_info.bpm)

        # The phrase ends are taken after the notes got their final times: the onset
        # correction may start the next word before the old end of the word before
        process_data.phrase_end_times = [data.end for data in process_data.transcribed_data if data.is_segment_end]

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(process_data)

    # Allowed keys for quantization
    allowed_notes_for_key = None
    if settings.quantize_to_key and not settings.ignore_audio:
        allowed_notes_for_key = get_allowed_notes_for_key(detected_key, detected_mode)

    # Create Midi_Segments
    if not settings.ignore_audio:
        process_data.midi_segments = create_midi_segments_from_transcribed_data(
            process_data.transcribed_data,
            process_data.pitched_data,
            allowed_notes_for_key
        )
    else:
        process_data.midi_segments = create_repitched_midi_segments_from_ultrastar_txt(process_data.pitched_data,
                                                                                       process_data.parsed_file)
        process_data.midi_segments = fill_unvoiced_notes(process_data.midi_segments)
        if settings.octave_correction:
            process_data.midi_segments = correct_octaves(process_data.midi_segments)

    if not process_data.midi_segments:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error: No singing was found in the audio, so there are no notes to write.')}")
        sys.exit(1)

    # Merge syllable segments
    if not settings.ignore_audio:
        process_data.midi_segments = fill_unvoiced_notes(process_data.midi_segments)
        process_data.midi_segments, process_data.transcribed_data = merge_syllable_segments(process_data.midi_segments,
                                                                                        process_data.transcribed_data,
                                                                                        process_data.media_info.bpm)
        process_data.midi_segments = simplify_melismas(process_data.midi_segments, process_data.pitched_data,
                                                       allowed_notes_for_key)
        if settings.octave_correction:
            process_data.midi_segments = correct_octaves(process_data.midi_segments)

    # Create plot
    if settings.create_plot:
        create_plots(process_data, settings.output_folder_path)

    # Create Ultrastar txt
    accurate_score, simple_score, ultrastar_file_output = CreateUltraStarTxt(process_data)

    # Create Midi
    if settings.create_midi:
        create_midi_file(process_data.media_info.bpm, settings.output_folder_path, process_data.midi_segments,
                         process_data.basename)

    # Sheet music
    create_sheet(process_data.midi_segments, settings.output_folder_path,
                 process_data.process_data_paths.cache_folder_path, settings.musescore_path, process_data.basename,
                 process_data.media_info)

    # Cleanup
    if not settings.keep_cache:
        remove_cache_folder(process_data.process_data_paths.cache_folder_path)

    # Print Support
    print_support()
    return ultrastar_file_output, simple_score, accurate_score


def split_syllables_into_segments(
        transcribed_data: list[TranscribedData],
        real_bpm: float) -> list[TranscribedData]:
    """Split every syllable into sub-segments

    This splits long syllables (including hyphenated ones) into smaller segments
    to allow detect for pitch changes within a syllable (e.g., singing a scale).
    """
    syllable_segment_size = get_sixteenth_note_second(real_bpm)
    thirtytwo_note = get_thirtytwo_note_second(real_bpm)

    segment_size_decimal_points = len(str(syllable_segment_size).split(".")[1])
    new_data = []

    for i, data in enumerate(transcribed_data):
        duration = data.end - data.start

        # If duration is less than or equal to a 16th note, don't split
        if duration <= syllable_segment_size:
            new_data.append(data)
            continue

        has_space = str(data.word).endswith(" ")
        first_segment = copy.deepcopy(data)
        filler_words_start = data.start + syllable_segment_size
        remainder = data.end - filler_words_start
        first_segment.end = filler_words_start
        if has_space:
            first_segment.word = first_segment.word[:-1]

        first_segment.is_word_end = False
        first_segment.is_segment_end = False
        new_data.append(first_segment)

        full_segments, partial_segment = divmod(remainder, syllable_segment_size)

        if full_segments >= 1:
            first_segment.is_hyphen = True
            for i in range(int(full_segments)):
                segment = TranscribedData()
                segment.word = "~"
                segment.start = filler_words_start + round(
                    i * syllable_segment_size, segment_size_decimal_points
                )
                segment.end = segment.start + syllable_segment_size
                segment.is_hyphen = True
                segment.is_word_end = False
                new_data.append(segment)

        # Only add a partial_segment if it's at least as long as a 32nd note
        # Otherwise add it to the last note
        if partial_segment >= thirtytwo_note:
            first_segment.is_hyphen = True
            segment = TranscribedData()
            segment.word = "~"
            segment.start = filler_words_start + round(
                full_segments * syllable_segment_size, segment_size_decimal_points
            )
            segment.end = segment.start + partial_segment
            segment.is_hyphen = True
            segment.is_word_end = False
            new_data.append(segment)
        elif full_segments >= 1 or len(new_data) > 0:
            # Add remaining time to the last note
            new_data[-1].end += partial_segment

        if has_space:
            new_data[-1].word += " "
            new_data[-1].is_word_end = True
        new_data[-1].is_segment_end = data.is_segment_end
    return new_data


def merge_syllable_segments(midi_segments: list[MidiSegment],
                            transcribed_data: list[TranscribedData],
                            real_bpm: float) -> tuple[list[MidiSegment], list[TranscribedData]]:
    """Merge sub-segments of a syllable where the pitch is the same

    This function handles two cases:
    1. Merge consecutive ~ segments with the SAME pitch (same note held)
    2. Detect and merge SLIDES: short ~ segments with ±1-2 semitone jumps between syllables
    Different syllables stay notes of their own, also on the same pitch: each one is sung anew.
    """

    sixteenth_note = get_sixteenth_note_second(real_bpm)
    # Slides are typically very short (1-2 16th notes)
    max_slide_duration = sixteenth_note * 2

    new_data = []
    new_midi_notes = []

    previous_data = None

    for i, data in enumerate(transcribed_data):
        # Check if previous element exists
        is_same_note = i > 0 and midi_segments[i].note == midi_segments[i - 1].note
        has_breath_pause = False

        if previous_data is not None:
            has_breath_pause = (data.start - previous_data.end) > sixteenth_note

        # Check if current word is a ~ segment (continuation marker)
        current_word_stripped = str(data.word).strip()
        is_tilde_segment = current_word_stripped == "~"

        # Slide detection: Detect short ~ segments with small pitch jumps
        is_potential_slide = False
        if i > 0 and is_tilde_segment:
            duration = data.end - data.start

            # Calculate pitch jump in semitones
            try:
                prev_midi = librosa.note_to_midi(midi_segments[i - 1].note)
                curr_midi = librosa.note_to_midi(midi_segments[i].note)
                semitone_diff = abs(curr_midi - prev_midi)

                # Slide: Short duration AND small pitch jump (1-2 semitones)
                is_potential_slide = (duration <= max_slide_duration and
                                     semitone_diff <= 2 and
                                     semitone_diff > 0)
            except:
                # Ignore slide detection on error
                pass

        should_merge_tilde = (is_tilde_segment and
                             previous_data is not None and
                             (is_same_note or is_potential_slide) and
                             not has_breath_pause)

        if should_merge_tilde:
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end

            # For slides: Keep the original note (not the transition note)
            if is_potential_slide and not is_same_note:
                new_midi_notes[-1].note = midi_segments[i - 1].note

            # Take over space and word_end flag from current segment
            # "~ " means end of word - add space to previous segment
            if str(data.word).endswith(" "):
                if not new_data[-1].word.endswith(" "):
                    new_data[-1].word += " "
                if not new_midi_notes[-1].word.endswith(" "):
                    new_midi_notes[-1].word += " "
                new_data[-1].is_word_end = True

        else:
            # Add as new segment
            new_data.append(data)
            new_midi_notes.append(midi_segments[i])

        previous_data = data

    return new_midi_notes, new_data


def create_audio_chunks(process_data):
    if not settings.ignore_audio:
        create_audio_chunks_from_transcribed_data(
            process_data.process_data_paths,
            process_data.transcribed_data)
    else:
        create_audio_chunks_from_ultrastar_data(
            process_data.process_data_paths,
            process_data.parsed_file
        )


def InitProcessData():
    settings.input_file_is_ultrastar_txt = settings.input_file_path.endswith(".txt")
    if settings.input_file_is_ultrastar_txt:
        # Parse Ultrastar txt
        (
            basename,
            settings.output_folder_path,
            audio_file_path,
            ultrastar_class,
            audio_extension,
        ) = parse_ultrastar_txt(settings.input_file_path, settings.output_folder_path)
        process_data = from_ultrastar_txt(ultrastar_class)
        process_data.basename = basename
        process_data.process_data_paths.audio_output_file_path = audio_file_path
        process_data.media_info.audio_extension = audio_extension
        # todo: ignore transcribe
        settings.ignore_audio = True

    elif settings.input_file_path.startswith("https:"):
        # Youtube
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info
        ) = download_from_youtube(settings.input_file_path, settings.output_folder_path, settings.cookiefile)
    else:
        # Audio/Video File
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info,
        ) = infos_from_audio_video_input_file()
    return process_data


def load_reference_lyrics(process_data) -> tuple[list, float]:
    """Lyrics the song may have: of the given file, of a file next to the input, or looked up online.

    Returns the lyrics and the share of their words they have to share with the transcription to be used.
    """
    lyrics_file = settings.lyrics_file or find_lyrics_file(settings.input_file_path)
    if lyrics_file:
        if not check_file_exists(lyrics_file):
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Error: Lyrics file not found:')} {lyrics_file}")
            sys.exit(1)
        print(f"{ULTRASINGER_HEAD} Reading lyrics from {blue_highlighted(lyrics_file)}")
        return [read_lyrics_file(lyrics_file)], MIN_MATCH_RATIO_GIVEN_LYRICS
    if settings.lyrics_lookup and process_data.media_info is not None:
        candidates = fetch_lyrics_from_lrclib(process_data.media_info.artist, process_data.media_info.title)
        if candidates:
            print(f"{ULTRASINGER_HEAD} Found lyrics at {blue_highlighted('lrclib.net')}")
        if not settings.lyrics_line_times:
            for lyrics in candidates:
                for word in lyrics:
                    word.line_start = word.line_stop = None
        return candidates, MIN_MATCH_RATIO
    return [], MIN_MATCH_RATIO


def TranscribeAudio(process_data):
    reference_lyrics, lyrics_min_match_ratio = load_reference_lyrics(process_data)
    transcription_result = transcribe_audio(process_data.process_data_paths.cache_folder_path,
                                            process_data.process_data_paths.processing_audio_path,
                                            reference_lyrics, lyrics_min_match_ratio)

    if process_data.media_info.language is None:
        process_data.media_info.language = transcription_result.detected_language

    process_data.transcribed_data = transcription_result.transcribed_data

    # Phrases end with the lines of the reference lyrics, else where the
    # transcription ends a sentence or sets a punctuation mark
    process_data.has_lyric_lines = transcription_result.has_lyric_lines
    for data in process_data.transcribed_data:
        data.is_segment_end = data.is_segment_end or (
            not transcription_result.has_lyric_lines and data.word.rstrip(" ").endswith(PHRASE_END_PUNCTUATION))

    # Hyphen
    # Todo: Is it really unnecessary?
    remove_unecessary_punctuations(process_data.transcribed_data)
    if settings.hyphenation:
        hyphen_words = hyphenate_each_word(process_data.media_info.language, process_data.transcribed_data)

        if hyphen_words is not None:
            process_data.transcribed_data = add_hyphen_to_data(process_data.transcribed_data, hyphen_words)

    process_data.transcribed_data = remove_silence_from_transcription_data(
        process_data.process_data_paths.processing_audio_path, process_data.transcribed_data
    )


def CreateUltraStarTxt(process_data: ProcessData):
    # Move instrumental and vocals
    if settings.create_karaoke and version.parse(settings.format_version.value) < version.parse(
            FormatVersion.V1_1_0.value):
        karaoke_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Karaoke]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.instrumental_audio_file_path, karaoke_output_path)

    if version.parse(settings.format_version.value) >= version.parse(FormatVersion.V1_1_0.value):
        instrumental_output_path = os.path.join(settings.output_folder_path,
                                                process_data.basename + " [Instrumental]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.instrumental_audio_file_path, instrumental_output_path)
        vocals_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Vocals]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.vocals_audio_file_path, vocals_output_path)

    # Create Ultrastar txt
    if not settings.ignore_audio:
        ultrastar_file_output = create_ultrastar_txt_from_automation(
            process_data.basename,
            settings.output_folder_path,
            process_data.midi_segments,
            process_data.media_info,
            settings.format_version,
            settings.create_karaoke,
            settings.APP_VERSION,
            process_data.phrase_end_times,
            process_data.has_lyric_lines
        )
    else:
        ultrastar_file_output = create_ultrastar_txt_from_midi_segments(
            settings.output_folder_path, settings.input_file_path, process_data.media_info.title,
            process_data.midi_segments
        )

    # Calc Points
    simple_score = None
    accurate_score = None
    if settings.calculate_score:
        simple_score, accurate_score = calculate_score_points(process_data, ultrastar_file_output)

    # Add calculated score to Ultrastar txt
    #Todo: Missing Karaoke
    if simple_score is not None:
        ultrastar_writer.add_score_to_ultrastar_txt(ultrastar_file_output, simple_score)
    return accurate_score, simple_score, ultrastar_file_output


def CreateProcessAudio(process_data) -> str:
    # Set processing audio to cache file
    process_data.process_data_paths.processing_audio_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + ".wav"
    )
    os_helper.create_folder(process_data.process_data_paths.cache_folder_path)

    # Separate vocal from audio
    audio_separation_folder_path = separate_vocal_from_audio(
        process_data.process_data_paths.cache_folder_path,
        process_data.process_data_paths.audio_output_file_path,
        settings.use_separated_vocal,
        settings.create_karaoke,
        settings.pytorch_device,
        settings.demucs_model,
        settings.skip_cache_vocal_separation
    )
    process_data.process_data_paths.vocals_audio_file_path = os.path.join(audio_separation_folder_path, "vocals.wav")
    process_data.process_data_paths.instrumental_audio_file_path = os.path.join(audio_separation_folder_path,
                                                                                "no_vocals.wav")

    if settings.use_separated_vocal:
        input_path = process_data.process_data_paths.vocals_audio_file_path
    else:
        input_path = process_data.process_data_paths.audio_output_file_path

    # Denoise vocal audio
    if settings.denoise:
        denoised_output_path = os.path.join(
            process_data.process_data_paths.cache_folder_path, process_data.basename + "_denoised.wav"
        )
        denoise_vocal_audio(input_path, denoised_output_path, settings.skip_cache_denoise_vocal_audio)
    else:
        denoised_output_path = input_path

    # Convert to mono audio
    mono_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mono.wav"
    )
    convert_audio_to_mono_wav(denoised_output_path, mono_output_path)

    # Mute silence sections
    mute_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mute.wav"
    )
    mute_no_singing_parts(mono_output_path, mute_output_path)

    # Define the audio file to process
    return mute_output_path


def transcribe_audio(cache_folder_path: str, processing_audio_path: str,
                     reference_lyrics: list = None,
                     lyrics_min_match_ratio: float = MIN_MATCH_RATIO) -> TranscriptionResult:
    """Transcribe audio with AI"""
    transcription_result = None
    whisper_align_model_string = None
    if settings.transcriber == "whisper":
        if not settings.whisper_align_model is None:
            whisper_align_model_string = settings.whisper_align_model.replace("/", "_")
        whisper_device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
        transcription_config = f"{settings.transcriber}_{settings.whisper_model.value}_{whisper_device}_{whisper_align_model_string}_{settings.whisper_batch_size}_{settings.whisper_compute_type}_{settings.language}"
        if reference_lyrics:
            lyrics_text = "//".join(" ".join(word.text + (f"/{word.line_start}" if word.line_end else "")
                                             for word in lyrics)
                                    for lyrics in reference_lyrics)
            lyrics_text += f"//{lyrics_min_match_ratio}"
            transcription_config += "_lyrics" + hashlib.md5(lyrics_text.encode("utf-8")).hexdigest()[:8]
        transcription_path = os.path.join(cache_folder_path, f"{transcription_config}.json")
        cached_transcription_available = check_file_exists(transcription_path)
        if settings.skip_cache_transcription or not cached_transcription_available:
            transcription_result = transcribe_with_whisper(
                processing_audio_path,
                settings.whisper_model,
                whisper_device,
                settings.whisper_align_model,
                settings.whisper_batch_size,
                settings.whisper_compute_type,
                settings.language,
                settings.keep_numbers,
                reference_lyrics,
                lyrics_min_match_ratio,
            )
            with open(transcription_path, "w", encoding=FILE_ENCODING) as file:
                file.write(transcription_result.to_json())
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached transcribed data")
            with open(transcription_path) as file:
                json = file.read()
                transcription_result = TranscriptionResult.from_json(json)
    else:
        raise NotImplementedError
    return transcription_result


def infos_from_audio_video_input_file() -> tuple[str, str, str, MediaInfo]:
    """Infos from audio/video input file"""
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0]

    artist, title = None, None
    if " - " in basename_without_ext:
        artist, title = basename_without_ext.split(" - ", 1)
    else:
        title = basename_without_ext

    song_info = search_musicbrainz(title, artist)
    basename_without_ext = f"{song_info.artist} - {song_info.title}"

    song_folder_output_path = os.path.join(settings.output_folder_path, basename_without_ext)
    song_folder_output_path = get_unused_song_output_dir(song_folder_output_path)
    os_helper.create_folder(song_folder_output_path)

    extension = os.path.splitext(basename)[1]
    if is_video_file(settings.input_file_path):
        print(f"{ULTRASINGER_HEAD} Video file detected - separating audio and video")

        video_with_audio_basename = f"{basename_without_ext}{extension}"
        video_with_audio_path = os.path.join(song_folder_output_path, video_with_audio_basename)
        os_helper.copy(settings.input_file_path, video_with_audio_path)

        # Separate audio and video
        ultrastar_audio_input_path, final_video_path, audio_ext, video_ext = separate_audio_video(
            video_with_audio_path, basename_without_ext, song_folder_output_path
        )
    else:
        # Audio file
        basename_with_ext = f"{basename_without_ext}{extension}"
        audio_ext = extension.lstrip('.')
        video_ext = None
        os_helper.copy(settings.input_file_path, song_folder_output_path)
        os_helper.rename(
            os.path.join(song_folder_output_path, os.path.basename(settings.input_file_path)),
            os.path.join(song_folder_output_path, basename_with_ext),
        )
        ultrastar_audio_input_path = os.path.join(song_folder_output_path, basename_with_ext)

    # Todo: Read ID3 tags
    if song_info.cover_image_data is not None:
        save_image(song_info.cover_image_data, basename_without_ext, song_folder_output_path)

    return (
        basename_without_ext,
        song_folder_output_path,
        ultrastar_audio_input_path,
        MediaInfo(
            artist=song_info.artist,
            title=song_info.title,
            year=song_info.year,
            genre=song_info.genres,
            cover_url=song_info.cover_url,
            audio_extension=audio_ext,
            video_extension=video_ext
        ),
    )


def pitch_audio(
        process_data_paths: ProcessDataPaths) -> PitchedData:
    """Pitch audio"""

    pitching_config = f"swiftf0_{settings.ignore_audio}"
    pitched_data_path = os.path.join(process_data_paths.cache_folder_path, f"{pitching_config}.json")
    cache_available = check_file_exists(pitched_data_path)

    if settings.skip_cache_pitch_detection or not cache_available:
        pitched_data = get_pitch_with_file(
            process_data_paths.processing_audio_path
        )

        pitched_data_json = pitched_data.to_json()
        with open(pitched_data_path, "w", encoding=FILE_ENCODING) as file:
            file.write(pitched_data_json)
    else:
        print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached pitch data")
        with open(pitched_data_path) as file:
            json = file.read()
            pitched_data = PitchedData.from_json(json)

    return pitched_data


def main(argv: list[str]) -> None:
    """Main function"""
    print_version(settings.APP_VERSION)
    init_settings(argv)
    check_requirements()
    if settings.interactive_mode:
        init_settings_interactive(settings)
    run()
    sys.exit()


def check_requirements() -> None:
    if not settings.force_cpu:
        settings.pytorch_device = check_gpu_support()
    print(f"{ULTRASINGER_HEAD} ----------------------")

    if not is_ffmpeg_available(settings.user_ffmpeg_path):
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} {blue_highlighted('FFmpeg')} {red_highlighted('is not available. Provide --ffmpeg ‘path’ or install FFmpeg with PATH')}")
        sys.exit(1)
    else:
        ffmpeg_path, ffprobe_path = get_ffmpeg_and_ffprobe_paths()
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('FFmpeg')} - using {red_highlighted(ffmpeg_path)}")
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('FFprobe')} - using {red_highlighted(ffprobe_path)}")

    print(f"{ULTRASINGER_HEAD} ----------------------")

def remove_cache_folder(cache_folder_path: str) -> None:
    """Remove cache folder"""
    os_helper.remove_folder(cache_folder_path)


def init_settings(argv: list[str]) -> Settings:
    """Init settings"""
    long, short = arg_options()
    opts, args = getopt.getopt(argv, short, long)
    if len(opts) == 0:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            settings.input_file_path = arg
        elif opt in ("-o", "--ofile"):
            settings.output_folder_path = arg
        elif opt in ("--whisper"):
            settings.transcriber = "whisper"

            #Addition of whisper model choice. Added error handling for unknown models.
            try:
                settings.whisper_model = WhisperModel(arg)
            except ValueError as ve:
                print(f"{ULTRASINGER_HEAD} The model {arg} is not a valid whisper model selection. Please use one of the following models: {blue_highlighted(', '.join([m.value for m in WhisperModel]))}")
                sys.exit()
        elif opt in ("--whisper_align_model"):
            settings.whisper_align_model = arg
        elif opt in ("--whisper_batch_size"):
            settings.whisper_batch_size = int(arg)
        elif opt in ("--whisper_compute_type"):
            settings.whisper_compute_type = arg
        elif opt in ("--keep_numbers"):
            settings.keep_numbers = True
        elif opt in ("--language"):
            settings.language = arg
        elif opt in ("--crepe"):
            settings.crepe_model_capacity = arg
        elif opt in ("--crepe_step_size"):
            settings.crepe_step_size = int(arg)
        elif opt in ("--plot"):
            settings.create_plot = True
        elif opt in ("--midi"):
            settings.create_midi = True
        elif opt in ("--disable_hyphenation"):
            settings.hyphenation = False
        elif opt in ("--disable_separation"):
            settings.use_separated_vocal = False
        elif opt in ("--disable_karaoke"):
            settings.create_karaoke = False
        elif opt in ("--create_audio_chunks"):
            settings.create_audio_chunks = arg
        elif opt in ("--ignore_audio"):
            settings.ignore_audio = True
        elif opt in ("--force_cpu"):
            settings.force_cpu = True
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif opt in ("--force_whisper_cpu"):
            settings.force_whisper_cpu = True
        elif opt in ("--force_crepe_cpu"):
            settings.force_crepe_cpu = True
        elif opt in ("--format_version"):
            if arg == FormatVersion.V0_3_0.value:
                settings.format_version = FormatVersion.V0_3_0
            elif arg == FormatVersion.V1_0_0.value:
                settings.format_version = FormatVersion.V1_0_0
            elif arg == FormatVersion.V1_1_0.value:
                settings.format_version = FormatVersion.V1_1_0
            elif arg == FormatVersion.V1_2_0.value:
                settings.format_version = FormatVersion.V1_2_0
            else:
                print(
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error: Format version')} {blue_highlighted(arg)} {red_highlighted('is not supported.')}"
                )
                sys.exit(1)
        elif opt in ("--keep_cache"):
            settings.keep_cache = True
        elif opt in ("--musescore_path"):
            settings.musescore_path = arg
        #Addition of demucs model choice. Work seems to be needed to make sure syntax is same for models. Added error handling for unknown models
        elif opt in ("--demucs"):
            try:
                settings.demucs_model = DemucsModel(arg)
            except ValueError as ve:
                print(f"{ULTRASINGER_HEAD} The model {arg} is not a valid demucs model selection. Please use one of the following models: {blue_highlighted(', '.join([m.value for m in DemucsModel]))}")
                sys.exit()
        elif opt in ("--cookiefile"):
            settings.cookiefile = arg
        elif opt in ("--interactive"):
            settings.interactive_mode = True
        elif opt in ("--quantize_to_key"):
            settings.quantize_to_key = True
        elif opt in ("--ffmpeg"):
            settings.user_ffmpeg_path = arg
        elif opt in ("--denoise"):
            settings.denoise = True
        elif opt in ("--lyrics_file"):
            settings.lyrics_file = arg
        elif opt in ("--disable_lyrics_lookup"):
            settings.lyrics_lookup = False
        elif opt in ("--disable_lyrics_line_times"):
            settings.lyrics_line_times = False
        elif opt in ("--disable_onset_correction"):
            settings.onset_correction = False
        elif opt in ("--disable_octave_correction"):
            settings.octave_correction = False
    if settings.output_folder_path == "":
        if settings.input_file_path.startswith("https:"):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_folder_path = os.path.join(dirname, "output")

    return settings


#For convenience, made True/False options into noargs
def arg_options():
    short = "hi:o:amv:"
    long = [
        "ifile=",
        "ofile=",
        "crepe=",
        "crepe_step_size=",
        "demucs=",
        "whisper=",
        "whisper_align_model=",
        "whisper_batch_size=",
        "whisper_compute_type=",
        "language=",
        "plot",
        "midi",
        "disable_hyphenation",
        "disable_separation",
        "disable_karaoke",
        "create_audio_chunks",
        "ignore_audio",
        "force_cpu",
        "force_whisper_cpu",
        "force_crepe_cpu",
        "format_version=",
        "keep_cache",
        "musescore_path=",
        "keep_numbers",
        "quantize_to_key",
        "interactive",
        "cookiefile=",
        "ffmpeg=",
        "lyrics_file=",
        "denoise",
        "disable_lyrics_lookup",
        "disable_lyrics_line_times",
        "disable_onset_correction",
        "disable_octave_correction",
    ]
    return long, short

if __name__ == "__main__":
    main(sys.argv[1:])
