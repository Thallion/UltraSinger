"""Whisper Speech Recognition Module"""
import gc
import json
import os
import inspect
import textwrap

# Set environment variable to handle cuDNN loading issues
# CUDA_MODULE_LOADING=LAZY allows PyTorch to continue even if some cuDNN modules are not found
#os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch

# Fix for PyTorch 2.6+ compatibility with omegaconf and old models
# MUST be done before importing whisperx or any other modules that use torch.load
# Monkey-patch torch.load to use weights_only=False for compatibility with older models
# This is necessary because PyTorch 2.6+ changed the default to weights_only=True
# but WhisperX/Pyannote models were saved with pickle and require weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False regardless of what was passed
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
from enum import Enum
from torch.cuda import OutOfMemoryError

from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, green_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData, from_whisper
from modules.Speech_Recognition.reference_lyrics import (
    MIN_MATCH_RATIO,
    choose_lyrics,
    correct_segments,
    plan_missed_lines,
)

#Addition for numbers to words
import re
import ast
from num2words import num2words

#Addition for numbers to words
re_split_preserve_space = re.compile(r'(\d+|\W+|\w+)')


# Alignment models that time sung words better than the defaults of whisperx.
# Measured against hand-made charts: the default German model starts every
# word about 0.1 s late (44 % of the word starts within 0.1 s of the chart,
# 82 % with this one). The default English model is the better one for English.
ALIGN_MODELS = {
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
}

SAMPLE_RATE = 16000
LANGUAGE_DETECTION_SECONDS = 30


def languages_with_align_model() -> set[str]:
    """Languages whisperx can align without a model given by the user"""
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH
    return set(DEFAULT_ALIGN_MODELS_TORCH) | set(DEFAULT_ALIGN_MODELS_HF) | set(ALIGN_MODELS)


def detect_language(whisper_model, audio, allowed_languages: set[str] | None) -> str | None:
    """Language of the first 30 s of singing, among ``allowed_languages`` when given.

    whisperx listens to the first 30 s of the file. Before the singing starts
    that is silence, for which whisper names any language (Latin for a song
    in English), and then no alignment model exists for it.
    """
    try:
        from whisperx.audio import log_mel_spectrogram, N_SAMPLES

        sung = (abs(audio) > 0.01).nonzero()[0]
        start = int(sung[0]) if len(sung) else 0
        start = max(0, min(start, len(audio) - N_SAMPLES))
        window = audio[start:start + N_SAMPLES]
        n_mels = whisper_model.model.feat_kwargs.get("feature_size") or 80
        segment = log_mel_spectrogram(window, n_mels=n_mels,
                                      padding=0 if len(window) >= N_SAMPLES else N_SAMPLES - len(window))
        encoder_output = whisper_model.model.encode(segment)
        results = whisper_model.model.model.detect_language(encoder_output)
        for language_token, probability in results[0]:
            language = language_token[2:-2]
            if allowed_languages is None or language in allowed_languages:
                print(f"{ULTRASINGER_HEAD} Detected language {blue_highlighted(language)} "
                      f"({probability:.0%}) in the first {LANGUAGE_DETECTION_SECONDS} s of singing")
                return language
    except Exception as exception:  # pylint: disable=broad-except
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Language detection failed:')} {exception}")
    return None


def free_gpu_memory() -> None:
    """Give the memory of deleted models back, so the next model (or song) fits on the GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


MEMORY_ERROR_MESSAGE = f"{ULTRASINGER_HEAD} {blue_highlighted('whisper')} ran out of GPU memory; reduce --whisper_batch_size or force usage of cpu with --force_cpu"

class WhisperModel(Enum):
    """Whisper model"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"

#Addition for numbers to words (Using previous code from louispan in PR#135)
def number_to_words(line,language='en'):
    # https://github.com/m-bain/whisperX
    # Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
    # Therefore, convert numbers to words
    out_tokens = []
    in_tokens = re_split_preserve_space.findall(line)
    for token in in_tokens:
        try:
            num = ast.literal_eval(token)
            try:
                out_tokens.append(num2words(num, lang=language))
            except NotImplementedError:
                print(
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language for number transcription. Keeping number as numeric characters for line: {line}, token: {token}"
                )
        except Exception:
            out_tokens.append(token)
    return ''.join(out_tokens) 

def replace_code_lines(source, start_token, end_token,
                       replacement, escape_tokens=True):
    """Replace the source code between `start_token` and `end_token`
    in `source` with `replacement`. The `start_token` portion is included
    in the replaced code. If `escape_tokens` is True (default),
    escape the tokens to avoid them being treated as a regular expression."""

    if escape_tokens:
        start_token = re.escape(start_token)
        end_token = re.escape(end_token)

    def replace_with_indent(match):
        indent = match.group(1)
        return textwrap.indent(replacement, indent)

    return re.sub(r"^(\s+)({}[\s\S]+?)(?=^\1{})".format(start_token, end_token),
                  replace_with_indent, source, flags=re.MULTILINE)

def transcribe_with_whisper(
    audio_path: str,
    model: WhisperModel,
    device="cpu",
    alignment_model: str = None,
    batch_size: int = 16,
    compute_type: str = None,
    language: str = None,
    keep_numbers: bool = False,
    reference_lyrics: list = None,
    lyrics_min_match_ratio: float = MIN_MATCH_RATIO,
    heard_text_cache_path: str = None,
) -> TranscriptionResult:
    """Transcribe with whisper

    reference_lyrics: lyrics the song may have (each a list of LyricWord); the words of
    those that fit the transcription best replace the transcribed ones
    heard_text_cache_path: file that keeps what whisper heard, before the lyrics correct it
    and the words are timed. With it, other lyrics do not need whisper again.
    """
    # Info: Regardless of the audio sampling rate used in the original audio file, whisper resample the audio signal to 16kHz (via ffmpeg). So the standard input from (44.1 or 48 kHz) should work.
    print(
        f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker"
    )
    if alignment_model is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(alignment_model)}")

    # Fixme: Why it is not none for cpu?
    #compute_type = "int8"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        torch.cuda.empty_cache()
        audio = whisperx.load_audio(audio_path)

        if heard_text_cache_path is not None and os.path.isfile(heard_text_cache_path):
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing what whisper heard")
            with open(heard_text_cache_path, encoding="utf-8") as file:
                result = json.load(file)
            language = result["used_language"]
        else:
            loaded_whisper_model = whisperx.load_model(
                model.value, language=language, device=device, compute_type=compute_type
            )

            print(f"{ULTRASINGER_HEAD} Transcribing {audio_path}")

            if language is None:
                language = detect_language(
                    loaded_whisper_model, audio,
                    None if alignment_model is not None else languages_with_align_model())

            # Songs with much text need more GPU memory: try again with half the batch size
            while True:
                try:
                    result = loaded_whisper_model.transcribe(
                        audio, batch_size=batch_size, language=language
                    )
                    break
                except (RuntimeError, OutOfMemoryError) as error:
                    if "out of memory" not in str(error).lower() or batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    free_gpu_memory()
                    print(f"{ULTRASINGER_HEAD} {blue_highlighted('whisper')} ran out of GPU memory; "
                          f"trying again with batch size {blue_highlighted(str(batch_size))}")

            if language is None:
                language = result["language"]

            # The alignment model needs the memory of the whisper model
            del loaded_whisper_model
            free_gpu_memory()

            if heard_text_cache_path is not None:
                result = {"language": result["language"], "used_language": language,
                          "segments": result["segments"]}
                with open(heard_text_cache_path, "w", encoding="utf-8") as file:
                    json.dump(result, file, ensure_ascii=False)

        detected_language = result["language"]

        # load alignment model and metadata
        if alignment_model is None:
            alignment_model = ALIGN_MODELS.get(language)
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=device, model_name=alignment_model
            )
        except ValueError as ve:
            print(
                f"{red_highlighted(f'{ve}')}"
                f"\n"
                f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language. "
                f"Try add it with --align_model [huggingface]."
            )
            raise ve

        # The words of known lyrics replace the transcribed ones before they are timed
        line_ends = None
        lyric_indexes = None
        lyrics = choose_lyrics(result["segments"], reference_lyrics or [], lyrics_min_match_ratio)
        if lyrics is not None:
            corrected = correct_segments(result["segments"], lyrics)
            if corrected is not None:
                result["segments"], line_ends, lyric_indexes = corrected

        #Addition for numbers to words (Using previous code from louispan in PR#135)
        if keep_numbers == False: 
            for obj in result["segments"]:
                obj["text"] = number_to_words(obj["text"],language)

        # align whisper output
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=True,
        )

        transcribed_data = convert_to_transcribed_data(result_aligned)

        # With reference lyrics, the phrases end where the lines of the lyrics end
        has_lyric_lines = line_ends is not None and len(line_ends) == len(transcribed_data)
        if has_lyric_lines:
            for data, line_end in zip(transcribed_data, line_ends):
                data.is_segment_end = line_end
            transcribed_data = add_missed_lines(
                transcribed_data, lyrics, lyric_indexes, model_a, metadata, audio, device, language, keep_numbers)

        del model_a
        free_gpu_memory()

        return TranscriptionResult(transcribed_data, detected_language, has_lyric_lines)
    except ValueError as value_error:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        if (
            "Requested float16 compute type, but the target device or backend do not support efficient float16 computation."
            in str(value_error.args[0])
        ):
            print(value_error)
            print(
                f"{ULTRASINGER_HEAD} Your GPU does not support efficient float16 computation; run UltraSinger with '--whisper_compute_type int8'"
            )

        raise value_error
    except OutOfMemoryError as oom_exception:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        print(oom_exception)
        print(MEMORY_ERROR_MESSAGE)
        raise oom_exception
    except Exception as exception:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        if "CUDA failed with error out of memory" in str(exception.args[0]):
            print(exception)
            print(MEMORY_ERROR_MESSAGE)
        raise exception
    finally:
        # Restore original torch.load after models are loaded
        # This ensures other modules (like pitch detection) are not affected by the monkey-patch
        torch.load = _original_torch_load
        free_gpu_memory()


def char_starts_per_word(segment: dict) -> list[list[float | None]] | None:
    """Start time of every character, grouped by the words of an aligned segment.

    None when the characters do not add up to the words of the segment.
    """
    chars = segment.get("chars")
    if not chars:
        return None

    groups = []
    current_text = ""
    current_starts = []
    for char in chars + [{"char": " "}]:
        if char["char"] == " ":
            if current_text:
                groups.append((current_text, current_starts))
            current_text = ""
            current_starts = []
        else:
            current_text += char["char"]
            current_starts.append(char.get("start"))

    words = segment["words"]
    if len(groups) != len(words) or any(text != word["word"] for (text, _), word in zip(groups, words)):
        return None
    return [starts for _, starts in groups]


# A line the transcription missed is taken when the alignment model hears its words at
# least this well on average (0 to 1); where nothing is sung it finds them, but badly
MIN_MISSED_LINE_SCORE = 0.25


def add_missed_lines(transcribed_data: list[TranscribedData], lyrics: list, lyric_indexes: list,
                     align_model, align_metadata, audio, device: str, language: str,
                     keep_numbers: bool) -> list[TranscribedData]:
    """Time the lines of synced lyrics that the transcription did not hear.

    Whisper leaves out quiet lines, mostly at the end of a song. The lyrics
    know their words and the synced lyrics about when they are sung, so the
    alignment model looks for them there.
    """
    placed = {lyric_index: (data.start, data.end)
              for data, lyric_index in zip(transcribed_data, lyric_indexes) if lyric_index is not None}
    plan, _ = plan_missed_lines(lyrics, placed, len(audio) / SAMPLE_RATE)
    if not plan:
        return transcribed_data

    entries = list(zip(transcribed_data, lyric_indexes))
    added = 0
    for segment, line_indexes in plan:
        if not keep_numbers:
            segment["text"] = number_to_words(segment["text"], language)
        try:
            aligned = whisperx.align([segment], align_model, align_metadata, audio, device,
                                     return_char_alignments=True)
        except Exception as exception:  # pylint: disable=broad-except
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Could not time a missed line:')} {exception}")
            continue
        words = [word for aligned_segment in aligned["segments"] for word in aligned_segment["words"]]
        if not words or any("start" not in word or "end" not in word for word in words):
            continue
        score = sum(word.get("score", 0.0) for word in words) / len(words)
        if score < MIN_MISSED_LINE_SCORE:
            # not heard there either: words of the line that were timed elsewhere stay where they are
            continue
        line = convert_to_transcribed_data(aligned)
        for data in line:
            data.is_segment_end = False
        line[-1].is_segment_end = True

        # words of the line that were squeezed in next to another line go
        moved = any(lyric_index in line_indexes for _, lyric_index in entries)
        entries = [(data, lyric_index) for data, lyric_index in entries if lyric_index not in line_indexes]
        entries.extend((data, None) for data in line)
        added += 1
        what = "Moved a line to its place" if moved else "Added a line the transcription did not hear"
        print(f"{ULTRASINGER_HEAD} {what} at {blue_highlighted(f'{line[0].start:.1f} s')}: {segment['text']}")

    if not added:
        return transcribed_data
    transcribed_data = sorted((data for data, _ in entries), key=lambda data: data.start)
    # a word that ran on through the rest in which a line was found ends where the line starts
    for data, following in zip(transcribed_data, transcribed_data[1:]):
        if data.end > following.start:
            data.end = max(data.start + 0.01, following.start)
    return transcribed_data


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        char_starts = char_starts_per_word(segment)
        for word_index, obj in enumerate(segment["words"]):
            vtd = from_whisper(obj)  # create custom Word object
            if char_starts is not None:
                vtd.char_starts = char_starts[word_index]
            vtd.is_segment_end = word_index == len(segment["words"]) - 1
            vtd.word = vtd.word + " "  # add space to end of word
            if "start" not in obj or "end" not in obj:
                #Addition for numbers to words (Using previous code from louispan in PR#135)
                if len(transcribed_data) == 0: # if the first word doesn't have any timing data
                    vtd.start = 0.0
                    vtd.end = 0.1
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                        f'Fixing it by placing it at beginning. At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                else:
                    previous = transcribed_data[-1] if len(transcribed_data) != 0 else TranscribedData()
                    vtd.start = previous.end + 0.1
                    vtd.end = previous.end + 0.2
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                          f'Fixing it by placing it after the previous word: "{previous.word}". At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                print(f"{red_highlighted(msg)}")
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data
