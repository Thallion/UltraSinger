"""Midi creator module"""

import bisect
import math
import os
from collections import Counter

import librosa
import numpy as np
import pretty_midi
import unidecode

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import (
    ULTRASINGER_HEAD, blue_highlighted,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Audio.key_detector import NOTE_NAMES

def create_midi_instrument(midi_segments: list[MidiSegment]) -> object:
    """Converts an Ultrastar data to a midi instrument"""

    print(f"{ULTRASINGER_HEAD} Creating midi instrument")

    instrument = pretty_midi.Instrument(program=0, name="Vocals")
    velocity = 100

    for i, midi_segment in enumerate(midi_segments):
        # pretty_midi rejects notes without a length
        if midi_segment.end <= midi_segment.start:
            continue
        note = pretty_midi.Note(velocity, librosa.note_to_midi(midi_segment.note), midi_segment.start, midi_segment.end)
        instrument.notes.append(note)

    return instrument

def sanitize_for_midi(text):
    """
    Sanitize text for MIDI compatibility.
    Uses unidecode to approximate characters to ASCII.
    """
    return unidecode.unidecode(text)

def __create_midi(instruments: list[object], bpm: float, midi_output: str, midi_segments: list[MidiSegment]) -> None:
    """Write instruments to midi file"""

    print(f"{ULTRASINGER_HEAD} Creating midi file -> {midi_output}")

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for i, midi_segment in enumerate(midi_segments):
        sanitized_word = sanitize_for_midi(midi_segment.word)
        midi_data.lyrics.append(pretty_midi.Lyric(text=sanitized_word, time=midi_segment.start))
    for instrument in instruments:
        midi_data.instruments.append(instrument)
    midi_data.write(midi_output)


class MidiCreator:
    """Docstring"""


def convert_frequencies_to_notes(frequency: [str]) -> list[list[str]]:
    """Converts frequencies to notes"""
    notes = []
    for freq in frequency:
        notes.append(librosa.hz_to_note(float(freq)))
    return notes


def most_frequent(array: [str]) -> list[tuple[str, int]]:
    """Get most frequent item in array"""
    return Counter(array).most_common(1)


FALLBACK_NOTE = "C4"
VOICED_CONFIDENCE = 0.5
# A pitch between two notes goes to the one of the key when it is at most this
# many semitones further away than the other one
KEY_PREFERENCE = 0.2


def weighted_median_pitch(frequencies: list[float], confidences: list[float],
                          threshold: float = VOICED_CONFIDENCE) -> float | None:
    """Confidence weighted median pitch in (fractional) midi notes, None when no frame is voiced.

    The median is taken over the pitch in semitones before it is rounded to a
    note. Counting note names instead lets a few frames of a slide or of noise
    win as soon as the held tone wobbles around the border of two notes.
    """
    pitches = []
    weights = []
    for frequency, confidence in zip(frequencies, confidences):
        if confidence > threshold and frequency > 0:
            pitches.append(float(librosa.hz_to_midi(float(frequency))))
            weights.append(float(confidence))
    if not pitches:
        return None

    order = np.argsort(pitches)
    sorted_pitches = np.asarray(pitches)[order]
    cumulative = np.cumsum(np.asarray(weights)[order])
    index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
    index = min(index, len(sorted_pitches) - 1)
    return float(sorted_pitches[index])


def weighted_median_note(frequencies: list[float], confidences: list[float],
                         threshold: float = VOICED_CONFIDENCE) -> str | None:
    """Note of the confidence weighted median pitch, None when no frame is voiced"""
    pitch = weighted_median_pitch(frequencies, confidences, threshold)
    if pitch is None:
        return None
    return librosa.midi_to_note(int(round(pitch)), unicode=False)


def estimate_tuning(pitched_data: PitchedData, threshold: float = 0.8) -> float:
    """How far the singing is off the standard pitch (A4 = 440 Hz) in semitones, -0.5 to 0.5.

    A recording tuned a quarter tone off puts every held tone on the border of
    two notes. Taking the offset out first lets the notes round the same way.
    """
    frequencies = np.asarray(pitched_data.frequencies, dtype=float)
    confidences = np.asarray(pitched_data.confidence, dtype=float)
    mask = (confidences > threshold) & (frequencies > 0)
    if int(mask.sum()) < 100:
        return 0.0
    pitches = 69 + 12 * np.log2(frequencies[mask] / 440.0)
    # circular mean of the distance to the nearest note
    angles = 2 * np.pi * (pitches - np.round(pitches))
    strength = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))
    if strength < 0.1:
        return 0.0
    return float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))) / (2 * np.pi))


def pitch_to_note(pitch: float, allowed_notes: set[str] = None) -> str:
    """Round a fractional midi pitch to a note, preferring a note of the key when it is close"""
    lower = math.floor(pitch)
    candidates = [lower, lower + 1]
    if allowed_notes is None:
        return librosa.midi_to_note(int(round(pitch)), unicode=False)

    def cost(candidate: int) -> float:
        in_key = NOTE_NAMES[candidate % 12] in allowed_notes
        return abs(pitch - candidate) - (KEY_PREFERENCE if in_key else 0.0)

    return librosa.midi_to_note(min(candidates, key=cost), unicode=False)


def find_nearest_index(array: list[float], value: float) -> int:
    """Nearest index in a sorted array"""
    if len(array) == 0:
        return 0
    idx = bisect.bisect_left(array, value)
    if idx >= len(array):
        return len(array) - 1
    if idx > 0 and math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx]):
        return idx - 1
    return idx


def create_midi_notes_from_pitched_data(start_times: list[float], end_times: list[float], words: list[str],
                                         pitched_data: PitchedData, allowed_notes: set[str] = None) -> list[MidiSegment]:
    """Create midi notes from pitched data

    Args:
        start_times: List of start times
        end_times: List of end times
        words: List of words/syllables
        pitched_data: Pitched data containing frequencies and confidence
        allowed_notes: Optional set of allowed note names for key quantization

    Returns:
        List of MidiSegments
    """
    print(f"{ULTRASINGER_HEAD} Creating midi_segments")

    tuning = estimate_tuning(pitched_data)
    if abs(tuning) >= 0.1:
        print(f"{ULTRASINGER_HEAD} The singing is tuned {blue_highlighted(f'{tuning * 100:+.0f} cents')} off; taking that into account")

    midi_segments = []

    for index, start_time in enumerate(start_times):
        end_time = end_times[index]
        word = str(words[index])

        midi_segment = create_midi_note_from_pitched_data(start_time, end_time, pitched_data, word, allowed_notes,
                                                          tuning)
        midi_segments.append(midi_segment)

    return midi_segments


def create_midi_note_from_pitched_data(start_time: float, end_time: float, pitched_data: PitchedData, word: str,
                                        allowed_notes: set[str] = None, tuning: float = 0.0) -> MidiSegment:
    """Create midi note from pitched data

    Args:
        start_time: Start time of the note
        end_time: End time of the note
        pitched_data: Pitched data containing frequencies and confidence
        word: The word/syllable for this note
        allowed_notes: Optional set of allowed note names (without octave) for key quantization
        tuning: Offset of the singing from the standard pitch in semitones

    Returns:
        One MidiSegment
    """

    start = find_nearest_index(pitched_data.times, start_time)
    end = find_nearest_index(pitched_data.times, end_time)

    if start == end:
        freqs = [pitched_data.frequencies[start]]
        confs = [pitched_data.confidence[start]]
    else:
        freqs = pitched_data.frequencies[start:end]
        confs = pitched_data.confidence[start:end]

    pitch = weighted_median_pitch(freqs, confs)
    voiced = pitch is not None
    if not voiced:
        # Nothing reliable was sung here; the neighbours decide later (see fill_unvoiced_notes)
        note = FALLBACK_NOTE
    else:
        note = pitch_to_note(pitch - tuning, allowed_notes)

    return MidiSegment(note, start_time, end_time, word, voiced, pitch - tuning if voiced else None)


# A "~" note shorter than this is an ornament or the wobble of a held tone, not a note of its own
MELISMA_MIN_SECONDS = 0.2
# "~" notes whose sung pitches are closer than this many semitones are one tone
MELISMA_TOLERANCE = 1.0
# Notes with a longer rest between them are not parts of one held syllable
MELISMA_MAX_REST_SECONDS = 0.06


def simplify_melismas(midi_segments: list[MidiSegment], pitched_data: PitchedData,
                      allowed_notes: set[str] = None) -> list[MidiSegment]:
    """Join the "~" notes of a syllable that are no notes of their own.

    A syllable that is held over several tones gets a "~" note for each tone.
    The pitch detection also sees a new tone where the singer only wobbles
    around the border of two notes, slides into the tone or adds a short
    ornament; hand-made charts have one note there. So within a syllable
      * a note shorter than MELISMA_MIN_SECONDS joins its neighbour, and
      * neighbours whose sung pitches differ by less than MELISMA_TOLERANCE
        semitones become one note,
    and a joined note gets the pitch that is sung over its whole length.

    Measured against hand-made charts of 14 songs: 1.30 -> 1.12 times as many
    notes as the charts, pitch agreement +1.3 points, 3.9 points more of the
    note starts are starts in the charts too (3.2 points fewer of theirs are found).
    """
    if not midi_segments:
        return midi_segments
    tuning = estimate_tuning(pitched_data)

    def join(first: MidiSegment, second: MidiSegment) -> MidiSegment:
        word = first.word
        if second.word.endswith(" ") and not word.endswith(" "):
            word += " "
        joined = create_midi_note_from_pitched_data(first.start, second.end, pitched_data, word, allowed_notes, tuning)
        if not joined.voiced:
            longer = first if first.end - first.start >= second.end - second.start else second
            joined.note, joined.voiced, joined.pitch = longer.note, longer.voiced, longer.pitch
        return joined

    result = []
    index = 0
    while index < len(midi_segments):
        # a syllable: its note and the "~" notes that continue it
        group = [midi_segments[index]]
        index += 1
        while (index < len(midi_segments) and midi_segments[index].word.strip() == "~"
               and midi_segments[index].start - group[-1].end <= MELISMA_MAX_REST_SECONDS):
            group.append(midi_segments[index])
            index += 1

        changed = True
        while changed and len(group) > 1:
            changed = False
            # the shortest note that is too short joins its neighbour (the one before, the first note the next)
            shortest = min(range(len(group)), key=lambda k: group[k].end - group[k].start)
            if group[shortest].end - group[shortest].start < MELISMA_MIN_SECONDS:
                first = max(shortest - 1, 0)
                group[first:first + 2] = [join(group[first], group[first + 1])]
                changed = True
                continue
            for k in range(len(group) - 1):
                a, b = group[k].pitch, group[k + 1].pitch
                if a is None or b is None or abs(a - b) < MELISMA_TOLERANCE:
                    group[k:k + 2] = [join(group[k], group[k + 1])]
                    changed = True
                    break
        result.extend(group)

    joined_count = len(midi_segments) - len(result)
    if joined_count:
        print(f"{ULTRASINGER_HEAD} Joined {blue_highlighted(str(joined_count))} \"~\" notes with the note they belong to")
    return result


def create_midi_segments_from_transcribed_data(transcribed_data: list[TranscribedData], pitched_data: PitchedData,
                                                allowed_notes: set[str] = None) -> list[MidiSegment]:
    """Create MIDI segments from transcribed data

    Args:
        transcribed_data: List of transcribed data segments
        pitched_data: Pitched data containing frequencies and confidence
        allowed_notes: Optional set of allowed note names for key quantization

    Returns:
        List of MidiSegments
    """
    start_times = []
    end_times = []
    words = []

    if transcribed_data:
        for i, midi_segment in enumerate(transcribed_data):
            start_times.append(midi_segment.start)
            end_times.append(midi_segment.end)
            words.append(midi_segment.word)
        midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words,
                                                            pitched_data, allowed_notes)
        return midi_segments
    return []


def create_repitched_midi_segments_from_ultrastar_txt(pitched_data: PitchedData, ultrastar_txt: UltrastarTxtValue) -> list[MidiSegment]:
    start_times = []
    end_times = []
    words = []

    for i, note_lines in enumerate(ultrastar_txt.UltrastarNoteLines):
        start_times.append(note_lines.startTime)
        end_times.append(note_lines.endTime)
        words.append(note_lines.word)
    midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words, pitched_data)
    return midi_segments


def create_midi_file(
        real_bpm: float,
        song_output: str,
        midi_segments: list[MidiSegment],
        basename_without_ext: str,
) -> None:
    """Create midi file"""
    print(f"{ULTRASINGER_HEAD} Creating Midi with {blue_highlighted('pretty_midi')}")

    voice_instrument = [
        create_midi_instrument(midi_segments)
    ]

    midi_output = os.path.join(song_output, f"{basename_without_ext}.mid")
    __create_midi(voice_instrument, real_bpm, midi_output, midi_segments)
