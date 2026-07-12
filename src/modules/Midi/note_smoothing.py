"""Smooth midi segments to reduce pitch detection artifacts"""

import re
import statistics

from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted

# A jump of this many semitones away from the local median is treated as a
# potential octave error of the pitch detection
OCTAVE_ERROR_THRESHOLD = 9
# After shifting by octaves the note must be at most this far from the local
# median for the shift to be accepted
OCTAVE_ERROR_MAX_RESIDUAL = 6
# A single note deviating this much from BOTH stable neighbours is an artifact
OUTLIER_THRESHOLD = 7
# Neighbours are considered stable when they are within this range of each other
OUTLIER_NEIGHBOUR_RANGE = 4
# Notes outside this range around the song's median are outside a plausible
# vocal range and get shifted back by octaves
MAX_SEMITONES_FROM_MEDIAN = 18
# Local window size (neighbours per side) for the octave error correction
LOCAL_WINDOW_SIZE = 3

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_OFFSETS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_PATTERN = re.compile(r"^([A-Ga-g])([#♯b♭]*)(-?\d+)$")


def note_to_midi(note: str) -> int | None:
    """Convert a note name like 'C#4' (also '♯'/'♭' variants) to a midi number"""
    match = _NOTE_PATTERN.match(note.strip())
    if match is None:
        return None
    name, accidentals, octave = match.groups()
    midi = 12 * (int(octave) + 1) + _NOTE_OFFSETS[name.upper()]
    for accidental in accidentals:
        midi += 1 if accidental in "#♯" else -1
    return midi


def midi_to_note(midi: int) -> str:
    """Convert a midi number to a note name like 'C#4'"""
    return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def smooth_midi_segments(
    midi_segments: list[MidiSegment], allowed_notes: set[str] = None
) -> list[MidiSegment]:
    """Reduce pitch detection artifacts in midi segments.

    Applies three corrections that keep the melody intact:
    1. Octave errors: notes detected one or more octaves away from their
       local context are shifted back by whole octaves.
    2. Single-note outliers: a lone note far away from two agreeing
       neighbours is replaced by the neighbours' pitch.
    3. Vocal range: notes far outside the song's median range are shifted
       back by whole octaves.

    Corrected notes are re-quantized to `allowed_notes` when given.
    """
    pitches = [note_to_midi(segment.note) for segment in midi_segments]
    valid_pitches = [pitch for pitch in pitches if pitch is not None]
    if len(valid_pitches) < 3:
        return midi_segments

    corrected_indexes = set()
    corrected_indexes |= __correct_octave_errors(pitches)
    corrected_indexes |= __correct_single_outliers(pitches)
    corrected_indexes |= __clamp_to_vocal_range(pitches)

    for index in corrected_indexes:
        note = midi_to_note(pitches[index])
        if allowed_notes is not None:
            note = __quantize_to_key(pitches[index], allowed_notes)
        midi_segments[index].note = note

    print(
        f"{ULTRASINGER_HEAD} Note smoothing: corrected {blue_highlighted(str(len(corrected_indexes)))} "
        f"of {blue_highlighted(str(len(midi_segments)))} note(s)"
    )
    return midi_segments


def __local_median(pitches: list[int | None], index: int) -> float | None:
    """Median of up to LOCAL_WINDOW_SIZE valid neighbours on each side, excluding index"""
    neighbours = []
    for offset in range(1, len(pitches)):
        for neighbour_index in (index - offset, index + offset):
            if 0 <= neighbour_index < len(pitches) and pitches[neighbour_index] is not None:
                neighbours.append(pitches[neighbour_index])
        if len(neighbours) >= LOCAL_WINDOW_SIZE * 2:
            break
    if not neighbours:
        return None
    return statistics.median(neighbours)


def __correct_octave_errors(pitches: list[int | None]) -> set[int]:
    corrected = set()
    for index, pitch in enumerate(pitches):
        if pitch is None:
            continue
        local_median = __local_median(pitches, index)
        if local_median is None or abs(pitch - local_median) < OCTAVE_ERROR_THRESHOLD:
            continue

        best_pitch = pitch
        best_distance = abs(pitch - local_median)
        for octave_shift in (-24, -12, 12, 24):
            shifted = pitch + octave_shift
            distance = abs(shifted - local_median)
            if distance < best_distance:
                best_distance = distance
                best_pitch = shifted

        if best_pitch != pitch and best_distance <= OCTAVE_ERROR_MAX_RESIDUAL:
            pitches[index] = best_pitch
            corrected.add(index)
    return corrected


def __correct_single_outliers(pitches: list[int | None]) -> set[int]:
    corrected = set()
    for index in range(1, len(pitches) - 1):
        pitch = pitches[index]
        previous = pitches[index - 1]
        following = pitches[index + 1]
        if pitch is None or previous is None or following is None:
            continue
        if (
            abs(pitch - previous) > OUTLIER_THRESHOLD
            and abs(pitch - following) > OUTLIER_THRESHOLD
            and abs(previous - following) <= OUTLIER_NEIGHBOUR_RANGE
        ):
            pitches[index] = round((previous + following) / 2)
            corrected.add(index)
    return corrected


def __clamp_to_vocal_range(pitches: list[int | None]) -> set[int]:
    corrected = set()
    valid_pitches = [pitch for pitch in pitches if pitch is not None]
    median = statistics.median(valid_pitches)
    for index, pitch in enumerate(pitches):
        if pitch is None:
            continue
        new_pitch = pitch
        while new_pitch > median + MAX_SEMITONES_FROM_MEDIAN:
            new_pitch -= 12
        while new_pitch < median - MAX_SEMITONES_FROM_MEDIAN:
            new_pitch += 12
        if new_pitch != pitch:
            pitches[index] = new_pitch
            corrected.add(index)
    return corrected


def __quantize_to_key(midi: int, allowed_notes: set[str]) -> str:
    """Return the note name of the nearest midi pitch whose pitch class is allowed"""
    for offset in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6):
        candidate = midi + offset
        if _NOTE_NAMES[candidate % 12] in allowed_notes:
            return midi_to_note(candidate)
    return midi_to_note(midi)
