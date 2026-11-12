"""Find the lines that are rapped, not sung"""

import bisect

import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.voicing import VOICED_CONFIDENCE

# A held tone: the pitch stays within this many semitones of its median ...
STABLE_TOLERANCE_SEMITONES = 0.5
# ... for at least this many frames of the pitch detection (about 0.1 s)
STABLE_MIN_FRAMES = 6
MIN_LINE_NOTES = 3
MIN_VOICED_FRAMES = 8

# Speech-like rap: fast syllables on a restless pitch without a held tone
SPEECH_MIN_SYLLABLES_PER_SECOND = 4.5
SPEECH_MAX_STABLE_SHARE = 0.55
SPEECH_MIN_JITTER_SEMITONES = 0.2
SPEECH_MAX_HELD_SECONDS = 0.36
# Monotone rap: fast syllables around one pitch, still without a held tone
MONOTONE_MIN_SYLLABLES_PER_SECOND = 4.0
MONOTONE_MAX_NOTE_SPREAD_SEMITONES = 1.0
MONOTONE_MIN_JITTER_SEMITONES = 0.13
MONOTONE_MAX_HELD_SECONDS = 0.6


def mark_rap_notes(midi_segments: list[MidiSegment], pitched_data: PitchedData,
                   linebreaks: set[int]) -> list[MidiSegment]:
    """Set ``rap`` on the notes of rapped lines.

    A rapped line cannot be scored by its pitch: the games have rap notes for
    it, which only want the syllable at the right time. Rap shows in the pitch
    of the voice: many syllables per second, no tone that is held, and a pitch
    that wanders like in speech or stays on one tone. Rap comes in passages, so
    a line is only taken with a rapped neighbour, and a single line between two
    rapped ones goes with them.

    The limits are set on the generated lines of 76 songs with hand-made charts,
    20 of them with rap or freestyle notes. They are cautious: about a third of
    the lines that are rap there is found. As many lines are marked that are
    sung notes there, nearly all of them in a few songs with fast, speech-like
    verses, which some charts write as rap and others do not.
    """
    times = np.asarray(pitched_data.times, dtype=float)
    if len(times) < 2 or not midi_segments:
        return midi_segments
    frequencies = np.asarray(pitched_data.frequencies, dtype=float)
    confidences = np.asarray(pitched_data.confidence, dtype=float)
    voiced = (frequencies > 0) & (confidences > VOICED_CONFIDENCE)
    pitch = np.full(len(times), np.nan)
    pitch[voiced] = 69 + 12 * np.log2(frequencies[voiced] / 440.0)
    frame = float(times[1] - times[0])

    bounds = sorted(linebreaks)
    lines = list(zip([0] + [i + 1 for i in bounds], bounds + [len(midi_segments) - 1]))
    flagged = [_is_rap_line(midi_segments[first:last + 1], times, pitch, frame) for first, last in lines]
    flagged = _passages(flagged)

    count = 0
    for (first, last), rap in zip(lines, flagged):
        for segment in midi_segments[first:last + 1]:
            segment.rap = rap
            count += rap
    if count:
        print(f"{ULTRASINGER_HEAD} Marked {blue_highlighted(str(count))} notes in "
              f"{blue_highlighted(str(sum(flagged)))} lines as rap")
    return midi_segments


def _is_rap_line(notes: list[MidiSegment], times: np.ndarray, pitch: np.ndarray, frame: float) -> bool:
    duration = notes[-1].end - notes[0].start
    if len(notes) < MIN_LINE_NOTES or duration <= 0:
        return False
    first = bisect.bisect_left(times, notes[0].start)
    last = bisect.bisect_left(times, notes[-1].end)
    line_pitch = pitch[first:last]
    is_voiced = ~np.isnan(line_pitch)
    if is_voiced.sum() < MIN_VOICED_FRAMES:
        return False

    syllables = sum(1 for note in notes if note.word.strip() != "~")
    rate = syllables / duration
    steps = np.abs(np.diff(line_pitch))
    steps = steps[~np.isnan(steps)]
    if len(steps) == 0:
        return False
    jitter = float(np.median(steps))
    stable = stable_frames(line_pitch)
    stable_share = float(stable[is_voiced].mean())
    held = _longest_run(stable) * frame

    if (rate >= SPEECH_MIN_SYLLABLES_PER_SECOND and stable_share <= SPEECH_MAX_STABLE_SHARE
            and jitter >= SPEECH_MIN_JITTER_SEMITONES and held <= SPEECH_MAX_HELD_SECONDS):
        return True

    note_pitches = []
    for note in notes:
        note_pitch = pitch[bisect.bisect_left(times, note.start):bisect.bisect_left(times, note.end)]
        note_pitch = note_pitch[~np.isnan(note_pitch)]
        if len(note_pitch) >= 2:
            note_pitches.append(float(np.median(note_pitch)))
    return (len(note_pitches) >= 3 and rate >= MONOTONE_MIN_SYLLABLES_PER_SECOND
            and float(np.std(note_pitches)) <= MONOTONE_MAX_NOTE_SPREAD_SEMITONES
            and jitter >= MONOTONE_MIN_JITTER_SEMITONES and held <= MONOTONE_MAX_HELD_SECONDS)


def stable_frames(pitch: np.ndarray) -> np.ndarray:
    """True for the frames of held tones: runs that stay near their median pitch"""
    count = len(pitch)
    stable = np.zeros(count, dtype=bool)
    i = 0
    while i < count:
        if np.isnan(pitch[i]):
            i += 1
            continue
        j = i + 1
        while (j < count and not np.isnan(pitch[j])
               and abs(pitch[j] - np.median(pitch[i:j])) <= STABLE_TOLERANCE_SEMITONES):
            j += 1
        if j - i >= STABLE_MIN_FRAMES:
            stable[i:j] = True
            i = j
        else:
            i += 1
    return stable


def _longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _passages(flagged: list[bool]) -> list[bool]:
    """Fill a single line between two rapped ones, then drop rapped lines that stand alone"""
    count = len(flagged)
    filled = list(flagged)
    for i in range(1, count - 1):
        if not flagged[i] and flagged[i - 1] and flagged[i + 1]:
            filled[i] = True
    return [filled[i] and ((i > 0 and filled[i - 1]) or (i < count - 1 and filled[i + 1]))
            for i in range(count)]
