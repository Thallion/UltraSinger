"""Clean up the notes after the pitch detection"""

import librosa

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Midi.MidiSegment import MidiSegment

# Notes with a longer rest between them belong to different phrases
PHRASE_REST_SECONDS = 1.0


def _to_midi(midi_segments: list[MidiSegment]) -> list[int]:
    return [int(librosa.note_to_midi(segment.note)) for segment in midi_segments]


def fill_unvoiced_notes(midi_segments: list[MidiSegment]) -> list[MidiSegment]:
    """Give notes without a sung pitch the pitch of the nearest voiced note.

    Consonants, breathy syllables and very short notes often have no frame
    the pitch detection is sure about. A random pitch there is a note nobody
    can hit; the pitch of the neighbour is what a human charter writes.
    """
    voiced_indices = [i for i, segment in enumerate(midi_segments) if segment.voiced]
    if not voiced_indices or len(voiced_indices) == len(midi_segments):
        return midi_segments

    filled = 0
    for i, segment in enumerate(midi_segments):
        if segment.voiced:
            continue
        previous_index = next((k for k in range(i - 1, -1, -1) if midi_segments[k].voiced), None)
        next_index = next((k for k in range(i + 1, len(midi_segments)) if midi_segments[k].voiced), None)

        candidates = []
        if previous_index is not None:
            candidates.append((segment.start - midi_segments[previous_index].end, 0, previous_index))
        if next_index is not None:
            candidates.append((midi_segments[next_index].start - segment.end, 1, next_index))
        _, _, source = min(candidates)
        segment.note = midi_segments[source].note
        filled += 1

    print(f"{ULTRASINGER_HEAD} Took the pitch of the neighbour for {blue_highlighted(str(filled))} unvoiced notes")
    return midi_segments


def correct_octaves(midi_segments: list[MidiSegment],
                    fidelity_weight: float = 2.0,
                    jump_hinge: float = 8.0) -> list[MidiSegment]:
    """Move notes the pitch detection put into the wrong octave back onto the melody.

    Every note keeps its tone; only its octave is chosen again (one down, as
    detected or one up) so that the whole song costs least:
      * a jump to the next note costs what it exceeds ``jump_hinge`` semitones by,
      * moving a note away from its detected octave costs ``fidelity_weight``.
    A single note or a short run in the wrong octave is cheaper to move than
    to keep, a longer passage that really is sung an octave apart is not.
    The games ignore the octave when they score, but they draw it.
    """
    if len(midi_segments) < 3:
        return midi_segments

    midis = _to_midi(midi_segments)
    shifts = (0, -1, 1)  # the detected octave first, so it wins ties
    cost = {shift: fidelity_weight * abs(shift) for shift in shifts}
    back_pointers = []
    for i in range(1, len(midis)):
        # After a long rest the melody may start anywhere
        new_phrase = midi_segments[i].start - midi_segments[i - 1].end > PHRASE_REST_SECONDS
        new_cost = {}
        pointer = {}
        for shift in shifts:
            candidate = midis[i] + 12 * shift
            best_cost, best_previous = float("inf"), 0
            for previous_shift in shifts:
                jump = abs(candidate - (midis[i - 1] + 12 * previous_shift))
                jump_cost = max(0.0, jump - jump_hinge)
                if new_phrase:
                    jump_cost *= 0.5
                total = cost[previous_shift] + jump_cost
                if total < best_cost:
                    best_cost, best_previous = total, previous_shift
            new_cost[shift] = best_cost + fidelity_weight * abs(shift)
            pointer[shift] = best_previous
        cost = new_cost
        back_pointers.append(pointer)

    shift = min(cost, key=cost.get)
    chosen = [shift]
    for pointer in reversed(back_pointers):
        shift = pointer[shift]
        chosen.append(shift)
    chosen.reverse()

    moved = 0
    for i, shift in enumerate(chosen):
        if shift != 0:
            midi_segments[i].note = librosa.midi_to_note(midis[i] + 12 * shift, unicode=False)
            moved += 1
    if moved:
        print(f"{ULTRASINGER_HEAD} Moved {blue_highlighted(str(moved))} notes into the octave of the melody")
    return midi_segments

