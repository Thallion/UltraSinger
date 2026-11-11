"""Where the lines of an UltraStar txt end"""

import bisect

from modules.Midi.MidiSegment import MidiSegment

# A rest of this many seconds after a word ends the line
BREAK_REST_SECONDS = 0.3
# With the lines of reference lyrics, only a long rest ends a line inside a line of the lyrics
BREAK_REST_SECONDS_LYRICS = 1.0
# Longer lines are split: they do not fit on the screen and squeeze the notes
MAX_LINE_CHARS = 45
MAX_LINE_NOTES = 12
# A line of at most this many notes that passes faster than this joins a neighbour
SHORT_LINE_NOTES = 2
SHORT_LINE_SECONDS = 1.0
# ... when the rest between them is shorter than this
JOIN_REST_SECONDS = 0.6


def compute_linebreaks(midi_segments: list[MidiSegment],
                       phrase_end_times: list[float] = None,
                       has_lyric_lines: bool = False) -> set[int]:
    """Indices of the notes after which a line ends.

    Hand-made charts end a line where the singer pauses and where a phrase of
    the lyrics ends, and keep lines short enough to read at a glance:
      1. a rest after a word ends the line,
      2. a line that is still too long is split, best at the end of a phrase
         of the transcription (``phrase_end_times``), at a rest, near its middle,
      3. a line of a word or two that flashes by joins the line it is sung closer to.
    With ``has_lyric_lines`` the phrases are the lines of reference lyrics: a
    line then ends there, and inside such a line only at a rest of
    BREAK_REST_SECONDS_LYRICS. (Measured on 16 songs against hand-made
    charts: 1.18 -> 0.98 times as many lines, line break F1 70.1 -> 72.1.)
    """
    count = len(midi_segments)
    if count < 2:
        return set()

    def rest_after(i: int) -> float:
        return max(0.0, midi_segments[i + 1].start - midi_segments[i].end)

    def is_word_end(i: int) -> bool:
        return midi_segments[i].word.endswith(" ")

    def chars(first: int, last: int) -> int:
        return len("".join(midi_segments[k].word for k in range(first, last + 1)).replace("~", "").strip())

    starts = [segment.start for segment in midi_segments]
    phrase_ends = set()
    for time in phrase_end_times or []:
        # the last note that starts before the phrase ends, moved on to the end of its word
        i = bisect.bisect_left(starts, time - 0.01) - 1
        while 0 <= i < count - 1 and not is_word_end(i):
            i += 1
        if 0 <= i < count - 1:
            phrase_ends.add(i)

    candidates = [i for i in range(count - 1) if is_word_end(i)]
    break_rest = BREAK_REST_SECONDS_LYRICS if has_lyric_lines else BREAK_REST_SECONDS
    breaks = {i for i in candidates if rest_after(i) > break_rest}
    if has_lyric_lines:
        breaks |= phrase_ends

    def lines() -> list[tuple[int, int]]:
        bounds = sorted(breaks)
        return list(zip([0] + [i + 1 for i in bounds], bounds + [count - 1]))

    # 2. split long lines
    pending = lines()
    while pending:
        first, last = pending.pop()
        if chars(first, last) <= MAX_LINE_CHARS and last - first + 1 <= MAX_LINE_NOTES:
            continue
        options = [i for i in candidates if first <= i < last]
        if not options:
            continue
        total = chars(first, last)

        def score(i: int) -> float:
            balance = abs(chars(first, i) - total / 2) / max(total, 1)  # 0 in the middle, 0.5 at the ends
            return (1.0 if i in phrase_ends else 0.0) + min(rest_after(i), BREAK_REST_SECONDS) * 2 - balance * 2

        best = max(options, key=score)
        breaks.add(best)
        pending.append((first, best))
        pending.append((best + 1, last))

    # 3. join lines that flash by
    changed = True
    while changed:
        changed = False
        current = lines()
        for k, (first, last) in enumerate(current):
            duration = midi_segments[last].end - midi_segments[first].start
            if last - first + 1 > SHORT_LINE_NOTES or duration >= SHORT_LINE_SECONDS:
                continue
            options = []
            if k > 0:
                options.append((rest_after(first - 1), first - 1, current[k - 1][0], last))
            if k + 1 < len(current):
                options.append((rest_after(last), last, first, current[k + 1][1]))
            for rest, break_index, joined_first, joined_last in sorted(options):
                if rest < JOIN_REST_SECONDS and chars(joined_first, joined_last) <= MAX_LINE_CHARS:
                    breaks.discard(break_index)
                    changed = True
                    break
            if changed:
                break

    return breaks
