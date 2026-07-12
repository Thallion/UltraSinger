"""Parse and align time-synced lyrics (LRC format, e.g. from lrclib.net)"""

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.lyrics_aligner import (
    MIN_ALIGNMENT_RATIO,
    normalize_word,
    split_lyrics_into_words,
)

# Line timestamps like [01:23.45], [1:23] or [01:23.456]
_LINE_TIMESTAMP_PATTERN = re.compile(r"\[(\d+):(\d{1,2}(?:\.\d+)?)\]")
# Enhanced LRC word timestamps like <01:23.45>
_WORD_TIMESTAMP_PATTERN = re.compile(r"<\d+:\d{1,2}(?:\.\d+)?>")

# An inserted (missing) word must at least be this long to be worth inserting
MIN_INSERTED_WORD_DURATION = 0.15
# An inserted word is never stretched longer than this
MAX_INSERTED_WORD_DURATION = 1.0
# Fallback duration when there is no transcribed word after the insertion
DEFAULT_INSERTED_WORD_DURATION = 0.4
# Keep a little distance to the surrounding transcribed words
INSERTED_WORD_PADDING = 0.05


@dataclass
class LyricLine:
    """One line of synced lyrics with its start time in seconds"""

    start: float
    words: list[str] = field(default_factory=list)


def parse_synced_lyrics(synced_lyrics: str) -> list[LyricLine]:
    """Parse LRC formatted lyrics into timed lines.

    Metadata tags like [ar:...] are skipped, lines with multiple timestamps
    are duplicated for each timestamp and enhanced word timestamps are
    stripped. The returned lines are sorted by start time.
    """
    lines = []
    for raw_line in synced_lyrics.splitlines():
        timestamps = _LINE_TIMESTAMP_PATTERN.findall(raw_line)
        if not timestamps:
            continue
        text = _LINE_TIMESTAMP_PATTERN.sub(" ", raw_line)
        text = _WORD_TIMESTAMP_PATTERN.sub(" ", text)
        words = split_lyrics_into_words(text)
        if not words:
            continue
        for minutes, seconds in timestamps:
            start = int(minutes) * 60 + float(seconds)
            lines.append(LyricLine(start=start, words=list(words)))
    lines.sort(key=lambda line: line.start)
    return lines


def synced_lyrics_to_plain(synced_lyrics: str) -> str:
    """Convert LRC formatted lyrics to plain lyrics text"""
    return "\n".join(
        " ".join(line.words) for line in parse_synced_lyrics(synced_lyrics)
    )


def align_synced_lyrics_to_transcription(
    transcribed_data: list[TranscribedData], synced_lyrics: str
) -> list[TranscribedData] | None:
    """Correct the transcription with time-synced reference lyrics.

    Misheard words are replaced like with plain reference lyrics. In addition
    the LRC line structure is used to mark line breaks on the transcribed
    words, and words that are missing from the transcription are inserted
    with timings derived from the line timestamps and the surrounding words.
    Returns a new transcribed data list, or None when the synced lyrics do
    not match the transcription well enough.
    """
    lyric_lines = parse_synced_lyrics(synced_lyrics)
    if not lyric_lines or not transcribed_data:
        return None

    reference_words = []
    reference_line_indexes = []
    line_windows = []
    for line_index, line in enumerate(lyric_lines):
        next_line_start = (
            lyric_lines[line_index + 1].start
            if line_index + 1 < len(lyric_lines)
            else math.inf
        )
        line_windows.append((line.start, next_line_start))
        for word in line.words:
            reference_words.append(word)
            reference_line_indexes.append(line_index)

    transcribed_normalized = [normalize_word(data.word) for data in transcribed_data]
    reference_normalized = [normalize_word(word) for word in reference_words]

    matcher = SequenceMatcher(
        None, transcribed_normalized, reference_normalized, autojunk=False
    )
    if matcher.ratio() < MIN_ALIGNMENT_RATIO:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Synced lyrics differ too much from the transcription. Skipping synced lyrics correction.')}"
        )
        return None

    corrections = 0
    # Each entry is (transcribed data or None, reference word index or None).
    # data None means the reference word was not transcribed (missing word),
    # reference index None means the transcribed word has no reference match.
    aligned: list[tuple[TranscribedData | None, int | None]] = []
    for tag, trans_start, trans_end, ref_start, ref_end in matcher.get_opcodes():
        if tag == "delete":
            # Transcribed words not in the reference (e.g. ad-libs) - keep
            for trans_index in range(trans_start, trans_end):
                aligned.append((transcribed_data[trans_index], None))
            continue
        if tag == "insert":
            for ref_index in range(ref_start, ref_end):
                aligned.append((None, ref_index))
            continue

        pair_count = min(trans_end - trans_start, ref_end - ref_start)
        for offset in range(pair_count):
            data = transcribed_data[trans_start + offset]
            corrections += __apply_reference_word(
                data, reference_words[ref_start + offset]
            )
            aligned.append((data, ref_start + offset))
        for trans_index in range(trans_start + pair_count, trans_end):
            aligned.append((transcribed_data[trans_index], None))
        for ref_index in range(ref_start + pair_count, ref_end):
            aligned.append((None, ref_index))

    new_data, line_indexes, inserted = __insert_missing_words(
        aligned, reference_words, reference_line_indexes, line_windows
    )
    line_breaks = __mark_line_ends(new_data, line_indexes)

    print(
        f"{ULTRASINGER_HEAD} Synced lyrics correction: {blue_highlighted(str(corrections))} word(s) corrected, "
        f"{blue_highlighted(str(inserted))} missing word(s) inserted, "
        f"{blue_highlighted(str(line_breaks))} line break(s) applied"
    )
    return new_data


def __apply_reference_word(data: TranscribedData, reference_word: str) -> int:
    """Replace a transcribed word with its reference word, keeping the timing.

    Returns 1 when the word actually changed (not only casing/punctuation).
    """
    # Preserve the trailing space that marks a word ending
    has_trailing_space = str(data.word).endswith(" ")
    new_word = reference_word + (" " if has_trailing_space else "")
    if data.word == new_word:
        return 0
    changed = normalize_word(data.word) != normalize_word(reference_word)
    data.word = new_word
    return 1 if changed else 0


def __insert_missing_words(
    aligned: list[tuple[TranscribedData | None, int | None]],
    reference_words: list[str],
    reference_line_indexes: list[int],
    line_windows: list[tuple[float, float]],
) -> tuple[list[TranscribedData], list[int | None], int]:
    """Build the final word list, inserting missing reference words.

    Missing words are only inserted when there is a large enough gap between
    the surrounding transcribed words, constrained to the time window of
    their LRC line. Returns the new data, the line index of each word (None
    for transcribed words without a reference match) and the insert count.
    """
    new_data: list[TranscribedData] = []
    line_indexes: list[int | None] = []
    inserted = 0

    position = 0
    while position < len(aligned):
        data, ref_index = aligned[position]
        if data is not None:
            new_data.append(data)
            line_indexes.append(
                reference_line_indexes[ref_index] if ref_index is not None else None
            )
            position += 1
            continue

        # Collect the run of missing words that belong to the same LRC line
        line_index = reference_line_indexes[ref_index]
        missing_ref_indexes = []
        while (
            position < len(aligned)
            and aligned[position][0] is None
            and reference_line_indexes[aligned[position][1]] == line_index
        ):
            missing_ref_indexes.append(aligned[position][1])
            position += 1

        previous_end = new_data[-1].end if new_data else 0.0
        next_start = __next_transcribed_start(aligned, position)

        missing_words = __create_missing_words(
            missing_ref_indexes,
            previous_end,
            next_start,
            reference_words,
            line_windows[line_index],
        )
        for missing_word in missing_words:
            new_data.append(missing_word)
            line_indexes.append(line_index)
        inserted += len(missing_words)

    return new_data, line_indexes, inserted


def __next_transcribed_start(
    aligned: list[tuple[TranscribedData | None, int | None]], position: int
) -> float:
    for data, _ in aligned[position:]:
        if data is not None:
            return data.start
    return math.inf


def __create_missing_words(
    missing_ref_indexes: list[int],
    previous_end: float,
    next_start: float,
    reference_words: list[str],
    line_window: tuple[float, float],
) -> list[TranscribedData]:
    line_start, line_end = line_window
    window_start = max(previous_end + INSERTED_WORD_PADDING, line_start)
    window_end = min(next_start - INSERTED_WORD_PADDING, line_end)

    count = len(missing_ref_indexes)
    if window_end == math.inf:
        # Missing words at the very end of the song without a following line
        window_end = window_start + count * DEFAULT_INSERTED_WORD_DURATION

    if (window_end - window_start) / count < MIN_INSERTED_WORD_DURATION:
        # Not enough room to sing the missing words - skip them
        return []

    word_duration = min(
        (window_end - window_start) / count, MAX_INSERTED_WORD_DURATION
    )
    missing_words = []
    for offset, ref_index in enumerate(missing_ref_indexes):
        start = window_start + offset * word_duration
        missing_words.append(
            TranscribedData(
                word=reference_words[ref_index] + " ",
                start=start,
                end=start + word_duration,
                is_word_end=True,
            )
        )
    return missing_words


def __mark_line_ends(
    new_data: list[TranscribedData], line_indexes: list[int | None]
) -> int:
    """Set is_line_end on the last word of each LRC line.

    Words without a reference match inherit the line of the previous word so
    that kept ad-libs do not split a line.
    """
    last_known_line = None
    for position, line_index in enumerate(line_indexes):
        if line_index is None:
            line_indexes[position] = last_known_line
        else:
            last_known_line = line_index

    line_breaks = 0
    for position in range(len(new_data) - 1):
        current_line = line_indexes[position]
        next_line = line_indexes[position + 1]
        if current_line is not None and next_line is not None and current_line != next_line:
            new_data[position].is_line_end = True
            line_breaks += 1
    return line_breaks
