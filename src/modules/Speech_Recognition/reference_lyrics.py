"""Use known lyrics of a song instead of the words the transcription heard

The transcription tells when which passage is sung, the reference lyrics tell
the right words. The words of the lyrics replace the transcribed ones before
the forced alignment, so the alignment times the right text.
"""

import json
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted

LRCLIB_URL = "https://lrclib.net/api"
REQUEST_TIMEOUT_SECONDS = 10

# Lyrics that share a smaller part of their words with the transcription belong to another song or version
MIN_MATCH_RATIO = 0.5
# Lyrics the user gave are the lyrics of the song; only a text that has nothing to do with it is refused
MIN_MATCH_RATIO_GIVEN_LYRICS = 0.2
# Transcribed words without a counterpart in the lyrics are kept from this many in a row on
# (a passage the lyrics lack); fewer are words the transcription made up
KEEP_UNMATCHED_WORDS = 8
# At most this many words of the lyrics in a row are added where the transcription has none
MAX_INSERTED_WORDS = 6

_SECTION_LABEL = re.compile(r"^\s*[\[(][^\])]*[\])]\s*$")
_LRC_TIMESTAMP = re.compile(r"\[\d+:\d+(?:[.:]\d+)?\]")
_LRC_LINE_TIME = re.compile(r"^\[(\d+):(\d+(?:[.:]\d+)?)\]")


@dataclass
class LyricWord:
    text: str
    line_end: bool = False
    # Synced lyrics: when the line of the word starts and when the next line starts (seconds)
    line_start: float | None = None
    line_stop: float | None = None


def normalize(word: str) -> str:
    """A word without case, accents and punctuation, for comparing"""
    word = unicodedata.normalize("NFKD", word.lower().replace("ß", "ss"))
    return "".join(char for char in word if char.isalnum())


def _line_time(line: str) -> float | None:
    """Start time of a line of synced lyrics ("[01:23.45] text"), None for a plain line"""
    match = _LRC_LINE_TIME.match(line.strip())
    if match is None:
        return None
    return int(match.group(1)) * 60 + float(match.group(2).replace(":", "."))


def parse_lyrics(text: str) -> list[LyricWord]:
    """Words of a lyrics text with the ends of its lines.

    Section labels like ``[Chorus]`` and empty lines are left out. The
    timestamps of synced lyrics (LRC) are kept as the times of the lines; a
    timestamp without text ends the line before it.
    """
    words = []
    previous_line = []
    for line in text.splitlines():
        time = _line_time(line)
        line = _LRC_TIMESTAMP.sub("", line).strip()
        if time is not None:
            for word in previous_line:
                if word.line_stop is None:
                    word.line_stop = time
        if not line or _SECTION_LABEL.match(line):
            if time is not None:
                previous_line = []
            continue
        line_words = [LyricWord(word, line_start=time) for word in line.split() if normalize(word)]
        if not line_words:
            continue
        line_words[-1].line_end = True
        words.extend(line_words)
        previous_line = line_words
    return words


def read_lyrics_file(path: str) -> list[LyricWord]:
    with open(path, "rb") as file:
        raw = file.read()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return parse_lyrics(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return parse_lyrics(raw.decode("utf-8", errors="replace"))


def find_lyrics_file(input_file_path: str) -> str | None:
    """A lyrics file lying next to the input file: ``<name> (Lyrics).txt`` or ``<name>.lyrics.txt``"""
    if not input_file_path or input_file_path.startswith("https:"):
        return None
    folder = os.path.dirname(os.path.abspath(input_file_path))
    name = os.path.splitext(os.path.basename(input_file_path))[0]
    for candidate in (f"{name} (Lyrics).txt", f"{name}.lyrics.txt"):
        path = os.path.join(folder, candidate)
        if os.path.isfile(path):
            return path
    return None


def _request_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "UltraSinger (https://github.com/rakuri255/UltraSinger)"})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


_BRACKETS_AT_END = re.compile(r"\s*[\[(][^\])]*[\])]\s*$")
_FEATURING = re.compile(r"\s+(feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)
MAX_LYRICS_CANDIDATES = 6


def clean_name(name: str) -> str:
    """An artist or title without additions like "(Album Version)" or "feat. X" at its end"""
    cleaned = name
    while cleaned and _BRACKETS_AT_END.search(cleaned):
        cleaned = _BRACKETS_AT_END.sub("", cleaned)
    cleaned = _FEATURING.sub("", cleaned).strip()
    return cleaned or name


def fetch_lyrics_from_lrclib(artist: str | None, title: str | None) -> list[list[LyricWord]]:
    """Look the lyrics of a song up at lrclib.net.

    Returns the lyrics of the entries found, best guess first; which of them
    fits the song shows only when they are compared with the transcription
    (see choose_lyrics). Empty when there are none or the lookup fails.
    """
    if not artist or not title:
        return []
    clean_artist, clean_title = clean_name(artist), clean_name(title)
    queries = [
        ("get", {"artist_name": artist, "track_name": title}),
        ("search", {"artist_name": clean_artist, "track_name": clean_title}),
        ("search", {"q": f"{clean_artist} {clean_title}"}),
        ("search", {"track_name": clean_title}),
    ]

    candidates = []
    seen = set()
    try:
        for endpoint, parameters in queries:
            try:
                found = _request_json(f"{LRCLIB_URL}/{endpoint}?{urllib.parse.urlencode(parameters)}")
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                continue
            for entry in found if isinstance(found, list) else [found]:
                if not isinstance(entry, dict) or entry.get("instrumental"):
                    continue
                # synced lyrics have the times of the lines, which place lines the transcription missed
                text = entry.get("syncedLyrics") or entry.get("plainLyrics")
                words = parse_lyrics(text) if text else []
                key = " ".join(normalize(word.text) for word in words)
                if words and key not in seen:
                    seen.add(key)
                    candidates.append(words)
            if len(candidates) >= MAX_LYRICS_CANDIDATES:
                break
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Lyrics lookup failed:')} {error}")
    return candidates[:MAX_LYRICS_CANDIDATES]


def matched_words(transcribed_words: list[str], lyric_words: list[LyricWord]) -> int:
    """Number of words that the transcription and the lyrics have in common, in the same order"""
    if not transcribed_words or not lyric_words:
        return 0
    matcher = SequenceMatcher(None, [normalize(word) for word in transcribed_words],
                              [normalize(word.text) for word in lyric_words], autojunk=False)
    return sum(block.size for block in matcher.get_matching_blocks())


def match_ratio(transcribed_words: list[str], lyric_words: list[LyricWord]) -> float:
    """Share of the words that the transcription and the lyrics have in common, in the same order.

    Counted against the shorter of the two: the transcription may hold a
    passage the lyrics lack, and the lyrics one the transcription missed.
    """
    if not transcribed_words or not lyric_words:
        return 0.0
    return matched_words(transcribed_words, lyric_words) / min(len(transcribed_words), len(lyric_words))


def choose_lyrics(segments: list[dict], candidates: list[list[LyricWord]],
                  min_match_ratio: float = MIN_MATCH_RATIO) -> list[LyricWord] | None:
    """The lyrics that fit the transcription best, None when none of them fits.

    The best are those with the most words in common with the transcription:
    lyrics that only hold a part of the song fit as well by their ratio, but
    cover fewer of the words.
    """
    transcribed = [word for segment in segments for word in segment["text"].split()]
    fitting = [lyric_words for lyric_words in candidates
               if match_ratio(transcribed, lyric_words) >= min_match_ratio]
    if not fitting:
        if candidates:
            best_ratio = max(match_ratio(transcribed, lyric_words) for lyric_words in candidates)
            print(f"{ULTRASINGER_HEAD} {red_highlighted('The lyrics do not fit the song')} "
                  f"(they share only {best_ratio:.0%} of their words with the transcription); "
                  f"keeping the transcription")
        return None
    return max(fitting, key=lambda lyric_words: matched_words(transcribed, lyric_words))


def whole_timed_lines(lyric_words: list[LyricWord], first: int, stop: int) -> set[int]:
    """Indexes of the words in ``first:stop`` whose whole line lies in that range and has a start time"""
    indexes = set()
    line = []
    for index in range(first, stop):
        starts_line = index == 0 or lyric_words[index - 1].line_end
        if starts_line:
            line = []
        elif not line:
            continue  # the line began before the range
        line.append(index)
        if lyric_words[index].line_end:
            if lyric_words[index].line_start is not None:
                indexes.update(line)
            line = []
    return indexes


def correct_segments(segments: list[dict],
                     lyric_words: list[LyricWord]) -> tuple[list[dict], list[bool], list[int | None]] | None:
    """Replace the text of the transcription segments with the words of the lyrics.

    Args:
        segments: Segments of the transcription (``text``, ``start``, ``end``) before the alignment
        lyric_words: Words of the reference lyrics

    Returns:
        The segments with the corrected text and, for every word of them in
        order, whether a line of the lyrics ends with it and which word of
        the lyrics it is (None for a kept transcribed word). None when there
        is nothing to correct. Whole lines of synced lyrics that the
        transcription lacks are left out: plan_missed_lines places them.
    """
    transcribed = []  # (word, segment index)
    for index, segment in enumerate(segments):
        for word in segment["text"].split():
            transcribed.append((word, index))
    if not transcribed or not lyric_words:
        return None

    ratio = match_ratio([word for word, _ in transcribed], lyric_words)

    matcher = SequenceMatcher(None, [normalize(word) for word, _ in transcribed],
                              [normalize(word.text) for word in lyric_words], autojunk=False)

    # (text, line end, segment index, index in the lyrics) of every word of the result
    result: list[tuple[str, bool, int, int | None]] = []
    replaced = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                lyric = lyric_words[j1 + k]
                result.append((lyric.text, lyric.line_end, transcribed[i1 + k][1], j1 + k))
        elif tag == "replace":
            # the words of the lyrics share the place of the transcribed ones
            for k in range(j1, j2):
                source = i1 + (k - j1) * (i2 - i1) // (j2 - j1)
                result.append((lyric_words[k].text, lyric_words[k].line_end, transcribed[source][1], k))
            replaced += j2 - j1
        elif tag == "delete":
            if i2 - i1 >= KEEP_UNMATCHED_WORDS:
                for k in range(i1, i2):
                    result.append((transcribed[k][0], False, transcribed[k][1], None))
        elif tag == "insert":
            missed_lines = whole_timed_lines(lyric_words, j1, j2)
            added = [k for k in range(j1, j2) if k not in missed_lines]
            if 0 < len(added) <= MAX_INSERTED_WORDS:
                index = transcribed[i1 - 1][1] if i1 > 0 else transcribed[0][1]
                for k in added:
                    result.append((lyric_words[k].text, lyric_words[k].line_end, index, k))
                replaced += len(added)

    corrected = []
    line_ends = []
    lyric_indexes = []
    for index, segment in enumerate(segments):
        words = [entry for entry in result if entry[2] == index]
        if not words:
            continue
        new_segment = dict(segment)
        new_segment["text"] = " ".join(entry[0] for entry in words)
        corrected.append(new_segment)
        line_ends.extend(entry[1] for entry in words)
        lyric_indexes.extend(entry[3] for entry in words)

    print(f"{ULTRASINGER_HEAD} Using the reference lyrics: {blue_highlighted(f'{ratio:.0%}')} of the words "
          f"fit the transcription, {blue_highlighted(str(replaced))} words corrected")
    return corrected, line_ends, lyric_indexes


# Synced lyrics place a line only when their times fit the song around it: the
# nearest lines found in both have to lie this close (seconds) around one common offset
MAX_OFFSET_SPREAD_SECONDS = 0.75
OFFSET_NEIGHBOURS = 3  # lines on each side
MIN_LINES_FOR_OFFSET = 3
WORD_BEFORE_MIN_SECONDS = 0.3
# A missed line is searched from this long before its time on
MISSED_LINE_LEAD_SECONDS = 0.3
# ... for at most this long per word, plus a fixed time
MISSED_LINE_SECONDS_PER_WORD = 1.2
MISSED_LINE_EXTRA_SECONDS = 1.5
MISSED_LINE_MIN_SECONDS_PER_WORD = 0.15
# A line whose words were timed further than this from the time of the line is misplaced
MISPLACED_LINE_SECONDS = 3.0


def plan_missed_lines(lyric_words: list[LyricWord], placed: dict[int, tuple[float, float]],
                      audio_seconds: float) -> tuple[list[tuple[dict, list[int]]], set[int]]:
    """Where to look for the lines of synced lyrics that the transcription did not hear.

    A line counts as not heard as well when its words were timed far from the
    time of the line: the transcription then lacked the line, and its words
    were squeezed in next to another line.

    Args:
        lyric_words: Words of the lyrics
        placed: Start and end time of the words of the lyrics that were timed, by their index
        audio_seconds: Length of the audio

    Returns:
        For each missed line a segment (``text``, ``start``, ``end``) for the
        forced alignment and the indexes of its words; and the indexes of the
        timed words that were misplaced and have to go. A line stays as it is
        where the times of the lyrics do not fit the song around it.
    """
    lines = []
    line = []
    for index, word in enumerate(lyric_words):
        line.append(index)
        if word.line_end:
            lines.append(line)
            line = []
    if line:
        lines.append(line)
    timed_lines = [line for line in lines if lyric_words[line[0]].line_start is not None]

    # How far each line that was timed lies from its time in the lyrics. The offset
    # may change within the song (a video with an interlude the album version lacks),
    # so a line is compared with the lines around it, not with the whole song.
    offsets = {}
    for number, line in enumerate(timed_lines):
        timed = [index for index in line if index in placed]
        if timed:
            offsets[number] = placed[timed[0]][0] - lyric_words[line[0]].line_start

    def local_offset(number: int, known: dict[int, float]) -> float | None:
        """Median offset of the nearest lines around a line, None when they do not agree"""
        before = [known[k] for k in sorted(known) if k < number][-OFFSET_NEIGHBOURS:]
        after = [known[k] for k in sorted(known) if k > number][:OFFSET_NEIGHBOURS]
        near = sorted(before + after)
        if len(near) < MIN_LINES_FOR_OFFSET:
            return None
        median = near[len(near) // 2]
        spread = sorted(abs(value - median) for value in near)[len(near) // 2]
        if spread > MAX_OFFSET_SPREAD_SECONDS:
            return None
        return median

    # A misplaced line hides the next one, as it is one of its neighbours: look
    # again without the lines found, until no more turn up
    good_offsets = dict(offsets)
    while True:
        found = {number for number in good_offsets
                 if (expected := local_offset(number, good_offsets)) is not None
                 and abs(good_offsets[number] - expected) > MISPLACED_LINE_SECONDS}
        if not found:
            break
        # the worst one first: taking all at once would also take good lines next to bad ones
        worst = max(found, key=lambda number: abs(good_offsets[number] - local_offset(number, good_offsets)))
        del good_offsets[worst]

    misplaced = set()
    missed = []  # (line, offset)
    for number, line in enumerate(timed_lines):
        if number in good_offsets:
            continue
        offset = local_offset(number, good_offsets)
        if offset is None:
            continue  # nothing reliable around the line: it stays as it is
        misplaced.update(index for index in line if index in placed)
        missed.append((line, offset))
    if not missed:
        return [], set()

    placed = {index: times for index, times in placed.items() if index not in misplaced}
    placed_indexes = sorted(placed)
    plan = []
    for line, offset in missed:
        first = lyric_words[line[0]]
        start = first.line_start + offset - MISSED_LINE_LEAD_SECONDS
        longest = MISSED_LINE_LEAD_SECONDS + MISSED_LINE_SECONDS_PER_WORD * len(line) + MISSED_LINE_EXTRA_SECONDS
        stop = min(start + longest, audio_seconds)
        if first.line_stop is not None:
            stop = min(stop, first.line_stop + offset)

        # never over words that have their time already; the word before may be cut
        # short, as the alignment lets the last word before a rest run on through it
        before = [index for index in placed_indexes if index < line[0]]
        after = [index for index in placed_indexes if index > line[-1]]
        if before:
            start = max(start, min(placed[before[-1]][1], placed[before[-1]][0] + WORD_BEFORE_MIN_SECONDS))
        if after:
            stop = min(stop, placed[after[0]][0])
        if stop - start < MISSED_LINE_MIN_SECONDS_PER_WORD * len(line):
            continue
        text = " ".join(lyric_words[index].text for index in line)
        plan.append(({"text": text, "start": round(max(0.0, start), 3), "end": round(stop, 3)}, line))
        # the next missed line must not lie over this one
        placed[line[-1]] = (start, stop)
        placed_indexes = sorted(placed)
    return plan, misplaced
