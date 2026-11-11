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


@dataclass
class LyricWord:
    text: str
    line_end: bool = False


def normalize(word: str) -> str:
    """A word without case, accents and punctuation, for comparing"""
    word = unicodedata.normalize("NFKD", word.lower().replace("ß", "ss"))
    return "".join(char for char in word if char.isalnum())


def parse_lyrics(text: str) -> list[LyricWord]:
    """Words of a lyrics text with the ends of its lines.

    Timestamps of synced lyrics, section labels like ``[Chorus]`` and empty
    lines are left out.
    """
    words = []
    for line in text.splitlines():
        line = _LRC_TIMESTAMP.sub("", line).strip()
        if not line or _SECTION_LABEL.match(line):
            continue
        line_words = [word for word in line.split() if normalize(word)]
        if not line_words:
            continue
        for word in line_words:
            words.append(LyricWord(word))
        words[-1].line_end = True
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
                text = entry.get("plainLyrics") or entry.get("syncedLyrics")
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


def correct_segments(segments: list[dict], lyric_words: list[LyricWord]) -> tuple[list[dict], list[bool]] | None:
    """Replace the text of the transcription segments with the words of the lyrics.

    Args:
        segments: Segments of the transcription (``text``, ``start``, ``end``) before the alignment
        lyric_words: Words of the reference lyrics

    Returns:
        The segments with the corrected text and, for every word of them in
        order, whether a line of the lyrics ends with it. None when the lyrics
        do not fit the transcription.
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

    # (text, line end, segment index) of every word of the result
    result: list[tuple[str, bool, int]] = []
    replaced = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                lyric = lyric_words[j1 + k]
                result.append((lyric.text, lyric.line_end, transcribed[i1 + k][1]))
        elif tag == "replace":
            # the words of the lyrics share the place of the transcribed ones
            for k in range(j1, j2):
                source = i1 + (k - j1) * (i2 - i1) // (j2 - j1)
                result.append((lyric_words[k].text, lyric_words[k].line_end, transcribed[source][1]))
            replaced += j2 - j1
        elif tag == "delete":
            if i2 - i1 >= KEEP_UNMATCHED_WORDS:
                for k in range(i1, i2):
                    result.append((transcribed[k][0], False, transcribed[k][1]))
        elif tag == "insert":
            if j2 - j1 <= MAX_INSERTED_WORDS:
                index = transcribed[i1 - 1][1] if i1 > 0 else transcribed[0][1]
                for k in range(j1, j2):
                    result.append((lyric_words[k].text, lyric_words[k].line_end, index))
                replaced += j2 - j1

    corrected = []
    line_ends = []
    for index, segment in enumerate(segments):
        words = [(text, line_end) for text, line_end, segment_index in result if segment_index == index]
        if not words:
            continue
        new_segment = dict(segment)
        new_segment["text"] = " ".join(text for text, _ in words)
        corrected.append(new_segment)
        line_ends.extend(line_end for _, line_end in words)

    print(f"{ULTRASINGER_HEAD} Using the reference lyrics: {blue_highlighted(f'{ratio:.0%}')} of the words "
          f"fit the transcription, {blue_highlighted(str(replaced))} words corrected")
    return corrected, line_ends
