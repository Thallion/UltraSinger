"""Align reference lyrics (e.g. official lyrics from the web) with transcribed words"""

import re
from difflib import SequenceMatcher

import Levenshtein

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted,
)

# Below this whole-song similarity the reference lyrics are considered to be
# the wrong song (or a very different version) and no correction is applied
MIN_ALIGNMENT_RATIO = 0.5

# Lines like "[Chorus]" or "(Bridge)" that are section markers, not lyrics
_SECTION_MARKER_PATTERN = re.compile(r"^\s*[\[(].*[\])]\s*$")
_EDGE_PUNCTUATION_PATTERN = re.compile(r"^\W+|\W+$", re.UNICODE)


def normalize_word(word: str) -> str:
    """Normalize a word for comparison: lowercase, unified apostrophes, no edge punctuation"""
    word = word.strip().lower().replace("’", "'").replace("`", "'")
    return _EDGE_PUNCTUATION_PATTERN.sub("", word)


def split_lyrics_into_words(lyrics_text: str) -> list[str]:
    """Split raw lyrics text into a flat list of words, skipping section markers"""
    words = []
    for line in lyrics_text.splitlines():
        if _SECTION_MARKER_PATTERN.match(line):
            continue
        for word in line.split():
            word = word.strip("\"“”«»()[]")
            if normalize_word(word):
                words.append(word)
    return words


def match_word_sequences(
    transcribed_words: list[str],
    reference_words: list[str],
    use_levenshtein: bool = False,
) -> tuple[float, list[tuple[str, int, int, int, int]]]:
    """Align two word sequences, returning the similarity ratio and opcodes.

    Both matchers return difflib-style opcodes and a comparable ratio.
    difflib's SequenceMatcher is the default; Levenshtein is faster on very
    long songs.
    """
    if use_levenshtein:
        return (
            Levenshtein.ratio(transcribed_words, reference_words),
            Levenshtein.opcodes(transcribed_words, reference_words),
        )
    matcher = SequenceMatcher(
        None, transcribed_words, reference_words, autojunk=False
    )
    return matcher.ratio(), matcher.get_opcodes()


def align_lyrics_to_transcription(
    transcribed_data, reference_lyrics: str, use_levenshtein: bool = False
) -> int:
    """Correct transcribed words with words from the reference lyrics.

    The two word sequences are aligned with a sequence matcher. Words that
    differ are replaced by the reference word; timings are always kept.
    Extra reference words (missed by transcription) are skipped, since they
    have no timing. Returns the number of corrected words, or -1 when the
    reference lyrics do not match the transcription well enough.
    """
    reference_words = split_lyrics_into_words(reference_lyrics)
    if not reference_words or not transcribed_data:
        return -1

    transcribed_normalized = [normalize_word(data.word) for data in transcribed_data]
    reference_normalized = [normalize_word(word) for word in reference_words]

    ratio, opcodes = match_word_sequences(
        transcribed_normalized, reference_normalized, use_levenshtein
    )
    if ratio < MIN_ALIGNMENT_RATIO:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Reference lyrics differ too much from the transcription. Skipping lyrics correction.')}"
        )
        return -1

    corrections = 0
    for tag, trans_start, trans_end, ref_start, ref_end in opcodes:
        if tag not in ("equal", "replace"):
            # 'insert': reference words without sung/timed counterpart - skip.
            # 'delete': transcribed words not in the reference (e.g. ad-libs) - keep.
            continue

        pair_count = min(trans_end - trans_start, ref_end - ref_start)
        for offset in range(pair_count):
            transcribed = transcribed_data[trans_start + offset]
            reference_word = reference_words[ref_start + offset]

            # Preserve the trailing space that marks a word ending
            has_trailing_space = str(transcribed.word).endswith(" ")
            new_word = reference_word + (" " if has_trailing_space else "")

            if transcribed.word != new_word:
                if normalize_word(transcribed.word) != normalize_word(reference_word):
                    corrections += 1
                transcribed.word = new_word

    print(
        f"{ULTRASINGER_HEAD} Lyrics correction: {blue_highlighted(str(corrections))} word(s) corrected with official lyrics"
    )
    return corrections
