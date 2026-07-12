#!/usr/bin/env python3
"""UltraSinger postprocessor v2: smooth notes, replace lyrics with reference text.

NOTE: This functionality is now built into the main UltraSinger pipeline and
enabled by default (see --disable_note_smoothing, --disable_lyrics_fetch and
--lyrics_file). This standalone script remains for postprocessing already
existing UltraStar txt files.
"""
import re
import sys
import os
import statistics
from difflib import SequenceMatcher


def load_reference_words(lyrics_file):
    """Load reference lyrics as flat list of words (preserving ~ as silences)."""
    with open(lyrics_file, "r", encoding="utf-8") as f:
        text = f.read()
    # Split into words, keep linebreak markers
    words = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            words.append('~')  # paragraph break as silence
            continue
        for w in re.split(r'\s+', line):
            w = w.strip('",.!?;:()[]\'"–—-')
            if w:
                words.append(w)
        words.append('~')  # line ending as silence
    return words


def smooth_notes(note_data):
    """Aggressive note smoothing: cap extremes, limit jumps, running average."""
    pitches = [n["pitch"] for n in note_data]
    median = statistics.median(pitches)

    # Cap: notes outside reasonable vocal range (±20 semitones from median)
    hard_lower = int(median - 20)
    hard_upper = int(median + 20)

    # First pass: hard cap
    for n in note_data:
        if n["pitch"] > hard_upper:
            n["pitch"] = hard_upper
        if n["pitch"] < hard_lower:
            n["pitch"] = hard_lower

    # Second pass: limit jump between consecutive notes to 8 semitones
    for _ in range(3):  # multiple passes for smoother result
        for i in range(1, len(note_data)):
            prev = note_data[i-1]["pitch"]
            curr = note_data[i]["pitch"]
            if abs(curr - prev) > 8:
                note_data[i]["pitch"] = prev + (8 if curr > prev else -8)

        for i in range(len(note_data) - 2, -1, -1):
            nxt = note_data[i+1]["pitch"]
            curr = note_data[i]["pitch"]
            if abs(curr - nxt) > 8:
                note_data[i]["pitch"] = nxt + (8 if curr > nxt else -8)

    # Third pass: running average with window 5
    smoothed = list(pitches)
    for _ in range(2):
        for i in range(2, len(note_data) - 2):
            window = [note_data[j]["pitch"] for j in range(i-2, i+3)]
            smoothed[i] = int(round(sum(window) / 5))
        for i in range(2, len(note_data) - 2):
            note_data[i]["pitch"] = smoothed[i]

    return note_data


def map_lyrics_to_notes(note_data, reference_words):
    """
    Map reference lyrics onto note timings.
    Keeps timing structure, silences (~), linebreaks (-) from original,
    but replaces words with reference words using sequence alignment.
    """
    # Build sentence groups from note data (separated by linebreaks '-')
    sentences = []
    current = []
    for note in note_data:
        current.append(note)
        if note["word"].strip() == '-' or (note["tag"] == '-'):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)

    # Build sentence texts from notes (only real words, not ~)
    sentence_texts = []
    for sent in sentences:
        words = [n["word"].strip() for n in sent if n["word"].strip() not in ('~', '-', '')]
        sentence_texts.append(words)

    # Build sentence texts from reference (split by '~' markers)
    ref_groups = []
    current_ref = []
    for w in reference_words:
        if w == '~':
            if current_ref:
                ref_groups.append(current_ref)
                current_ref = []
        else:
            current_ref.append(w)
    if current_ref:
        ref_groups.append(current_ref)

    # For each note sentence, find best matching reference sentence
    ref_pos = 0
    for sent_idx, sent in enumerate(sentences):
        sent_words = sentence_texts[sent_idx]
        if not sent_words:
            # Silence sentence: keep as is
            continue

        # Find best matching reference sentence near current position
        best_ratio = 0
        best_ref_idx = ref_pos
        for j in range(max(0, ref_pos - 2), min(len(ref_groups), ref_pos + 4)):
            if not ref_groups[j]:
                continue
            ref_text = ' '.join(ref_groups[j]).lower()
            sent_text = ' '.join(sent_words).lower()
            ratio = SequenceMatcher(None, ref_text, sent_text).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_ref_idx = j

        if best_ratio > 0.2:  # minimal threshold
            ref_words = ref_groups[best_ref_idx]
            ref_pos = best_ref_idx + 1

            # Distribute reference words across the sentence's non-silence notes
            note_idx = 0
            for w in ref_words:
                while note_idx < len(sent) and sent[note_idx]["word"].strip() in ('~', '-', ''):
                    note_idx += 1
                if note_idx >= len(sent):
                    break
                sent[note_idx]["word"] = w
                note_idx += 1

    return note_data


def process_txt(txt_file, lyrics_file=None):
    """Main processing function."""
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split('\n')

    note_data = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('-') and not stripped.startswith('--'):
            # Linebreak
            note_data.append({
                "line_idx": i,
                "start": 0, "duration": 0, "pitch": 0,
                "word": '-', "tag": '-',
            })
            continue

        parts = line.split(None, 4)
        if len(parts) >= 4 and parts[0] in (':', '*', 'F', 'G', 'R'):
            if parts[1].isdigit() and parts[2].isdigit():
                try:
                    note_data.append({
                        "line_idx": i,
                        "start": int(parts[1]),
                        "duration": int(parts[2]),
                        "pitch": int(parts[3]),
                        "word": parts[4].strip() if len(parts) > 4 else '~',
                        "tag": parts[0],
                    })
                except (ValueError, IndexError):
                    pass

    if not note_data:
        print("  -> Warnung: Keine Notenzeilen gefunden.")
        return

    # Step 1: Smooth notes
    note_data = smooth_notes(note_data)

    # Step 2: Replace lyrics if reference file exists
    if lyrics_file and os.path.exists(lyrics_file):
        print(f"  -> Korrigiere Lyrics mit: {os.path.basename(lyrics_file)}")
        ref_words = load_reference_words(lyrics_file)
        note_data = map_lyrics_to_notes(note_data, ref_words)

    # Reconstruct file
    new_lines = list(lines)
    for note in note_data:
        i = note["line_idx"]
        line = lines[i]

        if note["tag"] == '-':
            continue  # linebreaks unchanged

        parts = line.split(None, 4)
        if len(parts) >= 4:
            old_pitch = parts[3]
            old_word = parts[4].strip() if len(parts) > 4 else '~'

            # Replace pitch
            line = re.sub(
                rf'^({re.escape(parts[0])}\s+{re.escape(parts[1])}\s+{re.escape(parts[2])}\s+){re.escape(old_pitch)}(\s)',
                rf'\g<1>{note["pitch"]}\g<2>',
                line
            )

            # Replace word (only if it's not a silence)
            if old_word not in ('~', '-', '') and note["word"].strip() not in ('~', '-', ''):
                line = line[:line.rfind(old_word)] + note["word"] + line[line.rfind(old_word) + len(old_word):]

            new_lines[i] = line

    result = '\n'.join(new_lines)
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(result)

    pitches = [n["pitch"] for n in note_data if n["tag"] != '-']
    if pitches:
        print(f"  -> {len(pitches)} Noten geglättet (Range: {min(pitches)}–{max(pitches)})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 postprocess.py <ultrastar_file.txt> [reference_lyrics.txt]")
        sys.exit(1)

    txt_file = sys.argv[1]
    lyrics_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(txt_file):
        print(f"Error: File not found: {txt_file}")
        sys.exit(1)

    process_txt(txt_file, lyrics_file)
    print("  -> Postprocessing abgeschlossen.")
