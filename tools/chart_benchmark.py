#!/usr/bin/env python3
"""Measure generated UltraStar charts against hand-made reference charts.

Usage:
    python tools/chart_benchmark.py convert  SONGS_DIR WORK_DIR --label NAME [--filter TXT] [--limit N] [--args "..."]
    python tools/chart_benchmark.py evaluate SONGS_DIR WORK_DIR --label NAME
    python tools/chart_benchmark.py compare  WORK_DIR LABEL_A LABEL_B

convert and evaluate take --source usdb|own to use only the songs that USDB Syncer
downloaded (their folder has a .usdb file) or only the other ones.

SONGS_DIR is an UltraStar song library (one folder per song with a hand-made
txt and its audio). It is only read. Everything that is produced goes to
WORK_DIR, which must lie outside this repository.

All pitches are compared octave-folded, like the games do when scoring.
"""

import argparse
import json
import os
import re
import shlex
import statistics
import sys
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
FRAME = 0.01  # seconds
AUDIO_EXTENSIONS = (".mp3", ".ogg", ".m4a", ".aac", ".opus", ".flac", ".wav")


# --------------------------------------------------------------------------- #
# Chart parsing
# --------------------------------------------------------------------------- #

@dataclass
class Note:
    kind: str
    start: float
    end: float
    pitch: int
    text: str
    new_word: bool = True


@dataclass
class Chart:
    path: str
    tags: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    line_count: int = 1
    is_duet: bool = False
    line_starts: list = field(default_factory=list)

    @property
    def pitched(self):
        return [n for n in self.notes if n.kind in (":", "*")]


def read_text(path: str) -> str:
    with open(path, "rb") as file:
        raw = file.read()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_chart(path: str) -> Chart:
    chart = Chart(path)
    rows = []
    relative_offset = 0
    for line in read_text(path).splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            chart.tags[key.strip().upper()] = value.strip()
            continue
        if not line.strip():
            continue
        head = line[0]
        if head == "E":
            break
        if head == "P":
            if rows:
                chart.is_duet = True
            continue
        if head == "-":
            parts = line[1:].split()
            if chart.tags.get("RELATIVE", "").upper() == "YES" and parts:
                relative_offset += int(parts[-1])
            rows.append(None)
            continue
        if head in ":*FRG":
            match = re.match(r"^.\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)(?: (.*))?$", line)
            if not match:
                continue
            start, duration, pitch = (int(match.group(i)) for i in (1, 2, 3))
            rows.append((head, start + relative_offset, duration, pitch, match.group(4) or ""))

    bpm = float(chart.tags.get("BPM", "0").replace(",", "."))
    gap = float(chart.tags.get("GAP", "0").replace(",", ".")) / 1000
    if bpm <= 0:
        raise ValueError(f"No BPM in {path}")
    beat = 60 / (bpm * 4)

    new_word = True
    new_line = False
    for row in rows:
        if row is None:
            chart.line_count += 1
            new_word = True
            new_line = True
            continue
        kind, start, duration, pitch, text = row
        starts_word = new_word or text.startswith(" ")
        if new_line:
            chart.line_starts.append(gap + start * beat)
            new_line = False
        chart.notes.append(
            Note(kind, gap + start * beat, gap + (start + max(duration, 1)) * beat, pitch, text, starts_word)
        )
        new_word = text.endswith(" ")
    return chart


def is_generated(chart: Chart) -> bool:
    return "ultrasinger" in (chart.tags.get("CREATOR", "") + chart.tags.get("COMMENT", "")).lower()


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #

def find_songs(songs_dir: str) -> list[dict]:
    songs = []
    for name in sorted(os.listdir(songs_dir)):
        folder = os.path.join(songs_dir, name)
        if not os.path.isdir(folder):
            continue
        files = os.listdir(folder)
        txts = [f for f in files if f.lower().endswith(".txt")]
        source = "usdb" if any(f.lower().endswith(".usdb") for f in files) else "own"
        for txt in txts:
            try:
                chart = parse_chart(os.path.join(folder, txt))
            except (ValueError, OSError):
                continue
            if is_generated(chart) or chart.is_duet or len(chart.pitched) < 50:
                continue
            audio = chart.tags.get("AUDIO") or chart.tags.get("MP3") or ""
            audio_path = os.path.join(folder, audio.strip())
            if not audio or not os.path.isfile(audio_path) or not audio_path.lower().endswith(AUDIO_EXTENSIONS):
                continue
            songs.append({"name": name, "txt": chart.path, "audio": audio_path, "source": source,
                          "artist": chart.tags.get("ARTIST", ""), "title": chart.tags.get("TITLE", name)})
            break
    return songs


def select_songs(songs: list[dict], name_filter: str | None, limit: int | None,
                 source: str = "all") -> list[dict]:
    if source != "all":
        songs = [s for s in songs if s["source"] == source]
    if name_filter:
        wanted = [f.strip().lower() for f in name_filter.split(",") if f.strip()]
        songs = [s for s in songs if any(w in s["name"].lower() for w in wanted)]
    if limit and limit < len(songs):
        step = len(songs) / limit
        songs = [songs[int(i * step)] for i in range(limit)]
    return songs


def check_work_dir(work_dir: str) -> str:
    work_dir = os.path.abspath(work_dir)
    if os.path.commonpath([work_dir, REPO_ROOT]) == REPO_ROOT:
        sys.exit("The work directory must lie outside the repository.")
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


# --------------------------------------------------------------------------- #
# Convert
# --------------------------------------------------------------------------- #

def convert(args) -> None:
    work_dir = check_work_dir(args.work_dir)
    songs = select_songs(find_songs(args.songs_dir), args.filter, args.limit, args.source)
    print(f"Converting {len(songs)} songs as '{args.label}'")

    sys.path.insert(0, SRC_DIR)
    import UltraSinger  # pylint: disable=import-outside-toplevel
    from modules.ProcessData import ProcessData, MediaInfo  # pylint: disable=import-outside-toplevel
    from modules.DeviceDetection.device_detection import check_gpu_support  # pylint: disable=import-outside-toplevel

    settings = UltraSinger.settings
    extra_args = shlex.split(args.args or "")
    device = check_gpu_support()

    # The benchmark needs the chart only: no stems, sheets or network lookups.
    UltraSinger.convert_audio_format = lambda *a, **k: None
    UltraSinger.create_sheet = lambda *a, **k: None
    UltraSinger.print_support = lambda *a, **k: None

    failed = []
    for index, song in enumerate(songs, 1):
        out_dir = os.path.join(work_dir, "runs", args.label, song["name"])
        result_path = os.path.join(out_dir, "chart.txt")
        if os.path.isfile(result_path) and not args.force:
            continue
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== [{index}/{len(songs)}] {song['name']}")

        vars(settings).clear()
        UltraSinger.init_settings(["-i", song["audio"], "-o", out_dir] + extra_args)
        if "--force_cpu" not in extra_args:
            settings.pytorch_device = device
        settings.output_folder_path = out_dir
        settings.cache_override_path = os.path.join(work_dir, "cache", song["name"])
        settings.keep_cache = True
        settings.create_midi = False
        settings.create_karaoke = True  # keeps the separation step identical to a normal run
        settings.calculate_score = False
        for stage in args.refresh or []:
            setattr(settings, f"skip_cache_{stage}", True)

        def init_process_data(song=song, out_dir=out_dir):
            process_data = ProcessData()
            process_data.basename = "chart"
            process_data.process_data_paths.audio_output_file_path = song["audio"]
            process_data.media_info = MediaInfo(
                artist=song["artist"], title=song["title"],
                audio_extension=os.path.splitext(song["audio"])[1].lstrip("."))
            settings.output_folder_path = out_dir
            return process_data

        UltraSinger.InitProcessData = init_process_data
        try:
            UltraSinger.run()
        except (Exception, SystemExit) as error:  # pylint: disable=broad-except
            import traceback  # pylint: disable=import-outside-toplevel
            traceback.print_exc()
            failed.append((song["name"], repr(error)))
    if failed:
        print("\nFailed songs:")
        for name, error in failed:
            print(f"  {name}: {error}")


# --------------------------------------------------------------------------- #
# Evaluate
# --------------------------------------------------------------------------- #

def note_frames(notes: list, frame_count: int, offset: float = 0.0) -> list[int]:
    """Pitch class per frame, -1 where no note is"""
    frames = [-1] * frame_count
    for note in notes:
        first = max(0, int(round((note.start + offset) / FRAME)))
        last = min(frame_count, int(round((note.end + offset) / FRAME)))
        for i in range(first, last):
            frames[i] = note.pitch % 12
    return frames


def pitch_close(a: int, b: int, tolerance: int = 1) -> bool:
    diff = abs(a - b) % 12
    return min(diff, 12 - diff) <= tolerance


def load_vocal_pitch(cache_dir: str, frame_count: int) -> list[float] | None:
    """Sung pitch (fractional midi, UltraStar scale) per frame, None where unvoiced"""
    import math  # pylint: disable=import-outside-toplevel
    if not os.path.isdir(cache_dir):
        return None
    candidates = [f for f in os.listdir(cache_dir) if f.startswith("swiftf0") and f.endswith(".json")]
    if not candidates:
        return None
    with open(os.path.join(cache_dir, sorted(candidates)[0]), encoding="utf-8") as file:
        data = json.load(file)
    frames = [None] * frame_count
    for time, frequency, confidence in zip(data["times"], data["frequencies"], data["confidence"]):
        if confidence < 0.9 or frequency <= 0:
            continue
        index = int(round(time / FRAME))
        midi = 69 + 12 * math.log2(frequency / 440.0)
        for i in range(index - 1, index + 1):
            if 0 <= i < frame_count:
                frames[i] = midi
    return frames


def reference_fit(ref_notes: list, vocal: list, frame_count: int) -> tuple[float, float]:
    """Find the time offset at which the reference fits the sung pitch best"""
    best = (0.0, -1.0)
    for step in range(-40, 41):
        offset = step * 0.025
        frames = note_frames(ref_notes, frame_count, offset)
        hits = total = 0
        for i in range(0, frame_count, 2):
            if frames[i] >= 0 and vocal[i] is not None:
                total += 1
                diff = abs(vocal[i] - frames[i]) % 12
                if min(diff, 12 - diff) <= 1.0:
                    hits += 1
        fit = 100 * hits / total if total else 0.0
        # prefer the chart's own timing unless another offset is clearly better
        if fit > best[1] + (0.5 if offset != 0 else 0):
            best = (offset, fit)
    zero_frames = note_frames(ref_notes, frame_count, 0.0)
    hits = total = 0
    for i in range(0, frame_count, 2):
        if zero_frames[i] >= 0 and vocal[i] is not None:
            total += 1
            diff = abs(vocal[i] - zero_frames[i]) % 12
            hits += min(diff, 12 - diff) <= 1.0
    zero_fit = 100 * hits / total if total else 0.0
    if best[1] < zero_fit + 3.0:
        return 0.0, zero_fit
    return best


def normalize_word(word: str) -> str:
    word = unicodedata.normalize("NFKD", word.lower().replace("ß", "ss"))
    return "".join(c for c in word if c.isalnum())


def chart_words(chart: Chart) -> list[tuple[str, float]]:
    words = []
    for note in chart.notes:
        text = note.text.replace("~", "")
        if note.new_word or not words:
            words.append([text, note.start])
        else:
            words[-1][0] += text
    result = []
    for text, start in words:
        for part in text.split():
            normalized = normalize_word(part)
            if normalized:
                result.append((normalized, start))
    return result


def onset_hits(source: list[float], target: list[float], tolerance: float) -> float:
    if not source:
        return 0.0
    import bisect  # pylint: disable=import-outside-toplevel
    target = sorted(target)
    hits = 0
    for onset in source:
        i = bisect.bisect_left(target, onset)
        near = [target[j] for j in (i - 1, i) if 0 <= j < len(target)]
        if near and min(abs(onset - t) for t in near) <= tolerance:
            hits += 1
    return 100 * hits / len(source)


def octave_jumps(notes: list) -> float:
    """Jumps of 10+ semitones between neighbouring notes per 100 notes"""
    if len(notes) < 2:
        return 0.0
    jumps = sum(1 for a, b in zip(notes, notes[1:]) if abs(a.pitch - b.pitch) >= 10 and b.start - a.end < 1.0)
    return 100 * jumps / len(notes)


def evaluate_song(ref: Chart, gen: Chart, cache_dir: str) -> dict:
    end = max([n.end for n in ref.notes + gen.notes]) + 2.0
    frame_count = int(end / FRAME) + 1
    ref_pitched, gen_pitched = ref.pitched, gen.pitched

    vocal = load_vocal_pitch(cache_dir, frame_count)
    offset, fit = (0.0, None)
    if vocal is not None:
        offset, fit = reference_fit(ref_pitched, vocal, frame_count)

    ref_frames = note_frames(ref_pitched, frame_count, offset)
    gen_frames = note_frames(gen_pitched, frame_count)
    ref_any = note_frames(ref.notes, frame_count, offset)

    ref_time = sum(1 for f in ref_frames if f >= 0)
    gen_time = sum(1 for f in gen_frames if f >= 0)
    both = agree = covered = extra = 0
    for r, g, a in zip(ref_frames, gen_frames, ref_any):
        if r >= 0 and g >= 0:
            both += 1
            covered += 1
            if pitch_close(r, g):
                agree += 1
        if g >= 0 and a < 0:
            extra += 1

    agreement = 100 * agree / ref_time if ref_time else 0.0
    precision = 100 * agree / gen_time if gen_time else 0.0
    f1 = 2 * agreement * precision / (agreement + precision) if agreement + precision else 0.0

    ref_onsets = [n.start + offset for n in ref_pitched]
    gen_onsets = [n.start for n in gen_pitched]

    ref_words = [(w, t + offset) for w, t in chart_words(ref)]
    gen_words = chart_words(gen)
    matcher = SequenceMatcher(None, [w for w, _ in ref_words], [w for w, _ in gen_words], autojunk=False)
    found = 0
    start_100 = start_250 = 0
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            found += 1
            delta = abs(ref_words[block.a + k][1] - gen_words[block.b + k][1])
            start_100 += delta <= 0.1
            start_250 += delta <= 0.25

    ref_breaks = [t + offset for t in ref.line_starts]
    break_recall = onset_hits(ref_breaks, gen.line_starts, 0.5)
    break_precision = onset_hits(gen.line_starts, ref_breaks, 0.5)
    break_f1 = (2 * break_recall * break_precision / (break_recall + break_precision)
                if break_recall + break_precision else 0.0)

    # Rap: note time without a pitch to sing (rap and freestyle notes in hand-made charts).
    # Only time on which both charts have a note is compared: rap notes are short and leave gaps.
    ref_rap = note_frames([n for n in ref.notes if n.kind in "RGF"], frame_count, offset)
    gen_rap = note_frames([n for n in gen.notes if n.kind in "RGF"], frame_count)
    gen_any = note_frames(gen.notes, frame_count)
    ref_note_time = sum(1 for f in ref_any if f >= 0)
    gen_note_time = sum(1 for f in gen_any if f >= 0)
    ref_rap_time = sum(1 for f in ref_rap if f >= 0)
    gen_rap_time = sum(1 for f in gen_rap if f >= 0)
    ref_rap_shared = sum(1 for r, g in zip(ref_rap, gen_any) if r >= 0 and g >= 0)
    gen_rap_shared = sum(1 for a, g in zip(ref_any, gen_rap) if a >= 0 and g >= 0)
    rap_both = sum(1 for r, g in zip(ref_rap, gen_rap) if r >= 0 and g >= 0)

    gen_lengths = [n.end - n.start for n in gen_pitched]
    return {
        "ref_offset_ms": round(offset * 1000),
        "ref_fit_pct": fit,
        "chart_agreement_pct": agreement,
        "chart_precision_pct": precision,
        "chart_f1_pct": f1,
        "pitch_agree_pct": 100 * agree / both if both else 0.0,
        "ref_coverage_pct": 100 * covered / ref_time if ref_time else 0.0,
        "extra_time_pct": 100 * extra / gen_time if gen_time else 0.0,
        "onset_hit_50_pct": onset_hits(ref_onsets, gen_onsets, 0.05),
        "onset_hit_100_pct": onset_hits(ref_onsets, gen_onsets, 0.1),
        "onset_precision_100_pct": onset_hits(gen_onsets, ref_onsets, 0.1),
        "note_count_ratio": len(gen_pitched) / len(ref_pitched) if ref_pitched else 0.0,
        "line_count_ratio": gen.line_count / ref.line_count,
        "linebreak_f1_pct": break_f1,
        "median_note_ms": 1000 * statistics.median(gen_lengths) if gen_lengths else 0.0,
        "ref_median_note_ms": 1000 * statistics.median([n.end - n.start for n in ref_pitched]),
        "octave_jumps_per_100": octave_jumps(gen_pitched),
        "ref_octave_jumps_per_100": octave_jumps(ref_pitched),
        "lyrics_words_found_pct": 100 * found / len(ref_words) if ref_words else 0.0,
        "lyrics_precision_pct": 100 * found / len(gen_words) if gen_words else 0.0,
        "word_start_100_pct": 100 * start_100 / found if found else 0.0,
        "word_start_250_pct": 100 * start_250 / found if found else 0.0,
        "ref_rap_pct": 100 * ref_rap_time / ref_note_time if ref_note_time else 0.0,
        "rap_pct": 100 * gen_rap_time / gen_note_time if gen_note_time else 0.0,
        "rap_found_pct": 100 * rap_both / ref_rap_shared if ref_rap_shared else None,
        "rap_precision_pct": 100 * rap_both / gen_rap_shared if gen_rap_shared else None,
    }


SUMMARY_METRICS = [
    "chart_f1_pct", "chart_agreement_pct", "chart_precision_pct", "pitch_agree_pct", "ref_coverage_pct",
    "extra_time_pct", "onset_hit_50_pct", "onset_hit_100_pct", "onset_precision_100_pct", "note_count_ratio",
    "line_count_ratio", "linebreak_f1_pct", "median_note_ms", "octave_jumps_per_100", "lyrics_words_found_pct",
    "lyrics_precision_pct", "word_start_100_pct", "word_start_250_pct",
]


def evaluate(args) -> None:
    work_dir = check_work_dir(args.work_dir)
    songs = select_songs(find_songs(args.songs_dir), None, None, args.source)
    rows = {}
    for song in songs:
        gen_path = os.path.join(work_dir, "runs", args.label, song["name"], "chart.txt")
        if not os.path.isfile(gen_path):
            continue
        ref, gen = parse_chart(song["txt"]), parse_chart(gen_path)
        if not gen.pitched:
            continue
        row = evaluate_song(ref, gen, os.path.join(work_dir, "cache", song["name"]))
        row["reliable"] = row["ref_fit_pct"] is None or row["ref_fit_pct"] >= args.min_ref_fit
        row["source"] = song["source"]
        rows[song["name"]] = row

    reliable = [r for r in rows.values() if r["reliable"]]
    summary = {m: statistics.median(r[m] for r in reliable) for m in SUMMARY_METRICS} if reliable else {}
    summary_mean = {m: statistics.fmean(r[m] for r in reliable) for m in SUMMARY_METRICS} if reliable else {}
    report = {"label": args.label, "songs": len(rows), "reliable_songs": len(reliable),
              "median": summary, "mean": summary_mean, "rows": rows}
    os.makedirs(os.path.join(work_dir, "reports"), exist_ok=True)
    with open(os.path.join(work_dir, "reports", f"{args.label}.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=1, ensure_ascii=False)

    print(f"{args.label}: {len(rows)} songs, {len(reliable)} with a reliable reference")
    if args.verbose:
        for name, row in rows.items():
            flag = " " if row["reliable"] else "!"
            print(f"{flag} {name[:42]:42} f1 {row['chart_f1_pct']:5.1f} agr {row['chart_agreement_pct']:5.1f} "
                  f"prec {row['chart_precision_pct']:5.1f} on100 {row['onset_hit_100_pct']:5.1f} "
                  f"words {row['lyrics_words_found_pct']:5.1f} n {row['note_count_ratio']:4.2f} "
                  f"fit {row['ref_fit_pct'] or 0:5.1f} off {row['ref_offset_ms']:5d} "
                  f"rap {row['ref_rap_pct']:3.0f}/{row['rap_pct']:3.0f}")
    print(f"{'metric':28} {'median':>8} {'mean':>8}")
    for metric in SUMMARY_METRICS:
        if summary:
            print(f"{metric:28} {summary[metric]:8.2f} {summary_mean[metric]:8.2f}")
    ref_rap = sum(r["ref_rap_pct"] * r["rap_found_pct"] for r in reliable if r["rap_found_pct"] is not None)
    gen_rap = sum(r["rap_pct"] * r["rap_precision_pct"] for r in reliable if r["rap_precision_pct"] is not None)
    ref_total, gen_total = sum(r["ref_rap_pct"] for r in reliable), sum(r["rap_pct"] for r in reliable)
    if ref_total or gen_total:
        print(f"rap notes: {ref_rap / ref_total if ref_total else 0:.0f} % of the rap in the hand-made charts found, "
              f"{gen_rap / gen_total if gen_total else 0:.0f} % of the generated rap is rap there")


def compare(args) -> None:
    work_dir = check_work_dir(args.work_dir)
    reports = []
    for label in (args.label_a, args.label_b):
        with open(os.path.join(work_dir, "reports", f"{label}.json"), encoding="utf-8") as file:
            reports.append(json.load(file))
    a, b = reports
    common = [n for n in a["rows"] if n in b["rows"] and a["rows"][n]["reliable"] and b["rows"][n]["reliable"]]
    print(f"{len(common)} songs in both reports")
    print(f"{'metric':28} {args.label_a[:12]:>12} {args.label_b[:12]:>12} {'diff':>8} {'better':>7} {'worse':>6}")
    for metric in SUMMARY_METRICS:
        values_a = [a["rows"][n][metric] for n in common]
        values_b = [b["rows"][n][metric] for n in common]
        if not common:
            continue
        mean_a, mean_b = statistics.fmean(values_a), statistics.fmean(values_b)
        better = sum(1 for x, y in zip(values_a, values_b) if y > x + 1e-9)
        worse = sum(1 for x, y in zip(values_a, values_b) if y < x - 1e-9)
        print(f"{metric:28} {mean_a:12.2f} {mean_b:12.2f} {mean_b - mean_a:+8.2f} {better:7d} {worse:6d}")


SOURCE_HELP = "usdb: only songs downloaded by USDB Syncer (folder has a .usdb file), own: only the others"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("convert")
    p.add_argument("songs_dir")
    p.add_argument("work_dir")
    p.add_argument("--label", required=True)
    p.add_argument("--filter", help="comma separated parts of song folder names")
    p.add_argument("--limit", type=int, help="evenly spread sample of this many songs")
    p.add_argument("--args", help="extra UltraSinger arguments")
    p.add_argument("--force", action="store_true", help="convert songs again that have a result")
    p.add_argument("--source", choices=["all", "usdb", "own"], default="all", help=SOURCE_HELP)
    p.add_argument("--refresh", nargs="*",
                   choices=["vocal_separation", "denoise_vocal_audio", "transcription", "pitch_detection"],
                   help="cached stages to compute again")
    p.set_defaults(func=convert)

    p = sub.add_parser("evaluate")
    p.add_argument("songs_dir")
    p.add_argument("work_dir")
    p.add_argument("--label", required=True)
    p.add_argument("--min-ref-fit", type=float, default=50.0)
    p.add_argument("--source", choices=["all", "usdb", "own"], default="all", help=SOURCE_HELP)
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=evaluate)

    p = sub.add_parser("compare")
    p.add_argument("work_dir")
    p.add_argument("label_a")
    p.add_argument("label_b")
    p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
