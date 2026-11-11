"""Move note starts onto the onsets heard in the vocal track"""

import bisect

import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData

# The forced alignment tends to start a word late, hardly ever early
SNAP_BACK_SECONDS = 0.2
SNAP_FORWARD_SECONDS = 0.05
MIN_NOTE_SECONDS = 0.05


def detect_vocal_onsets(audio_path: str, hop_length: int = 256) -> list[float]:
    """Times in seconds at which a new sound starts in the vocal track"""
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = librosa.to_mono(audio.T)
    if sample_rate != 22050:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=22050)
        sample_rate = 22050

    envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    frames = librosa.onset.onset_detect(onset_envelope=envelope, sr=sample_rate, hop_length=hop_length,
                                        backtrack=False)
    return [float(t) for t in librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)]


def snap_to_onsets(transcribed_data: list[TranscribedData], onsets: list[float]) -> list[TranscribedData]:
    """Start each note at the onset closest to its start, when one is near.

    A start moves at most ``SNAP_BACK_SECONDS`` earlier or ``SNAP_FORWARD_SECONDS``
    later, never before the start of the note before it, and the note before
    ends where the moved one starts.
    """
    if not onsets or not transcribed_data:
        return transcribed_data

    snapped = 0
    for i, data in enumerate(transcribed_data):
        low = bisect.bisect_left(onsets, data.start - SNAP_BACK_SECONDS)
        high = bisect.bisect_right(onsets, data.start + SNAP_FORWARD_SECONDS)
        if low >= high:
            continue
        target = min(onsets[low:high], key=lambda onset: abs(onset - data.start))

        if i > 0:
            previous = transcribed_data[i - 1]
            if target < previous.start + MIN_NOTE_SECONDS:
                continue
        if target > data.end - MIN_NOTE_SECONDS or target == data.start:
            continue

        if i > 0 and transcribed_data[i - 1].end > target:
            transcribed_data[i - 1].end = target
        data.start = target
        snapped += 1

    print(f"{ULTRASINGER_HEAD} Moved {blue_highlighted(str(snapped))} of "
          f"{blue_highlighted(str(len(transcribed_data)))} note starts onto an onset of the singing")
    return transcribed_data
