import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


FALLBACK_BPM = 120.0


def get_bpm_from_data(data, sampling_rate):
    """Get real bpm from audio data"""
    try:
        onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)
        wav_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sampling_rate)
        bpm = float(wav_tempo[0])
    except (ZeroDivisionError, ValueError, IndexError, librosa.ParameterError) as error:
        # e.g. silence or a file too short to find a tempo in
        print(f"{ULTRASINGER_HEAD} BPM detection failed ({error}); using {blue_highlighted(str(FALLBACK_BPM))}")
        return FALLBACK_BPM

    if not bpm > 0 or bpm != bpm or bpm == float("inf"):
        print(f"{ULTRASINGER_HEAD} No BPM found; using {blue_highlighted(str(FALLBACK_BPM))}")
        return FALLBACK_BPM

    print(f"{ULTRASINGER_HEAD} BPM is {blue_highlighted(str(round(bpm, 2)))}")
    return bpm


def get_bpm_from_file(wav_file: str) -> float:
    """Get real bpm from audio file"""
    data, sampling_rate = sf.read(wav_file, dtype='float32')
    # Transpose if stereo to match librosa's expected format
    if len(data.shape) > 1:
        data = data.T
    # Convert to mono if stereo
    if data.ndim > 1:
        data = librosa.to_mono(data)
    return get_bpm_from_data(data, sampling_rate)
