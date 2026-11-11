"""Fit the notes to where the pitch detection hears singing"""

import bisect

import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.TranscribedData import TranscribedData

VOICED_CONFIDENCE = 0.5
# Unvoiced time at the end of a note that is left alone (the last consonant)
END_TOLERANCE_SECONDS = 0.03
# Unvoiced time at the start of a note that is left alone (the first consonants)
START_TOLERANCE_SECONDS = 0.12
START_LEAD_SECONDS = 0.04
MIN_NOTE_SECONDS = 0.05


def trim_to_voicing(transcribed_data: list[TranscribedData], pitched_data: PitchedData) -> list[TranscribedData]:
    """End every note where its singing ends, and start it where the singing starts.

    The forced alignment gives a word all the time until the next word, also
    the rest after it; some alignment models start a word long before it is
    sung. A note over a rest cannot be hit, so the pitch detection decides:
    the note ends with its last voiced frame and starts shortly before its
    first one. A note without any voiced frame stays as it is.
    """
    times = pitched_data.times
    if len(times) < 2:
        return transcribed_data
    voiced = np.asarray(pitched_data.confidence, dtype=float) > VOICED_CONFIDENCE
    frame = times[1] - times[0]

    trimmed = 0
    for data in transcribed_data:
        first = bisect.bisect_left(times, data.start)
        last = bisect.bisect_left(times, data.end)
        indices = np.nonzero(voiced[first:last])[0]
        if len(indices) == 0:
            continue

        changed = False
        voiced_end = times[first + int(indices[-1])] + frame
        if data.end - voiced_end > END_TOLERANCE_SECONDS:
            data.end = max(voiced_end, data.start + MIN_NOTE_SECONDS)
            changed = True

        voiced_start = times[first + int(indices[0])]
        if voiced_start - data.start > START_TOLERANCE_SECONDS:
            data.start = min(voiced_start - START_LEAD_SECONDS, data.end - MIN_NOTE_SECONDS)
            changed = True
        trimmed += changed

    print(f"{ULTRASINGER_HEAD} Fitted {blue_highlighted(str(trimmed))} notes to the singing")
    return transcribed_data
