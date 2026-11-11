from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class TranscribedData:
    """Transcribed data from json file"""

    confidence: float = 0
    word: str = ""
    start: float = 0
    end: float = 0
    is_hyphen: bool = False
    is_word_end: bool = True
    is_segment_end: bool = False  # Last word of a segment (sentence) of the transcription
    # Start time of each character of the word from the forced alignment (None where unknown)
    char_starts: Optional[list[Optional[float]]] = None


def from_whisper(whisper_dict) -> TranscribedData:
    transcribed_data = TranscribedData()
    if "score" in whisper_dict:
        transcribed_data.confidence = whisper_dict["score"]
    if "word" in whisper_dict:
        transcribed_data.word = whisper_dict["word"]
    if "start" in whisper_dict:
        transcribed_data.start = whisper_dict["start"]
    if "end" in whisper_dict:
        transcribed_data.end = whisper_dict["end"]
    return transcribed_data
