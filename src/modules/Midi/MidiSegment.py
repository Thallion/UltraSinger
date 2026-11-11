from dataclasses import dataclass


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  voiced: bool = True  # False when no sung pitch was found for the note
