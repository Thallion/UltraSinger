from dataclasses import dataclass


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  is_line_end: bool = False
