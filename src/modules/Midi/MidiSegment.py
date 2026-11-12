from dataclasses import dataclass


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  voiced: bool = True  # False when no sung pitch was found for the note
  pitch: float | None = None  # Sung pitch in midi notes before rounding (tuning offset taken out)
  rap: bool = False  # The note is rapped: it is written as a rap note, which is scored without its pitch
