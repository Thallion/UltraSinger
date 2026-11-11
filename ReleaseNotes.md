# Version: next
- Changes:
  - Reference lyrics: `--lyrics_file`, a `<name> (Lyrics).txt` next to the input, or a lookup at lrclib.net (`--disable_lyrics_lookup`). The words of the lyrics replace the transcribed ones before they are timed
  - Better alignment model for German (the default one starts sung words about 0.1 s late)
  - Language detection in the first 30 s of singing, among the languages that can be aligned (no more "No default align-model for language: la")
  - Note pitch from the confidence weighted median of the pitch track, tuning offset of the recording taken out
  - Octave correction: single notes and short runs in the wrong octave are moved back onto the melody (`--disable_octave_correction`)
  - Notes without a sung pitch take the pitch of their neighbour
  - Notes end where their singing ends; note starts move onto the onsets of the vocal track (`--disable_onset_correction`)
  - Syllables start where the forced alignment hears their first letter, and stay notes of their own on the same pitch
  - Short "~" notes inside a held syllable join the note they belong to (1.30 -> 1.12 times as many notes as hand-made charts)
  - An mp3 with a cover picture is no longer taken for a video; MusicBrainz requests time out; hits that do not fit the file name are refused
  - Line breaks at rests and phrase ends, long lines are split, lines that flash by are joined
  - Key quantization is off by default and only prefers the notes of the key (`--quantize_to_key`); moving every note into the key made the charts worse
  - GPU memory of the models is given back after use (several songs in one process, 8 GB cards); the transcription is tried again with half the batch size when the memory runs out
  - Denoising of the vocals is off by default (`--denoise`)
  - Notes never get a length of 0 beats; no crash without score calculation or without any singing
  - CUDA install scripts keep the locked versions of the other packages (NumPy)
  - BPM fallback when no tempo is found, MuseScore is found on the PATH, tag values with a colon are read completely, artist and title of the input are kept when MusicBrainz finds nothing
  - `tools/chart_benchmark.py` measures generated charts against hand-made ones
  - `ultrasinger_batch.sh` converts a folder

# Version: 0.0.13dev16
Date: 2026.03.04
- Changes:
  - Download Cover from MusicBrainz
  - Fix selection of whisper model
  - Added colab notebook
  - Optimized whisper language check with confidence
  - Improved arguments, so you dont need to use true or false
  - Added interactive mode
  - Use user defined ffmpeg path
  - Show GPU Name and VRAM
  - No linebreak before the last 'E' word for UltraStar txt files
  - Download yt video/audio only once
  - Support for video as input
  - Optimise scale detection
  - Added quantization by key
  - Changed installer to uv
  - Drop crepe for SwiftF0
  - upgrade to python 3.12

# Version: 0.0.12
Date: 2024.12.19
- Changes:
  - Reduce Memory usage by clearing cache in whisper
  - Add lyrics to midi file
  - Split word by note changes
  - Upgrade UltraStar Format to Version 1.2.0 (Use of VIDEOURL)
  - Use yt with cookies
  - Some docker container improvements
  - Fix keep-cache option
  - Fix numbers in lyrics and transcribtion
  - Fix model path option
  - Fix error in PDF sheet generation
  - Fix hypen language download
  - Fix install scripts
  - Some bug fixes and improved error handling and logs

# Version: 0.0.11
Date: 2024.07.06
- Changes:
  - Better linebreak calculation
  - Remove cache folder when finished
  - Remove audio from yt video
  - Added install and start scripts
  - Added Docker support
  - Added sheet music generation

# Version: 0.0.10
Date: 2024.05.01
- Fix:
  - remove whitespace from the beginning and end of title and artist
  - image conversion to jpeg for transparent or RGBA
  - index out of range error in when list is empty from musicbrainz

# Version: 0.0.9
Date: 2024.02.06
- Fix:
  - Re-Pitch mode now re-pitch the audio again
  - Re-Pitch mode now show the text and lines in plot

# Version: 0.0.8
Date: 2024.01.03
- Changes:
  - Plot words
- Fix:
  - Missing word lines in plot

# Version: 0.0.7
Date: 2023.12.29
- Changes:
  - Added format version support for 0.3.0, 1.0.0 and 1.1.0

# Version: 0.0.6
Date: 2023.12.28
- Changes:
  - Optimized the removing of silence in transcription data
  - Mute the processing audio in parts where no singing is detected
  - Plot muted audio

# Version: 0.0.5
Date: 2023.12.23
- Changes:
  - Format GENRE string
  - Extract year from date

# Version: 0.0.4
Date: 2023.12.16
- Changes:
  - Optimized the conversion to Mono
  - Removed limitation to mp3 and wav audio formats
  - Added option float32 to demucs