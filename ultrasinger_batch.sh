#!/bin/bash
# UltraSinger Batch – verarbeitet MP4/MP3-Dateien in einem Ordner
# Nutzung: ./ultrasinger_batch.sh /pfad/zum/ordner
# Optionen: --disable_karaoke (keine zusaetzliche Karaoke-TXT)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SONG_DIR="${1%/}"

if [ -z "$SONG_DIR" ]; then
    echo "Fehler: Bitte einen Ordner mit MP3/MP4-Dateien angeben."
    echo "Nutzung: $0 /pfad/zum/ordner"
    exit 1
fi

if [ ! -d "$SONG_DIR" ]; then
    echo "Fehler: Ordner '$SONG_DIR' existiert nicht."
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Fehler: .venv nicht gefunden. Bitte führe zuerst das Install-Skript aus."
    exit 1
fi

cd "$SCRIPT_DIR"
source .venv/bin/activate

# Aufraeumen: generierte Dateien aus vorherigen Laeufen loeschen
# (nur UltraSinger-generierte TXTs, nicht user-eigene Lyrics-Dateien)
for f in "$SONG_DIR"/*.txt; do
    if [[ "$f" != *"(Lyrics).txt" ]] && [[ "$f" != *"(Lyrics) .txt" ]]; then
        rm -f "$f" 2>/dev/null
    fi
done
for f in "$SONG_DIR"/*"[CO].jpg" "$SONG_DIR"/*"[Vocals]."* "$SONG_DIR"/*"[Instrumental]."* "$SONG_DIR"/*.mid; do
    rm -f "$f" 2>/dev/null
done
# Audio-lose MP4s aus frueheren Laeufen entfernen
for f in "$SONG_DIR"/*.mp4; do
    if ! ffprobe -v error -select_streams a -show_entries stream=codec_type -of default=noprint_wrappers=1:nokey=1 "$f" 2>/dev/null | grep -q .; then
        rm -f "$f" 2>/dev/null
    fi
done

shopt -s nullglob
MP4S=("$SONG_DIR"/*.mp4)
MP3S=("$SONG_DIR"/*.mp3)
shopt -u nullglob

declare -A HAS_MP4
for mp4 in "${MP4S[@]}"; do
    base="$(basename "$mp4" .mp4)"
    HAS_MP4["$base"]=1
done

for mp3 in "${MP3S[@]}"; do
    base="$(basename "$mp3" .mp3)"
    if [ -z "${HAS_MP4[$base]}" ]; then
        MP4S+=("$mp3")
    fi
done

if [ ${#MP4S[@]} -eq 0 ]; then
    echo "Keine neuen MP3/MP4-Dateien in '$SONG_DIR' gefunden."
    exit 1
fi

echo "==> UltraSinger Batch <=="
echo "Ordner: $SONG_DIR"
echo "Dateien: ${#MP4S[@]}"
echo "Optionen: --disable_karaoke"
echo

TEMP_OUTPUT="/tmp/ultrasinger_batch"
rm -rf "$TEMP_OUTPUT"
mkdir -p "$TEMP_OUTPUT"

for FILE in "${MP4S[@]}"; do
    echo "--- Verarbeite: $(basename "$FILE") ---"

    python "$SCRIPT_DIR/src/UltraSinger.py" \
        -i "$FILE" \
        -o "$TEMP_OUTPUT" \
        --disable_karaoke

    OUTPUT_SUBDIR=$(ls -td "$TEMP_OUTPUT"/*/ 2>/dev/null | head -1)

    if [ -z "$OUTPUT_SUBDIR" ]; then
        echo "  -> Warnung: Kein Output-Ordner gefunden"
        continue
    fi

    mv "$OUTPUT_SUBDIR"/* "$SONG_DIR/" 2>/dev/null || true
    rm -rf "$OUTPUT_SUBDIR"

    # Postprocessing: Noten glätten + optionale Lyrics-Korrektur
    TXT_FILE=$(ls -t "$SONG_DIR"/*.txt 2>/dev/null | grep -v "(Lyrics)" | head -1)
    if [ -n "$TXT_FILE" ]; then
        LYRICS_FILE=$(find "$SONG_DIR" -maxdepth 1 -name "* (Lyrics).txt" -print -quit 2>/dev/null)
        if [ -n "$LYRICS_FILE" ] && [ -f "$LYRICS_FILE" ]; then
            python "$SCRIPT_DIR/postprocess.py" "$TXT_FILE" "$LYRICS_FILE"
        else
            python "$SCRIPT_DIR/postprocess.py" "$TXT_FILE"
        fi
    fi

    echo "  -> Dateien nach '$SONG_DIR' verschoben."
    echo "  -> Fertig."
    echo
done

rm -rf "$TEMP_OUTPUT"
echo "==> Alle Dateien verarbeitet! =="
