#!/bin/bash
# UltraSinger Batch – verarbeitet alle MP4/MP3-Dateien eines Ordners
#
# Nutzung: ./ultrasinger_batch.sh /pfad/zum/ordner [weitere UltraSinger-Optionen]
#   z.B.   ./ultrasinger_batch.sh ~/songs/neu --language de
#
# - Gibt es zu einer Datei "Name.mp4" eine Textdatei "Name (Lyrics).txt" im
#   selben Ordner, nimmt UltraSinger deren Text und Zeilen (sonst wird der
#   Text bei lrclib.net gesucht; abschalten mit --disable_lyrics_lookup).
# - Existiert zu einer MP3 eine gleichnamige MP4, wird nur die MP4 verarbeitet.
# - Die Ergebnisse landen im selben Ordner. TXT-Dateien, die UltraSinger in
#   einem frueheren Lauf erzeugt hat, werden vorher entfernt; die anderen
#   Ergebnisdateien werden ueberschrieben, alles andere bleibt unberuehrt.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SONG_DIR="${1%/}"
shift || true
EXTRA_ARGS=("$@")

if [ -z "$SONG_DIR" ]; then
    echo "Fehler: Bitte einen Ordner mit MP3/MP4-Dateien angeben."
    echo "Nutzung: $0 /pfad/zum/ordner [weitere UltraSinger-Optionen]"
    exit 1
fi

if [ ! -d "$SONG_DIR" ]; then
    echo "Fehler: Ordner '$SONG_DIR' existiert nicht."
    exit 1
fi

if [ ! -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    echo "Fehler: .venv nicht gefunden. Bitte führe zuerst das Install-Skript aus."
    exit 1
fi
PYTHON="$SCRIPT_DIR/.venv/bin/python"

has_audio() {
    ffprobe -v error -select_streams a -show_entries stream=codec_type \
        -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null | grep -q .
}

is_generated_txt() {
    grep -q -m1 '^#CREATOR:UltraSinger' "$1" 2>/dev/null
}

shopt -s nullglob

# Aufraeumen: nur was UltraSinger in frueheren Laeufen erzeugt hat
for f in "$SONG_DIR"/*.txt; do
    if is_generated_txt "$f"; then
        rm -f "$f"
    fi
done
# Cover, [Vocals], [Instrumental] und .mid werden vom neuen Lauf ueberschrieben;
# sie vorher zu loeschen wuerde ein Cover kosten, wenn MusicBrainz diesmal keins findet

# Eingabedateien: MP4s mit Tonspur, dazu MP3s ohne gleichnamige MP4 mit Tonspur.
# (Eine MP4 ohne Tonspur ist das Video, das ein frueherer Lauf erzeugt hat: sie
# bleibt liegen, und das Lied wird aus der gleichnamigen MP3 neu erzeugt.)
FILES=()
declare -A HAS_MP4
for mp4 in "$SONG_DIR"/*.mp4; do
    if has_audio "$mp4"; then
        FILES+=("$mp4")
        HAS_MP4["$(basename "$mp4" .mp4)"]=1
    fi
done
for mp3 in "$SONG_DIR"/*.mp3; do
    base="$(basename "$mp3" .mp3)"
    case "$base" in *"[Vocals]"|*"[Instrumental]") continue ;; esac
    if [ -z "${HAS_MP4[$base]}" ]; then
        FILES+=("$mp3")
    fi
done
shopt -u nullglob

if [ ${#FILES[@]} -eq 0 ]; then
    echo "Keine MP3/MP4-Dateien in '$SONG_DIR' gefunden."
    exit 1
fi

echo "==> UltraSinger Batch <=="
echo "Ordner:   $SONG_DIR"
echo "Dateien:  ${#FILES[@]}"
echo "Optionen: --disable_karaoke ${EXTRA_ARGS[*]}"
echo

TEMP_OUTPUT="$(mktemp -d "${TMPDIR:-/tmp}/ultrasinger_batch.XXXXXX")"
trap 'rm -rf "$TEMP_OUTPUT"' EXIT

FAILED=()
for FILE in "${FILES[@]}"; do
    echo "--- Verarbeite: $(basename "$FILE") ---"

    if ! "$PYTHON" "$SCRIPT_DIR/src/UltraSinger.py" -i "$FILE" -o "$TEMP_OUTPUT" \
            --disable_karaoke "${EXTRA_ARGS[@]}"; then
        echo "  -> Fehler bei $(basename "$FILE"), weiter mit der naechsten Datei."
        FAILED+=("$(basename "$FILE")")
        rm -rf "$TEMP_OUTPUT"/*
        continue
    fi

    OUTPUT_SUBDIR=$(ls -td "$TEMP_OUTPUT"/*/ 2>/dev/null | head -1)
    if [ -z "$OUTPUT_SUBDIR" ]; then
        echo "  -> Warnung: Kein Output-Ordner gefunden"
        FAILED+=("$(basename "$FILE")")
        continue
    fi

    # Die Eingabedatei selbst nicht ueberschreiben
    for out in "$OUTPUT_SUBDIR"*; do
        [ -d "$out" ] && continue
        target="$SONG_DIR/$(basename "$out")"
        if [ "$target" = "$FILE" ]; then
            continue
        fi
        mv -f "$out" "$target"
    done
    rm -rf "$OUTPUT_SUBDIR"

    # Liegt zum Ergebnis ein Video im Ordner, das die TXT nicht nennt (Eingabe war
    # die MP3 eines frueheren Laufs), wird es eingetragen
    for txt in "$SONG_DIR"/*.txt; do
        is_generated_txt "$txt" || continue
        video="${txt%.txt}.mp4"
        if [ -f "$video" ] && ! grep -q '^#VIDEO:' "$txt"; then
            sed -i "0,/^#MP3:.*/s||&\n#VIDEO:$(basename "$video" | sed 's/[&|\\]/\\&/g')|" "$txt"
        fi
    done

    echo "  -> Dateien nach '$SONG_DIR' verschoben."
    echo
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "==> Fertig, aber ${#FAILED[@]} Datei(en) fehlgeschlagen:"
    printf '    %s\n' "${FAILED[@]}"
    exit 1
fi
echo "==> Alle Dateien verarbeitet! =="
