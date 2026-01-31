from pathlib import Path
import csv

# -------- CONFIG --------
STEP1_CSV = Path("data/step1_rewritten.csv")
AUDIO_DIR = Path("outputs/audio")
OUT_MANIFEST = Path("data/step2_audio_manifest.csv")
# ------------------------

OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

# Index all wav files by dialog_id
wav_index = {
    wav.stem: wav.resolve().as_posix()
    for wav in AUDIO_DIR.glob("*.wav")
}

rows = []

with open(STEP1_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for item in reader:
        dialog_id = item["dialog_id"]

        audio_path = wav_index.get(dialog_id)
        success = audio_path is not None

        rows.append(
            {
                "dialog_id": dialog_id,
                "audio_path": audio_path,
                "text": item.get("rewritten_text", item.get("utterance", "")),
                "emotion": item.get("emotion", "neutral"),
                "speech_act": item.get("speech_act", "statement"),
                "intent": item.get("intent", "inform"),
                "success": success,
            }
        )

# Write manifest
with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Manifest rebuilt: {OUT_MANIFEST}")
print(f"🎧 Found audio: {sum(r['success'] for r in rows)}/{len(rows)}")
