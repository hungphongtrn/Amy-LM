import os
import torch
import json
import soundfile as sf
from datasets import load_dataset, Audio
import nemo.collections.asr as nemo_asr

# CONFIG
SRC_REPO = "fixie-ai/gigaspeech"
TGT_REPO = "hungphongtrn/speech-time-alignment"
BATCH_SIZE = 512  # Adjust based on VRAM
HF_TOKEN = os.getenv("HF_TOKEN")
AUDIO_OUTPUT_DIR = "data/audio"
SILENCE_TOKEN = "<silence>"


def preprocess_with_silence(raw_word_list):
    """
    Injects <silence> into ANY gap between words, no matter how small.
    Also adds trailing silence to "inf".
    """
    new_word_list = []
    cursor = 0.0
    
    if not raw_word_list:
        raw_word_list = []

    sorted_words = sorted(raw_word_list, key=lambda x: x['start'])

    for word in sorted_words:
        w_start = word['start']
        w_end = word['end']
        w_text = word.get('text') or word.get('word', "")

        # STRICT CONTINUITY CHECK
        # If the next word starts after the cursor, there IS silence.
        # Even if the gap is 0.00001s.
        if w_start > cursor:
            new_word_list.append({
                "text": SILENCE_TOKEN,
                "start": cursor,
                "end": w_start
            })
        
        # Add the actual word
        new_word_list.append({
            "text": w_text,
            "start": w_start,
            "end": w_end
        })

        # Update cursor to the end of the current word
        cursor = max(cursor, w_end)

    # ALWAYS add trailing silence (End of Audio -> Infinity)
    new_word_list.append({
        "text": SILENCE_TOKEN,
        "start": cursor,
        "end": -1
    })

    new_text = " ".join([w["text"] for w in new_word_list])
    
    return new_text, new_word_list


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting Alignment on {device}")
    
    # 0. Create audio output directory
    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
    print(f"📁 Audio files will be saved to: {AUDIO_OUTPUT_DIR}")
    
    # 1. Load Model
    model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
    model.eval()
    if torch.cuda.is_available():
        model = model.to(device)

    # 2. Load Source Data
    ds = load_dataset(SRC_REPO, "dev", split="dev")
    # Cast audio to 16kHz (standard for ASR models)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    
    # 3. First Pass: Save all audio files
    print("💾 Saving all audio files...")
    # def save_audio(batch):
    #     for audio, seg_id in zip(batch['audio'], batch['segment_id']):
    #         audio_path = os.path.join(AUDIO_OUTPUT_DIR, f"{seg_id}.wav")
    #         if os.path.exists(audio_path):
    #             continue
    #         sf.write(audio_path, audio['array'], audio['sampling_rate'])
    #     return {}
    
    # ds.map(save_audio, batched=True, batch_size=BATCH_SIZE)
    print(f"✅ All audio files saved to {AUDIO_OUTPUT_DIR}")
    
    # 4. Second Pass: Transcribe and extract timestamps
    print("🎤 Transcribing audio files and extracting timestamps (with SILENCE_TOKEN logic)...")
    def transcribe_batch(batch):
        # Build paths to saved audio files
        audio_paths = [os.path.join(AUDIO_OUTPUT_DIR, f"{seg_id}.wav") for seg_id in batch['segment_id']]
        
        # Run transcription with timestamps
        outputs = model.transcribe(audio_paths, timestamps=True)
        
        # Extract word-level timestamps and create alignment JSON with silence tokens
        aligned_data = []
        for output in outputs:
            raw_words = output.timestamp.get('word', [])
            # Skip items with no transcription data
            if raw_words is None or len(raw_words) == 0:
                continue
            
            # Apply STRICT silence logic
            full_text, full_words = preprocess_with_silence(raw_words)
            
            alignment = {
                "word": full_words,  # Now includes silence tokens
                "segment": output.timestamp.get('segment', []),
                "char": output.timestamp.get('char', []),
                "text": full_text,  # Updated text with silence tokens
                "original_text": output.text  # Keep original for reference
            }
            aligned_data.append(json.dumps(alignment))
        
        return {"alignment_json": aligned_data}

    # 5. Process - keep segment_id and alignment_json
    # Remove all columns except segment_id
    columns_to_remove = [col for col in ds.column_names if col not in ['segment_id']]
    ds_aligned = ds.map(
        transcribe_batch, 
        batched=True, 
        batch_size=BATCH_SIZE, 
        remove_columns=columns_to_remove
    )
    # audio_paths = [os.path.join(AUDIO_OUTPUT_DIR, f"{seg_id}.wav") for seg_id in ds['segment_id']]
    # outputs = model.transcribe(audio_paths, timestamps=True, batch_size=BATCH_SIZE)
    # aligned_data = []
    # for output in outputs:
    #     alignment = {
    #         "word": output.timestamp.get('word', []),
    #         "segment": output.timestamp.get('segment', []),
    #         "char": output.timestamp.get('char', []),
    #         "text": output.text
    #     }
    #     aligned_data.append(json.dumps(alignment))
    # ds_aligned = ds.add_column("alignment_json", aligned_data)

    # # Remove all columns except segment_id and alignment_json
    # columns_to_remove = [col for col in ds.column_names if col not in ['segment_id', 'alignment_json']]
    # ds_aligned = ds_aligned.remove_columns(columns_to_remove)

    # Push to Hub
    print("🚀 Pushing alignment data to hub...")
    ds_aligned.push_to_hub(TGT_REPO, token=HF_TOKEN)
    print("✅ Alignment Data Pushed!")

if __name__ == "__main__":
    main()

