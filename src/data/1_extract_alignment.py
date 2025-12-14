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
    print("🎤 Transcribing audio files and extracting timestamps...")
    def transcribe_batch(batch):
        # Build paths to saved audio files
        audio_paths = [os.path.join(AUDIO_OUTPUT_DIR, f"{seg_id}.wav") for seg_id in batch['segment_id']]
        
        # Run transcription with timestamps
        outputs = model.transcribe(audio_paths, timestamps=True)
        
        # Extract word-level timestamps and create alignment JSON
        aligned_data = []
        for output in outputs:
            if output.timestamp.get('word') is None:
                continue
            
            alignment = {
                "word": output.timestamp.get('word', []),
                "segment": output.timestamp.get('segment', []),
                "char": output.timestamp.get('char', []),
                "text": output.text
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

