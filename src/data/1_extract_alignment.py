import os
import torch
import json
import tempfile
from datasets import load_dataset, Dataset, Features, Value, Sequence
from nemo.collections.asr.models import ASRModel

# CONFIG
SRC_REPO = "speechcolab/gigaspeech"
TGT_REPO = "hungphongtrn/speech-time-alignment"
BATCH_SIZE = 16  # Adjust based on VRAM
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    print(f"🚀 Starting Alignment on {torch.cuda.get_device_name(0)}")
    
    # 1. Load NeMo Model (Parakeet TDT)
    # Automatically downloads the model
    model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3").cuda()
    model.eval()

    # 2. Load Source Data
    # using "validation" split as the dev subset
    ds = load_dataset(SRC_REPO, "dev", split="validation", token=HF_TOKEN)
    
    # 3. Batch Processing Function
    def align_batch(batch):
        audio_paths = []
        ids = batch['segment_id']
        
        # Save temp files for NeMo (it expects file paths)
        temp_dir = tempfile.mkdtemp()
        for i, audio_array in enumerate(batch['audio']):
            # Assuming audio is decoded by HF to {'array': ..., 'sampling_rate': ...}
            # We need to save it as wav for NeMo
            import soundfile as sf
            path = os.path.join(temp_dir, f"{ids[i]}.wav")
            sf.write(path, audio_array['array'], audio_array['sampling_rate'])
            audio_paths.append(path)

        # Run Transcription with Timestamps
        # Parakeet TDT v3 supports native timestamp generation
        with torch.no_grad():
             outputs = model.transcribe(paths2audio_files=audio_paths, batch_size=len(audio_paths), timestamps=True)
        
        aligned_data = []
        # outputs is a list of objects with .text and .timestamp attributes
        for out in outputs:
            # out.timestamp['word'] contains list of {'word': ..., 'start': ..., 'end': ...}
            word_timestamps = out.timestamp['word']
            aligned_data.append(json.dumps(word_timestamps)) # Store as JSON string
            
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return {"alignment_json": aligned_data}

    # 4. Process
    # We use batching to maximize throughput
    ds_aligned = ds.map(align_batch, batched=True, batch_size=BATCH_SIZE, remove_columns=["audio"])

    # 5. Push
    ds_aligned.push_to_hub(TGT_REPO, token=HF_TOKEN)
    print("✅ Alignment Data Pushed!")

if __name__ == "__main__":
    main()

