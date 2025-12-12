import torch
import json
import numpy as np
from datasets import load_dataset, Features, Value, Array2D
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# CONFIG
ALIGN_REPO = "hungphongtrn/speech-time-alignment"
TGT_REPO = "hungphongtrn/llm-features"
BATCH_SIZE = 8
MODEL_ID = "Qwen/Qwen3-1.7B"

# Set up logging to catch errors without stopping execution
logging.basicConfig(filename="extraction_errors.log", level=logging.ERROR)


def get_token_timestamps(token_ids, word_timings, tokenizer):
    """
    Aligns Qwen tokens to NeMo word-level timestamps using a Shared Window strategy.
    """
    token_strs = [
        tokenizer.decode([t], clean_up_tokenization_spaces=False) for t in token_ids
    ]

    aligned_times = []
    word_idx = 0
    current_word_chars_left = 0
    current_word_start = 0.0
    current_word_end = 0.0

    for tok_str in token_strs:
        clean_tok = tok_str.strip()
        tok_len = len(clean_tok)

        # CASE 1: Empty/Special Token -> Attach to previous
        if tok_len == 0:
            last_end = aligned_times[-1][1] if aligned_times else 0.0
            aligned_times.append([last_end, last_end])
            continue

        # CASE 2: Load Next Word if budget exhausted
        if current_word_chars_left <= 0:
            if word_idx < len(word_timings):
                w_info = word_timings[word_idx]
                current_word_start = w_info["start"]
                current_word_end = w_info["end"]

                # Robust key access
                ref_text = w_info.get("word") or w_info.get("text") or clean_tok
                current_word_chars_left = len(ref_text)
                word_idx += 1
            else:
                # No words left
                last_end = aligned_times[-1][1] if aligned_times else 0.0
                aligned_times.append([last_end, last_end])
                continue

        # CASE 3: Shared Window
        aligned_times.append([current_word_start, current_word_end])
        current_word_chars_left -= tok_len

    return np.array(aligned_times, dtype=np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Extracting LLM Feats on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    ds = load_dataset(ALIGN_REPO, split="train")

    def extract_batch(batch):
        # 1. Forward Pass (Batch Level)
        # Note: batch["text"] must exist in ALIGN_REPO
        inputs = tokenizer(
            batch["text"], return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            states = outputs.hidden_states[-2]  # Layer 27

        results_feat = []
        results_time = []

        # 2. Process Individual Samples
        for i, text in enumerate(batch["text"]):
            try:
                valid_len = inputs.attention_mask[i].sum().item()

                # Extract Feature
                feat = states[i, :valid_len, :].cpu().numpy().astype(np.float16)

                # Extract Timestamps
                # Key Check: Ensure 'alignment_json' exists, otherwise fallback/fail gracefully
                align_raw = batch.get("alignment_json", [None])[i]
                if align_raw is None:
                    raise ValueError("Missing alignment_json")

                align_data = json.loads(align_raw)
                token_ids = inputs.input_ids[i][:valid_len].cpu().tolist()

                times = get_token_timestamps(token_ids, align_data, tokenizer)

                # Shape Correction
                if len(times) != valid_len:
                    # Pad or Truncate
                    new_times = np.zeros((valid_len, 2), dtype=np.float32)
                    min_len = min(len(times), valid_len)
                    if min_len > 0:
                        new_times[:min_len] = times[:min_len]
                        # Propagate last valid timestamp to remaining tokens (safer than 0)
                        new_times[min_len:] = times[min_len - 1]
                    times = new_times

                results_feat.append(feat)
                results_time.append(times)

            except Exception as e:
                # Log error and return dummy data to keep batch size consistent
                # This prevents the whole map() job from crashing
                logging.error(f"Error processing sample {i} in batch: {e}")
                print(f"⚠️ Error in sample: {e}")

                # Return Zero-dummies matching the shape
                # Feat: [valid_len, 2048], Times: [valid_len, 2]
                # We need to re-calculate valid_len locally or assume 1
                v_len = inputs.attention_mask[i].sum().item()
                results_feat.append(np.zeros((v_len, 2048), dtype=np.float16))
                results_time.append(np.zeros((v_len, 2), dtype=np.float32))

        return {"llm_feat": results_feat, "llm_times": results_time}

    # Define Schema explicitly to ensure correct storage types
    features = ds.features.copy()
    features["llm_feat"] = Array2D(shape=(None, 2048), dtype="float16")
    features["llm_times"] = Array2D(shape=(None, 2), dtype="float32")

    # Run Map
    ds_llm = ds.map(extract_batch, batched=True, batch_size=BATCH_SIZE)

    # Push
    ds_llm.push_to_hub(TGT_REPO)
    print("✅ LLM Features Pushed!")


if __name__ == "__main__":
    main()
