import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# CONFIG
ALIGN_REPO = "hungphongtrn/speech-time-alignment"
TGT_REPO = "hungphongtrn/llm-features"
BATCH_SIZE = 4
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
    
    # Handle empty timings case
    if not word_timings:
        return [[0.0, 0.0] for _ in token_ids]

    w_info = word_timings[word_idx]
    current_word_start = w_info["start"]
    current_word_end = w_info["end"] # Can be float or "inf"
    
    ref_text = w_info.get("text") or w_info.get("word") or ""
    current_word_chars_left = len(ref_text)

    for tok_str in token_strs:
        clean_tok = tok_str.strip()
        tok_len = len(clean_tok)

        # CASE 1: Empty/Special Token -> Attach to previous
        if tok_len == 0:
            last_end = aligned_times[-1][1] if aligned_times else current_word_start
            aligned_times.append([last_end, last_end])
            continue

        # CASE 2: Load Next Word if budget exhausted
        if current_word_chars_left <= 0:
            if word_idx + 1 < len(word_timings):
                word_idx += 1
                w_info = word_timings[word_idx]
                current_word_start = w_info["start"]
                current_word_end = w_info["end"]
                
                ref_text = w_info.get("text") or w_info.get("word") or clean_tok
                current_word_chars_left = len(ref_text)
            else:
                # No words left. Propagate "inf" if that was the last endpoint.
                last_end = aligned_times[-1][1] if aligned_times else 0.0
                aligned_times.append([last_end, last_end])
                continue

        # CASE 3: Shared Window
        aligned_times.append([current_word_start, current_word_end])
        current_word_chars_left -= tok_len

    return aligned_times


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Extracting LLM Features on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    ds = load_dataset(ALIGN_REPO, split="dev")

    def extract_batch(batch):
        texts = []
        word_alignments_list = []
        
        # 1. Parse alignment data (already contains silence tokens)
        for align_json_str in batch["alignment_json"]:
            try:
                align_data = json.loads(align_json_str)
                words = align_data.get("word", [])
                
                # Use the text field which already includes silence tokens
                text = align_data.get("text", "")
                
                # If text is not available, reconstruct from words
                if not text and words:
                    text = " ".join([w.get("text") or w.get("word", "") for w in words])
                
                texts.append(text)
                word_alignments_list.append(words)
            except Exception as e:
                logging.error(f"Error parsing alignment_json: {e}")
                texts.append("")
                word_alignments_list.append([])
        
        # 2. Tokenize & Forward
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            states = outputs.hidden_states[-2]

        results_feat = []
        results_time = []

        # 3. Extract & Align
        for i, text in enumerate(texts):
            try:
                valid_len = inputs.attention_mask[i].sum().item()
                feat = states[i, :valid_len, :].cpu().numpy().astype(np.float16)
                
                word_alignments = word_alignments_list[i]
                if not word_alignments:
                    results_feat.append(feat.tolist())
                    results_time.append([[0.0, 0.0]] * valid_len)
                    continue

                token_ids = inputs.input_ids[i][:valid_len].cpu().tolist()
                times = get_token_timestamps(token_ids, word_alignments, tokenizer)

                # Shape Correction
                if len(times) != valid_len:
                    last_valid = times[-1] if times else [0.0, 0.0]
                    new_times = [[0.0, 0.0] for _ in range(valid_len)]
                    min_len = min(len(times), valid_len)
                    if min_len > 0:
                        new_times[:min_len] = times[:min_len]
                        for j in range(min_len, valid_len):
                            new_times[j] = last_valid
                    times = new_times

                results_feat.append(feat.tolist())
                results_time.append(times)

            except Exception as e:
                logging.error(f"Error processing sample {i}: {e}")
                v_len = inputs.attention_mask[i].sum().item()
                results_feat.append([[0.0]*2048]*v_len)
                results_time.append([[0.0, 0.0]]*v_len)

        return {"llm_feat": results_feat, "llm_times": results_time}

    ds_llm = ds.map(extract_batch, batched=True, batch_size=BATCH_SIZE)
    ds_llm.push_to_hub(TGT_REPO)
    print("✅ LLM Features Pushed!")

if __name__ == "__main__":
    main()