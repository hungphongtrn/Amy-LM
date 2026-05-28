from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn.functional as F

TARGET_SAMPLE_RATE = 16000

YES_TOKENS = {"yes", "true", "sarcastic", "irony", "ironic"}
NO_TOKENS = {"no", "false", "not", "non-sarcastic"}


def parse_response(response: str) -> int:
    response_lower = response.lower().strip()
    if not response_lower:
        return 0
    first_word = response_lower.split()[0].rstrip(".,;:!?")
    if first_word in YES_TOKENS:
        return 1
    if first_word in NO_TOKENS:
        return 0
    for word in response_lower.split():
        if word.rstrip(".,;:!?") in YES_TOKENS:
            return 1
    return 0


def resample_audio(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio
    try:
        import torchaudio

        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        return resampler(audio.unsqueeze(0)).squeeze(0)
    except ImportError:
        warnings.warn(
            f"torchaudio not available; passing audio at {sample_rate} Hz as-is. "
            f"FACodec expects {TARGET_SAMPLE_RATE} Hz — results may be invalid."
        )
        return audio


class AmyInference:
    def __init__(self, model: Any, encoder: Any, tokenizer: Any, device: torch.device | str = "cpu"):
        self.model = model
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.lm = self.model.get_language_model()

    def compute_fused_h(
        self,
        audio: torch.Tensor,
        prosody_indices: torch.Tensor,
        timbre_vector: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.amy_moss.encode_enriched_audio_embeds(
            audio.to(self.device),
            prosody_indices=prosody_indices.to(self.device),
            timbre_vector=timbre_vector.to(self.device),
        )

    def _assemble_inputs(self, fused_h: torch.Tensor, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        chat = [{"role": "user", "content": instruction}]
        prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        instruction_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        instruction_embeds = self.lm.get_input_embeddings()(instruction_ids)
        lm_dtype = next(self.lm.parameters()).dtype
        inputs_embeds = torch.cat([fused_h.to(dtype=lm_dtype), instruction_embeds], dim=1)
        attention_mask = torch.ones((1, inputs_embeds.shape[1]), device=self.device, dtype=torch.long)
        return inputs_embeds, attention_mask

    def predict(
        self,
        audio: torch.Tensor,
        instruction: str,
        prosody_indices: torch.Tensor,
        timbre_vector: torch.Tensor,
    ) -> dict[str, Any]:
        fused_h = self.compute_fused_h(audio, prosody_indices, timbre_vector)
        inputs_embeds, attention_mask = self._assemble_inputs(fused_h, instruction)
        outputs = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=32,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"prediction": parse_response(response), "response": response}

    def evaluate(self, samples: list[dict[str, Any]], max_samples: int | None = None) -> dict[str, Any]:
        selected = samples[:max_samples] if max_samples is not None else samples
        results: list[dict[str, Any]] = []
        correct = 0
        for idx, sample in enumerate(selected):
            prediction = int(sample["pred_fn"]()) if "pred_fn" in sample else int(sample["prediction"])
            label = int(sample["label"])
            is_correct = prediction == label
            if is_correct:
                correct += 1
            results.append({"idx": idx, "prediction": prediction, "label": label, "correct": is_correct})
        total = len(selected)
        return {"accuracy": (correct / total) if total else 0.0, "correct": correct, "total": total, "results": results}
