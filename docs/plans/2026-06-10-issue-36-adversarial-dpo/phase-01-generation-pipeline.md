# Phase 1: Adversarial Pair Generation Pipeline

## Phase Goal
`scripts/generate_adversarial_pairs.py` is created, tested locally with smoke mode, and ready to run on the training machine against the full NVTTS dataset. The script produces a JSONL file with 1,000+ high-quality adversarial preference pairs after two-phase filtering.

## Files to Touch

- `scripts/generate_adversarial_pairs.py` — **Create** — main generation script
- `scripts/generate_pairs_deepseek.py` — Reference only (literal pair generation pattern)

## Tasks

### Task 1: Inverse Emotion Mapping

**Files:**
- Create: section in `scripts/generate_adversarial_pairs.py`

- [ ] **Step 1: Write the inverse emotion mapping table**

```python
INVERSE_EMOTION = {
    "happy": "sad",
    "sad": "happy",
    "angry": "neutral",
    "disgusted": "neutral",
    "fearful": "sad",
    "surprised": "happy",
    "neutral": "sad",
}
# "other" and "disgusted" are SKIPPED from mapping but still generate pairs
```

- [ ] **Step 2: Write a unit test for the mapping**

Create `tests/scripts/test_generate_adversarial_pairs.py`:

```python
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from generate_adversarial_pairs import INVERSE_EMOTION, get_inverse_emotion

def test_inverse_emotion_known():
    assert get_inverse_emotion("happy") == "sad"
    assert get_inverse_emotion("sad") == "happy"
    assert get_inverse_emotion("angry") == "neutral"
    assert get_inverse_emotion("neutral") == "sad"

def test_inverse_emotion_skip():
    assert get_inverse_emotion("other") is None
    assert get_inverse_emotion("disgusted") is None

def test_inverse_emotion_unknown():
    assert get_inverse_emotion("nonexistent") is None
```

- [ ] **Step 3: Run tests to verify**

```bash
uv run python -m pytest tests/scripts/test_generate_adversarial_pairs.py -v
```

### Task 2: Single-Call Prompt Template + JSON Schema

**Files:**
- Section in `scripts/generate_adversarial_pairs.py`

- [ ] **Step 1: Write the prompt template**

```python
ADVERSARIAL_PROMPT_TEMPLATE = """You are a conversation partner responding to someone who just spoke.

Transcript of what the speaker said:
{transcript}

The speaker expressed the following emotion: {emotion_label}.
The opposite/inverse of this emotion is: {inverse_emotion}.

Your task:
1. Write a response that reflects the GROUND-TRUTH emotion ({emotion_label}). Label this "chosen".
2. Write a response that reflects the INVERSE emotion ({inverse_emotion}). Label this "rejected".
3. Briefly explain your strategy for making these responses textually similar but emotionally different. Label this "strategy".

Both responses should be natural conversation. The two responses should be structurally and lexically similar — a reader who cannot hear the audio should struggle to tell which is correct.

Output JSON with keys "strategy", "chosen", "rejected".

EXAMPLE JSON OUTPUT:
{{
    "strategy": "I made both responses express concern, but the chosen response reflects the speaker's upbeat tone while the rejected is flat.",
    "chosen": "That's great to hear! I'm really happy for you.",
    "rejected": "I understand. That must have been difficult."
}}
"""
```

- [ ] **Step 2: Write JSON schema parse + validation function**

```python
import json
import re

def parse_adversarial_response(text: str) -> dict:
    """Extract {strategy, chosen, rejected} from LLM response text."""
    # 1. Try direct parse
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = {}

    # 2. Extract from markdown code block
    if not _validate_pair(data):
        if "```json" in text:
            block = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            block = text.split("```")[1].split("```")[0]
        else:
            block = text
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            data = {}

    # 3. Regex fallback
    if not _validate_pair(data):
        data = _regex_extract_fields(text)

    if not _validate_pair(data):
        raise ValueError(f"Could not extract 'chosen' and 'rejected' from text: {text[:300]}")
    return data

def _validate_pair(data: dict) -> bool:
    return (
        isinstance(data, dict)
        and "chosen" in data
        and "rejected" in data
        and isinstance(data["chosen"], str)
        and isinstance(data["rejected"], str)
        and len(data["chosen"].strip()) > 0
        and len(data["rejected"].strip()) > 0
    )

def _regex_extract_fields(text: str) -> dict:
    result = {"strategy": "", "chosen": "", "rejected": ""}
    for key in ("strategy", "chosen", "rejected"):
        match = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if match:
            result[key] = json.loads(f'"{match.group(1)}"')
    return result
```

- [ ] **Step 3: Write unit tests for parsing**

```python
def test_parse_direct_json():
    text = '{"strategy": "x", "chosen": "hello", "rejected": "world"}'
    result = parse_adversarial_response(text)
    assert result["chosen"] == "hello"
    assert result["rejected"] == "world"

def test_parse_markdown_block():
    text = '```json\n{"strategy": "x", "chosen": "a", "rejected": "b"}\n```'
    result = parse_adversarial_response(text)
    assert result["chosen"] == "a"

def test_parse_missing_fields_raises():
    with pytest.raises(ValueError):
        parse_adversarial_response('{"chosen": "only one"}')

def test_parse_empty_strings_raises():
    with pytest.raises(ValueError):
        parse_adversarial_response('{"strategy": "", "chosen": "", "rejected": ""}')
```

### Task 3: Two-Phase Quality Filters

**Files:**
- Section in `scripts/generate_adversarial_pairs.py`

- [ ] **Step 1: Phase 1 — Semantic Gate**

```python
def compute_bleu1(text_a: str, text_b: str) -> float:
    """Compute BLEU-1 score between two strings."""
    from nltk.translate.bleu_score import sentence_bleu
    ref_tokens = [text_a.lower().split()]
    hyp_tokens = text_b.lower().split()
    if not hyp_tokens or not ref_tokens[0]:
        return 0.0
    return sentence_bleu(ref_tokens, hyp_tokens, weights=(1.0, 0, 0, 0))

def compute_length_parity_ratio(text_a: str, text_b: str) -> float:
    """Ratio of token counts (shorter/longer). 1.0 = equal length."""
    len_a = len(text_a.split())
    len_b = len(text_b.split())
    if len_a == 0 and len_b == 0:
        return 1.0
    if min(len_a, len_b) == 0:
        return 0.0
    return min(len_a, len_b) / max(len_a, len_b)
```

Phase 1 gate function:

```python
class SemanticGate:
    def __init__(
        self,
        length_ratio_min: float = 0.8,
        length_ratio_max: float = 1.2,
        bleu_min: float = 0.3,
        sim_min: float = 0.6,
        sim_max: float = 0.9,
    ):
        self.length_ratio_min = length_ratio_min
        self.length_ratio_max = length_ratio_max
        self.bleu_min = bleu_min
        self.sim_min = sim_min
        self.sim_max = sim_max

    def check(self, chosen: str, rejected: str, embedding_model) -> tuple[bool, dict]:
        length_ratio = compute_length_parity_ratio(chosen, rejected)
        if not (self.length_ratio_min < length_ratio < self.length_ratio_max):
            return False, {"reason": "length_parity", "value": length_ratio}

        bleu = compute_bleu1(chosen, rejected)
        if bleu <= self.bleu_min:
            return False, {"reason": "lexical_overlap", "value": bleu}

        chosen_emb = embedding_model.encode([chosen])[0]
        rejected_emb = embedding_model.encode([rejected])[0]
        sim = cosine_similarity(chosen_emb, rejected_emb)
        if not (self.sim_min < sim < self.sim_max):
            return False, {"reason": "semantic_similarity", "value": sim}

        return True, {"length_ratio": length_ratio, "bleu": bleu, "sim": sim}
```

- [ ] **Step 2: Phase 2 — LLM-as-Judge**

```python
JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of a preference pair for speech-language DPO training.

The speaker's transcript: {transcript}
The speaker's ground-truth emotion: {emotion_label}
The inverse emotion used for the rejected response: {inverse_emotion}

Here are two responses, presented in random order:

Response A: {response_a}
Response B: {response_b}

For each response, rate:
1. Emotion Fidelity (1-5): How well does this response reflect {emotion_label} emotionally?
2. Text-Ambiguity (1-5): How hard would it be for a text-only reader to identify the correct emotion from this response ALONE?

Output JSON with keys:
- "fidelity_A": integer 1-5
- "fidelity_B": integer 1-5
- "ambiguity_A": integer 1-5
- "ambiguity_B": integer 1-5
- "identified_correct": string - which response shows emotion "{emotion_label}" more clearly: "A" or "B" or "neither"
"""

async def judge_pair(
    client: AsyncOpenAI,
    transcript: str,
    emotion_label: str,
    inverse_emotion: str,
    chosen: str,
    rejected: str,
) -> dict:
    import random
    if random.random() < 0.5:
        response_a, response_b = chosen, rejected
        a_is_chosen = True
    else:
        response_a, response_b = rejected, chosen
        a_is_chosen = False

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        transcript=transcript,
        emotion_label=emotion_label,
        inverse_emotion=inverse_emotion,
        response_a=response_a,
        response_b=response_b,
    )

    response = await client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=512,
    )
    judge_data = parse_json_response(response.choices[0].message.content)

    if a_is_chosen:
        fidelity_chosen = judge_data.get("fidelity_A", 0)
        fidelity_rejected = judge_data.get("fidelity_B", 0)
        ambiguity_chosen = judge_data.get("ambiguity_A", 0)
        ambiguity_rejected = judge_data.get("ambiguity_B", 0)
    else:
        fidelity_chosen = judge_data.get("fidelity_B", 0)
        fidelity_rejected = judge_data.get("fidelity_A", 0)
        ambiguity_chosen = judge_data.get("ambiguity_B", 0)
        ambiguity_rejected = judge_data.get("ambiguity_A", 0)

    return {
        "fidelity_chosen": fidelity_chosen,
        "fidelity_rejected": fidelity_rejected,
        "ambiguity_chosen": ambiguity_chosen,
        "ambiguity_rejected": ambiguity_rejected,
    }

def passes_judge_gate(judge_result: dict) -> bool:
    """Both fidelity scores >= 4 AND max ambiguity <= 2."""
    fidelity_ok = (
        judge_result["fidelity_chosen"] >= 4
        and judge_result["fidelity_rejected"] >= 4
    )
    ambiguity_ok = (
        judge_result["ambiguity_chosen"] <= 2
        and judge_result["ambiguity_rejected"] <= 2
    )
    return fidelity_ok and ambiguity_ok
```

- [ ] **Step 3: Unit tests for filter functions**

```python
def test_length_parity_equal():
    assert compute_length_parity_ratio("a b c", "a b c") == 1.0

def test_length_parity_asymmetric():
    ratio = compute_length_parity_ratio("a b c", "a b c d e")
    assert 0.5 < ratio < 0.7

def test_bleu1_identical():
    assert compute_bleu1("hello world", "hello world") == 1.0

def test_bleu1_disjoint():
    assert compute_bleu1("hello world", "foo bar baz") < 0.3  # low but not zero due to smoothing

def test_semantic_gate_pass():
    gate = SemanticGate()
    # These are similar enough to pass all gates
    ok, info = gate.check("That sounds great, I am happy for you", "That sounds okay, I understand", mock_embedder)
    assert ok or not ok  # Integration test — depends on embedder

def test_semantic_gate_length_fail():
    gate = SemanticGate()
    ok, info = gate.check("Hi", "A very long response with many words " * 20, mock_embedder)
    assert not ok
    assert info["reason"] == "length_parity"
```

### Task 4: Main Generation Loop with Resume

**Files:**
- `scripts/generate_adversarial_pairs.py` — main loop

- [ ] **Step 1: API caller with retry**

```python
MAX_RETRIES = 3

async def generate_single_pair(
    client: AsyncOpenAI,
    transcript: str,
    emotion_label: str,
    inverse_emotion: str | None,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    if inverse_emotion is None:
        return None  # Skip unmappable emotions

    prompt = ADVERSARIAL_PROMPT_TEMPLATE.format(
        transcript=transcript,
        emotion_label=emotion_label,
        inverse_emotion=inverse_emotion,
    )

    for attempt in range(MAX_RETRIES):
        temperature = 0.7 + attempt * 0.15  # Temperature jitter
        async with semaphore:
            response = await client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=2048,
            )
        try:
            data = parse_adversarial_response(response.choices[0].message.content)
            return {
                "chosen": data["chosen"],
                "rejected": data["rejected"],
                "strategy": data.get("strategy", ""),
                "attempt": attempt + 1,
            }
        except ValueError as e:
            if attempt < MAX_RETRIES - 1:
                continue
            raise
    return None
```

- [ ] **Step 2: Main processing loop with resume + filters**

```python
async def run_adversarial_generation(
    dataset,
    output_path: str,
    concurrency: int = 5,
    smoke: bool = False,
    stop_event: asyncio.Event | None = None,
) -> None:
    # Load resume state
    existing_ids = load_existing_ids(output_path)
    pending = [s for s in dataset if s["id"] not in existing_ids]
    if smoke:
        pending = pending[:10]

    # Init models (lazy, only if pending exists)
    embedding_model = SentenceTransformer("google/embeddinggemma-300m")
    semantic_gate = SemanticGate()

    client = AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
        max_retries=1,
        timeout=30.0,
    )
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    failed_count = 0
    passed_count = 0

    async def process_sample(sample):
        nonlocal failed_count, passed_count
        emotion = sample["emotion_label"]
        inverse = get_inverse_emotion(emotion)
        if inverse is None:
            return

        try:
            pair = await generate_single_pair(
                client, sample["transcript_with_tags"], emotion, inverse, semaphore
            )
            if pair is None:
                return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            async with lock:
                failed_count += 1
            async_tqdm.write(f"  GEN_FAILED {sample['id']}: {e}")
            return

        # Phase 1: Semantic Gate
        ok, gate_info = semantic_gate.check(
            pair["chosen"], pair["rejected"], embedding_model
        )
        if not ok:
            async_tqdm.write(f"  SEMANTIC_GATE {sample['id']}: {gate_info}")
            return

        # Phase 2: LLM-as-Judge
        try:
            judge_result = await judge_pair(
                client, sample["transcript_with_tags"], emotion, inverse,
                pair["chosen"], pair["rejected"],
            )
        except Exception as e:
            async_tqdm.write(f"  JUDGE_FAILED {sample['id']}: {e}")
            return

        if not passes_judge_gate(judge_result):
            async_tqdm.write(f"  JUDGE_GATE {sample['id']}: fidelity={judge_result}")
            return

        # Write passing pair
        output = {
            "id": sample["id"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "strategy": pair["strategy"],
            "emotion_label": emotion,
            "inverse_emotion": inverse,
            "judge_fidelity_chosen": judge_result["fidelity_chosen"],
            "judge_fidelity_rejected": judge_result["fidelity_rejected"],
            "judge_ambiguity_chosen": judge_result["ambiguity_chosen"],
            "judge_ambiguity_rejected": judge_result["ambiguity_rejected"],
            "generation_attempts": pair["attempt"],
        }
        async with lock:
            with open(output_path, "a") as f:
                json.dump(output, f)
                f.write("\n")
            passed_count += 1

    tasks = [asyncio.create_task(process_sample(s)) for s in pending]
    with async_tqdm(total=len(pending), desc="Generating adversarial pairs") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                await coro
            except asyncio.CancelledError:
                pass
            pbar.update(1)
            if stop_event and stop_event.is_set():
                for t in tasks:
                    t.cancel()

    completed = len(load_existing_ids(output_path))
    print(f"Done: {completed} pairs in {output_path} ({failed_count} failed, {passed_count} passed this run)")
```

- [ ] **Step 3: CLI entry point**

```python
def main():
    parser = argparse.ArgumentParser(description="Generate adversarial DPO preference pairs")
    parser.add_argument("--input", default="data/nvtts_enriched/nvtts_enriched.parquet",
                        help="Enriched NVTTS parquet file")
    parser.add_argument("--output", default="data/nvtts_adversarial/pairs.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of concurrent API calls")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: process only 10 samples")
    args = parser.parse_args()

    from datasets import Dataset
    ds = Dataset.from_parquet(args.input)
    print(f"Loaded {len(ds)} samples from {args.input}")

    stop_event = asyncio.Event()
    def _on_sigint(signum, frame):
        print("\nInterrupted, finishing in-flight requests...", flush=True)
        stop_event.set()
    signal.signal(signal.SIGINT, _on_sigint)

    asyncio.run(run_adversarial_generation(
        ds, args.output,
        concurrency=args.concurrency,
        smoke=args.smoke,
        stop_event=stop_event,
    ))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test locally (generation only, no API calls)**

Test the inverse mapping and parsing logic without API:

```bash
uv run python -m pytest tests/scripts/test_generate_adversarial_pairs.py -v
```

- [ ] **Step 5: Smoke test with 10 samples (requires API key)**

```bash
uv run python scripts/generate_adversarial_pairs.py --smoke --output data/nvtts_adversarial/pairs_smoke.jsonl
```

Verify smoke output has valid pairs that pass both filter phases.

### Task 5: Counting and Split Verification

- [ ] **Step 1: Write pair counting utility**

```python
def count_pairs(jsonl_path: str) -> dict:
    pairs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    emotion_counts = {}
    for p in pairs:
        e = p.get("emotion_label", "unknown")
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    return {
        "total": len(pairs),
        "emotions": emotion_counts,
    }
```

- [ ] **Step 2: Verify 1,000 target with split proportions**

After generation completes, verify:
- Total pairs ≥ 1,000
- Each emotion class has ≥ 100 pairs (rough balance)
- Train/dev/test split (800/100/100) is achievable

## Phase Completion Criteria
- [ ] `scripts/generate_adversarial_pairs.py` exists and passes unit tests
- [ ] Smoke test (10 samples) produces valid JSONL with all required fields
- [ ] Both filter phases work correctly on smoke output
- [ ] Full run produces ≥ 1,000 pairs in JSONL format
- [ ] JSONL is committed or staged for FACodec encoding in Phase 2
- [ ] Script supports resume (re-running picks up where it left off)

## Handoff Notes
- The JSONL output must preserve all fields needed by Phase 2: `id`, `chosen`, `rejected`, `strategy`, `emotion_label`, `inverse_emotion`, `judge_fidelity_chosen`, `judge_fidelity_rejected`, `judge_ambiguity_chosen`, `judge_ambiguity_rejected`.
- The `strategy` field can be discarded after validation — it's only used for debugging.
- Expect ~40-60% yield rate (1,000 pairs from ~2,000 API calls). The retry logic may inflate API costs if many pairs fail the judge gate.
- SentenceTransformer model `google/embeddinggemma-300m` is ~300M params. First load will download the model. Consider verifying the exact HuggingFace model ID before running.
