# Agent Instructions

This is developer machine so do not install packages, running command, etc on this machine. After development, it needs to be committed, pushed and pulled on training machine to be able to run. THIS IS IMPORTANT

**Branch: `exp/amylm-facodec`** — Amy LM with FACodec. No Mimi, no compressor trainer.

## Project Context

This is **Amy-LM**, a research project developing "The Semantic Bridge" - a disentangled neural audio codec that separates speech into interpretable layers. This branch focuses on the Amy LM + FACodec integration path only.

## Key Files

| File | Purpose |
|------|---------|
| **CONTEXT.md** | Domain glossary and terminology. **Check this first** for vocabulary. |
| docs/ideation/proposal.md | Research proposal with methodology |
| docs/ideation/amy.md | Architecture details and model specification |
| scripts/preprocess.py | Preprocessing orchestrator (FACodec encoding) |
| tests/ | Test suite (108 tests) |

## Decision Log

- **2026-05-30**: `lambda_p` and `lambda_t` are learnable `nn.Parameter`s initialized to 1.0, NOT CLI flags. Added backward hooks in `AmyTrainer` to log their gradient values (`grad_p`/`grad_t`) each epoch for gradient-flow debugging.
- **2026-06-10 (Phase 2)**: Ported lambda gradient hooks to `AmyDPOTrainer` (same pattern as `AmyTrainer`). PEFT-compatible via `unwrap_model` + `__getattr__` forwarding chain. Extended `PreferenceDatasetProcessor` with `process_adversarial_dataset()` and `ADVERSARIAL_FEATURES` schema. Adversarial pairs have `cosine_similarity=0.0` (always passes literal-pair filter). `rationale_chosen`/`rationale_rejected` set to empty for adversarial pairs.

## Before Starting Any Task

1. **Read CONTEXT.md** to understand domain terminology
2. Check relevant docs in `docs/` for historical context
3. **Grill first, implement later.** For issues involving domain decisions (schema, data flow, architecture), use `grill-with-docs` to resolve terminology and design choices before touching code. Update CONTEXT.md, AGENTS.md, and the GitHub issue with decisions as they crystallize. Only begin implementation after the grilling session is complete and the plan is committed.

## Project Structure

```
Amy-LM/ (branch: exp/amylm-facodec)
├── CONTEXT.md             # Domain glossary (start here)
├── AGENTS.md              # This file
├── docs/
│   ├── ideation/          # Research proposals and architecture
│   └── training_records/  # Epoch-by-epoch training logs
├── src/
│   ├── models/            # Amy LM modules (embedding, fusion, pooling)
│   ├── preprocessing/     # FACodec encoder, dataset processor, reporting
│   └── data/              # Feature extraction scripts
├── scripts/               # Preprocessing and utilities
├── tests/
│   ├── models/            # Model unit tests
│   └── preprocessing/     # Preprocessing integration/unit tests
└── vendor/Amphion/        # FACodec dependency
```

## Common Tasks

- **Understand architecture**: Read `docs/ideation/amy.md`
- **Run preprocessing**: `uv run python scripts/preprocess.py --dataset ...`
- **Understand domain terms**: Read `CONTEXT.md`
- **Run tests**: `uv run python -m pytest tests/`

## Coding Conventions

- Use `uv run python` for running Python scripts
- This project uses `uv` for Python package management
- API keys are loaded from environment or `.env` files

## Testing

- **Always load models on GPU.** All model constructors (`MossAudioWrapper`, `AmyForProsodyClassification`, `BaselineClassifier`) auto-detect GPU by default (`device=None` → `"cuda" if available else "cpu"`) and use `torch.float16` precision. Tests must use the shared `device` fixture from `tests/conftest.py` and the `require_gpu` fixture for tests that perform forward/backward passes.
- **Heavy model tests (4B MOSS-Audio, training loops, full forward/backward) are skipped on CPU** via `require_gpu` fixture. These tests are designed for GPU and will timeout or OOM on CPU. Accept that tests passing init/lambda/logic checks on CPU are sufficient; full verification happens on GPU.
- **Never create or instantiate models on CPU.** Always create/load models directly on GPU. Creating on CPU and then moving to GPU is also not accepted — it hangs the machine. Use `.to("cuda")` or `device="cuda"` at construction time. If GPU is unavailable, use a minimal config for shape checks only.
- To run only fast tests (no heavy model forward): `uv run python -m pytest tests/ --ignore=tests/training -k "not (train_epoch or evaluate or save_load_roundtrip or training_step)"`
- Training scripts (`scripts/train_amy_classifier.py`) and heavy test suites are GPU-only. Use `nohup` for long-running GPU jobs (see Long-Running Tasks above).

## Long-Running Tasks

For long-running tasks (e.g., preprocessing large datasets, training), always use `nohup` and provide the log path to the user. The user will inform when the task is complete for review.

If a command could be a long-running process, **do not run it directly** — it may crash the machine. Instead, prompt the user to run it themselves with both console output (for humans) and log output (for you to debug later). This keeps the machine stable and preserves the logs for post-mortem analysis.

Example:
```bash
mkdir -p logs
nohup python scripts/preprocess.py --dataset ... > logs/preprocess_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Log: logs/preprocess_<timestamp>.log"
```

## Agent skills

### Engineering workflow

Before using skills under `skills/engineering/`, consult `WORKFLOW.md` to choose the appropriate workflow. Treat `WORKFLOW.md` as the workflow router and operating model; treat individual `SKILL.md` files as detailed procedures.

### Issue tracker

Issues live in GitHub Issues on `hungphongtrn/Amy-LM`. See `docs/agents/issue-tracker.md`.

### Triage labels

Standard label vocabulary (bug, enhancement, needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` at the root. See `docs/agents/domain.md`.

---

**Branch: `exp/amylm-facodec`** — Amy LM with FACodec only. No Mimi or compressor trainer code exists on this branch.
