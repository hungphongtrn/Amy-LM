# Agent Instructions

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

## Before Starting Any Task

1. **Read CONTEXT.md** to understand domain terminology
2. Check relevant docs in `docs/` for historical context

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

## Long-Running Tasks

For long-running tasks (e.g., preprocessing large datasets, training), always use `nohup` and provide the log path to the user. The user will inform when the task is complete for review.

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
