# Agent Instructions

## Project Context

This is **Amy-LM**, a research project developing "The Semantic Bridge" - a disentangled neural audio codec that separates speech into interpretable layers.

## Key Files

| File | Purpose |
|------|---------|
| **STATE.md** | Project state, training metrics, current issues, and next steps. **Check this first** for latest status. |
| docs/ideation/proposal.md | Research proposal with methodology |
| docs/ideation/amy.md | Architecture details and model specification |
| docs/training_records/*.md | Per-epoch training logs with metrics |
| train.py | Main training script |
| benchmark/prosody_validation/ | Evaluation pipeline for prosody understanding |

## Before Starting Any Task

1. **Read STATE.md** to understand current project status, open issues, and recommended actions
2. Check relevant training records in `docs/training_records/` for historical context
3. Verify if your task aligns with next steps listed in STATE.md

## Project Structure

```
Amy-LM/
├── STATE.md              # Current project state (start here)
├── AGENTS.md             # This file
├── docs/
│   ├── ideation/         # Research proposals and architecture
│   └── training_records/ # Epoch-by-epoch training logs
├── src/                  # Source code
│   ├── models/           # Neural network implementations
│   ├── trainer/          # Training loop and optimization
│   └── utils/            # Utilities
├── benchmark/            # Evaluation pipelines
│   └── prosody_validation/
└── train.py              # Main entry point
```

## Common Tasks

- **Check training status**: Read latest file in `docs/training_records/`
- **Understand architecture**: Read `docs/ideation/amy.md`
- **Run evaluations**: Check `benchmark/prosody_validation/README.md`
- **Modify training**: Edit `src/trainer/` or `train.py`
- **Review recent changes**: Check STATE.md "Latest Commits" section

## Coding Conventions

- Use `uv run python` for running Python scripts
- This project uses `uv` for Python package management
- API keys are loaded from environment or `.env` files
- Training configs are in YAML format
- Prefer async patterns for API calls

## Agent skills

### Issue tracker

Issues live in GitHub Issues on `hungphongtrn/Amy-LM`. See `docs/agents/issue-tracker.md`.

### Triage labels

Standard label vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` at the root. See `docs/agents/domain.md`.
