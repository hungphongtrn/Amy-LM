# Phase 3: CLI integration

## Phase Goal
`--use-lora`, `--lora-r`, `--lora-alpha`, `--lora-dropout` flags on `train_amy_classifier.py` work end-to-end. Model is conditionally wrapped with LoRA at construction.

**Depends on:** Phase 2

## Files to Touch (preliminary)

| File | Action |
|------|--------|
| `scripts/train_amy_classifier.py` | Add argparse flags, wire LoRA wrapping |

## Tasks (stub — to be detailed after Phase 2)

- Add `--use-lora` (store_true), `--lora-r` (int, default 8), `--lora-alpha` (int, default 16), `--lora-dropout` (float, default 0.05)
- In `main()`: after model construction, if `args.use_lora and not is_baseline`, call `wrap_classifier_with_lora(model, r=args.lora_r, ...)`
- Pass `is_lora=args.use_lora` to `AmyTrainer`
- Smoke test: `uv run python scripts/train_amy_classifier.py --data-path ... --mode amy --use-lora --epochs 1`
