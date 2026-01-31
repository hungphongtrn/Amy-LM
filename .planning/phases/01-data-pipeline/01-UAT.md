---
status: diagnosed
phase: 01-data-pipeline
source: 01-01-SUMMARY.md, 01-02-SUMMARY.md, 01-03-SUMMARY.md
started: 2026-01-31T12:00:00Z
updated: 2026-01-31T12:10:00Z
---

## Current Test

[testing complete]

## Summary

total: 5
passed: 3
issues: 2
pending: 0
skipped: 0

## Gaps

- truth: "User can run a single command to load/parse `.data/` into a structured sample table without manual editing"
  status: failed
  reason: "User reported: ModuleNotFoundError: No module named 'proactive_sat'"
  severity: blocker
  test: 1
  root_cause: "Package 'proactive_sat' is not installed. The pyproject.toml defines 'amy-lm' project but proactive_sat code is in src/proactive_sat/ without proper package configuration."
  artifacts:
    - path: "pyproject.toml"
      issue: "Does not include proactive_sat as installable package"
    - path: "src/proactive_sat/__init__.py"
      issue: "Package exists but not configured for import"
    - path: "src/proactive_sat/data_pipeline/run_pipeline.py"
      issue: "Imports proactive_sat module which is not available"
  missing:
    - "Package configuration to make proactive_sat importable"
    - "Or PYTHONPATH setup for development mode"
  debug_session: ""
- truth: "User can inspect any sample and see both original text and a lexical-neutralized transcript"
  status: failed
  reason: "User reported: This neutralization is using rule-based. I want to use LLM for this task."
  severity: major
  test: 2
  root_cause: "User unaware that LLM mode exists. Feature already implemented: use `--neutralizer openai` flag and set OPENAI_API_KEY environment variable."
  artifacts:
    - path: "src/proactive_sat/data_pipeline/run_pipeline.py"
      issue: "Already supports --neutralizer openai option (lines 106-110)"
    - path: "src/proactive_sat/data_pipeline/neutralize.py"
      issue: "Already implements _call_openai_api() for LLM-based neutralization"
  missing:
    - "Documentation/instruction on using LLM mode"
    - "OPENAI_API_KEY environment setup"
  debug_session: ""

### 2. Inspect sample with original and neutralized text
expected: |
  Running `uv run python -c "from datasets import load_from_disk; ds = load_from_disk('.data/proactive_sat/hf_dataset'); s = ds['train'][0]; print('source_text:', s['source_text']); print('neutral_text:', s['neutral_text'])"` shows:
  
  - source_text: Original emotional text (e.g., "I can't believe you did that!")
  - neutral_text: Lexically neutralized version (e.g., "I did that")
result: issue
reported: "This neutralization is using rule-based. I want to use LLM for this task."
severity: major

### 3. Inspect sample with prosody instructions
expected: |
  Running `uv run python -c "from datasets import load_from_disk; ds = load_from_disk('.data/proactive_sat/hf_dataset'); s = ds['train'][0]; print('prosody_style:', s['prosody_style']); print('control_instruction:', s['control_speaker_instruction']); print('trigger_instruction:', s['trigger_speaker_instruction'])"` shows:
  
  - prosody_style: Emotion category (sarcastic, frustrated, or distressed)
  - control_speaker_instruction: Flat/factual delivery instruction
  - trigger_speaker_instruction: Prosodically rich instruction matching prosody_style
result: pass

### 4. Verify HF dataset has exactly 200 samples
expected: |
  Running `uv run python -c "from datasets import load_from_disk; ds = load_from_disk('.data/proactive_sat/hf_dataset'); print('Total samples:', len(ds['train']))"` outputs:
  
  Total samples: 200
result: pass

### 5. Verify Control/Trigger metadata for prosodic injection
expected: |
  Running `uv run python -c "from datasets import load_from_disk; ds = load_from_disk('.data/proactive_sat/hf_dataset'); s = ds['train'][0]; print('control_text == neutral_text:', s['control_text'] == s['neutral_text']); print('trigger_text == neutral_text:', s['trigger_text'] == s['neutral_text'])"` outputs:
  
  control_text == neutral_text: True
  trigger_text == neutral_text: True
  
  This confirms prosody-only modification (text stays same, only instructions differ).
result: pass

## Summary

total: 5
passed: 2
issues: 2
pending: 1
skipped: 0

## Gaps

[none yet]
