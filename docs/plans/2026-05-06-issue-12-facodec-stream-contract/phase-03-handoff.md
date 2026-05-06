# Phase 3: Handoff

## Phase Goal
Finalize the spike by publishing the blocker recommendation for issue #8, ensuring all acceptance criteria are met, and defining actionable follow-up tasks for issues #6 and #7 implementers.

## Phase 2 Learnings Applied
- Complete field migration table with exact types and shapes
- Complete module interface specs (TimbreProjection, AcousticEmbedding, ContentEmbedding, corrected ProsodyEmbedding and ResidualFusion)
- Full test audit with specific line references
- `D_timbre = 256` hard-confirmed

## Files to Touch

| File | Action | Responsibility |
|------|--------|----------------|
| `docs/spikes/issue-12-facodec-stream-contract.md` | Finalize | Ensure all acceptance criteria sections are complete |
| `CONTEXT.md` | Verify | Already updated during grill session; verify no gaps |

No production code changes. This phase is documentation finalization.

---

## Task 1: Finalize Blocker Recommendation and Acceptance Criteria Check

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Cross-check acceptance criteria against the spike report**

Read the spike report and verify each acceptance criterion from issue #12:

```markdown
## Acceptance Criteria Status

- [x] FACodec tensor contract verified against Amphion source/checkpoint
  - Local checkpoint run confirmed: vq_id=(6,B,80), spk_embs=(B,256)
  - Amphion README evidence recorded with source URL

- [x] CONTEXT.md updated with Acoustic Stream, FACodec Content Stream, per-stream lambda, Stream Dimensionality Contract, Stream Activation Config
  - Already updated during 2026-05-06 grill session; verified in Phase 1

- [x] #6 schema documented with corrected field names, shapes, and types
  - Field migration table, FACodecEncoder return contract, HF Features schema completed in Phase 2 Task 1

- [x] #7 module set documented with corrected modules and interfaces
  - Module migration table, interface specs for all 6 modules completed in Phase 2 Task 2

- [ ] Blocker recommendation for #8 documented (pending this task)
```

- [ ] **Step 2: Write the final blocker recommendation**

Add or update the `## Issue #8 Blocker Recommendation` section:

```markdown
## Issue #8 Blocker Recommendation

**Status: APPROVED TO PROCEED with conditions.**

The FACodec contract has been verified against a real checkpoint run. All tensor
shapes match the documented contract. The corrected specs for #6 preprocessing and
#7 modules are published and ready.

### Conditions for issue #8

1. **Preprocessing contract (#6) must be implemented first** — #8 consumes
   `prosody_codebooks_idx` and `timbre_vector` from the preprocessed dataset.
   These fields do not exist in the current dataset output format.

2. **Module refactor (#7) must be implemented first** — #8 uses `TimbreProjection`
   (not `TimbreEmbedding`) and per-stream lambda gates in `ResidualFusion`. Neither
   exists in the current codebase.

3. **Minimum viable surface:** Issue #8 can start coding in parallel with #6/#7
   fixes by implementing against the corrected interface specs in this spike report.
   The actual integration will fail until #6/#7 land, but module code can be written.

4. **Recommended first experiment:** Prosody + Timbre (λ_p enabled, λ_t enabled,
   λ_c disabled, λ_a disabled). See the Acoustic Stream Recommendation in the
   data flow section for rationale.

### What changes between current state and corrected state

| Component | Current (broken) | Corrected |
|-----------|-----------------|-----------|
| Preprocessing output field | `timbre_codebooks_idx: Sequence(int64)` | `acoustic_codebooks_idx: Sequence(Sequence(int64))` |
| Missing preprocessing field | *(none)* | `timbre_vector: Sequence(float32)` D=256 |
| Timbre module | `TimbreEmbedding(B,)` discrete lookup | `TimbreProjection(B, 256)` continuous projection |
| Fusion gates | Single `_lambda` for (prosody+timbre) | Per-stream `lambda_p, lambda_c, lambda_a, lambda_t` |
| Acoustic stream module | *(none)* | `AcousticEmbedding(B, 3, T80)` |
| Content stream module | *(none)* | `ContentEmbedding(B, 2, T80)` |
```

- [ ] **Step 3: Commit**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: finalize blocker recommendation and acceptance criteria"
```

---

## Task 2: Define Follow-Up Implementation Tasks

**Files:**
- Modify: `docs/spikes/issue-12-facodec-stream-contract.md`

- [ ] **Step 1: Add follow-up issue checklist**

Add to the end of the report:

```markdown
## Follow-Up Implementation Issues

### Issue #6 (Preprocessing)
- [ ] Rename `timbre_codebooks_idx` → `acoustic_codebooks_idx` in HF Features schema
- [ ] Stop averaging vq_id[3:6] — preserve 3-codebook axis as `[3, T80]`
- [ ] Capture `spk_embs` from decoder output (4th return value)
- [ ] Add `timbre_vector` field to schema: `Sequence(float32)`, D=256
- [ ] Update `content_codebooks_idx` to preserve `[2, T80]` codebook axis
- [ ] Update `FACodecEncoder.encode()` return contract: 3-tuple → 4-tuple
- [ ] Update mock mode to match real tensor shapes
- [ ] Update tests: 20+ assertions in `test_facodec_encoder.py`, 4 areas in `test_dataset_processor.py`
- [ ] Update CLI and reporting references to old field name

### Issue #7 (Deep Modules)
- [ ] Replace `TimbreEmbedding` with `TimbreProjection` (discrete → continuous)
- [ ] Create `AcousticEmbedding` module (3 codebook tables, sum strategy)
- [ ] Create `ContentEmbedding` module (2 codebook tables, sum strategy)
- [ ] Update `ProsodyEmbedding` input shape from `[B, T80]` to `[B, 1, T80]`
- [ ] Replace single lambda with per-stream lambdas in `ResidualFusion`
- [ ] Add Stream Activation Config support to `ResidualFusion`
- [ ] Update `__init__.py` exports
- [ ] Update tests: replace TimbreEmbedding tests, add Acoustic/Content tests, expand fusion tests

### Issue #8 (MOSS-Audio Residual Extension)
- [ ] Consume `audio` for online semantic encoding via MOSS-Audio
- [ ] Consume `prosody_codebooks_idx` → ProsodyEmbedding → TemporalPool → P_t
- [ ] Consume `timbre_vector` → TimbreProjection → broadcast → T_t
- [ ] Wire ResidualFusion with per-stream lambdas inside MOSS-Audio forward path
- [ ] Implement Stream Activation Config YAML
- [ ] First experiment: Prosody only (λ_p enabled, all others disabled)
- [ ] Second experiment: Prosody + Timbre (λ_p + λ_t enabled)
```

- [ ] **Step 2: Commit**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: define follow-up implementation tasks for #6, #7, #8"
```

---

## Task 3: Verify CONTEXT.md Completeness

**Files:**
- Audit: `CONTEXT.md`

- [ ] **Step 1: Cross-check CONTEXT.md against spike findings**

Read `CONTEXT.md` and verify these entries are present and correct:
- FACodec definition: 1 prosody + 2 content + 3 acoustic + 1 global timbre vector ✓
- Acoustic Stream (a_t): `vq_id[3:]`, not to be conflated with Timbre Vector ✓
- Timbre Vector: utterance-level from `spk_embs` ✓
- Stream Dimensionality Contract: all streams output D_moss ✓
- Stream Activation Config: YAML controlling enabled streams ✓
- Residual Summation: per-stream λ gates ✓
- Flagged ambiguity: `timbre_codebooks_idx` was a mistake ✓

If any are missing or incorrect, add a note to the spike report. If all are present, add a confirmation:

```markdown
## CONTEXT.md Verification

All canonical terms verified against spike findings:
- FACodec tensor mapping matches confirmed shapes
- Acoustic Stream / Timbre Vector distinction is clear
- Stream Dimensionality Contract is documented
- Stream Activation Config is defined
- Per-stream lambdas are described
- Flagged ambiguity about `timbre_codebooks_idx` is documented

CONTEXT.md requires no changes from this spike.
```

- [ ] **Step 3: Commit**

```bash
git add docs/spikes/issue-12-facodec-stream-contract.md
git commit -m "docs: verify CONTEXT.md completeness against spike findings"
```

---

## Phase Completion Criteria
- [ ] All 5 acceptance criteria checked off in the spike report
- [ ] Blocker recommendation for #8 documented with explicit conditions
- [ ] Follow-up implementation checklist for #6, #7, #8 written
- [ ] CONTEXT.md verified as complete against spike findings
- [ ] Each task committed separately

## Handoff Notes
- The spike is complete when all acceptance criteria are checked off and the follow-up list is published.
- Issue #12 can be closed after Phase 3 with the spike report as the artifact.
- The old single-file plan (`docs/plans/2026-05-06-issue-12-facodec-stream-contract.md`) can be archived or deleted since this layered plan supersedes it.
