# Decision Log

## 2026-05-18: Group B as a single plan
**Context:** Issue #14 has 9 sub-issues across 4 groups. Group B has 4 issues (#20, #21, #22, #23) that form a sequential data pipeline with one parallel component.
**Decision:** Create a single progressive-disclosure plan for all Group B issues rather than separate plans per issue. Phase them sequentially (Enrichment → Generation → Filter) with the FACodec processor built in Phase 1 alongside enrichment.
**Rationale:** The issues share a common data flow and same intermediate formats. A unified plan prevents the format contract between scripts from drifting.
**Consequences:** Phase documents cover multiple GitHub issues. Each phase completion updates the corresponding issue.

## 2026-05-18: NV tags stored as emoji-to-text mapping, not kept as emojis
**Context:** NVTTS stores paralinguistic tags as emojis (🌬️, 🤣, 😷) in the `Result` column. DeepSeek prompts need readable text labels.
**Decision:** A dedicated mapping module (#17) converts emojis to `[Tag]` format before prompts. The mapping is bidirectional (emoji→tag and tag→emoji) for debuggability.
**Rationale:** DeepSeek may not reliably interpret arbitrary emojis as paralinguistic vocalizations. Text labels are unambiguous.
**Consequences:** The `transcript_with_tags` column in output uses `[Tag]` format, not emojis.

## 2026-05-18: Cosine similarity stored, not used for hard filtering
**Context:** After pair generation, cosine similarity between chosen and rejected responses indicates whether paralinguistic context was decisive.
**Decision:** Store similarity as a column. Manual threshold selection later. No samples dropped during the pipeline.
**Rationale:** The optimal threshold is unknown until empirical review. Dropping samples early is irreversible.
**Consequences:** The intermediate parquet retains all samples. The training script or a post-hoc filtering step applies the threshold.

## 2026-05-18: PreferenceDatasetProcessor skips content and acoustic streams
**Context:** The DPO experiment only uses prosody and timbre. FACodec encoding produces 4 streams by default.
**Decision:** Modify PreferenceDatasetProcessor to encode only prosody + timbre, skipping content and acoustic codebooks entirely.
**Rationale:** Saves compute (fewer VQ lookups) and storage (fewer columns). Content/acoustic can be added back for ablation studies if needed.
**Consequences:** Output parquet has no `content_codebooks_idx` or `acoustic_codebooks_idx` columns.

## 2026-05-18: Async batched DeepSeek API with per-sample checkpointing
**Context:** ~4K API calls with potential rate limits and failures. Need to resume without redoing completed work.
**Decision:** Async batched calls with configurable concurrency. Append each completed sample to JSONL immediately. On restart, load existing JSONL to get completed IDs and skip them.
**Rationale:** JSONL append is trivially resumable and requires no locking. Async calls maximize throughput under API constraints.
**Consequences:** Partial output file is always valid. Resume is O(n) to scan completed IDs.

## 2026-05-18: No prompt column in final schema
**Context:** Original issue #14 described a per-sample `prompt` column for DPO training input.
**Decision:** Drop the prompt column. Use a uniform system prompt applied via MOSS-Audio's chat template for all samples.
**Rationale:** The model must derive all paralinguistic understanding from the audio alone. Per-sample prompts would leak context and defeat the hypothesis.
**Consequences:** Final parquet has no `prompt` column. DPO training collator applies the system prompt uniformly.
