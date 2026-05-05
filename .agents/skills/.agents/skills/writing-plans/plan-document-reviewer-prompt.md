# Plan Document Reviewer Prompt Template

Use this template when dispatching a plan document reviewer subagent.

**Purpose:** Verify the current phase plan is complete, matches the spec, and is ready for implementation.

**Dispatch after:** A phase document is written (not after the entire plan — phases are progressive).

```
Task tool (general-purpose):
  description: "Review plan phase document"
  prompt: |
    You are a plan document reviewer. Verify this phase plan is complete and ready for implementation.

    **Phase to review:** [PHASE_FILE_PATH]
    **Strategy document:** [STRATEGY_FILE_PATH]
    **Spec for reference:** [SPEC_FILE_PATH]

    ## What to Check

    | Category | What to Look For |
    |----------|------------------|
    | Completeness | Current phase has no TODOs/placeholders, every step has concrete content |
    | Spec Alignment | Phase covers its portion of the spec requirements |
    | Task Decomposition | Tasks have clear boundaries, steps are actionable (2-5 min each) |
    | Buildability | Could an engineer follow this phase without getting stuck? |
    | Phase Boundaries | Clear completion criteria, handoff notes for next phase |

    ## Important: Progressive Disclosure Context

    This plan uses **progressive disclosure**:
    - The CURRENT phase should be fully detailed (no placeholders)
    - FUTURE phases are intentionally stubbed — do NOT flag them as incomplete
    - The strategy document provides context across phases

    Only review the current phase for completeness. Stubs in future phases are by design.

    ## Calibration

    **Only flag issues that would cause real problems during implementation.**
    An implementer building the wrong thing or getting stuck is an issue.
    Minor wording, stylistic preferences, and "nice to have" suggestions are not.

    Approve unless there are serious gaps:
    - Missing requirements from the spec for THIS phase
    - Contradictory steps
    - Placeholder content in the CURRENT phase
    - Tasks so vague they can't be acted on

    ## Output Format

    ## Phase Review

    **Status:** Approved | Issues Found

    **Issues (if any):**
    - [Task X, Step Y]: [specific issue] - [why it matters for implementation]

    **Phase Boundary Check:**
    - Completion criteria: Clear / Unclear
    - Handoff notes: Present / Missing

    **Recommendations (advisory, do not block approval):**
    - [suggestions for improvement]
```

**Reviewer returns:** Status, Issues (if any), Phase Boundary Check, Recommendations

---

# Strategy Document Reviewer Prompt Template

Use this for reviewing the initial strategy.md before any implementation.

```
Task tool (general-purpose):
  description: "Review plan strategy document"
  prompt: |
    You are reviewing a plan strategy document. This is the high-level direction, not detailed tasks.

    **Strategy to review:** [STRATEGY_FILE_PATH]
    **Spec for reference:** [SPEC_FILE_PATH]

    ## What to Check

    | Category | What to Look For |
    |----------|------------------|
    | Goal Clarity | Can you understand what we're building in one sentence? |
    | Architecture | Is the high-level approach sensible? |
    | Phase Sequence | Do phases flow logically? Are dependencies clear? |
    | Scope Boundaries | Is each phase's outcome testable and bounded? |
    | Open Questions | Are unknowns identified? |

    ## Output Format

    ## Strategy Review

    **Status:** Approved | Needs Revision

    **Concerns (if any):**
    - [Area]: [concern and why it matters]

    **Phase Sequence Check:**
    - Phase dependencies: Clear / Unclear
    - Scope per phase: Well-bounded / Overlapping

    **Questions for Planner:**
    - [questions that should be resolved before Phase 1 detailed planning]
```
