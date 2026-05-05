---
name: writing-plans
description: Use when you have a spec or requirements for a multi-step task, before touching code. Creates implementation plans in progressive disclosure layers rather than one massive document.
---

# Writing Plans (Progressive Disclosure)

## Overview

Build implementation plans in **progressive disclosure layers** rather than one massive document. Start with high-level strategy, then expand phases iteratively as you approach implementation. This prevents plan fatigue and allows adaptation based on early learning.

**Core principle:** Reveal detail only when needed. The implementer shouldn't need to read 50 tasks to understand the first 3.

**Announce at start:** "I'm using the writing-plans skill to create an implementation plan with progressive disclosure."

## Plan Structure (Layered)

```
docs/plans/YYYY-MM-DD-<feature-name>/
├── README.md              # Executive summary + current status
├── strategy.md            # Goals, architecture, constraints
├── phase-01-<name>.md     # Phase 1: high-level tasks + detailed steps
├── phase-02-<name>.md     # Phase 2: stubbed initially, detailed before implementation
├── phase-03-<name>.md     # Phase 3: stubbed initially, detailed before implementation
└── decisions.md           # ADR-style log of decisions made during planning
```

## Layer 1: Strategy Document (strategy.md)

Create this first. It defines the overall direction without prescribing every detail.

```markdown
# [Feature Name] - Strategy

## Goal
One sentence describing what this builds.

## Architecture
2-3 sentences about the high-level approach. What are the major components?

## Tech Stack
Key technologies/libraries.

## Constraints & Assumptions
- Known limitations
- Dependencies on other systems
- Risks or unknowns

## Phases (High-Level)

### Phase 1: [Name] - Foundation
**Outcome:** What's working after this phase?
**Rough scope:** 2-3 sentences about what's included

### Phase 2: [Name] - Core Feature
**Outcome:** What's working after this phase?
**Rough scope:** 2-3 sentences about what's included
**Depends on:** Phase 1

### Phase 3: [Name] - Polish/Integration
**Outcome:** What's working after this phase?
**Rough scope:** 2-3 sentences about what's included
**Depends on:** Phase 2

## Open Questions
Questions that need answers before or during implementation.
```

## Layer 2: Phase Documents

Each phase gets its own file. **Critical:** Only write detailed task specifications for the current/next phase. Future phases remain stubbed with just high-level descriptions until you're ready to implement them.

### Phase Document Structure

```markdown
# Phase N: [Name]

## Phase Goal
What specific, testable outcome marks this phase complete?

## Files to Touch
Before detailed tasks, map the file structure:

- `path/to/file.py` - Responsibility
- `path/to/test.py` - Test coverage
- `path/to/existing.py` - Modifications needed

## Tasks

### Task 1: [Component Name]

**Files:**
- Create: `exact/path/to/file.py`
- Test: `tests/path/to/test.py`

- [ ] **Step 1: Write the failing test**

```python
def test_specific_behavior():
    result = function(input)
    assert result == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/path/test.py::test_name -v`
Expected: FAIL with "function not defined"

- [ ] **Step 3: Write minimal implementation**

```python
def function(input):
    return expected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/path/test.py::test_name -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/path/test.py src/path/file.py
git commit -m "feat: add specific feature"
```

### Task 2: [Next Component]
...

## Phase Completion Criteria
- [ ] All tasks complete
- [ ] Tests passing
- [ ] Can demonstrate: [specific behavior]

## Handoff Notes
What the next phase implementer needs to know.
```

## Layer 3: Living README (README.md)

The README is the entry point. It stays current as phases complete.

```markdown
# [Feature Name] Implementation Plan

> **For agentic workers:** Use subagent-driven-development or executing-plans. Start with the current phase — don't read ahead.

## Quick Status
- **Current Phase:** Phase 1 - Foundation
- **Next Up:** Phase 2 - Core Feature (pending Phase 1 completion)
- **Overall Progress:** 0/3 phases complete

## Start Here
New implementer? Read in this order:
1. [strategy.md](./strategy.md) - Understand the big picture (5 min)
2. [Current phase document] - Only the phase you're implementing (15 min)
3. [decisions.md](./decisions.md) - Context on choices made (optional, 5 min)

**Do NOT read future phases.** They're stubbed and will change based on Phase 1 learnings.

## Phase Overview

| Phase | Status | Outcome | Document |
|-------|--------|---------|----------|
| 1 - Foundation | 🔲 Not Started | [Outcome] | [phase-01-*.md](./phase-01-*.md) |
| 2 - Core | 🔲 Pending | [Outcome] | Stub only |
| 3 - Polish | 🔲 Pending | [Outcome] | Stub only |

## Key Decisions
See [decisions.md](./decisions.md) for rationale on major choices.
```

## The Progressive Disclosure Workflow

### Step 1: Create Strategy Document (15-30 min)
- Define goal, architecture, phases at high level
- Identify open questions
- Create phase stubs (just names + outcomes, no details)
- **Stop here** — don't write detailed tasks for all phases

### Step 2: Detail Phase 1 Only (30-60 min)
- Write complete task specifications for Phase 1 only
- Include all code snippets, commands, file paths
- Leave Phase 2+ as stubs with just "Phase 2: [Name] - TBD"

### Step 3: Implement Phase 1
- Hand off to subagent-driven-development or executing-plans
- As Phase 1 completes, learnings emerge

### Step 4: Adapt & Detail Phase 2
- Update strategy.md if learnings change the approach
- Now (and only now) write detailed Phase 2 tasks
- Phase 2 plan incorporates Phase 1 reality, not speculation

### Step 5: Repeat
- Continue pattern: implement → learn → adapt next phase → repeat

## Decision Log (decisions.md)

Track why things are the way they are. This becomes crucial context for later phases and future maintainers.

```markdown
# Decision Log

## 2024-01-15: Chose [Library A] over [Library B]
**Context:** We needed X, Y, Z. Both libraries supported this.
**Decision:** Use Library A
**Rationale:** Better TypeScript support, smaller bundle
**Consequences:** Requires polyfill for older browsers

## 2024-01-20: Changed Phase 2 approach after Phase 1
**Context:** Original plan assumed X, but Phase 1 revealed Y
**Decision:** Split Phase 2 into 2a and 2b
**Rationale:** The monolithic approach would create too much risk
```

## File Granularity

**Each step is one action (2-5 minutes):**
- "Write the failing test" - step
- "Run it to make sure it fails" - step
- "Implement the minimal code to make the test pass" - step
- "Run the tests and make sure they pass" - step
- "Commit" - step

## No Placeholders in Detailed Tasks

When you DO write detailed tasks for a phase, they must be complete:

- ✅ "TBD", "TODO", "implement later" — allowed ONLY in future phase stubs
- ✅ "Add appropriate error handling" — allowed ONLY in future phase stubs
- ❌ In current phase: Every step must contain actual content
- ❌ In current phase: Code blocks required for code steps
- ❌ In current phase: Exact file paths always

## Execution Handoff

**"Plan initialized with progressive disclosure.**

**Strategy complete and saved to `docs/plans/<feature-name>/`.**

**Phase 1 is detailed and ready for implementation.**

**Phases 2+ are intentionally stubbed — they will be detailed after Phase 1 completes and we incorporate learnings.**

**Execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks

**2. Inline Execution** - Execute tasks in this session using executing-plans

**Which approach for Phase 1?"**

## After Each Phase Completes

1. Update README.md status table
2. Write decision log entries for anything learned
3. Review and update strategy.md if needed
4. NOW write detailed tasks for next phase
5. Offer to continue with next phase

## Why Progressive Disclosure?

**Problems with massive plans:**
- Plan becomes outdated as implementation reveals reality
- Reader overwhelmed by 50 tasks when they only need the first 3
- No room to adapt — sunk cost in detailed planning for unproven approaches
- Decision rationale lost in task noise

**Benefits of progressive disclosure:**
- Plans adapt to reality
- Implementer focuses only on current work
- Decisions documented explicitly
- Less planning waste on speculative future work
