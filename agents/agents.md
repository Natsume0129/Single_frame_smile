# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Handoff Documents

**Keep one canonical handoff current. Preserve old context as backup.**

When the user asks for a handoff document, says "给我交接文档", or asks to update project state:
- Treat it as a request to update the current canonical handoff, not to create a parallel duplicate.
- Prefer a project-level file such as `codex_project_handoff.md`, `project_handoff.md`, or `handoff.md` over feature-specific handoffs.
- If multiple handoff files exist, read them first, choose the broadest and most current one as canonical, then rename superseded feature-specific handoffs to `*.backup_YYYYMMDD.md`.
- Never delete old handoff content. Back it up before replacing or superseding it.

Before editing a handoff:
- Read the existing handoff, relevant prompt/task files, recent implementation files, and current verification state.
- Check the worktree state so the handoff distinguishes committed work, local edits, generated outputs, and unrelated dirty files.
- Verify facts that can be verified locally: key paths, file names, test results, row counts, output directories, and known environment limitations.

When updating the handoff:
- Preserve the existing useful structure, headings, and historical context unless it is clearly obsolete.
- Replace stale status with current status and mark the update date.
- Include concrete paths, commands, validation results, known risks, open questions, and the next recommended steps.
- Clearly label information as verified, inferred, or not yet verified when certainty differs.
- Add only relevant supplements. Do not turn the handoff into a full project diary.
- Make the first section immediately useful for the next agent: what changed, what is canonical, what is safe to run, and what should not be touched.

After editing:
- State which handoff file was updated and which files, if any, were renamed as backups.
- Mention if no tests were run because the change was documentation-only.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
