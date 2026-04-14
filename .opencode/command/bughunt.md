---
description: "Fix a bug: RCA → regression test → fix → error KB entry"
agent: orchestrator
model: openai/gpt-5.4
temperature: 0.2
---

# Bug Hunt

**Problem**: $ARGUMENTS

## Protocol

1) Load mission and error-kb skills:

   ```bash
   skill "core_spec_mission"
   skill "knowledge_spec_error_kb"
   ```

2) **Understand before fixing**:
   - Read the stack trace carefully
   - Reproduce the issue (write a failing test first)
   - Build a mental model of what SHOULD happen
   - Identify the gap between expected and actual

3) **Root Cause Analysis**:
   - WHY did this happen, not just WHAT happened
   - Is this a symptom or the root cause?
   - Are there similar bugs lurking nearby?
   - What assumption was violated?

4) **Fix with evidence**:
   - Write regression test BEFORE fixing
   - Fix the root cause, not the symptom
   - Verify fix doesn't break other things
   - Run full test suite

5) **Document in Error KB**:
   - Add entry to `docs/agent/error-kb.md`
   - Include: symptoms, root cause, fix, prevention
   - Make it findable for future similar issues

## Deliverables

- [ ] Failing test that reproduces the bug
- [ ] Root cause identified and documented
- [ ] Fix applied (minimal, focused)
- [ ] All tests pass (new + existing)
- [ ] Error KB entry added
- [ ] Spec updated if behavior was unclear

## Anti-Patterns to Avoid

- Fixing symptoms without understanding cause
- "It works now" without understanding why
- Skipping the regression test
- Not checking for similar issues elsewhere
