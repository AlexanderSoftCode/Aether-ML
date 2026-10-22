<!-- Title: conventional commit, e.g. `fix(conv): ...`, `chore: ...` -->

## Context

<!-- Why this PR exists and what prompted it, in 2-4 sentences. Say how the
commits are split, and which commits are mechanical (review with `git diff -w`). -->

## <Area or fix>

<!-- One section per fix or area. Open with the old behavior and what it broke
(concrete example or numbers), then what changed. Leave this out for one-topic PRs. -->

## Behavior changes

<!-- Anything a user could notice: new exceptions, changed outputs, changes to
seeding or serialization. Write "None." if there are none. -->

## Testing

- `python3 -m unittest discover tests`: N tests pass on NumPy and CuPy (M skipped).
- <New regression tests, each confirmed to fail on the old code.>
- <Manual or GPU checks, with the hardware used.>

## Follow-ups (out of scope)

<!-- Known gaps or work saved for later. Delete this section if there are none. -->
