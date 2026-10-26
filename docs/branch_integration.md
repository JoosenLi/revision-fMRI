# Branch integration

Remote state checked on 2026-10-26: main at 85f790b includes PR #1 (methodology audit), PR #2
(encoding reference), and PR #3 (DeepNSD source reuse).

Recommended remaining review/merge order:

1. `implementation/laion-data-adapter` into main: NIfTI/TSV adapter and pilot tools.
2. `implementation/roi-response-protocol` into main: explicit scientific primitives
   and collaborator-neutral README execution guidance.
3. `implementation/cached-evaluation` into main: immutable raw feature artifacts,
   pinned explicit splits and the shared cache-to-evaluation path.

These branches are stacked: each includes its predecessor. Reviewing and merging
in order makes each subsequent main comparison smaller. Prefer merge commits
while the stack remains open, preserving ancestry. If using squash/rebase merges,
rebase the remaining branches onto the resulting main before opening their PRs;
otherwise earlier commits can reappear in commit lists. Do not independently
cherry-pick the same changes from multiple branches. The next-stage branch also
incorporates the existing main merge history.

The older `setup/methodology-audit` and `implementation/encoding-reference`
branches, plus `implementation/deepnsd-reuse`, need no additional merge. No branches were deleted. No automatic merge
or direct main push was performed. Scientific changes still need three-person
review; an implementation branch is not an approved experimental protocol.
