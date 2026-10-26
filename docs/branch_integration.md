# Branch integration

Remote state checked on 2026-10-26: main at d3ccf69 includes PR #1 (methodology audit), PR #2
(encoding reference), PR #3 (DeepNSD source reuse), and PR #4 (LAION data adapter).

Recommended remaining review/merge order:

1. `implementation/roi-response-protocol` into main: explicit scientific primitives
   and collaborator-neutral README execution guidance.
2. `implementation/cached-evaluation` into main: immutable raw feature artifacts
   and the shared cache-to-evaluation path.
3. `implementation/official-laion-splits` into main: pinned official shared-pool
   assignments, trial coverage diagnostics and upstream parity validation.

These branches are stacked: each includes its predecessor. Reviewing and merging
in order makes each subsequent main comparison smaller. Prefer merge commits
while the stack remains open, preserving ancestry. If using squash/rebase merges,
rebase the remaining branches onto the resulting main before opening their PRs;
otherwise earlier commits can reappear in commit lists. Do not independently
cherry-pick the same changes from multiple branches. The next-stage branch also
incorporates the existing main merge history.

The older `setup/methodology-audit` and `implementation/encoding-reference`
branches, plus `implementation/deepnsd-reuse` and
`implementation/laion-data-adapter`, need no additional merge. No branches were deleted. No automatic merge
or direct main push was performed. Scientific changes still need three-person
review; an implementation branch is not an approved experimental protocol.
