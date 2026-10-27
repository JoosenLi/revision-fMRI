# Official LAION shared-pool splits — secondary analyses

The importer reads the original assignment files from
[LAION-fMRI commit cf82765](https://github.com/ViCCo-Group/LAION-fMRI/tree/cf82765ced029de81c3bec7254b8cbd6a65e3686/laion_fmri/splits).
It never regenerates random, CLIP-cluster, tau or OOD partitions. This is a named
LAION adaptation (D09) for **secondary extension analyses**, not the primary
replication protocol. The primary analysis uses the original-style even/odd
split-half over 1121 non-OOD images (561/560); see
[primary LAION protocol](primary_laion_protocol.md).

The package includes only a small lock manifest of paths, sizes and SHA-256
hashes (`revision_fmri/resources/laion_shared_splits.json`). Actual official JSON
files and the inspected reference loader live under configured data_root:
`official_splits/<commit>/laion_fmri/splits/`. All 13 files total 693161 bytes.
No stimulus images, beta arrays or model weights are part of this acquisition.

```bash
python scripts/download_official_splits.py --config configs/paths.local.yaml
python scripts/validate_official_splits.py --config configs/paths.local.yaml
```

The downloader verifies pinned hashes before writing, verifies existing files,
and never overwrites them. Interrupted writes fail verification on rerun and
require explicit inspection. The validation script verifies the inspected
upstream Python file's hash before loading it and compares every assignment
and its order against the upstream `get_train_test_ids` function. Unit tests
are entirely offline and do not require these downloaded resources.

## API

```python
from revision_fmri import load_official_split, partition_trials
from revision_fmri.config import load_config

config = load_config()
split = load_official_split(config, name="tau", dataset_release=reviewed_release)
# split is RecordedSplit, usable directly with evaluate_cached_model.
partition = partition_trials(trial_labels, split=split)
# All repeats of a label have the same membership.
```

The dataset_release is explicitly supplied and must match response provenance
when evaluating through the cached API; the source repository revision alone
is not a verified release identity for a collaborator's local beta dataset.
The result carries the pinned commit and exact source JSON digest. The protocol
ID contains the pool, split name, variant and commit. No ID sorting, filename
normalization or automatic sample intersection occurs.

The upstream JSON's `pool` value is a human-readable description. As in the
upstream loader, the API pool identifier is `shared`, derived from the resource
location. Both the original description and counts are checked against the lock.

Supported: shared-pool `tau`, `cluster_k5_0` … `cluster_k5_4`, `random_0` …
`random_4`, and full `ood`, with the existing variant 0. Subject/pooled pools,
OOD-category subsetting, new source revisions and fold-score aggregation are
not implemented here. Unknown choices fail; there is no silent fallback.

## Coverage and repeats

`partition_trials` returns train, test and excluded boolean masks, and the missing
train/test image IDs. Repeated labels remain together. Trials outside the pool
are explicitly excluded, matching upstream exact-string membership behavior.
The default requires every official train/test image to appear. For diagnostics
only, `require_complete=False` reports missing images without changing the split.
This option does not authorize a reduced scientific evaluation. The existing
EvaluationSplit coverage check still rejects incomplete response sets.

`validate_holdout_family` checks five distinct folds with matching release,
revision and pool, equal image universes, disjoint held-out sets and complete
held-out coverage. It is called separately for cluster and random families.
It does not perform layer selection or statistical averaging.

## Verified source parity — 2026-10-26

All train/test lists, including their order, matched the inspected upstream loader.
Repeated-trial masks and outside-pool exclusion also matched upstream. The nine
new unit tests and actual upstream comparison passed against the built wheel:


| Split | Train | Test |
| --- | ---: | ---: |
| tau | 897 | 224 |
| cluster_k5_0 | 847 | 274 |
| cluster_k5_1 | 873 | 248 |
| cluster_k5_2 | 875 | 246 |
| cluster_k5_3 | 896 | 225 |
| cluster_k5_4 | 993 | 128 |
| random_0 | 896 | 225 |
| random_1 … random_4 | 897 | 224 each |
| ood | 1121 | 371 |

Both five-fold families passed disjoint/exhaustive checks. No upstream generation
algorithm was run and no model was used to alter assignments.

The existing sub-01/ses-01 pilot's 1044 trials do not cover the shared pool.
For tau: 340 train trials, 93 test trials, 611 excluded trials; 639 train images
and 149 test images are absent. For OOD: 433 train trials, 36 test trials,
575 excluded trials; 788 train and 348 test images are absent. These are metadata
coverage diagnostics, not model scores. Further session coverage must be obtained
before a full official shared-pool experiment.
