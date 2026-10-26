# Validation record — 2026-10-24

Scope: audit documents, model/source inventories and shared API foundation.
This is not a validation of full Conwell numerical reproduction or LAION results.

| Check | Result |
|---|---|
| Local host | is248155 |
| New project environment | Isolated `.venv`, editable package install successful |
| Unit tests | `.venv/bin/python -m pytest -q`: 20 passed |
| Dependency consistency | `pip check`: no broken requirements |
| Import behavior | All seven API modules import without importing torch |
| Configuration | Local YAML loaded with explicit CUDA allowance and separate feature/ridge precision |
| Private-file exclusions | Git ignore behavior checked in a temporary repository; local AGENTS/YAML, environment, source cache and data excluded; shared AGENTS/template/code/manifest included |
| Publication inventory | 224 unique model IDs; controlled comparison membership checked |
| Source continuity | Eight key method files byte-identical between audited current commit and 2024-11-02 release |
| Packaging | Initial wheel build/install smoke passed; updated YAML package editable build/install passed in isolated environment |

The installed environment uses NumPy 2.2.6, SciPy 1.15.3, PyYAML 6.0.3 and
pytest 9.1.1. This environment validates the new foundation only. It is **not**
asserted to reproduce the original sklearn, torchvision, timm or R environment.
The full local dependency snapshot is ignored under `.venv/installed-packages.txt`.

No datasets, checkpoint weights or large feature caches were downloaded; no vision
network was trained/fine-tuned; no GPU inference experiment or remote job ran.
GPU inspection was read-only. No institutional data were changed, and configured
resource subdirectories were not created or assumed populated. Ridge/SRP, model
loading/extraction, LAION ROI reconstruction and statistical backends remain
unimplemented and have no numerical parity claim.

At the initial validation, the workspace exposed only a `.git` placeholder;
private-file rules were checked in a temporary repository. After the collaboration
repository was supplied, local Git was initialized and connected to
`https://github.com/JoosenLi/revision-fMRI`. The audit/API foundation is prepared
on `setup/methodology-audit` for review against an empty initialization commit on
`main`. Publication success must be verified against remote commit IDs.

## Encoding reference update — 2026-10-26

On `is248155`, 42 tests passed in the project environment with NumPy 1.26.4, SciPy 1.11.4 and sklearn 1.3.2. The earlier 2026-10-24 environment snapshot above describes the foundation-only run. Unmodified DeepNSD RidgeCVMod was executed on five synthetic fixtures; explicit source hash and runtime versions are stored with the fixtures. Small-array SRP/ridge/veRSA and held-out layer evaluation are now implemented. The positional normalization issue is documented as D19. Original historical environment, full-data numerical parity, float32 ridge, GPU execution, data/model/ROI/statistical adapters remain unverified or pending. No real-data/GPU experiments or network training ran. See `encoding_reference.md`.

## Source-reuse update — 2026-10-26

53 tests passed after routing the canonical backend through vendored DeepNSD ridge, SRP and RSA code. Added tests verify original source bytes, bundled license resources, actual invocation of upstream RidgeCVMod, compatibility-version rejection, canonical settings and original even/odd splitting. Scientific experiment restrictions remain unchanged.

## Shared API update — 2026-10-26

61 tests passed on `is248155`, including public pipeline evaluation, equivalent
reports across different local roots, JSON/digest roundtrip, overwrite and path
traversal rejection, protocol mismatch, ID alignment and precision enforcement.
The facade exposes API version 1 and preserves the underlying source backend;
data/model/hook/statistics adapters remain contracts, not completed backends.

The wheel built with the host packaging tools and passed source-hash checks and
the synthetic public-API example when imported directly from the wheel outside
the checkout, using the project's pinned numerical environment. `pip check`
passed. A first build attempt with `.venv --no-build-isolation` failed because
that environment lacks `wheel` and a sufficiently recent setuptools; it was
not used for packaging. This does not affect the numerical runtime checks.

## Local data adapter and authorized acquisition — 2026-10-26

83 tests pass, including genuine tiny NIfTI/TSV files, shuffled trial rows,
run-local and session-global indexing, unequal repeats, missing/duplicate indices,
invalid session/ROI identity, affine/shape mismatches, nonfinite data and explicit
aggregation policy. Downloader tests verify conditional object identity, checksum
receipts, no overwrite, partial-file retention and path containment.

The authorized public pilot acquisition completed: sub-01/ses-01 beta NIfTI
(1,155,267,333 bytes), seven JSON/TSV metadata files and a 67,979-byte
laion-general mask. Storage uses local YAML and preserves original S3 keys;
receipts record ETags and SHA-256. No gated image agreement was submitted and
no raw stimulus images, checkpoints or raw BOLD were downloaded.

Real-data I/O smoke: beta shape `(143, 143, 143, 1044)`, 1044 trial rows,
769 unique labels; mask has 12020 selected voxels with matching geometry.
Three singly presented images returned a finite `(3, 12020)` ResponseSet through
the public adapter. This is not a whole-session finite-data check, an OTC
selection, an encoding experiment or a complete reproduction. The initial
session-global-index attempt failed, correctly detecting the actual run-local
indices; D20 documents the explicit correction and pinned official-loader evidence.

Pilot beta SHA-256: `0828db6bea2d0674cddbe20460296bb43e8f414d530e6fc7f7f49ede5ca97275`.
Trial TSV SHA-256: `77c7e3c1360272bec1384b019e87306949acd777adf9a8093d93e1a9b931df73`.
Mask SHA-256: `fe3ceabbb9c80a6937c095e640154938dbce6e4fa69ae1fb3e1250fa9f7a30f0`.

Long tasks now run locally per explicit user instruction; private workstation
rules were updated and remain ignored. No GPU experiment or training was run.

## ROI and response primitives — 2026-10-26

98 unit tests passed, including analytic OTC union/threshold/identity cases and
response normalization scope, axis, dtype and held-out perturbation checks.
These are tiny synthetic tests; no new full-session preprocessing, ROI generation
or GPU experiment was run. The shared README now delegates machine details to
ignored local instructions.

## Cached evaluation — 2026-10-26

118 tests passed. The 20 new cache/split tests also passed against the built wheel,
with the package imported from that wheel rather than the source tree. Dependency
consistency and diff whitespace checks passed. Importing the package did not
import torch. Tests use only tiny synthetic arrays and temporary configured roots.

Cached and direct evaluation agree on metrics and per-voxel alphas. Copying the
same artifacts to different roots produces identical reports. Tests cover source
identity changes, image order/precision changes, tampered metadata/values,
incomplete writes, no overwrite, symlinks, split digest/partition errors,
candidate mismatches and response/split dataset-release mismatches.

No existing shared cache was overwritten, no scientific method was changed,
and no real feature extraction, additional dataset/checkpoint download, GPU
experiment or whole-data evaluation ran. Official LAION split import and vision
loaders/hooks remain unimplemented.

## Official LAION split import — 2026-10-26

127 offline tests passed. Separately, all twelve downloaded official shared-pool
partitions matched the pinned upstream loader
`cf82765ced029de81c3bec7254b8cbd6a65e3686`, including complete ID lists and order.
Both five-fold families passed disjointness/exhaustive-coverage checks. Pilot TSV
coverage was checked without reading beta values; the single session is incomplete
for official shared-pool evaluation. See `official_laion_splits.md` for counts.
No new beta/checkpoint/image download, GPU work or scientific evaluation occurred.

The nine new split tests and official-source parity checks also passed with the
package imported from the built wheel. Repeated-label masks and outside-pool
exclusion matched upstream. The downloader reverified all existing small files
without replacing them; dependency and whitespace checks passed.
