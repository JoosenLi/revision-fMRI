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
