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
