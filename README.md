# re:vision — Conwell 2024 / LAION-fMRI

Methodological fidelity first. Start with [original_paper.md](docs/original_paper.md), [methodology_audit.md](docs/methodology_audit.md), [deviations.md](docs/deviations.md), [model_manifest.md](docs/model_manifest.md), and [roi_mapping.md](docs/roi_mapping.md). The [implementation and unit-test plan](docs/implementation_plan.md) distinguishes working foundations from future numerical backends.

The shared `revision_fmri` package currently implements local configuration/checkpoint validation, data and feature contracts, raw cRSA and train-only layer selection. A [small-array encoding reference](docs/encoding_reference.md) now implements SRP, explicit-normalization float64 Pearson-LOOCV ridge, veRSA and held-out layer evaluation. Dataset/model/hook/statistical backends remain pending. **It does not yet run the complete reproduction.** No vision training/fine-tuning, GPU experiment, large dataset or checkpoint download is part of this stage.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
cp configs/paths.example.yaml configs/paths.local.yaml
# Edit the ignored local file to point to your existing resources.
export REVISION_FMRI_CONFIG="$PWD/configs/paths.local.yaml"
python -m pip install -e '.[test,encoding]'
python -m pytest -q
```

Every collaborator keeps their own roots. Paths are relative to the YAML file, and checkpoint bindings are relative to `checkpoint_root`. Alternatively pass the config filename directly to `load_config`. Storage and stage-specific compute precision are read from the same YAML loader.
Features default to float32; fidelity-reference ridge defaults to float64.
No paths are inferred from the source author's machine. Do not commit local YAML files, datasets, checkpoints or feature caches.

```python
from revision_fmri.config import load_config
from revision_fmri.evaluation import evaluate_crsa, select_layer

settings = load_config()
# Dataset/feature adapters must supply ID-aligned ResponseSet/FeatureSet objects.
# score = evaluate_crsa(raw_features, responses)
# selected = select_layer(training_scores_for_one_subject_roi_metric)
```

The 224-model inventory is `manifests/models.json`. None is yet approved for feature extraction; missing checkpoint/library/transform provenance must be resolved first. Reference results remain metadata, never optimization targets.

Run local debugging/tests on `is248155`; use its RTX 6000 GPUs for future authorized model work, with no CPU fallback. Long authorized tasks go through `ssh jz`. Current array/configuration tests neither initialize GPUs nor train a network.

## Provenance

Audit source revision and per-file digests are in `docs/provenance/source_inventory.json`; public source URLs are recorded there and in the audit. `.audit_sources/` is an ignored inspection cache, not a runtime dependency. Rebuild `manifests/models.json` from an existing cache with:

```bash
python scripts/build_audit_manifest.py --sources .audit_sources --output manifests/models.json
```

No upstream source is imported at package runtime. DeepNSD is GPL-3.0; any future vendoring/port must retain applicable notices and choose compatible redistribution terms. Source documents/PDFs remain in the local ignored audit cache. The current numerical API is an independently written small formula implementation, not a copied legacy ridge backend.

Shared development rules live in `AGENTS.md`. Local workstation instructions live in ignored `AGENTS.local.md`; agents must read it when local environment details are needed. Scientific changes require review by all three collaborators before merge.

The encoding reference revealed an additional source discrepancy: on the pinned sklearn 1.3.2 environment, the released positional preprocessing call enables L2 normalization. See D19 before choosing a protocol; this is not claimed as the recovered historical environment.
