# re:vision — Conwell 2024 / LAION-fMRI

Methodological fidelity first. Start with [original_paper.md](docs/original_paper.md), [methodology_audit.md](docs/methodology_audit.md), [deviations.md](docs/deviations.md), [model_manifest.md](docs/model_manifest.md), and [roi_mapping.md](docs/roi_mapping.md). The [implementation and unit-test plan](docs/implementation_plan.md) distinguishes working foundations from future numerical backends.

The shared `revision_fmri` package exposes a [standard API](docs/shared_api.md) for configuration, ID-aligned data/features, evaluation and portable results. `ReproductionPipeline` encapsulates original DeepNSD SRP, float64 Pearson-LOOCV ridge and RSA source, with documented train-only layer selection. A [local NIfTI/TSV data adapter](docs/laion_data_adapter.md) is implemented; ROI construction and model/hook/statistical backends remain pending. **It does not yet run the complete reproduction.** No vision training/fine-tuning or GPU experiment has run. A public pilot data download is authorized; checkpoint downloads remain pending.

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
from revision_fmri import DeepNSDProtocol, ReproductionPipeline

# Supply a reviewed projection count/policy and ID-aligned input objects.
pipeline = ReproductionPipeline.from_config(protocol=DeepNSDProtocol(
    "team-reviewed-laion-v1", projection_count, projection_policy))
report = pipeline.evaluate_model(features=candidate_features,
                                 responses=responses, split=split)
pipeline.save_report(report, run_id="subject01-model01-otc-v1")
```

For a runnable small-array demonstration, see `examples/evaluate_synthetic.py`.

The 224-model inventory is `manifests/models.json`. None is yet approved for feature extraction; missing checkpoint/library/transform provenance must be resolved first. Reference results remain metadata, never optimization targets.

Each collaborator keeps workstation details and execution rules in ignored
`AGENTS.local.md`, and runtime paths/device settings in their local YAML.
Current array/configuration tests neither initialize GPUs nor train a network.

## Provenance

Audit source revision and per-file digests are in `docs/provenance/source_inventory.json`; public source URLs are recorded there and in the audit. `.audit_sources/` is an ignored inspection cache, not a runtime dependency. Rebuild `manifests/models.json` from an existing cache with:

```bash
python scripts/build_audit_manifest.py --sources .audit_sources --output manifests/models.json
```

The canonical backend now directly executes pinned DeepNSD source. `ridge_gcv_mod.py` is byte-identical; RSA/SRP function bodies are exact excerpts with minimal imports. See [upstream reuse](docs/upstream_reuse.md) for provenance, adaptation boundaries and usage. Original GPL-3.0 and scikit-learn BSD notices are bundled; the combined project is distributed under the root [LICENSE](LICENSE). Source PDFs remain in the ignored audit cache. The independent SVD backend is retained only for numerical cross-checks.

Shared development rules live in `AGENTS.md`. Local workstation instructions live in ignored `AGENTS.local.md`; agents must read it when local environment details are needed. Scientific changes require review by all three collaborators before merge.

The encoding reference revealed an additional source discrepancy: on the pinned sklearn 1.3.2 environment, the released positional preprocessing call enables L2 normalization. See D19 before choosing a protocol; this is not claimed as the recovered historical environment.
