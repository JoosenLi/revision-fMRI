# re:vision — LAION-fMRI Replication

## Scientific goal

Faithfully reproduce the methodology of Conwell et al. (Nature Communications,
2024) using LAION-fMRI beta responses. Prioritize scientific fidelity,
reproducibility and shared interfaces over speed or model ranking. Read
`docs/methodology_audit.md`, `docs/deviations.md` and the relevant model/ROI
manifest before implementing scientific behavior. Distinguish paper evidence,
released-code behavior, LAION adaptations and unresolved details.

## Local environment and configuration

Read `AGENTS.local.md`, if present, when local hardware, environments, storage
or execution details are needed. It is an explicitly referenced local convention,
not a substitute for program configuration. Do not copy its institutional paths
into shared code, documentation, tests or manifests.

All runtime storage paths and compute settings must come through
`revision_fmri.config.load_config`, using an explicit YAML filename or
`REVISION_FMRI_CONFIG`. Each collaborator creates ignored
`configs/paths.local.yaml` from `configs/paths.example.yaml`.
Do not hardcode local paths or GPU indices. Keep large data, checkpoints,
features, temporary scientific arrays and results on configured storage roots.
Code and small test fixtures can remain local. Never modify institutional source
data or overwrite shared caches without explicit authorization.

## Shared APIs

Use and extend the single shared package:

- `revision_fmri.data`: image/voxel IDs, responses, repeat and ROI provenance.
- `revision_fmri.models`: model identity, local checkpoint verification, loaders.
- `revision_fmri.features`: transforms, hook outputs and feature provenance.
- `revision_fmri.metrics`: cRSA and future shared veRSA metric primitives.
- `revision_fmri.evaluation`: splits, encoding, layer selection and evaluation.
- `revision_fmri.statistics`: statistical specifications and future adapters.

Do not create parallel implementations of cRSA, veRSA, ROI selection, splitting
or configuration inside analysis scripts. Some interfaces are contracts only;
do not present them as validated numerical backends.

## Scientific requirements

- Do not train or fine-tune vision networks. Freeze pretrained models and use eval.
- Match stimulus IDs explicitly to beta rows; preserve all repeats within splits.
- Preserve checkpoint-specific transforms and the audited hook/layer procedure.
- Never use test responses to fit, tune or select layers in independent evaluation.
  The released test-dependent layer filter is a documented discrepancy; reproducing
  it requires a separately named legacy protocol, never silent inclusion.
- Never silently substitute modern defaults for historical methods or hyperparameters.
- Keep feature inference/storage float32 by default. The original ridge path uses
  float64; use float64 for the fidelity reference. A float32 ridge path requires
  explicit configuration, documented deviation and numerical parity evidence
  covering alpha selection, predictions, RSA scores and selected layers. Do not
  enable mixed precision, float16/bfloat16 or TF32 without review.
- Record data/split/ROI/checkpoint/transform/layer hashes, versions, precision and
  seeds in results and cache keys. Reference scores are not optimization targets.
- Changes to metric definitions, original hyperparameters, ROI/reliability rules,
  repetition handling, splits or precision policy require review by all three
  collaborators before merging.

## GPU and execution

Check available GPU memory and running processes before model runs; configured
GPU IDs are an allowance, not proof that a GPU is idle. Do not load an entire
dataset into GPU memory. Never fall back to CPU vision training or silently
move model inference to CPU. Tiny CPU unit tests and reference linear algebra
are permitted and are not vision-network training.

The current phase authorizes documentation, APIs and small tests only. Do not
download large datasets/checkpoints or launch GPU experiments until separately
instructed. Use local execution/remote-job rules from `AGENTS.local.md`.

## Development and merging

Use English for code, comments and technical documentation; discuss this project
with the user in Chinese. Work on feature branches when Git is initialized;
keep changes small and independently reviewable. Run relevant tests and package
checks before opening a PR. State which backends remain unimplemented and which
checks were not run. No direct pushes to main after initialization. No datasets,
checkpoints, local settings or secrets in Git. Destructive operations need
explicit authorization. Do not initialize a remote or push merely because this
file describes a Git workflow.
