# DeepNSD source attribution and adaptation notice

Source: https://github.com/ColinConwell/DeepNSD
Commit: 36f6a34727dccb5eec8fa49364b210a049a388d7
Authors: Colin Conwell and the DeepNSD contributors.
Paper: Conwell, Prince, Kay, Alvarez & Konkle (2024), Nature Communications 15,
9383, https://doi.org/10.1038/s41467-024-53147-y.
Upstream license: GNU General Public License version 3; complete text in LICENSE.

Adapted by the re:vision collaboration on 2026-10-26:

- ridge_gcv_mod.py: copied byte-for-byte, including original comments. No fixes
  to normalization, private sklearn calls, alpha scoring or stored predictions.
- rsa.py: compare_rdms copied verbatim from model_opts/mapping_methods.py;
  module imports reduced to its dependencies.
- reduction.py: the SRP/JL functions and their cache helpers copied verbatim
  from model_opts/feature_reduction.py; module imports reduced to dependencies.
  The public adaptation calls only the ndarray, no-cache path.
- vision.py: get_torchvision_model and get_torchvision_transforms copied
  verbatim from model_opts/model_options.py; only torchvision.transforms is
  imported at module scope. The public adapter uses a fixed architecture and
  disables pretrained downloading before loading verified local weights.
- manifest.json: exact original file hashes, excerpt line ranges/function hashes
  and redistributed file hashes. scripts/vendor_deepnsd.py reproduces extraction.

The ridge file identifies scikit-learn's sklearn/linear_model/_ridge.py as its
source. scikit-learn's BSD 3-Clause notice is retained in SKLEARN_COPYING.
No other DeepNSD methods, checkpoints or participant data are bundled here.
The surrounding validation/configuration adapters are separate from these files.
The combined project distribution carries GPL-3.0 in the repository root LICENSE.
