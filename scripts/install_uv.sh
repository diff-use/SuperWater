#!/usr/bin/env bash
#
# Create a reproducible, uv-managed SuperWater environment from uv.lock.
#
# Resolves the full GPU stack (PyTorch 2.8 + CUDA 12.6, the matching PyTorch Geometric
# extension wheels, e3nn, rdkit, fair-esm, ...) pinned in pyproject.toml / uv.lock, and
# installs the superwater package + console scripts into ./.venv.
#
# Usage:
#     bash scripts/install_uv.sh                       # runtime env
#     EXTRAS="cu126 dev" bash scripts/install_uv.sh    # also install pytest
#
# Requires the `uv` launcher and an NVIDIA driver supporting CUDA 12.6. For a different
# CUDA build, edit the [[tool.uv.index]] URLs in pyproject.toml and re-run, or use the
# conda installer (scripts/install.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EXTRAS="${EXTRAS:-cu126}"
EXTRA_ARGS=()
for e in ${EXTRAS}; do EXTRA_ARGS+=(--extra "${e}"); done

echo ">> Syncing uv environment (.venv) with extras: ${EXTRAS}"
uv sync "${EXTRA_ARGS[@]}"

echo ">> Verifying the installation"
uv run --no-sync python - <<'PY'
import torch
import e3nn, torch_cluster, torch_scatter, torch_geometric  # noqa: F401
import rdkit, Bio, esm, superwater  # noqa: F401
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
print(f"torch_geometric {torch_geometric.__version__}  (pinned 2.6.1)")
print("SuperWater uv environment imports successfully.")
PY

cat <<EOF

Done. Run commands through the locked env, e.g.:

  uv run superwater-predict --config examples/configs/predict_5srf.yaml

EOF
