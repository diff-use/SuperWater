#!/usr/bin/env bash
#
# Create a uv-managed SuperWater environment for CUDA 11.8 training/inference.

set -euo pipefail

ENV_DIR="${ENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX="https://download.pytorch.org/whl/cu118"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo ">> [1/4] Creating uv virtualenv at ${ENV_DIR} with Python ${PYTHON_VERSION}"
uv venv "${ENV_DIR}" --python "${PYTHON_VERSION}"

echo ">> [2/4] Installing PyTorch CUDA 11.8 wheels"
uv pip install --python "${ENV_DIR}/bin/python" \
  --index-url "${TORCH_INDEX}" \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

echo ">> [3/4] Installing SuperWater dependencies"
uv pip install --python "${ENV_DIR}/bin/python" -r requirements-uv-cu118.txt

echo ">> [4/4] Installing SuperWater editable package"
uv pip install --python "${ENV_DIR}/bin/python" -e .

"${ENV_DIR}/bin/python" - <<'PY'
import torch
import e3nn, torch_cluster, torch_scatter, torch_geometric  # noqa: F401
import rdkit, Bio, superwater  # noqa: F401
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
print("SuperWater uv environment imports successfully.")
PY

cat <<EOF

Done. Activate with:

  source ${ENV_DIR}/bin/activate

EOF
