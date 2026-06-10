#!/usr/bin/env bash
# Bootstrap the server-side Python venv for this project.
#
# Run this ONCE on the Linux server (with two RTX 6000 Ada GPUs):
#
#     ssh bobyard-server-6000
#     cd ~/james/irrigation_symbol_recognition
#     bash scripts/setup_server.sh
#
# Subsequent updates only require `git pull` + (if requirements.txt
# changed) re-running this script.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
REPO_ROOT="$(pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

echo "=========================================================="
echo "Setting up venv at: ${VENV_DIR}"
echo "Repo root         : ${REPO_ROOT}"
echo "=========================================================="

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH" >&2
    exit 1
fi

# Refuse to run on macOS — the wheels here are linux/CUDA-specific.
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: setup_server.sh is for the Linux server only."
    echo "       On the MacBook, set up the venv with:"
    echo "         python -m venv .venv && source .venv/bin/activate"
    echo "         pip install -r requirements.txt"
    exit 1
fi

PY_VERSION="$(python3 -c 'import sys; print("{}.{}".format(sys.version_info.major, sys.version_info.minor))')"
echo "Found python3 ${PY_VERSION}"

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "Creating venv..."
    python3 -m venv "${VENV_DIR}"
else
    echo "Reusing existing venv."
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

echo "----------------------------------------------------------"
echo "Installing requirements.txt (this pulls a CUDA-enabled"
echo "torch wheel from PyPI on Linux; takes a few minutes the"
echo "first time)..."
echo "----------------------------------------------------------"
pip install -r "${REPO_ROOT}/requirements.txt"

echo "----------------------------------------------------------"
echo "Sanity check"
echo "----------------------------------------------------------"
python - <<'PYCHECK'
import sys
print(f"python   : {sys.version.split()[0]}")
import torch
print(f"torch    : {torch.__version__}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}         : {name}")
from ultralytics import YOLO
print("ultralytics: imported OK")

sys.path.insert(0, "src")
from irrigation_symbol_recognition.data import YoloDataset
from irrigation_symbol_recognition.detection import filter_yolo_dataset, slice_data, nms_boxes
print("project pkg: imported OK")

from irrigation_symbol_recognition.utils.config import resolve_path, load_dataset_config
root = resolve_path(load_dataset_config()["dataset"]["root"])
print(f"dataset root resolved to: {root}")
print(f"  exists?              : {root.exists()}")
print(f"  data.yaml exists?    : {(root / 'data.yaml').exists()}")
PYCHECK

echo "=========================================================="
echo "Done. To use the venv interactively:"
echo "    source .venv/bin/activate"
echo "Then:"
echo "    python scripts/inspect_dataset.py"
echo "    python scripts/prepare_detection_dataset.py"
echo "    python scripts/train_detection.py --device cuda:0"
echo "=========================================================="
