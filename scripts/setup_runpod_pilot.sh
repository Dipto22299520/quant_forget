#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=0
python -m venv --system-site-packages /workspace/acl-pilot-venv
/workspace/acl-pilot-venv/bin/python -m pip install 'transformers>=4.57,<5' 'accelerate>=1.10,<2' 'peft>=0.14,<0.19' 'datasets>=3,<5' scikit-learn scipy numpy pandas matplotlib pyyaml tqdm sacrebleu evaluate
/workspace/acl-pilot-venv/bin/python -m pip freeze > pilot_environment.txt
export PYTHONPATH="$PWD/src"
/workspace/acl-pilot-venv/bin/python -m collapse.cli check-env
/workspace/acl-pilot-venv/bin/python tests/test_units.py
/workspace/acl-pilot-venv/bin/python tests/test_audit_fixes.py
echo 'Setup and tests finished. Activate /workspace/acl-pilot-venv/bin/activate to run pilots.'
