#!/usr/bin/env bash
# Run without Docker. Usage: ./run_local.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
# Aliyun mirror; drop the flag outside mainland China.
pip install -q --index-url https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
streamlit run app.py
