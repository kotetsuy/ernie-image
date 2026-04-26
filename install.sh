#!/usr/bin/env bash
# ERNIE-Image-Turbo on AMD Ryzen AI MAX+ 395 (gfx1151 / Strix Halo) セットアップ
#
# 前提:
#   - Ubuntu 24.04
#   - ROCm 7.2.x が `--no-dkms` でインストール済み
#   - amdgpu カーネルモジュールがロードされている
#   - BIOS で UMA Frame Buffer Size を最大化済み
#   - Python 3.12 が利用可能
#
# 使い方:
#   bash install.sh
#   # 完了後、別ターミナルで `hf auth login --token <HF_TOKEN>` を実行
#   # その後 `source venv/bin/activate && python run_ernie.py` で推論

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[1/3] Python 3.12 venv を作成"
python3.12 -m venv venv
# shellcheck source=/dev/null
source venv/bin/activate
pip install --upgrade pip wheel

echo "[2/3] PyTorch (AMD TheRock gfx1151 ネイティブ nightly, ROCm 7.x) をインストール"
# 注: 公式 PyTorch ROCm 6.4 wheel は arch list に gfx1151 を含まないため使えない
#     (実行時に HIP error: invalid device function でクラッシュする)
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
    torch torchvision torchaudio

echo "[3/3] diffusers (git版) と依存パッケージをインストール"
# ErnieImagePipeline は新しいので git 版 diffusers が必須
pip install \
    git+https://github.com/huggingface/diffusers \
    transformers accelerate sentencepiece protobuf \
    "huggingface_hub[cli]"

echo
echo "=========================================================="
echo "インストール完了。"
echo
echo "次に Hugging Face にログインしてください (token は https://huggingface.co/settings/tokens で取得):"
echo
echo "  source venv/bin/activate"
echo "  hf auth login --token <YOUR_HF_TOKEN>"
echo
echo "その後、推論を実行:"
echo
echo "  source venv/bin/activate"
echo "  python run_ernie.py"
echo
echo "初回実行時はモデル (約80GB) を Hugging Face からダウンロードします。"
echo "=========================================================="
