# ERNIE-Image-Turbo on NucBox EVO X2 (gfx1151)

Baidu の ERNIE-Image-Turbo (text-to-image, 8B DiT) を AMD Ryzen AI MAX+ 395 (Strix Halo, gfx1151) 上で diffusers 経由で動かすセットアップ。

## ハードウェア / 環境

| 項目 | 値 |
|---|---|
| マシン | GMKtec NucBox EVO X2 |
| CPU/iGPU | AMD Ryzen AI MAX+ 395 (Strix Halo) |
| GPU arch | gfx1151 (RDNA 3.5) |
| メモリ | 96GB(統合メモリ、BIOS で 48GB を VRAM に割当) |
| OS | Ubuntu 24.04 |
| ROCm | 7.2.1(`--no-dkms` + OEM カーネル) |
| Python | 3.12.3 |

## セットアップでハマったポイント

### 1. 公式 PyTorch ROCm wheel は gfx1151 で動かない

`pip install torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4`
で入る公式 PyTorch nightly は arch list に gfx1151 を**含まない**:

```
arch list: ['gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx942',
            'gfx1030', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1200', 'gfx1201']
gcnArchName: gfx1151
```

`HSA_OVERRIDE_GFX_VERSION=11.5.1` を設定しても、対応カーネルが wheel に存在しないため実行時に
`HIP error: invalid device function` でクラッシュする。

**解決策**: AMD TheRock の gfx1151 ネイティブ nightly wheel を使う。

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision torchaudio
```

これで `torch-2.10.0+rocm7.13.0` + `rocm-sdk-libraries-gfx1151` が入り、
`arch list: ['gfx1151']` でネイティブに動く。

### 2. ErnieImagePipeline の API 変更

CLAUDE.md(初版)に書かれていた `pipe.enable_vae_slicing()` / `pipe.enable_vae_tiling()` は
最新の diffusers では廃止され、VAE モジュール側に移動している:

```python
# 旧 (動かない)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# 新
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

### 3. AOTriton Flash Attention は実験フラグで有効化

`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` を設定すると、
SDPA で AMD GPU 向け Flash Attention 相当が有効化される。
本セットアップでは **約2.5倍の高速化**が確認できた。

### 4. cpu_offload は不要

VRAM 48GB 環境では ERNIE-Image-Turbo (Peak ~37GB at 1024x1024) を
全量 GPU に載せられる。`enable_model_cpu_offload()` を `pipe.to("cuda")` に
置き換えると **さらに2倍弱の高速化**。

## インストール

```bash
bash install.sh
```

完了後、Hugging Face にログイン(別ターミナル推奨):

```bash
source venv/bin/activate
hf auth login --token <HF_TOKEN>   # https://huggingface.co/settings/tokens で取得
```

## 実行

```bash
source venv/bin/activate
python run_ernie.py
```

初回は Hugging Face からモデル(約 80GB)をダウンロードするため、
ディスク空きと時間が必要(高速回線で5分程度)。

## 最終構成 (`run_ernie.py`)

- gfx1151 ネイティブ PyTorch 2.10 + ROCm 7.13 (TheRock)
- AOTriton Flash Attention 有効 (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`)
- 全モデル GPU 常駐 (`pipe.to("cuda")`)
- VAE slicing/tiling 有効
- bfloat16, 8 step

## 性能(640x640 / 1024x1024 比較)

8 step・guidance_scale=1.0・bf16 で計測。

| 解像度 | 設定 | 総時間 | s/step | Peak VRAM |
|---|---|---|---|---|
| 640x640  | 初期 (cpu_offload, AOTriton無効) | 52.4s | 6.55 | 17.8 GB |
| 640x640  | + AOTriton 有効                  | 21.6s | 2.70 | 16.6 GB |
| 640x640  | + 全GPU常駐 (本構成)             | 12.2s | 1.52 | 33.4 GB |
| 1024x1024| + 全GPU常駐 (本構成)             | 31.3s | 3.91 | 37.2 GB |

最終的に初期比で **約4.3倍の高速化**。
1024x1024 で 31秒/枚 は CLAUDE.md の予想性能(1〜3分/枚)を大きく上回る結果。

## プロンプトと画像

プロンプトと画像サイズはrun_ernie.pyにじかに書き込む

### プロンプト1
ずんだもんが秋葉原の電気街を歩いている、アニメ風、明るい色彩、夕方

### プロンプト2
富士山を背景に金閣寺が光輝いている、金閣寺の前には池、写真風、明るい色彩、昼間

## 参考リンク

- [AMD TheRock - gfx1151 PyTorch wheels](https://github.com/ROCm/TheRock/discussions/655)
- [ERNIE-Image-Turbo on Hugging Face](https://huggingface.co/baidu/ERNIE-Image-Turbo)
- [Strix Halo gfx1151: 93 ML experiments (ROCm/ROCm Issue #6034)](https://github.com/ROCm/ROCm/issues/6034)
