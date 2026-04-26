# ERNIE-Image-Turbo on NucBox EVO X2 (gfx1151)

Setup for running Baidu's ERNIE-Image-Turbo (text-to-image, 8B DiT) on AMD Ryzen AI MAX+ 395 (Strix Halo, gfx1151) via diffusers.

## Hardware / Environment

| Item | Value |
|---|---|
| Machine | GMKtec NucBox EVO X2 |
| CPU/iGPU | AMD Ryzen AI MAX+ 395 (Strix Halo) |
| GPU arch | gfx1151 (RDNA 3.5) |
| Memory | 96GB (unified memory, 48GB allocated to VRAM in BIOS) |
| OS | Ubuntu 24.04 |
| ROCm | 7.2.1 (`--no-dkms` + OEM kernel) |
| Python | 3.12.3 |

## Setup gotchas

### 1. The official PyTorch ROCm wheel does not work on gfx1151

The official PyTorch nightly installed via
`pip install torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4`
does **not** include gfx1151 in its arch list:

```
arch list: ['gfx900', 'gfx906', 'gfx908', 'gfx90a', 'gfx942',
            'gfx1030', 'gfx1100', 'gfx1101', 'gfx1102', 'gfx1200', 'gfx1201']
gcnArchName: gfx1151
```

Even with `HSA_OVERRIDE_GFX_VERSION=11.5.1`, the matching kernels are absent
from the wheel, so it crashes at runtime with
`HIP error: invalid device function`.

**Solution**: use AMD TheRock's gfx1151-native nightly wheel.

```bash
pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision torchaudio
```

This installs `torch-2.10.0+rocm7.13.0` together with `rocm-sdk-libraries-gfx1151`,
and runs natively with `arch list: ['gfx1151']`.

### 2. ErnieImagePipeline API change

`pipe.enable_vae_slicing()` / `pipe.enable_vae_tiling()`, mentioned in the
initial CLAUDE.md, have been removed in recent diffusers and moved onto the
VAE module:

```python
# Old (no longer works)
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# New
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

### 3. AOTriton Flash Attention is gated behind an experimental flag

Setting `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` enables a Flash
Attention–equivalent path on AMD GPUs through SDPA. In this setup it gave
**about a 2.5× speedup**.

### 4. cpu_offload is unnecessary

With 48GB of VRAM, ERNIE-Image-Turbo (peak ~37GB at 1024x1024) fits entirely
on the GPU. Replacing `enable_model_cpu_offload()` with `pipe.to("cuda")`
yields **another ~2× speedup**.

## Install

```bash
bash install.sh
```

After it finishes, log into Hugging Face (in a separate terminal is fine):

```bash
source venv/bin/activate
hf auth login --token <HF_TOKEN>   # get one at https://huggingface.co/settings/tokens
```

## Run

```bash
source venv/bin/activate
python run_ernie.py
```

The first run downloads the model (~80GB) from Hugging Face, so make sure
you have the disk space and time (about 5 minutes on a fast connection).

## Final configuration (`run_ernie.py`)

- gfx1151-native PyTorch 2.10 + ROCm 7.13 (TheRock)
- AOTriton Flash Attention enabled (`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`)
- Full model resident on GPU (`pipe.to("cuda")`)
- VAE slicing/tiling enabled
- bfloat16, 8 steps

## Performance (640x640 vs 1024x1024)

Measured at 8 steps, guidance_scale=1.0, bf16.

| Resolution | Configuration | Total time | s/step | Peak VRAM |
|---|---|---|---|---|
| 640x640   | initial (cpu_offload, AOTriton off) | 52.4s | 6.55 | 17.8 GB |
| 640x640   | + AOTriton enabled                  | 21.6s | 2.70 | 16.6 GB |
| 640x640   | + full GPU residency (this setup)   | 12.2s | 1.52 | 33.4 GB |
| 1024x1024 | + full GPU residency (this setup)   | 31.3s | 3.91 | 37.2 GB |

End-to-end this is **about a 4.3× speedup** over the initial config.
31s per 1024x1024 image is well ahead of the 1–3 min/image estimate in
CLAUDE.md.

## Prompts and images

The prompt and image size are written directly in `run_ernie.py`.

### Prompt 1
Zundamon walking through Akihabara's electric town, anime style, bright colors, evening.

### Prompt 2
Kinkaku-ji glowing in front of Mt. Fuji with a pond in the foreground, photographic style, bright colors, daytime.

## References

- [AMD TheRock - gfx1151 PyTorch wheels](https://github.com/ROCm/TheRock/discussions/655)
- [ERNIE-Image-Turbo on Hugging Face](https://huggingface.co/baidu/ERNIE-Image-Turbo)
- [Strix Halo gfx1151: 93 ML experiments (ROCm/ROCm Issue #6034)](https://github.com/ROCm/ROCm/issues/6034)
