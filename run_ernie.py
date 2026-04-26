import os

os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.5.1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
from diffusers import ErnieImagePipeline

print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

pipe = ErnieImagePipeline.from_pretrained(
    "baidu/ERNIE-Image-Turbo",
    torch_dtype=torch.bfloat16,
)

pipe = pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
# pipe.enable_attention_slicing()  # OOM時に有効化

# prompt = "ずんだもんが秋葉原の電気街を歩いている、アニメ風、明るい色彩、夕方"
prompt = (
    "富士山を背景に金閣寺が光輝いている、金閣寺の前には池、写真風、明るい色彩、昼間"
)


with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=8,
        guidance_scale=1.0,
        use_pe=True,
    ).images[0]

image.save("output.png")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
