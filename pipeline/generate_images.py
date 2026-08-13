"""Stage 2: generate one image per row with a local Stable Diffusion model.

- Deterministic: seed = row index, so a rerun reproduces identical images.
- Resumable: existing ``row_{idx}.png`` files are skipped, so the job can be
  interrupted and restarted (important for the full 4601-row run).
- GPU-aware: uses fp16 on CUDA, falls back to fp32 on CPU/MPS.

Filenames (``row_{idx:05d}.png``) match the convention in ``enrich_csv.py``.
"""
from __future__ import annotations

import os

import pandas as pd

from pipeline.prompt_builder import row_to_prompt

# SDXL: far more photorealistic + finer per-row detail than SD 2.1, so the images
# (and their captions) vary more between rows. AutoPipelineForText2Image resolves
# the right pipeline class from the model id, so FLUX / SD3 also work as a drop-in
# --model-id (note: FLUX ignores negative_prompt).
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
NEGATIVE_PROMPT = "blurry, distorted, text, watermark, low quality, cartoon"


def _load_pipeline(model_id: str):
    """Lazily import diffusers/torch so the module is importable without them."""
    import torch
    from diffusers import AutoPipelineForText2Image

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id, torch_dtype=dtype, use_safetensors=True
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # Disable the NSFW checker (false-positives blank valid house images). Only
    # SD-class pipelines have one; SDXL/FLUX don't, so guard the attribute.
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    # SDXL's UNet is big; slicing keeps 1024px generation within reach on smaller GPUs.
    if device == "cuda":
        pipe.enable_attention_slicing()
    print(f"🖼️  loaded {model_id} on {device} ({dtype})")
    return pipe, device


def generate_images(
    csv: str,
    out_dir: str,
    model_id: str = DEFAULT_MODEL_ID,
    limit: int | None = None,
    steps: int = 30,
    guidance: float = 7.5,
    size: int = 1024,  # SDXL native; 512 degrades SDXL output
) -> None:
    import torch

    df = pd.read_csv(csv)
    if limit is not None:
        df = df.head(limit)
    df = df.reset_index(drop=True)
    os.makedirs(out_dir, exist_ok=True)

    pending = [
        i for i in range(len(df))
        if not os.path.exists(os.path.join(out_dir, f"row_{i:05d}.png"))
    ]
    print(f"generating {len(pending)}/{len(df)} images (skipping existing) into {out_dir}")
    if not pending:
        return

    pipe, device = _load_pipeline(model_id)

    # Not every pipeline accepts negative_prompt (e.g. FLUX) — only pass what the
    # active pipeline's __call__ actually supports.
    import inspect
    supported = set(inspect.signature(pipe.__call__).parameters)
    extra = {k: v for k, v in {"negative_prompt": NEGATIVE_PROMPT}.items() if k in supported}

    for n, i in enumerate(pending):
        prompt = row_to_prompt(df.iloc[i])
        generator = torch.Generator(device=device).manual_seed(i)
        image = pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=size,
            width=size,
            generator=generator,
            **extra,
        ).images[0]
        image.save(os.path.join(out_dir, f"row_{i:05d}.png"))
        if (n + 1) % 25 == 0 or n + 1 == len(pending):
            print(f"  [{n + 1}/{len(pending)}] row {i}: {prompt[:60]}...")

    print(f"✅ done: {out_dir}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate house images with Stable Diffusion.")
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--size", type=int, default=1024)
    args = p.parse_args()

    generate_images(
        csv=args.csv,
        out_dir=args.out_dir,
        model_id=args.model_id,
        limit=args.limit,
        steps=args.steps,
        guidance=args.guidance,
        size=args.size,
    )
