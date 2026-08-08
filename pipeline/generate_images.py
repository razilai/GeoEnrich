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

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-2-1-base"
NEGATIVE_PROMPT = "blurry, distorted, text, watermark, low quality, cartoon"


def _load_pipeline(model_id: str):
    """Lazily import diffusers/torch so the module is importable without them."""
    import torch
    from diffusers import StableDiffusionPipeline

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    # Disable the NSFW checker: it can blank out valid house images as false positives.
    pipe.safety_checker = None
    print(f"🖼️  loaded {model_id} on {device} ({dtype})")
    return pipe, device


def generate_images(
    csv: str,
    out_dir: str,
    model_id: str = DEFAULT_MODEL_ID,
    limit: int | None = None,
    steps: int = 30,
    guidance: float = 7.5,
    size: int = 512,
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

    for n, i in enumerate(pending):
        prompt = row_to_prompt(df.iloc[i])
        generator = torch.Generator(device=device).manual_seed(i)
        image = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=size,
            width=size,
            generator=generator,
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
    p.add_argument("--size", type=int, default=512)
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
