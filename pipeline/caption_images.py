"""Stage 2.5: caption each generated image with a VLM → the text modality.

This *replaces* the deterministic ``row_to_prompt`` template as the source of the
``description`` column. The template was a pure function of the tabular columns,
so the text modality carried no signal beyond the table (see the text-eval
verdict: joint_signal fails for most learners). Captioning the *generated* image
instead routes the text through Stable Diffusion's stochastic / hallucinated
pixels — detail that is not a deterministic function of the row — giving the
text a chance to be complementary to the table.

Pipeline chain per row:  row_to_prompt (fixed) -> SD image -> VLM caption -> text.

- Resumable: an existing caption for ``row_{idx}.png`` in ``out_csv`` is reused,
  so the (slow, GPU-heavy) BLIP-2 pass can be interrupted and restarted.
- GPU-aware: fp16 on CUDA, fp32 elsewhere. BLIP-2 2.7B on CPU is impractically
  slow — expect this stage to want a GPU.

Filenames (``row_{idx:05d}.png``) match generate_images.py / enrich_csv.py.
"""
from __future__ import annotations

import os

import pandas as pd

# BLIP-2 (OPT-2.7B) — open weights, no gated-repo auth needed, richer than BLIP.
DEFAULT_CAPTIONER = "Salesforce/blip2-opt-2.7b"


def _load_captioner(model_id: str):
    """Lazily import torch/transformers so the module stays importable without them."""
    import torch
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    print(f"📝 loaded {model_id} on {device} ({dtype})")
    return processor, model, device, dtype


def caption_images(
    image_folder: str,
    out_csv: str,
    model_id: str = DEFAULT_CAPTIONER,
    limit: int | None = None,
    max_new_tokens: int = 40,
    image_col: str = "image",
    caption_col: str = "description",
) -> pd.DataFrame:
    """Caption ``row_*.png`` in ``image_folder``; write ``(image, description)`` to
    ``out_csv`` and return the DataFrame. Row order follows the filename index so it
    aligns with the raw CSV rows in enrich_csv.py."""
    from PIL import Image

    files = sorted(
        f for f in os.listdir(image_folder)
        if f.startswith("row_") and f.endswith(".png")
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(
            f"no row_*.png images in {image_folder}; run Stage 2 (generate_images) first."
        )

    # Resume: reuse captions already written on a previous (interrupted) run.
    done: dict[str, str] = {}
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        if image_col in prev and caption_col in prev:
            done = dict(zip(prev[image_col], prev[caption_col].astype(str)))

    pending = [f for f in files if f not in done]
    print(f"captioning {len(pending)}/{len(files)} images (reusing existing) from {image_folder}")

    processor = model = device = dtype = None
    if pending:
        import torch
        processor, model, device, dtype = _load_captioner(model_id)

    rows = []
    for n, fn in enumerate(files):
        if fn in done:
            caption = done[fn]
        else:
            img = Image.open(os.path.join(image_folder, fn)).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device, dtype)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens)
            caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            if (n + 1) % 25 == 0 or n + 1 == len(files):
                print(f"  [{n + 1}/{len(files)}] {fn}: {caption[:60]}...")
        rows.append({image_col: fn, caption_col: caption})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✅ wrote {out_csv} | {len(df)} captions")
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Caption generated images with a VLM.")
    p.add_argument("--image-folder", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--model-id", default=DEFAULT_CAPTIONER)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=40)
    args = p.parse_args()

    caption_images(
        image_folder=args.image_folder,
        out_csv=args.out_csv,
        model_id=args.model_id,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
    )
