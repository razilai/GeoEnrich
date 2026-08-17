"""Stage 5: evaluate MulTaBench curation eligibility on the enriched dataset.

Runs the 5 required learners (TabM, CatBoost, LightGBM, TabPFN v2, TabPFN v2.5)
across the curation conditions and checks the two criteria (Joint Signal + TAR
Gain) per modality, for at least 3 of the 5 learners.

We bypass MulTaBench's Kaggle download path and drive its baseline pipeline
directly via ``evaluate_on_loaded_dataset``, constructing a ``MultimodalDataset``
whose feature set we control per condition:

  structured           : tabular only (numeric + label-encoded categoricals)
  unstructured_text     : the free-text ``description`` column only
  unstructured_image    : the ``image`` filename column only
  joint_text_frozen     : structured + text (frozen E5)
  joint_text_tar        : structured + text (LoRA-tuned E5)          [tune_e5]
  joint_image_frozen    : structured + image (frozen DINO)
  joint_image_tar       : structured + image (LoRA-tuned DINO)       [tune_dino]

Metric: R^2 (regression). Categoricals are integer-encoded so the "structured"
baseline is unambiguously tabular (no text encoder leaks into it).
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the vendored MulTaBench clone (at the repo root) importable.
from airbnb_surroundings import config

sys.path.insert(0, os.path.join(config.ROOT, "MulTaBench"))

import pandas as pd
from dataclasses import asdict

from tabstar.training.devices import get_device

from multabench.constants import DEVICE
from multabench.datasets.all_datasets import KaggleDatasetID
from multabench.datasets.curation import MultimodalDataset
from multabench.datasets.objects import SupervisedTask
from multabench.baselines.benchmarks.evaluate import evaluate_on_loaded_dataset
from multabench.baselines.lgbm import LightGBM
from multabench.baselines.catboost import CatBoost
from multabench.baselines.tabm import TabM
from multabench.baselines.tabpfnv2 import TabPFNv2, TabPFNv2p5
from multabench.finetune.train_args import DinoTrainArgs, E5TrainArgs

# ── Encoder cache (optimizations A + B) ─────────────────────────────────────
# The E5/DINO fit (frozen encode AND LoRA finetune) happens inside every
# learner's fit_preprocessor, but its output is learner-independent: it depends
# only on the text/image column values, the train-row set, the tune flag, and the
# train kwargs — never on model_cls. eval.py runs 5 learners × several conditions,
# so the same encoder was being fit up to 5× (A: the LoRA finetune, the dominant
# cost) and even twice per learner for frozen text (unstructured_text +
# joint_text_frozen share the same column — B).
#
# We memoize the fitted encoders across the whole evaluate_dataset run. The key
# includes a content hash of the exact columns + rows fed in, so the two split
# groups (USE_VAL_SPLIT learners get x_train−val; TabPFN gets full x_train) fall
# into distinct cache entries and never cross-contaminate. Learners inside a group
# share rows deterministically (do_split is seeded), so they hit the cache.
#
# Patched into the abstract_model namespace (which imports these by name), so the
# MulTaBench baselines stay untouched.
import hashlib

from multabench.baselines import abstract_model as _abstract_model
from multabench.baselines.preprocessing.feature_types import detect_image_features as _detect_image_features

_ENCODER_CACHE: dict = {}


def _hash_frame(obj) -> str:
    """Stable content hash of a DataFrame/Series (values + index)."""
    return hashlib.sha1(pd.util.hash_pandas_object(obj, index=True).values.tobytes()).hexdigest()


def _kwargs_key(d) -> tuple:
    return tuple(sorted((d or {}).items()))


_orig_fit_text = _abstract_model.fit_text_encoders
_orig_fit_image = _abstract_model.fit_image_encoders


def _cached_fit_text_encoders(**kw):
    cols = sorted(kw.get("text_features") or set())
    if not cols:
        return _orig_fit_text(**kw)
    tune, y = kw.get("tune_e5", False), kw.get("y")
    key = ("text", kw.get("e5_model_name"), tune, _kwargs_key(kw.get("e5_train_kwargs")),
           kw.get("is_cls"), kw.get("d_output"), kw.get("pca_components"), kw.get("no_pca"),
           tuple(cols), _hash_frame(kw["x"][cols]),
           _hash_frame(y) if (tune and y is not None) else None)
    if key in _ENCODER_CACHE:
        print(f"  ♻️  reuse E5 encoder (tune={tune}) — cache hit")
        return _ENCODER_CACHE[key]
    enc = _orig_fit_text(**kw)
    _ENCODER_CACHE[key] = enc
    return enc


def _cached_fit_image_encoders(**kw):
    x = kw["x"]
    img_cols = sorted(_detect_image_features(x=x))
    if not img_cols or kw.get("image_folder") is None:
        return _orig_fit_image(**kw)
    tune, y = kw.get("tune_dino", False), kw.get("y")
    key = ("image", kw.get("dino_model_name"), kw.get("image_folder"), tune,
           _kwargs_key(kw.get("dino_train_kwargs")), kw.get("is_cls"), kw.get("d_output"),
           kw.get("pca_components"), kw.get("no_pca"), tuple(img_cols),
           _hash_frame(x[img_cols]), _hash_frame(y) if (tune and y is not None) else None)
    if key in _ENCODER_CACHE:
        print(f"  ♻️  reuse DINO encoder (tune={tune}) — cache hit")
        return _ENCODER_CACHE[key]
    out = _orig_fit_image(**kw)
    _ENCODER_CACHE[key] = out
    return out


_abstract_model.fit_text_encoders = _cached_fit_text_encoders
_abstract_model.fit_image_encoders = _cached_fit_image_encoders
# ────────────────────────────────────────────────────────────────────────────

# ── E5 finetune/encode speedups (option B: monkeypatched, baselines untouched) ──
# Measured: the text column is ≤122 tokens, but E5 finetune + encode padded every
# example to 512 (attention is O(L²)), and everything ran in fp32. We:
#   • pad to 256 (conservative headroom over the 122-token max) for training
#     (via _e5_kwargs(max_length=...)) and encoding (via the wrapper below);
#   • run E5 *training* in bf16 (fp16 fallback on pre-Ampere) through the Trainer;
#   • enable TF32 for remaining fp32 matmuls (helps the fp32 encode too);
#   • raise the inference-encode batch (E5 hardcoded 32) to 256.
# Encode stays fp32: autocast there yields bf16 tensors that break the encoder's
# `.cpu().numpy()`; the pad-256 + batch-256 + TF32 wins already dominate the encode.
import torch

from multabench.e5 import e5_finetune as _e5_finetune
from multabench.baselines.preprocessing import text_embeddings as _text_embeddings

_MAX_LEN = 256       # data max ≈122 tok; 256 = conservative (was 512 → ~4x less attention)
_ENCODE_BATCH = 256  # inference-encode batch (E5 default was 32)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _amp_dtype():
    """bf16 if the GPU supports it, else fp16; None on CPU (no mixed precision)."""
    if not torch.cuda.is_available():
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# Inject bf16/fp16 into the Trainer without editing e5_finetune (it does
# `from transformers import TrainingArguments`, so patch that bound name).
_orig_training_args = _e5_finetune.TrainingArguments


def _mixed_precision_training_args(*args, **kwargs):
    dt = _amp_dtype()
    if dt is not None and "bf16" not in kwargs and "fp16" not in kwargs:
        kwargs["bf16" if dt is torch.bfloat16 else "fp16"] = True
    return _orig_training_args(*args, **kwargs)


_e5_finetune.TrainingArguments = _mixed_precision_training_args

# Faster encode: shorter padding + bigger batch. text_embeddings imports
# encode_texts_with_e5 by name, so patch it in that namespace (covers frozen + tuned).
_orig_encode = _text_embeddings.encode_texts_with_e5


def _fast_encode_texts_with_e5(*args, **kwargs):
    kwargs.setdefault("batch_size", _ENCODE_BATCH)
    kwargs.setdefault("max_length", _MAX_LEN)
    return _orig_encode(*args, **kwargs)


_text_embeddings.encode_texts_with_e5 = _fast_encode_texts_with_e5
# ────────────────────────────────────────────────────────────────────────────

DATASET_ID = KaggleDatasetID.REG_IMAGE_HOUSE_PRICE_KING_COUNTY
LEARNERS = {
    "tabm": TabM,
    "cat": CatBoost,
    "light": LightGBM,
    "tabpfnv2": TabPFNv2,
    "tabpfnv2p5": TabPFNv2p5,
}
TEXT_COL = "surroundings_summary"
IMAGE_COL = "image"


def _dino_kwargs(epochs: int | None = None) -> dict:
    a = DinoTrainArgs()
    return dict(lora_rank=a.lora_rank, img_layers=a.img_layers, learning_rate=a.learning_rate,
                epochs=epochs or a.epochs, patience=a.patience, weight_decay=a.weight_decay,
                batch_size=a.batch_size)


def _e5_kwargs(epochs: int | None = None) -> dict:
    a = E5TrainArgs()
    return dict(lora_rank=a.lora_rank, text_layers=a.text_layers, learning_rate=a.learning_rate,
                epochs=epochs or a.epochs, patience=a.patience, weight_decay=a.weight_decay,
                batch_size=a.batch_size, max_length=_MAX_LEN)


def build_structured(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Numeric columns as-is; string columns integer-encoded (categorical)."""
    feats = df.drop(columns=[c for c in (target, TEXT_COL, IMAGE_COL) if c in df.columns])
    out = {}
    for col in feats.columns:
        s = feats[col]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.5:
            out[col] = num.fillna(num.median())
        else:
            out[col] = s.astype("category").cat.codes  # integer categorical
    return pd.DataFrame(out, index=df.index)


def score(model_cls, x, y, task, image_folder, fold, train_examples, device,
          tune_e5=False, tune_dino=False, e5_epochs=None, dino_epochs=None) -> float:
    dataset = MultimodalDataset(x=x.reset_index(drop=True), y=y.reset_index(drop=True),
                                task_type=task, dataset_id=DATASET_ID,
                                image_folder=image_folder)
    ret = evaluate_on_loaded_dataset(
        model_cls=model_cls, dataset=dataset, fold=fold, device=device,
        train_examples=train_examples,
        tune_e5=tune_e5, e5_train_kwargs=_e5_kwargs(e5_epochs) if tune_e5 else None,
        tune_dino=tune_dino, dino_train_kwargs=_dino_kwargs(dino_epochs) if tune_dino else None,
    )
    return float(ret["test_score"])


def evaluate_dataset(csv, image_folder, target, task, fold, train_examples,
                     with_image: bool, e5_epochs=None, dino_epochs=None, with_tar=True):
    _ENCODER_CACHE.clear()  # fresh encoder cache per run (A+B share fits across learners)
    device = get_device(device=DEVICE)
    df = pd.read_csv(csv)
    y = df[target]
    x_struct = build_structured(df, target)
    x_text = df[[TEXT_COL]]
    x_img = df[[IMAGE_COL]] if with_image else None

    rows = []
    for name, cls in LEARNERS.items():
        def run(x, **kw):
            return score(cls, x, y, task, image_folder, fold, train_examples, device,
                         e5_epochs=e5_epochs, dino_epochs=dino_epochs, **kw)

        rec = {"learner": name}
        rec["structured"] = run(x_struct)
        rec["unstructured_text"] = run(x_text)
        rec["joint_text_frozen"] = run(pd.concat([x_struct, x_text], axis=1))
        rec["joint_text_tar"] = (run(pd.concat([x_struct, x_text], axis=1), tune_e5=True)
                                 if with_tar else float("nan"))
        if with_image:
            rec["unstructured_image"] = run(x_img)
            rec["joint_image_frozen"] = run(pd.concat([x_struct, x_img], axis=1))
            rec["joint_image_tar"] = (run(pd.concat([x_struct, x_img], axis=1), tune_dino=True)
                                      if with_tar else float("nan"))
        rows.append(rec)
        print(f"  {name}: {rec}")
    return pd.DataFrame(rows)


def _criteria(report: pd.DataFrame, modality: str):
    """Return per-learner (joint_signal, tar_gain) booleans for a modality."""
    frozen, tar, unstruct = (f"joint_{modality}_frozen", f"joint_{modality}_tar",
                             f"unstructured_{modality}")
    out = {}
    for _, r in report.iterrows():
        joint_signal = (r[frozen] > r["structured"]) and (r[frozen] > r[unstruct])
        tar_gain = r[tar] > r[frozen]
        out[r["learner"]] = (bool(joint_signal), bool(tar_gain))
    return out


def report_verdict(report: pd.DataFrame, modality: str) -> bool:
    crit = _criteria(report, modality)
    n_pass = sum(1 for js, tg in crit.values() if js and tg)
    print(f"\n── {modality.upper()} modality ──")
    for learner, (js, tg) in crit.items():
        print(f"  {learner:12s} joint_signal={js!s:5s}  tar_gain={tg!s:5s}  "
              f"{'✅ PASS' if js and tg else '❌'}")
    ok = n_pass >= 3
    print(f"  → {n_pass}/5 learners pass both criteria  "
          f"({'ELIGIBLE' if ok else 'NOT eligible'} for {modality})")
    return ok


def main():
    p = argparse.ArgumentParser(description="MulTaBench curation eligibility eval.")
    p.add_argument("--csv", required=True)
    p.add_argument("--image-folder", required=True)
    p.add_argument("--target", default="price")
    p.add_argument("--task", choices=["reg", "bin", "mul"], default="reg")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--train-examples", type=int, default=10_000)
    p.add_argument("--out", default=config.EVAL_REPORT_CSV)
    p.add_argument("--no-image", action="store_true",
                   help="old text-tabular dataset only (skip image conditions)")
    p.add_argument("--e5-epochs", type=int, default=None, help="override E5 LoRA epochs (TAR)")
    p.add_argument("--dino-epochs", type=int, default=None, help="override DINO LoRA epochs (TAR)")
    p.add_argument("--no-tar", action="store_true",
                   help="skip TAR (LoRA) conditions; joint-signal only. TAR requires a GPU.")
    args = p.parse_args()

    task = {"reg": SupervisedTask.REGRESSION, "bin": SupervisedTask.BINARY,
            "mul": SupervisedTask.MULTICLASS}[args.task]
    with_image = not args.no_image

    report = evaluate_dataset(args.csv, args.image_folder, args.target, task,
                              args.fold, args.train_examples, with_image,
                              e5_epochs=args.e5_epochs, dino_epochs=args.dino_epochs,
                              with_tar=not args.no_tar)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    report.to_csv(args.out, index=False)
    print(f"\n✅ wrote {args.out}")

    text_ok = report_verdict(report, "text")
    if with_image:
        image_ok = report_verdict(report, "image")
        print(f"\n=== TRI-MODAL VERDICT: "
              f"{'✅ ELIGIBLE' if (text_ok and image_ok) else '❌ NOT eligible'} "
              f"(text={text_ok}, image={image_ok}) ===")


if __name__ == "__main__":
    main()
