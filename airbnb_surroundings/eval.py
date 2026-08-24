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

# ── E5 speedups + cross-learner encoder cache (now first-class in the MulTaBench fork) ──
# The E5 finetune bf16/fp16 + TF32 speedups now live in the fork itself, so nothing here
# has to touch it. The remaining knobs are dataset-specific, so we pass them to MulTaBench
# as *supported config* rather than monkeypatching: this dataset's text column is ≤122
# tokens, so we pad to 256 (vs the 512 default — attention is O(L²)) for both finetune
# (via _e5_kwargs) and encode (via e5_encode_kwargs), and raise the encode batch (default
# 32) to 256.
#
# The cross-learner encoder cache — the E5/DINO fit is learner-independent but ran up to
# 5× per run — is now an opt-in MulTaBench feature. We turn it on for the whole run with
# the encoder_cache() context manager in evaluate_dataset (it clears on entry/exit and
# keeps the val-split learner groups isolated by hashing the exact rows fed in).
from multabench.baselines.encoder_cache import encoder_cache

_MAX_LEN = 256                                                    # data max ≈122 tok; 256 = conservative (vs 512 default)
_E5_ENCODE_KWARGS = {"batch_size": 256, "max_length": _MAX_LEN}   # encode: bigger batch (default 32) + shorter padding
# ────────────────────────────────────────────────────────────────────────────

# This local NYC Airbnb regression screen follows MulTaBench's Airbnb regression
# configuration.  The old King County enum was removed from the vendored fork.
DATASET_ID = KaggleDatasetID.REG_IMAGE_HOUSES_AIRBNB_SEATTLE
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
        e5_encode_kwargs=_E5_ENCODE_KWARGS,
        tune_dino=tune_dino, dino_train_kwargs=_dino_kwargs(dino_epochs) if tune_dino else None,
    )
    return float(ret["test_score"])


def evaluate_dataset(csv, image_folder, target, task, fold, train_examples,
                     with_image: bool, e5_epochs=None, dino_epochs=None, with_tar=True):
    device = get_device(device=DEVICE)
    df = pd.read_csv(csv)
    y = df[target]
    x_struct = build_structured(df, target)
    x_text = df[[TEXT_COL]]
    x_img = df[[IMAGE_COL]] if with_image else None

    rows = []
    # Share the learner-independent E5/DINO fits across all learners + conditions in this run.
    with encoder_cache():
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
