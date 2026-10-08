"""Stage 5: evaluate MulTaBench curation eligibility on the enriched dataset.

Runs the 5 required learners (TabM, CatBoost, LightGBM, TabPFN v2, TabPFN v2.5)
across the curation conditions and checks the two criteria (Joint Signal + TAR
Gain) per modality, for at least 3 of the 5 learners. The full evaluation uses
the MulTaBench protocol's five folds (0--4) and judges each learner's mean score
across them; ``--light`` runs a single-fold screen for quick iteration.

MulTaBench's Kaggle download path is bypassed: its baseline pipeline is driven
directly via ``evaluate_on_loaded_dataset`` on a ``MultimodalDataset`` whose
feature set is chosen per condition:

  structured          : tabular only (numeric + label-encoded categoricals)
  unstructured_text   : the ``surroundings_summary`` text column only
  unstructured_image  : the ``image`` filename column only
  joint_text_frozen   : structured + text (frozen E5)
  joint_text_tar      : structured + text (LoRA-tuned E5)      [tune_e5]
  joint_image_frozen  : structured + image (frozen DINO)
  joint_image_tar     : structured + image (LoRA-tuned DINO)   [tune_dino]

Metric: R^2 (regression). Categoricals are integer-encoded so the "structured"
baseline is unambiguously tabular (no text encoder leaks into it).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable

from src import config

# Make the vendored MulTaBench clone (at the repo root) importable.
sys.path.insert(0, os.path.join(config.ROOT, "MulTaBench"))

import pandas as pd
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
from multabench.baselines.encoder_cache import encoder_cache
from multabench.finetune.train_args import DinoTrainArgs, E5TrainArgs

# The text column is <=122 tokens, so E5 pads to 256 instead of the 512 default (attention
# is O(L^2)) in both finetune and encode; encoding also uses a bigger batch (default 32).
_MAX_LEN = 256
_E5_ENCODE_KWARGS = {"batch_size": 256, "max_length": _MAX_LEN}

# Reuses MulTaBench's Airbnb regression configuration for this NYC dataset.
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
FULL_FOLDS = tuple(range(5))
CURATION_MARGIN = 0.001


def _lora_kwargs(a, layers: str, epochs: int | None = None, **extra) -> dict:
    """LoRA finetune kwargs from a MulTaBench TrainArgs default, with optional epochs override."""
    return dict(lora_rank=a.lora_rank, **{layers: getattr(a, layers)},
                learning_rate=a.learning_rate, epochs=epochs or a.epochs, patience=a.patience,
                weight_decay=a.weight_decay, batch_size=a.batch_size, **extra)


def _dino_kwargs(epochs: int | None = None) -> dict:
    return _lora_kwargs(DinoTrainArgs(), "img_layers", epochs)


def _e5_kwargs(epochs: int | None = None) -> dict:
    return _lora_kwargs(E5TrainArgs(), "text_layers", epochs, max_length=_MAX_LEN)


def build_structured(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Numeric columns as-is; string columns integer-encoded (categorical)."""
    feats = df.drop(columns=[c for c in (target, TEXT_COL, IMAGE_COL) if c in df.columns])
    out = {}
    for col, s in feats.items():
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


def evaluate_dataset(csv, image_folder, target, task, folds: Iterable[int], train_examples,
                     with_image: bool, e5_epochs=None, dino_epochs=None, with_tar=True):
    """Score every learner under every curation condition, one row per (fold, learner)."""
    device = get_device(device=DEVICE)
    df = pd.read_csv(csv)
    y = df[target]
    x_struct = build_structured(df, target)
    modalities = {"text": (df[[TEXT_COL]], "tune_e5")}
    if with_image:
        modalities["image"] = (df[[IMAGE_COL]], "tune_dino")

    rows = []
    # Share the learner-independent E5/DINO fits across all learners + conditions in this run.
    with encoder_cache():
        for fold in folds:
            print(f"\n── Fold {fold} ──")
            for name, cls in LEARNERS.items():
                def run(x, **kw):
                    return score(cls, x, y, task, image_folder, fold, train_examples, device,
                                 e5_epochs=e5_epochs, dino_epochs=dino_epochs, **kw)

                rec = {"fold": fold, "learner": name, "structured": run(x_struct)}
                for modality, (x_mod, tune_flag) in modalities.items():
                    joint = pd.concat([x_struct, x_mod], axis=1)
                    rec[f"unstructured_{modality}"] = run(x_mod)
                    rec[f"joint_{modality}_frozen"] = run(joint)
                    rec[f"joint_{modality}_tar"] = (run(joint, **{tune_flag: True})
                                                    if with_tar else float("nan"))
                rows.append(rec)
                print(f"  {name}: {rec}")
    return pd.DataFrame(rows)


def _mean_scores_by_learner(report: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-fold report used by the MulTaBench curation rule."""
    score_columns = [c for c in report.columns if c not in {"fold", "learner"}]
    return report.groupby("learner", as_index=False)[score_columns].mean()


def _criteria(report: pd.DataFrame, modality: str, *, margin: float = 0.0):
    """Return per-learner (joint_signal, tar_gain) booleans for a modality."""
    frozen, tar = f"joint_{modality}_frozen", f"joint_{modality}_tar"
    return {
        r["learner"]: (
            bool(r[frozen] - max(r["structured"], r[f"unstructured_{modality}"]) > margin),
            bool(r[tar] - r[frozen] > margin),
        )
        for _, r in report.iterrows()
    }


def report_verdict(report: pd.DataFrame, modality: str, *, margin: float = 0.0) -> bool:
    """Print the committee verdict from the mean score of each learner's folds."""
    fold_count = report["fold"].nunique() if "fold" in report else 1
    crit = _criteria(_mean_scores_by_learner(report), modality, margin=margin)
    n_pass = sum(js and tg for js, tg in crit.values())
    print(f"\n── {modality.upper()} modality ({fold_count}-fold mean) ──")
    for learner, (js, tg) in crit.items():
        print(f"  {learner:12s} joint_signal={js!s:5s}  tar_gain={tg!s:5s}  "
              f"{'✅ PASS' if js and tg else '❌'}")
    ok = n_pass >= 3
    margin_note = f" (margin > {margin:.3f})" if margin else ""
    print(f"  → {n_pass}/5 learners pass both criteria{margin_note}  "
          f"({'ELIGIBLE' if ok else 'NOT eligible'} for {modality})")
    return ok


def main():
    p = argparse.ArgumentParser(description="MulTaBench curation eligibility eval.")
    p.add_argument("--csv", required=True)
    p.add_argument("--image-folder", required=True)
    p.add_argument("--target", default="price")
    p.add_argument("--task", choices=["reg", "bin", "mul"], default="reg")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--light", action="store_true",
                      help="run the former single-fold screen (uses --fold, default: 0)")
    mode.add_argument("--full", action="store_true",
                      help="run the required five-fold MulTaBench evaluation (the default)")
    p.add_argument("--fold", type=int, default=0,
                   help="fold for --light only (default: 0)")
    p.add_argument("--train-examples", type=int, default=10_000)
    p.add_argument("--out", default=config.EVAL_REPORT_CSV)
    p.add_argument("--no-image", action="store_true",
                   help="old text-tabular dataset only (skip image conditions)")
    p.add_argument("--e5-epochs", type=int, default=None, help="override E5 LoRA epochs (TAR)")
    p.add_argument("--dino-epochs", type=int, default=None, help="override DINO LoRA epochs (TAR)")
    p.add_argument("--no-tar", action="store_true",
                   help="skip TAR (LoRA) conditions; joint-signal only. TAR requires a GPU.")
    args = p.parse_args()

    if not args.light and args.fold != 0:
        p.error("--fold is only available with --light; --full always runs folds 0--4")

    task = {"reg": SupervisedTask.REGRESSION, "bin": SupervisedTask.BINARY,
            "mul": SupervisedTask.MULTICLASS}[args.task]
    with_image = not args.no_image

    folds = (args.fold,) if args.light else FULL_FOLDS
    margin = 0.0 if args.light else CURATION_MARGIN
    print(f"\nRunning {'light single-fold' if args.light else 'full five-fold'} evaluation "
          f"(folds: {', '.join(map(str, folds))})")
    report = evaluate_dataset(args.csv, args.image_folder, args.target, task, folds,
                              args.train_examples, with_image,
                              e5_epochs=args.e5_epochs, dino_epochs=args.dino_epochs,
                              with_tar=not args.no_tar)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    report.to_csv(args.out, index=False)
    print(f"\n✅ wrote {args.out}")

    text_ok = report_verdict(report, "text", margin=margin)
    if with_image:
        image_ok = report_verdict(report, "image", margin=margin)
        print(f"\n=== TRI-MODAL VERDICT: "
              f"{'✅ ELIGIBLE' if (text_ok and image_ok) else '❌ NOT eligible'} "
              f"(text={text_ok}, image={image_ok}) ===")


if __name__ == "__main__":
    main()
