"""Single entry point: `uv run main` runs the whole pipeline correctly.

The heavy deps (tabstar, torch, autogluon, geopandas, pyrosm, pydantic-ai) all
live in MulTaBench/.venv — the one env built by init.sh. `uv run` activates the
thin project .venv instead, which has no tabstar, so importing the eval there
fails with "tabstar not installed". This launcher sidesteps that: every stage is
run with MulTaBench/.venv's interpreter, regardless of which env `uv run` picked.

Stages (each skipped when its output already exists, so it is safe to re-run).
Each runs as a module in the airbnb_surroundings package (installed -e into
MulTaBench/.venv by init.sh):
    0. clean     -m airbnb_surroundings.clean     -> data/airbnb.csv (manual/upstream,
                 PySpark; run once from the raw scrape — not in this auto-chain)
    1. build     -m airbnb_surroundings.build     -> artifacts/airbnb_enriched.csv
    2. describe  -m airbnb_surroundings.describe   -> artifacts/airbnb_described.csv (LLM key)
    3. eval      -m airbnb_surroundings.eval       -> results/eval_report.csv

On a fresh GPU box the usual flow is: build + describe locally, `scripts/sync.sh`
the CSVs over, then run only the eval here — so the build stage is skipped when
a described CSV is already present.

Extra args after `--` are forwarded verbatim to the eval, overriding the
defaults, e.g.  `uv run main -- --no-tar`  (joint-signal only, no GPU).
"""

from __future__ import annotations

import os
import subprocess
import sys

# Anchor to the repo root, which is the cwd `uv run main` executes from. Do NOT
# use __file__: this module ships as an installed wheel, so __file__ resolves to
# .venv/site-packages, not the project tree where init.sh / src / the CSVs live.
HERE = os.getcwd()
VENV_PY = os.path.join(HERE, "MulTaBench", ".venv", "bin", "python")
INIT_SH = os.path.join(HERE, "init.sh")

ARTIFACTS = os.path.join(HERE, "artifacts")
ENRICHED = os.path.join(ARTIFACTS, "airbnb_enriched.csv")
DESCRIBED = os.path.join(ARTIFACTS, "airbnb_described.csv")

# text-tabular dataset (no image modality); TAR (LoRA) runs since the box has a GPU.
DEFAULT_EVAL_ARGS = [
    "--no-image",
    "--target", "price",
    "--csv", DESCRIBED,
    "--image-folder", "/dev/null",
]


def sh(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a subprocess from the repo root, aborting the pipeline on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=HERE, check=True, env=env)


def pin_gpu() -> None:
    """Pin one GPU for the whole run, before any child imports torch.

    tabstar's get_device treats CUDA_VISIBLE_DEVICES=<int> as the single-GPU
    contract, and the LoRA finetune asserts it is set (else HF Trainer would
    DataParallel across every visible GPU). It must exist BEFORE torch imports,
    so we set it in this parent env, which every subprocess inherits. Honor an
    existing value; else fall back to the GPU index (constants.GPU) or 0.
    """
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("GPU") or "0"
        # GPU drove constants.DEVICE=cuda:{GPU}; now that only one GPU is
        # visible (remapped to index 0), drop it so DEVICE stays None and
        # get_device returns plain "cuda" for the single visible device.
        os.environ.pop("GPU", None)


def ensure_env() -> None:
    """Build MulTaBench/.venv (clone + deps) via init.sh if it isn't there yet."""
    if os.path.exists(VENV_PY):
        return
    print("MulTaBench/.venv missing — bootstrapping via init.sh", flush=True)
    sh(["bash", INIT_SH])
    if not os.path.exists(VENV_PY):
        sys.exit(f"init.sh finished but {VENV_PY} still absent — check its output")


def main() -> None:
    extra_eval_args = sys.argv[1:]  # anything after `uv run main`

    pin_gpu()  # CUDA_VISIBLE_DEVICES=0 for the whole run (before any torch import)

    ensure_env()

    # 1. build: Overture Places POI enrichment. Reads the public S3 Parquet over
    #    the network (no local extracts). On a GPU box with only the synced CSVs,
    #    ENRICHED already exists, so this is skipped and we use what's there.
    if not os.path.exists(ENRICHED):
        sh([VENV_PY, "-m", "airbnb_surroundings.build"])

    # 2. describe: LLM surroundings summary. Skipped once the described CSV exists.
    if not os.path.exists(DESCRIBED):
        if os.path.exists(ENRICHED):
            sh([VENV_PY, "-m", "airbnb_surroundings.describe"])
        else:
            sys.exit(f"neither {DESCRIBED} nor {ENRICHED} present — nothing to "
                     "describe; build the dataset locally and sync it over first")

    # 3. eval: the tabstar-dependent stage that was failing under `uv run`.
    #    Extra args augment the defaults (e.g. `-- --no-tar` toggles the flag)
    #    rather than replacing them, so --csv/--image-folder are always present.
    sh([VENV_PY, "-m", "airbnb_surroundings.eval", *DEFAULT_EVAL_ARGS, *extra_eval_args])


if __name__ == "__main__":
    main()
