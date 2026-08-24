"""uv entry points for the pipeline. `uv run main` runs the whole chain; the
per-stage scripts run one stage each.

The heavy deps (tabstar, torch, autogluon, geopandas, duckdb, pyspark,
pydantic-ai) all live in MulTaBench/.venv — the one env built by init.sh. `uv
run` activates the thin project .venv instead, which has no tabstar, so importing
the eval there fails with "tabstar not installed". These launchers sidestep that:
every stage runs with MulTaBench/.venv's interpreter, regardless of which env
`uv run` picked.

Per-stage (each forwards its args; run any in isolation):
    uv run clean [RAW.csv]   0. data/raw/airbnb_nyc.csv -> data/processed/airbnb.csv (PySpark)
    uv run build             1. -> data/processed/airbnb_enriched.csv   (Overture POIs)
    uv run describe [N]      2. -> data/processed/airbnb_described.csv   (LLM key; N = top-N test)
    uv run enrich            -> data/processed/airbnb_described_16.csv  (fixed prompt 16)
    uv run eval [--no-tar]   3. -> results/eval_report.csv              (needs GPU)

Whole chain:
    uv run main              runs build -> describe -> eval, skipping any stage
                             whose output already exists (safe to re-run). clean
                             is upstream/manual, NOT part of this auto-chain.
    uv run main -- --no-tar  extra args after `--` augment the eval defaults.

On a fresh GPU box the usual flow is: clean + build + describe locally,
`scripts/sync.sh` the CSVs over, then `uv run eval` here.
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

PROCESSED = os.path.join(HERE, "data", "processed")
ENRICHED = os.path.join(PROCESSED, "airbnb_enriched.csv")
DESCRIBED = os.path.join(PROCESSED, "airbnb_described.csv")

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


def _stage(module: str, argv: list[str], *, gpu: bool = False) -> None:
    """Run one pipeline stage in MulTaBench/.venv, forwarding CLI args."""
    if gpu:
        pin_gpu()
    ensure_env()
    sh([VENV_PY, "-m", f"airbnb_surroundings.{module}", *argv])


# Per-stage entry points (registered as uv scripts in pyproject.toml). Each runs
# the stage in MulTaBench/.venv and forwards args, so `uv run build`, `uv run
# describe 10`, etc. Just Work regardless of which env `uv run` activated.
def clean() -> None:
    """`uv run clean [RAW.csv]` — stage 0: data/raw -> data/processed/airbnb.csv (PySpark)."""
    _stage("clean", sys.argv[1:])


def clean_pandas() -> None:
    """`uv run clean-pandas [RAW.csv]` — stage 0, pandas twin of clean (no JVM)."""
    _stage("clean_pandas", sys.argv[1:])


def build() -> None:
    """`uv run build` — stage 1: -> data/processed/airbnb_enriched.csv."""
    _stage("build", sys.argv[1:])


def describe() -> None:
    """`uv run describe [N]` — stage 2: -> data/processed/airbnb_described.csv (LLM key)."""
    _stage("describe", sys.argv[1:])


def enrich() -> None:
    """`uv run enrich` — full fine-schema corpus with the fixed prompt-16 profile."""
    _stage("enrich", sys.argv[1:])


def evaluate() -> None:
    """`uv run eval [extra]` — stage 3: MulTaBench eligibility (needs GPU + tabstar).

    Extra args augment the defaults, e.g. `uv run eval --no-tar` (joint-signal only).
    """
    _stage("eval", [*DEFAULT_EVAL_ARGS, *sys.argv[1:]], gpu=True)


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
