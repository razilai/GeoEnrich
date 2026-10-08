"""uv entry points for the pipeline. `uv run main` runs the dataset-build chain;
the per-stage scripts run one stage each.

The heavy deps (tabstar, torch, autogluon, geopandas, duckdb, pydantic-ai) all
live in MulTaBench/.venv — the one env built by init.sh. `uv run` activates the
thin project .venv instead, so these launchers run every stage with
MulTaBench/.venv's interpreter, regardless of which env `uv run` picked.

Per-stage (each forwards its args; run any in isolation):
    uv run clean [RAW.csv]        0. data/raw/airbnb_nyc.csv -> data/processed/airbnb.csv (pandas)
    uv run build                  1. -> data/processed/airbnb_enriched.csv   (Overture POIs)
    uv run describe --confirm [N] 2. -> data/processed/airbnb_described_<prompt>.csv
                                     (spends LLM credits, so --confirm is mandatory)

Whole chain:
    uv run main --confirm    runs build -> describe, skipping any stage whose
                             output already exists (safe to re-run). clean is
                             upstream/manual, NOT part of this auto-chain.
    uv run main --prompt 08 --confirm   the same chain with prompt variant 08.

Benchmark-facing stages (stage, bench, report) run with HF_HUB_OFFLINE=1, and
device selection is the benchmark's own `--device` flag.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib

# Anchor to the repo root, which is the cwd `uv run main` executes from. Do NOT
# use __file__: this module ships as an installed wheel, so __file__ resolves to
# .venv/site-packages, not the project tree where init.sh / src / the CSVs live.
HERE = os.getcwd()
VENV_PY = os.path.join(HERE, "MulTaBench", ".venv", "bin", "python")
INIT_SH = os.path.join(HERE, "init.sh")

PROCESSED = os.path.join(HERE, "data", "processed")
ENRICHED = os.path.join(PROCESSED, "airbnb_enriched.csv")

# Stages that touch the benchmark: the Hugging Face hub must never be reached.
BENCHMARK_STAGES = frozenset({"stage", "bench", "report"})

with open(os.path.join(HERE, "src", "prompts.toml"), "rb") as f:
    _PROMPTS_TOML = tomllib.load(f)
DEFAULT_PROMPT = _PROMPTS_TOML["default"]
PROMPTS = sorted(_PROMPTS_TOML["prompts"])


def described(prompt: str) -> str:
    """Described CSV for one prompt variant (mirrors config.described_csv)."""
    return os.path.join(PROCESSED, f"airbnb_described_{prompt}.csv")


def pop_prompt(argv: list[str]) -> tuple[str, list[str]]:
    """Split `--prompt ID` off argv; the rest is forwarded to the stage."""
    prompt, rest = DEFAULT_PROMPT, []
    args = iter(argv)
    for arg in args:
        if arg == "--prompt":
            prompt = next(args, prompt)
        elif arg.startswith("--prompt="):
            prompt = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    if prompt not in PROMPTS:
        sys.exit(f"unknown --prompt {prompt!r}; choose from {PROMPTS}")
    return prompt, rest


def stage_env(module: str) -> dict[str, str]:
    """Environment for one stage; benchmark-facing stages run hub-offline."""
    env = dict(os.environ)
    if module in BENCHMARK_STAGES:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def sh(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run a subprocess from the repo root, aborting the pipeline on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=HERE, check=True, env=env)


def ensure_env() -> None:
    """Build MulTaBench/.venv (clone + deps) via init.sh if it isn't there yet."""
    if os.path.exists(VENV_PY):
        return
    print("MulTaBench/.venv missing — bootstrapping via init.sh", flush=True)
    sh(["bash", INIT_SH])
    if not os.path.exists(VENV_PY):
        sys.exit(f"init.sh finished but {VENV_PY} still absent — check its output")


def _stage(module: str, argv: list[str]) -> None:
    """Run one pipeline stage in MulTaBench/.venv, forwarding CLI args."""
    ensure_env()
    sh([VENV_PY, "-m", f"src.{module}", *argv], env=stage_env(module))


# Per-stage entry points (registered as uv scripts in pyproject.toml).
def clean() -> None:
    """`uv run clean [RAW.csv]` — stage 0: data/raw -> data/processed/airbnb.csv (pandas)."""
    _stage("clean", sys.argv[1:])


def build() -> None:
    """`uv run build` — stage 1: -> data/processed/airbnb_enriched.csv."""
    _stage("build", sys.argv[1:])


def describe() -> None:
    """`uv run describe --confirm [N] [--prompt ID]` — stage 2: -> airbnb_described_<ID>.csv."""
    _stage("describe", sys.argv[1:])


def main() -> None:
    prompt, rest = pop_prompt(sys.argv[1:])
    described_csv = described(prompt)
    ensure_env()

    # 1. build: Overture Places POI enrichment. Skipped once ENRICHED exists.
    if not os.path.exists(ENRICHED):
        sh([VENV_PY, "-m", "src.build"], env=stage_env("build"))

    # 2. describe: LLM surroundings summary. Skipped once the described CSV exists.
    if not os.path.exists(described_csv):
        if os.path.exists(ENRICHED):
            sh(
                [VENV_PY, "-m", "src.describe", "--prompt", prompt, *rest],
                env=stage_env("describe"),
            )
        else:
            sys.exit(
                f"neither {described_csv} nor {ENRICHED} present — nothing to "
                "describe; build the dataset first"
            )


if __name__ == "__main__":
    main()
