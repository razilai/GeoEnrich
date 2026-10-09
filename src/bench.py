"""Run the benchmark grid with the official entry point; no evaluation logic lives here.

The pipeline logic is the pure `plan_runs`: a grid in, the benchmark's own `Run`s out. `main` is a thin shell
that skips runs with an existing result (the sweep tool's `pending_runs`), then runs the
rest one after another with `benchmark.py --device`, a failed run never stopping the
others (benchmark.py records the failure in the run's own result JSON).

The sweep tool's local launcher cannot pass `--device`, so its list/skip/launch steps
are used directly instead of its CLI.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

from multabench.benchmark.runs import Run
from multabench.benchmark.splits import SIZE_10K
from multabench.benchmark.sweep import pending_runs
from multabench.e5.constants import TextEncoder

from src import config, stage

COMMITTEE = ("light", "cat", "tabm", "tabpfnv2", "tabpfnv2p5")
SCREEN_SPLITS = (0,)
FINAL_SPLITS = (0, 1, 2, 3, 4, 5)
FROZEN = TextEncoder.E5_SMALL
TAR = TextEncoder.E5_SMALL_TAR

# The probe is a single learner on a single split of one joint dataset.
PROBE_LEARNER = "light"
PROBE_SPLIT = 0

Pair = tuple[str, TextEncoder]  # (registered dataset key, text encoder)


@dataclass(frozen=True)
class Grid:
    pairs: tuple[Pair, ...]
    learners: tuple[str, ...]
    splits: tuple[int, ...]

    @property
    def total(self) -> int:
        return len(self.pairs) * len(self.learners) * len(self.splits)


def _pairs(prompt: str) -> tuple[Pair, ...]:
    """The four conditions of one staged corpus: structured, text_only, joint frozen and joint TAR."""
    key = lambda condition: stage.dataset_key(stage.ARM, prompt, condition)  # noqa: E731
    return (("structured", FROZEN), (key("text_only"), FROZEN), (key("joint"), FROZEN), (key("joint"), TAR))


def screen_grid(prompt: str) -> Grid:
    """The final grid on split 0 only."""
    return Grid(_pairs(prompt), COMMITTEE, SCREEN_SPLITS)


def final_grid(prompt: str) -> Grid:
    return Grid(_pairs(prompt), COMMITTEE, FINAL_SPLITS)


def probe_grid(prompt: str) -> Grid:
    """Two runs, one TAR and one frozen, on the same joint dataset, to size a grid from."""
    joint = stage.dataset_key(stage.ARM, prompt, "joint")
    return Grid(((joint, FROZEN), (joint, TAR)), (PROBE_LEARNER,), (PROBE_SPLIT,))


def plan_runs(grid: Grid, size: str = SIZE_10K) -> list[Run]:
    """Every (pair, learner, split) as the benchmark's `Run`.

    Dataset names are built from the stage's naming rule and not looked up in the
    benchmark registry, so planning works before `stage` has registered anything.
    """
    return [
        Run(learner, stage.enum_name(key), encoder, None, size, split)
        for key, encoder in grid.pairs
        for learner in grid.learners
        for split in grid.splits
    ]


def run_command(run: Run, python: str, output_dir: str, device: str) -> list[str]:
    """The exact argument list for one run: the benchmark's own command plus `--device`."""
    return [*shlex.split(run.command(python=python, output_dir=output_dir)), "--device", device]


def offline_env() -> dict[str, str]:
    return {**os.environ, "HF_HUB_OFFLINE": "1"}


# --- Entry point -----------------------------------------------------------------------
def execute(runs: list[Run], output_dir: str, device: str, repo_dir: str) -> int:
    """Run each pending run in its own process; returns the number of failed runs."""
    pending = pending_runs(runs, output_dir=output_dir)
    print(f"{len(runs)} runs, {len(runs) - len(pending)} already have a result, {len(pending)} to run")
    failed = 0
    for i, run in enumerate(pending, start=1):
        cmd = run_command(run, sys.executable, output_dir, device)
        print(f"[{i}/{len(pending)}] {' '.join(cmd)}", flush=True)
        failed += subprocess.run(cmd, cwd=repo_dir, env=offline_env()).returncode != 0
    print(f"Done: {len(pending) - failed} succeeded, {failed} failed")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", choices=["screen", "final", "probe"], required=True)
    parser.add_argument("--confirm", action="store_true", help="required: the grid occupies the GPU")
    parser.add_argument("--device", default="cuda", help="forwarded to benchmark.py (e.g. cuda:1)")
    parser.add_argument("--prompt", required=True, help="the prompt id `stage` was run with")
    parser.add_argument("--output_dir", default=os.path.join(config.RESULTS_DIR, "runs"))
    args = parser.parse_args()

    if not args.confirm:
        raise SystemExit("bench occupies the GPU; re-run with --confirm")
    grid = {"screen": screen_grid, "final": final_grid, "probe": probe_grid}[args.grid](args.prompt)
    repo_dir = os.path.join(config.ROOT, "MulTaBench")
    failed = execute(plan_runs(grid), os.path.abspath(args.output_dir), args.device, repo_dir)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
