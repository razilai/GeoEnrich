# GeoEnrich-NYC paper analysis

This directory is the reproducibility layer for the paper in `paper/paper.md`.
It deliberately does **not** reimplement MulTaBench. Evaluation uses the
public upstream repository, [alanarazi7/MulTaBench](https://github.com/alanarazi7/MulTaBench), by calling its documented
`benchmark.py` command directly. No local evaluator, split generator, encoder,
learner, LoRA implementation, or copied MulTaBench code exists here.

`MulTaBench/` is assumed to be the available benchmark checkout, with its
documented environment and credentials already configured.

## 1. Configure the release candidate

Copy the template and fill in the dataset metadata before a final run:

```bash
cp analysis/config.example.json analysis/config.json
python analysis/summarize_dataset.py --config analysis/config.json --out analysis/output
```

## 2. Run the benchmark

The benchmark runner and result export are being rebuilt around the official
sweep tool (see `.scratch/official-eval/`). Until then, no run script lives
here: register the dataset in the official repository's supported flow and run
`MulTaBench/benchmark.py` directly.

## 3. Extract paper assets

After `uv run report --grid final`, render the paper assets from its two CSVs
(`results/metrics_final.csv`, `results/verdict_final.csv`) and nothing else:

```bash
MulTaBench/.venv/bin/python analysis/make_paper_assets.py --grid final [--config analysis/config.json]
```

Outputs go to `results/paper/` (`--out` to change):

- `tables/table_2_results.{csv,tex}` — per arm and committee learner: the four
  condition scores, the `latlon` reference, joint signal and TAR gain deltas, and
  whether the arm is eligible.
- `tables/appendix_a4_per_split.{csv,tex}` — every run's score and error.
- `methods.md` — the evaluation-protocol paragraph (six splits, score and error,
  leaderboard non-comparability, the deliberate `landmark_redacted` token).
- `figures/figure_2_scores.pdf` — split-mean score per condition, ±1 SEM.
- `figures/figure_1_curation_pipeline.pdf` — only with `--config`.

The logic lives in `src/paper.py`; a score is the benchmark's negated RMSE
(higher is better), so it is not comparable to the published leaderboard.

## Design boundary

The upstream README documents a CLI for datasets registered with MulTaBench.
The analysis folder adds no local-data path: register the release candidate
through that flow, then evaluate it through `benchmark.py`.
