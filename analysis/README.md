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

`make_paper_assets.py` is rewritten in a later step to read the new results CSV.

The outputs are deterministic, reviewable files:

- `tables/table_1_dataset_card.{csv,tex}` — compact data/curation card.
- `tables/appendix_a1_field_inventory.csv` — field inventory and missingness.
- `tables/table_2_multabench_results.{csv,tex}` — the main five-learner table.
- `tables/appendix_a4_per_fold.{csv,tex}` — every official fold score.
- `tables/curation_verdict.json` — C1/C2 pass flags and the 3/5 verdict.
- `figures/figure_1_curation_pipeline.pdf` — the required curation diagram.
- `figures/figure_2_multabench_scores.pdf` — fold-mean scores with error bars.

`table_2` uses fold-mean R², with
`Δjoint = joint_frozen − max(structured, text)` and
`ΔTAR = joint_tar − joint_frozen`.  A positive value is required; the overall
verdict is pass only when at least three learners pass both conditions.

## Design boundary

The upstream README documents a CLI for datasets registered with MulTaBench.
The analysis folder adds no local-data path: register the release candidate
through that flow, then evaluate it through `benchmark.py`.
