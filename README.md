# GeoEnrich-NYC

A text-tabular dataset of 14,990 New York City Airbnb listings for nightly-price regression, in which each
listing's coordinates are replaced by an LLM-generated description of its surroundings, built from
Overture Maps places data.

**Dataset:** [huggingface.co/datasets/ElyR120/geoenrich-nyc](https://huggingface.co/datasets/ElyR120/geoenrich-nyc)
(CC BY 4.0, in [MulTaBench](https://github.com/alanarazi7/MulTaBench)'s format: typed `data.parquet` + `metadata.json`)

## The dataset

| | |
|---|---|
| Rows | 14,990 |
| Target | `price`: $\log(1 + \text{nightly price in USD})$ |
| Structured columns (8) | `ratings`, `guests`, `bathrooms`, `num_bedrooms`, `property_number_of_reviews`, `minimum_nights`, `room_type` (categorical), `is_superhost` (boolean) |
| Text column (1) | `surroundings_summary`: one or two sentences describing the listing's surroundings |
| Not included | coordinates, listing IDs, host names |

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/ElyR120/geoenrich-nyc/data.parquet")
```

Example summary: *"This busy neighbourhood features far more discount stores than most neighbourhoods, with
historic sites steps away. Central Park is about 10 minutes away."*

## How it is built

1. **Clean.** Start from Inside Airbnb's NYC snapshot (scraped 14–23 June 2026, 30,259 listings). Drop
   listings with a missing or non-positive price, missing coordinates, or more than 25% of features missing
   (21,155 left), then sample 15,000 at random (seed 0).
2. **Enrich.** Join each listing to the Overture Maps places (release `2026-08-19.0`, confidence ≥ 0.6)
   within 450 m, across 20 categories curated from Overture's level-2 taxonomy. The 10 listings with no
   place nearby are dropped, leaving 14,990.
3. **Aggregate.** Rank each category count, and the total, against the citywide distribution and keep the
   percentile. Match the 77 curated landmarks in `src/landmarks.json` within 800 m, by name.
4. **Describe.** Render the record as an evidence block of word bands (busyness, outer-15% category
   contrasts, one proximity cue within 150 m, up to three landmarks) and ask Gemini 3.5 Flash-Lite
   (`google/gemini-3.5-flash-lite`, temperature 0.4, through OpenRouter's batch API) for a neutral summary.
   The LLM never sees coordinates, exact counts, exact distances, or price.

Generation is sampled, so reruns differ; the released text column is the reference.

## Results

Evaluated with the MulTaBench curation criterion (latest `master`, pinned in `init.sh`): RMSE on
$\log(1+\text{price})$, mean over six splits, lower is better.

| Learner | Structured | Text-only | Joint frozen | Joint TAR |
|---|---|---|---|---|
| LightGBM | 0.501 | 0.726 | 0.442 | 0.435 |
| CatBoost | 0.498 | 0.722 | 0.435 | 0.431 |
| TabM | 0.505 | 0.720 | 0.434 | 0.432 |
| TabPFNv2 | 0.490 | 0.715 | 0.422 | 0.418 |
| TabPFNv2.5 | 0.490 | 0.710 | 0.415 | 0.412 |

All five learners show joint signal (joint frozen beats both unimodal baselines) and TAR gain (fine-tuning
the E5-small text encoder on the target beats frozen embeddings), so the dataset passes the criterion 5/5.
The TAR gain (0.002–0.007) is of the same order as the split-to-split standard deviation. See the paper for
limitations.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/). `init.sh` clones MulTaBench at the pinned commit and builds
`MulTaBench/.venv`, which every stage runs in. Put the raw snapshot at `data/raw/airbnb_nyc.csv` and
`OPENROUTER_API_KEY` in `.env`. `describe` defaults to the paper's model,
`google/gemini-3.5-flash-lite:batch`; set `LLM_MODEL` to override it.

```bash
bash init.sh
uv run clean                      # raw snapshot -> data/processed/airbnb.csv
uv run build                      # + Overture places -> airbnb_enriched.csv
uv run describe --confirm         # LLM summaries (spends credits) -> airbnb_described_16.csv
uv run stage --prompt 16          # register the conditions in the local HF cache (nothing uploaded)
uv run bench --grid final --prompt 16 --confirm   # benchmark grid (GPU)
uv run report --grid final        # metrics and verdict -> results/
```

`scripts/evaluate.sh --prompt 16` runs stage → bench → report in one go. Run the tests with
`MulTaBench/.venv/bin/python -m pytest tests`.

## Layout

| Path | Contents |
|---|---|
| `src/` | one module per pipeline stage; `config.py` holds radii, categories and the Overture release |
| `src/prompts.toml` | the prompt (`16` is the released one) |
| `src/landmarks.json` | the 77 curated landmarks |
| `run.py` | `uv run` entry points |
| `analysis/` | dataset statistics and paper assets |
| `paper/latex/` | the paper |
| `docs/adr/` | design decisions |

## License

Dataset: CC BY 4.0. Attribute Inside Airbnb (CC BY 4.0) and the Overture Maps Foundation
(CDLA-Permissive-2.0).
