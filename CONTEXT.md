# GeoEnrich-NYC

Curation of a geospatial text-tabular dataset candidate for the MulTaBench
benchmark: NYC Airbnb listings whose coordinates are replaced by a
natural-language description of each listing's surroundings.

## Language

### The three independent axes

These were conflated as "variant" and must stay distinct; every evaluation row
is identified by all three plus a learner and a split.

**Prompt**:
One entry in `src/prompts.toml` (`05`, `08`, `16`), pairing an evidence view
with the system instruction written for it. The screen selects exactly one;
the losers are then deleted, and this axis disappears from the final design.
_Avoid_: variant, prompt variant, template

**Arm**:
One dataset built for the paper's comparison — `latlon`, `enriched`,
`landmark_only`, or `landmark_redacted` — differing only in what stands in for
location. All arms share the same rows and tabular columns, except `latlon`,
which keeps the coordinates and carries no text.
_Avoid_: variant, censored variant, version, v1/v2/v3/v4

**Condition**:
One of the four curation inputs — `structured`, `text_only`, `joint_frozen`,
`joint_tar`. Not a MulTaBench flag: each is a registered dataset id whose
parquet carries the right columns, and the two joint conditions differ only by
`--text_encoder e5-small` versus `e5-small-tar`.
_Avoid_: multimodal state, mode, setting

### Pipeline stages

The stage names are also the CLI verbs: `clean → build → describe → arms →
stage → bench → report`.

**Enrichment**:
Assigning every listing the Overture POIs within `DOORSTEP` (150 m), `RADIUS`
(450 m) and `LANDMARK_RADIUS` (800 m) of its coordinates, then dropping the
coordinates.
_Avoid_: joining, geocoding

**Evidence**:
The rendered view of a listing's surroundings JSON that a prompt receives:
citywide percentile bands, doorstep proximity cues, and named landmarks. Never
the raw counts, the coordinates, or the price.
_Avoid_: context, features, input

**Summary**:
The one-or-two-sentence LLM output in the `surroundings_summary` column. The
dataset's only text feature.
_Avoid_: description, caption, blurb

**Landmark**:
One of the 77 curated attractions in `src/landmarks.json`, matched by distance
to its geometry rather than by POI name, and named verbatim in a summary. The
unit the `landmark_only` and `landmark_redacted` arms censor on.
_Avoid_: POI, attraction, anchor

**Staging**:
Writing an arm-and-condition pair as `data.parquet` + `metadata.json` into the
local Hugging Face cache layout, with column types declared explicitly so the
benchmark reads `room_type` as categorical and `surroundings_summary` as text.
Nothing is uploaded; the cache is read offline.
_Avoid_: publishing, uploading, registering

### Evaluation

**Split**:
One of the benchmark's six outer train/test partitions — TabArena's 3-fold
67/33 scheme, repeated twice.
_Avoid_: fold

**Score**:
A run's `test_score`: negated RMSE for this regression target, so higher is
better. RMSE itself is `test_error`. R² is not reported.
_Avoid_: R², accuracy, metric

**TAR**:
Target-aware representation — the text encoder fine-tuned on the regression
target with LoRA, requested as `--text_encoder e5-small-tar`.
_Avoid_: finetune, FT

**Joint signal**:
A learner's mean `joint_frozen` score exceeds both its mean `structured` and
its mean `text_only` score by more than 0.001.
_Avoid_: C1, criterion 1

**TAR gain**:
A learner's mean `joint_tar` score exceeds its mean `joint_frozen` score by
more than 0.001.
_Avoid_: C2, awareness gain

**Eligible**:
An arm for which at least 3 of the 5 committee learners (`light`, `cat`,
`tabm`, `tabpfnv2`, `tabpfnv2p5`) show both joint signal and TAR gain. The
property the paper claims.
_Avoid_: passing, curated, accepted
