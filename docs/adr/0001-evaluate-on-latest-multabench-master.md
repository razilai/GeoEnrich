# Evaluate on the latest MulTaBench master, not the paper_version tag

Status: accepted

MulTaBench's `master` is moving towards a living leaderboard and has diverged
sharply from the `paper_version` tag that its own README points at for
reproducing the published paper. We evaluate GeoEnrich-NYC on the latest
`master`, pinned to commit `d88821d`, on the principle that the benchmark's
current measurements are the ones worth adhering to.

## Considered Options

**`paper_version` (tag `3bb9507`)** implements the curation protocol this project's
claim is phrased in: a `--multimodal_state` flag giving the four conditions
directly, R² as the regression metric, five folds, a first-class local-CSV
dataset mechanism needing no upload, and — decisively — a runnable implementation
of the curation criterion itself in its leaderboard analysis code. Its cost is a
hard dependency on Weights & Biases, which it refuses to run without.

**The author's fork** adds a cross-learner encoder cache that makes a full grid
several times faster, but is based on a commit well behind `paper_version` and
would have had to be maintained against it.

**Latest `master`** writes each run's result to a local JSON file and collects
them into a CSV, removing the W&B dependency entirely (upstream commit
`23e46a7`). It also removed, in later commits, every one of the things listed
above: the four-condition flag, the Kaggle/URL/OpenML dataset mechanisms in
favour of Hugging Face only, R² in favour of TabArena's RMSE, five folds in
favour of a repeated 3-fold scheme giving six splits, and the curation criterion
implementation along with the rest of the paper analysis code.

## Consequences

Three consequences are surprising enough to state, because a future reader will
otherwise assume a mistake.

**The reported metric is negated RMSE, not R².** Every prior Airbnb-pricing
result this project cites uses R², and so does the published MulTaBench
leaderboard, so our scores are not directly comparable to either. R² is
deliberately not computed, on the rule that the results table contains nothing
the official repository does not produce.

**The curation verdict is implemented in this repository.** This looks like the
reimplementation the project otherwise avoids, and it is not a lapse: `master`
deleted the official implementation, and `paper_version`'s version hardcodes five
folds and asserts a complete five-fold grid, so it cannot consume six splits
unmodified. Our module reproduces the published criterion exactly — three-decimal
rounding of state means before differencing, a 0.001 margin on both deltas, and a
three-of-five quorum over the committee — and is labelled as this project's
computation of MulTaBench's criterion rather than as MulTaBench's own.

**The four conditions are registered datasets, not a flag.** With
`--multimodal_state` gone, `structured` and `text_only` are separate dataset ids
whose stored columns define the condition, and the two joint conditions are one
id run under the frozen and the TAR text encoder. Because `master` reads feature
types from stored dtypes, this makes the column type declarations load-bearing:
an inferred object dtype would be classified as text and silently corrupt every
condition.

Also accepted: `master` has no embedding cache, and the fine-tune is
learner-independent, so most of the grid's LoRA fine-tunes are recomputations.
That cost was the fork's reason to exist and is given up here deliberately.
