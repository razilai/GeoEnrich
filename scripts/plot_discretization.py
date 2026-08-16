"""Per-category discretization histograms (density buckets).

Two 13-panel figures over the enriched corpus:
  1. Deviation bands   — count percentile vs corpus (the density token in the prompt)
  2. Proximity bands   — nearest_m -> doorstep / steps / short walk

Run: python scripts/plot_discretization.py  (writes PNGs to artifacts/)
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from airbnb_surroundings import config as C
from airbnb_surroundings import describe as D

# density buckets in display order
BUCKETS = list(D._DISPLAY.keys())

# --- deviation bands: replicate describe._relative but always return a bin so
# every listing lands somewhere (the prompt omits the middle third as "typical";
# here we surface it so the histogram covers the full corpus).
DEV_ORDER = ["none", "almost none", "well below", "below",
             "typical", "above", "well above", "far more"]


def dev_band(bucket, v):
    arr = D._REF.get(bucket)
    if arr is None or len(arr) == 0:
        return "typical"
    if v <= 0:
        return "none"
    pct = np.searchsorted(arr, v, side="left") / len(arr)
    if pct >= 0.95:
        return "far more"
    if pct >= 0.80:
        return "well above"
    if pct >= 0.65:
        return "above"
    if pct <= 0.05:
        return "almost none"
    if pct <= 0.20:
        return "well below"
    if pct <= 0.35:
        return "below"
    return "typical"


# --- proximity bands: describe._prox_word over nearest_m (present buckets only)
PROX_ORDER = ["doorstep", "steps", "short walk"]


def prox_band(m):
    return "doorstep" if m <= 50 else "steps" if m <= 150 else "short walk"


def _grid(counts_by_bucket, order, title, colors, out):
    """13-panel bar grid, one panel per bucket, shared band x-axis."""
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(title, fontsize=15, y=0.995)
    axes = axes.ravel()
    for ax, b in zip(axes, BUCKETS):
        c = counts_by_bucket[b]
        vals = [c.get(k, 0) for k in order]
        tot = sum(vals) or 1
        ax.bar(range(len(order)), vals, color=colors)
        ax.set_title(D._DISPLAY[b], fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        for i, v in enumerate(vals):  # pct labels
            if v:
                ax.text(i, v, f"{100*v/tot:.0f}%", ha="center", va="bottom", fontsize=6)
    for ax in axes[len(BUCKETS):]:  # hide the 3 spare panels (16-13)
        ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=120)
    print("wrote", out)


def main():
    df = pd.read_csv(C.ENRICHED_CSV)
    surr = df["surroundings"].map(lambda s: json.loads(s).get("cats", {}))
    D.load_reference(df)  # fills D._REF for dev_band

    dev = {b: {} for b in BUCKETS}
    prox = {b: {} for b in BUCKETS}
    for cats in surr:
        for b in BUCKETS:
            e = cats.get(b)  # [c150, c400, nearest_m] or None if absent (>450m)
            v = e[1] if e else 0
            db = dev_band(b, v)
            dev[b][db] = dev[b].get(db, 0) + 1
            if e:  # proximity only meaningful when bucket present
                pb = prox_band(e[2])
                prox[b][pb] = prox[b].get(pb, 0) + 1

    os.makedirs(C.ARTIFACTS_DIR, exist_ok=True)
    _grid(dev, DEV_ORDER,
          f"Density deviation-band distribution per category (n={len(df)})",
          plt.cm.RdBu_r(np.linspace(0.1, 0.9, len(DEV_ORDER))),
          os.path.join(C.ARTIFACTS_DIR, "discretization_deviation.png"))
    _grid(prox, PROX_ORDER,
          f"Proximity-band distribution per category, present buckets (n={len(df)})",
          ["#2a6", "#7c5", "#cc8"],
          os.path.join(C.ARTIFACTS_DIR, "discretization_proximity.png"))


if __name__ == "__main__":
    main()
