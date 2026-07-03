"""
explore.py — run this ONCE on your machine to see exactly what each feed returns.

It pulls every feed and, for each, prints:
  - number of rows and columns
  - every column name
  - the data type
  - one real sample value (first non-blank)

This is the ground-truth field inventory to design your visualizations against.
It writes a tidy summary to ./out/field_inventory.csv so you can browse it in Excel.

Run:  python explore.py
      python explore.py --no-eia    (skip the electricity feed if you have no key)
"""

from __future__ import annotations
import sys
import pandas as pd

import ingest as ig   # reuses the pullers you already have

FEEDS = {
    # name             puller                lens
    "ai_models":       (ig.pull_models,      "demand"),
    "ai_companies":    (ig.pull_companies,   "demand"),
    "gpu_clusters":    (ig.pull_clusters,    "capacity"),
    "data_centers":    (ig.pull_datacenters, "capacity+power"),
    "ml_hardware":     (ig.pull_hardware,    "capacity"),
    "ai_chip_sales":   (ig.pull_chipsales,   "capacity"),
    "ai_chip_owners":  (ig.pull_chipowners,  "capacity"),
}


def first_sample(series: pd.Series):
    """Return the first non-null value, trimmed, or '(all blank)'."""
    s = series.dropna()
    if s.empty:
        return "(all blank)"
    val = str(s.iloc[0])
    return val[:60] + ("..." if len(val) > 60 else "")


def describe(name: str, lens: str, df: pd.DataFrame, rows: list):
    print("\n" + "=" * 70)
    if df.empty:
        print(f"{name}  [{lens}]  —  NO DATA (feed unreachable or empty)")
        return
    print(f"{name}  [{lens}]  —  {len(df):,} rows  x  {df.shape[1]} columns")
    print("-" * 70)
    print(f"{'COLUMN':<34}{'TYPE':<12}SAMPLE")
    print("-" * 70)
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = first_sample(df[col])
        print(f"{col[:33]:<34}{dtype:<12}{sample}")
        rows.append({"feed": name, "lens": lens, "column": col,
                     "dtype": dtype, "sample_value": sample})


def main(use_eia: bool = True):
    inventory: list[dict] = []

    for name, (puller, lens) in FEEDS.items():
        df = ig.safe(puller, name)
        describe(name, lens, df, inventory)

    if use_eia:
        try:
            eia = ig.pull_eia(ig.C.EIA_ROUTES["hourly_demand"])
            describe("eia_hourly_demand", "power", eia, inventory)
        except Exception as e:
            print(f"\neia_hourly_demand [power] — SKIPPED: {e}")

    inv = pd.DataFrame(inventory)
    ig.OUT.mkdir(exist_ok=True)
    out_path = ig.OUT / "field_inventory.csv"
    inv.to_csv(out_path, index=False)
    print("\n" + "=" * 70)
    print(f"Field inventory saved to: {out_path}")
    print("Open that file in Excel to browse every field across all feeds.")


if __name__ == "__main__":
    main(use_eia="--no-eia" not in sys.argv)
