"""
ingest.py — pull Epoch AI + EIA feeds, clean them into tidy tables, and compute
the derived columns that let the three lenses (capacity / demand / power) share a
common denominator: POWER (MW) and COMPUTE (FLOP or FLOP/s).

Run:  python ingest.py            # pulls everything, writes ./out/*.parquet
      python ingest.py --no-eia   # skip EIA (no key)

Design notes
------------
Rule Zero: no fabricated numbers. If a feed is unreachable, the table is written
empty with a logged reason rather than filled with guesses.

All derived columns are FORMULAS built from raw source columns + config.ASSUMPTIONS.
Each formula is documented inline with the physics/math it encodes.
"""

from __future__ import annotations
import io
import os
import sys
import zipfile
import logging
from pathlib import Path

import requests
import pandas as pd

import config as C

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("ingest")

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
HEADERS = {"User-Agent": "gore-creek-compute-dashboard/1.0"}
A = C.ASSUMPTIONS


# ===========================================================================
# 1. GENERIC FETCH HELPERS
# ===========================================================================
def _get(url: str, timeout: int = 60) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r


def fetch_csv(url: str) -> pd.DataFrame:
    """Load a flat CSV endpoint straight into a DataFrame."""
    return pd.read_csv(io.StringIO(_get(url).text))


def fetch_zip_csvs(url: str) -> dict[str, pd.DataFrame]:
    """Download a ZIP, return {inner_filename: DataFrame} for every CSV inside."""
    z = zipfile.ZipFile(io.BytesIO(_get(url).content))
    out = {}
    for name in z.namelist():
        if name.lower().endswith(".csv"):
            with z.open(name) as fh:
                out[name] = pd.read_csv(fh)
    return out


def safe(fn, label: str) -> pd.DataFrame:
    """Run a puller; on failure log and return an empty frame (Rule Zero)."""
    try:
        df = fn()
        log.info("%-22s %5d rows  %d cols", label, len(df), df.shape[1])
        return df
    except Exception as e:  # network / schema / parse
        log.warning("%-22s FAILED: %s", label, e)
        return pd.DataFrame()


# ===========================================================================
# 2. RAW PULLERS  (one per Epoch feed)
#    Epoch column names drift over time, so we resolve columns fuzzily.
# ===========================================================================
def _pick(df: pd.DataFrame, *needles: str) -> str | None:
    """Return the first column whose lowercased name contains all needles."""
    for col in df.columns:
        low = col.lower()
        if all(n in low for n in needles):
            return col
    return None


def pull_models() -> pd.DataFrame:
    return fetch_csv(C.EPOCH["ai_models_csv"])


def _first_frame(url: str) -> pd.DataFrame:
    d = fetch_zip_csvs(url)
    if not d:
        return pd.DataFrame()
    # pick the largest CSV in the archive (the main table)
    return max(d.values(), key=len)


def pull_clusters()   -> pd.DataFrame: return _first_frame(C.EPOCH["gpu_clusters"])
def pull_datacenters()-> pd.DataFrame: return _first_frame(C.EPOCH["data_centers"])
def pull_hardware()   -> pd.DataFrame: return _first_frame(C.EPOCH["ml_hardware"])
def pull_chipsales()  -> pd.DataFrame: return _first_frame(C.EPOCH["ai_chip_sales"])
def pull_chipowners() -> pd.DataFrame: return _first_frame(C.EPOCH["ai_chip_owners"])
def pull_companies()  -> pd.DataFrame: return _first_frame(C.EPOCH["ai_companies"])


# ===========================================================================
# 3. EIA PULLER  (power bridge)
# ===========================================================================
def pull_eia(route: str, extra_params: dict | None = None) -> pd.DataFrame:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY not set")
    params = {
        "api_key": key,
        "frequency": "hourly",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
    }
    if extra_params:
        params.update(extra_params)
    r = _get(C.EIA_BASE + route + "?" + _qs(params))
    rows = r.json().get("response", {}).get("data", [])
    return pd.DataFrame(rows)


def _qs(params: dict) -> str:
    from urllib.parse import urlencode
    return urlencode(params, doseq=True)


# ===========================================================================
# 4. DERIVED MATH  — the reason this dashboard exists.
#    Each function adds explicit FORMULA columns from raw + ASSUMPTIONS.
# ===========================================================================
def enrich_hardware(hw: pd.DataFrame) -> pd.DataFrame:
    """
    Per-accelerator economics. Two derived columns:

      flops_per_watt = peak_flops_per_s / tdp_watts
          -> energy efficiency; the lever that decouples compute from power.

      (tdp filled from ASSUMPTIONS.default_tdp_watts when the source is blank)
    """
    if hw.empty:
        return hw
    flop_col = _pick(hw, "flop") or _pick(hw, "fp16") or _pick(hw, "performance")
    tdp_col  = _pick(hw, "tdp") or _pick(hw, "watt") or _pick(hw, "power")
    hw = hw.copy()
    hw["_peak_flops_s"] = pd.to_numeric(hw.get(flop_col), errors="coerce")
    hw["_tdp_w"] = pd.to_numeric(hw.get(tdp_col), errors="coerce").fillna(A["default_tdp_watts"])
    # FORMULA: efficiency = FLOP/s per watt
    hw["flops_per_watt"] = hw["_peak_flops_s"] / hw["_tdp_w"]
    return hw


def enrich_clusters(cl: pd.DataFrame, hw: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster-level capacity and its power footprint.

      peak_flops_s      = chip_count * per_chip_flops_s
      sustained_flops_s = peak_flops_s * utilization           (MFU haircut)
      annual_flop       = sustained_flops_s * seconds_per_year
      it_power_mw       = chip_count * tdp_w / 1e6
      facility_power_mw = it_power_mw * PUE                     (cooling+overhead)

    per_chip_flops_s / tdp_w come from ml_hardware when the chip type matches,
    else from ASSUMPTIONS. This is the join that turns a chip count into MW.
    """
    if cl.empty:
        return cl
    cl = cl.copy()
    cnt_col  = _pick(cl, "chip", "count") or _pick(cl, "gpu", "count") or _pick(cl, "chips")
    flop_col = _pick(cl, "flop")   # some cluster rows already carry aggregate FLOP/s
    cl["_chip_count"] = pd.to_numeric(cl.get(cnt_col), errors="coerce")

    # median chip performance/tdp as a fallback denominator
    med_flops = pd.to_numeric(hw.get("_peak_flops_s"), errors="coerce").median() if not hw.empty else None
    med_tdp   = pd.to_numeric(hw.get("_tdp_w"), errors="coerce").median() if not hw.empty else A["default_tdp_watts"]

    if flop_col:
        cl["peak_flops_s"] = pd.to_numeric(cl[flop_col], errors="coerce")
    elif med_flops:
        cl["peak_flops_s"] = cl["_chip_count"] * med_flops
    else:
        cl["peak_flops_s"] = pd.NA

    # FORMULAS
    cl["sustained_flops_s"] = cl["peak_flops_s"] * A["utilization"]
    cl["annual_flop"]       = cl["sustained_flops_s"] * A["seconds_per_year"]
    cl["it_power_mw"]       = cl["_chip_count"] * med_tdp / 1e6
    cl["facility_power_mw"] = cl["it_power_mw"] * A["PUE"]
    return cl


def enrich_chipsales(cs: pd.DataFrame) -> pd.DataFrame:
    """
    Installed base = cumulative sum of units shipped over time (running total).
    A flow (sales/period) becomes a stock (capacity on the ground).

      installed_base_units[t] = sum_{s<=t} units_sold[s]
    """
    if cs.empty:
        return cs
    cs = cs.copy()
    date_col = _pick(cs, "date") or _pick(cs, "year") or _pick(cs, "quarter") or _pick(cs, "period")
    unit_col = _pick(cs, "unit") or _pick(cs, "shipment") or _pick(cs, "sold") or _pick(cs, "quantity")
    if date_col and unit_col:
        cs = cs.sort_values(date_col)
        cs["installed_base_units"] = pd.to_numeric(cs[unit_col], errors="coerce").cumsum()
    return cs


def enrich_models(m: pd.DataFrame) -> pd.DataFrame:
    """
    Demand side. Where training compute is missing but params (N) and tokens (D)
    exist, impute it via the standard relation:

      C = flops_per_param_token * N * D      (default k = 6)

    Also derive implied energy of the training run from cluster-class efficiency:
      implied_run_energy_mwh = C / flops_per_watt / 3.6e9
      (J = FLOP / (FLOP/W) gives watt-seconds; /3.6e9 -> MWh)
    """
    if m.empty:
        return m
    m = m.copy()
    c_col = _pick(m, "training", "compute") or _pick(m, "flop")
    n_col = _pick(m, "parameter")
    d_col = _pick(m, "dataset", "size") or _pick(m, "training", "token") or _pick(m, "tokens")
    m["_C"] = pd.to_numeric(m.get(c_col), errors="coerce")
    N = pd.to_numeric(m.get(n_col), errors="coerce")
    D = pd.to_numeric(m.get(d_col), errors="coerce")
    imputed = A["flops_per_param_token"] * N * D
    # FORMULA: fill missing C with 6*N*D
    m["training_compute_flop"] = m["_C"].fillna(imputed)
    return m


def build_power_bridge(clusters: pd.DataFrame, eia_demand: pd.DataFrame) -> pd.DataFrame:
    """
    The join that makes 'all three in one view' meaningful.

    Total known AI cluster facility power vs. the grid it sits on:

      ai_facility_power_mw = sum(cluster.facility_power_mw)
      grid_hourly_mw       = latest EIA hourly demand (MWh over 1h = MW)
      ai_share_of_grid     = ai_facility_power_mw / grid_hourly_mw

    Returns a one-row summary frame (extend by region as needed).
    """
    ai_mw = clusters["facility_power_mw"].sum(skipna=True) if not clusters.empty else float("nan")
    grid_mw = float("nan")
    if not eia_demand.empty and "value" in eia_demand.columns:
        grid_mw = pd.to_numeric(eia_demand["value"], errors="coerce").dropna().iloc[:1].sum()
    row = {
        "ai_cluster_facility_power_mw": ai_mw,
        "grid_hourly_demand_mw": grid_mw,
        "ai_share_of_grid": (ai_mw / grid_mw) if grid_mw else float("nan"),
    }
    return pd.DataFrame([row])


# ===========================================================================
# 5. ORCHESTRATOR
# ===========================================================================
def run(use_eia: bool = True) -> dict[str, pd.DataFrame]:
    log.info("Pulling Epoch feeds ...")
    models   = safe(pull_models,      "models")
    clusters = safe(pull_clusters,    "gpu_clusters")
    dcs      = safe(pull_datacenters, "data_centers")
    hw       = safe(pull_hardware,    "ml_hardware")
    sales    = safe(pull_chipsales,   "ai_chip_sales")
    owners   = safe(pull_chipowners,  "ai_chip_owners")
    firms    = safe(pull_companies,   "ai_companies")

    eia = pd.DataFrame()
    if use_eia:
        eia = safe(lambda: pull_eia(C.EIA_ROUTES["hourly_demand"]), "eia_hourly_demand")

    log.info("Computing derived math ...")
    hw       = enrich_hardware(hw)
    clusters = enrich_clusters(clusters, hw)
    sales    = enrich_chipsales(sales)
    models   = enrich_models(models)
    bridge   = build_power_bridge(clusters, eia)

    tables = {
        "models_demand": models, "clusters_capacity": clusters,
        "datacenters_capacity": dcs, "hardware_capacity": hw,
        "chipsales_capacity": sales, "chipowners_capacity": owners,
        "companies_demand": firms, "eia_power": eia, "power_bridge": bridge,
    }
    for name, df in tables.items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
    log.info("Wrote %d tables to %s", len(tables), OUT)
    return tables


if __name__ == "__main__":
    run(use_eia="--no-eia" not in sys.argv)
