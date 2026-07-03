"""
config.py — endpoints and physical assumptions for the compute-capacity dashboard.

Everything in ASSUMPTIONS is an INPUT you can override (the equivalent of a blue
input cell in Excel). Everything computed from these downstream is a FORMULA.
Sources: Epoch AI (CC-BY) and EIA Open Data (public domain, key required).
"""

# ---------------------------------------------------------------------------
# EPOCH AI ENDPOINTS  (verified live 2026-07-01; all CC-BY licensed)
# ZIPs contain one or more CSVs. Models also exposes a daily-updated flat CSV.
# ---------------------------------------------------------------------------
EPOCH = {
    "ai_models_csv":     "https://epoch.ai/data/all_ai_models.csv",          # daily CSV
    "ai_models":         "https://epoch.ai/data/ai_models.zip",
    "gpu_clusters":      "https://epoch.ai/data/gpu_clusters.zip",
    "data_centers":      "https://epoch.ai/data/data_centers/data_centers.zip",
    "ml_hardware":       "https://epoch.ai/data/ml_hardware.zip",
    "ai_chip_sales":     "https://epoch.ai/data/ai_chip_sales.zip",
    "ai_chip_owners":    "https://epoch.ai/data/ai_chip_owners.zip",
    "ai_chip_components":"https://epoch.ai/data/ai_chip_components.zip",
    "ai_companies":      "https://epoch.ai/data/ai_companies.zip",
}

# ---------------------------------------------------------------------------
# EIA OPEN DATA (v2 REST API). Register a free key: https://www.eia.gov/opendata/
# Set env var EIA_API_KEY. Route below = hourly demand by balancing authority.
# ---------------------------------------------------------------------------
EIA_BASE = "https://api.eia.gov/v2"
EIA_ROUTES = {
    # hourly electricity demand (MWh) per balancing authority
    "hourly_demand": "/electricity/rto/region-data/data/",
    # annual generating capacity by state/source (MW) — supply side of the grid
    "operable_capacity": "/electricity/operating-generator-capacity/data/",
}

# ---------------------------------------------------------------------------
# ASSUMPTIONS  (INPUTS — override freely). Sources noted for each.
# ---------------------------------------------------------------------------
ASSUMPTIONS = {
    # Power Usage Effectiveness: total facility power / IT power.
    # Hyperscale AI campuses run ~1.1–1.3; 1.2 is a common planning midpoint.
    "PUE": 1.2,

    # Fraction of nameplate FLOP/s actually realized during sustained training.
    # Empirical MFU (Model FLOPs Utilization) for large runs ~0.3–0.5.
    "utilization": 0.40,

    # Chinchilla / Kaplan compute-accounting constant: C = k * N * D.
    # Standard k = 6 (fwd+bwd passes, dense transformer). In your OFI-adjacent
    # notes this is the same C = 6ND relation.
    "flops_per_param_token": 6,

    # Seconds per year, for converting a sustained FLOP/s rate into annual FLOP.
    "seconds_per_year": 365.25 * 24 * 3600,

    # Fallback per-chip thermal design power (W) if ml_hardware lacks a value.
    "default_tdp_watts": 700,   # ~ H100 SXM class
}
