"""
dashboard.py — turn the tidy tables from ingest.py into (1) Excel-friendly CSV
copies and (2) a single self-contained HTML dashboard with three lenses:
compute CAPACITY, compute DEMAND, and POWER.

Design rules (inherited from ingest.py):
  * Rule Zero — no fabricated numbers. If a feed/series is empty, the tile or
    chart renders a greyed "no data" state; it is never back-filled with guesses.
  * Every headline number is a FORMULA over real source columns, documented
    inline. No magic constants baked into the output.

The HTML embeds its data inline as a JS object (no external data fetch) and
draws charts with Chart.js from a CDN — so the committed docs/index.html is a
faithful, standalone snapshot that GitHub Pages can serve directly.
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import pandas as pd

# docs/ is the GitHub Pages source folder (Settings -> Pages -> /docs).
DOCS = Path(__file__).parent / "docs"


# ===========================================================================
# CSV EXPORT
# ===========================================================================
def write_csvs(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Write one CSV per table next to the parquet files, for opening in Excel."""
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)


# ===========================================================================
# SMALL HELPERS  (numeric hygiene so we never emit NaN into JSON)
# ===========================================================================
def _num(series: pd.Series) -> pd.Series:
    """Coerce to numeric; non-parseable -> NaN."""
    return pd.to_numeric(series, errors="coerce")


def _clean(x):
    """NaN/inf -> None so json.dumps(allow_nan=False) stays valid."""
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _decimal_year(value) -> float | None:
    """
    Convert a date to a decimal year (e.g. 2025-07-02 -> 2025.50) so charts can
    use a plain numeric x-axis with no extra date-adapter dependency.

        decimal_year = year + (t - Jan1) / (NextJan1 - Jan1)   # leap-year exact
    """
    dt = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(dt):
        return None
    y = dt.year
    start = pd.Timestamp(year=y, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=y + 1, month=1, day=1, tz="UTC")
    return round(y + (dt - start) / (end - start), 4)


def _human(x) -> str:
    """Compact human number: 1.2k / 3.4M / 5.6B / 7.8T, else scientific."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    a = abs(x)
    if a >= 1e15:
        return f"{x:.2e}"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if a >= div:
            return f"{x / div:.1f}{suf}"
    return f"{x:,.0f}"


def _col(df: pd.DataFrame, *needles: str):
    """First column whose lowercased name contains all needles (fuzzy match)."""
    for c in df.columns:
        low = c.lower()
        if all(n in low for n in needles):
            return c
    return None


def _xy(xs, ys) -> list[dict]:
    """Zip into Chart.js scatter points, dropping any pair with a missing value."""
    pts = []
    for x, y in zip(xs, ys):
        x, y = _clean(x), _clean(y)
        if x is not None and y is not None:
            pts.append({"x": x, "y": y})
    return pts


# ===========================================================================
# SERIES + KPI BUILDERS  (all math explicit)
# ===========================================================================
def _scoreboard(t: dict[str, pd.DataFrame]) -> list[dict]:
    """Six headline KPIs, two per lens. Each value is a formula over real cols."""
    cl = t["clusters_capacity"]
    m = t["models_demand"]
    co = t["companies_demand"]

    tiles: list[dict] = []

    # --- CAPACITY ---------------------------------------------------------
    # Total known AI cluster nameplate = sum of measured-preferred MW.
    if not cl.empty and "power_capacity_mw" in cl:
        mw = _num(cl["power_capacity_mw"])
        tiles.append(_tile("Known cluster power capacity", _human(mw.sum()) + " MW",
                           f"Σ over {int(mw.notna().sum())} clusters", "capacity"))
        tiles.append(_tile("Tracked GPU clusters", f"{int(mw.notna().sum()):,}",
                           "gpu_clusters feed", "capacity"))
    else:
        tiles += [_tile_nodata("Known cluster power capacity", "capacity"),
                  _tile_nodata("Tracked GPU clusters", "capacity")]

    # --- DEMAND -----------------------------------------------------------
    # Frontier training compute = max training_compute_flop (real + 6ND imputed).
    if not m.empty and "training_compute_flop" in m:
        c = _num(m["training_compute_flop"])
        big = int((c >= 1e25).sum())  # count of >= 1e25 FLOP runs
        tiles.append(_tile("Frontier training run", f"{c.max():.2e} FLOP",
                           "max Training compute (FLOP)", "demand"))
        tiles.append(_tile("Models ≥ 1e25 FLOP", f"{big:,}",
                           "count of frontier-scale runs", "demand"))
    else:
        tiles += [_tile_nodata("Frontier training run", "demand"),
                  _tile_nodata("Models ≥ 1e25 FLOP", "demand")]

    # --- POWER ------------------------------------------------------------
    # Total AI annualized revenue = Σ latest annualized revenue per company.
    if not co.empty:
        rev_c = _col(co, "annualized", "revenue") or _col(co, "revenue", "annu")
        date_c = _col(co, "date")
        name_c = _col(co, "company")
        if rev_c and date_c and name_c:
            cc = co[[name_c, date_c, rev_c]].copy()
            cc[rev_c] = _num(cc[rev_c])
            cc = cc.dropna(subset=[rev_c]).sort_values(date_c)
            latest = cc.groupby(name_c).tail(1)  # newest datapoint per company
            tiles.append(_tile("AI annualized revenue", "$" + _human(latest[rev_c].sum()),
                               f"Σ latest of {latest[name_c].nunique()} firms", "power"))
        else:
            tiles.append(_tile_nodata("AI annualized revenue", "power"))
    else:
        tiles.append(_tile_nodata("AI annualized revenue", "power"))

    # Grid demand snapshot from the power bridge (actual demand, summed BAs).
    br = t["power_bridge"]
    if not br.empty and _clean(br.iloc[0].get("grid_actual_demand_mw")) is not None:
        r = br.iloc[0]
        tiles.append(_tile("US actual grid demand", _human(r["grid_actual_demand_mw"]) + " MW",
                           f"EIA {r.get('grid_region') or '?'} actual demand", "power"))
    else:
        tiles.append(_tile_nodata("US actual grid demand", "power"))

    return tiles


def _tile(label, value, sub, lens):
    return {"label": label, "value": value, "sub": sub, "lens": lens, "nodata": False}


def _tile_nodata(label, lens):
    return {"label": label, "value": "no data", "sub": "feed unreachable", "lens": lens, "nodata": True}


def _charts(t: dict[str, pd.DataFrame]) -> dict:
    """Build the six trend series. Empty inputs yield [] -> 'no data' in the UI."""
    charts: dict = {}

    # (a) CAPACITY — cumulative cluster power capacity (MW) by first-operational date.
    #     cum_mw[t] = Σ_{first_op <= t} power_capacity_mw
    cl = t["clusters_capacity"]
    a = []
    if not cl.empty and "power_capacity_mw" in cl:
        dcol = _col(cl, "first", "operational", "date") or _col(cl, "operational", "date")
        if dcol:
            g = cl[[dcol, "power_capacity_mw"]].copy()
            g["_yr"] = g[dcol].map(_decimal_year)
            g["_mw"] = _num(g["power_capacity_mw"])
            g = g.dropna(subset=["_yr", "_mw"]).sort_values("_yr")
            g["_cum"] = g["_mw"].cumsum()
            a = _xy(g["_yr"], g["_cum"])
    charts["cap_cum_mw"] = a

    # (b) CAPACITY — cumulative AI-chip installed base (units) over time.
    #     units aggregated per end-date, then running total (a flow -> a stock).
    cs = t["chipsales_capacity"]
    b = []
    if not cs.empty:
        dcol = _col(cs, "end", "date") or _col(cs, "date")
        ucol = _col(cs, "number", "units", "median") or _col(cs, "units", "median") or _col(cs, "unit")
        if dcol and ucol:
            g = cs[[dcol, ucol]].copy()
            g["_yr"] = g[dcol].map(_decimal_year)
            g["_u"] = _num(g[ucol])
            g = g.dropna(subset=["_yr", "_u"])
            g = g.groupby("_yr", as_index=False)["_u"].sum().sort_values("_yr")
            g["_cum"] = g["_u"].cumsum()
            b = _xy(g["_yr"], g["_cum"])
    charts["cap_cum_units"] = b

    # (c) DEMAND — training compute vs publication date (log-y frontier scatter).
    m = t["models_demand"]
    c = []
    if not m.empty and "training_compute_flop" in m:
        dcol = _col(m, "publication", "date") or _col(m, "publication")
        if dcol:
            g = m[[dcol, "training_compute_flop"]].copy()
            g["_yr"] = g[dcol].map(_decimal_year)
            g["_c"] = _num(g["training_compute_flop"])
            g = g.dropna(subset=["_yr", "_c"])
            g = g[g["_c"] > 0]  # log scale needs strictly positive
            c = _xy(g["_yr"], g["_c"])
    charts["dem_compute"] = c

    # (d) DEMAND — AI-company annualized revenue over time (log-y scatter).
    co = t["companies_demand"]
    d = []
    if not co.empty:
        rcol = _col(co, "annualized", "revenue") or _col(co, "revenue", "annu")
        dcol = _col(co, "date")
        if rcol and dcol:
            g = co[[dcol, rcol]].copy()
            g["_yr"] = g[dcol].map(_decimal_year)
            g["_r"] = _num(g[rcol])
            g = g.dropna(subset=["_yr", "_r"])
            g = g[g["_r"] > 0]
            d = _xy(g["_yr"], g["_r"])
    charts["dem_revenue"] = d

    # (e) POWER — recent US actual demand (MW): the lower-48 total 'US48'.
    #     We use the authoritative US48 aggregate (NOT a sum across respondents,
    #     which would double-count nested regions/BAs), type==D (actual).
    eia = t["eia_power"]
    e_labels, e_vals = [], []
    if not eia.empty and {"type", "value", "period", "respondent"}.issubset(eia.columns):
        d0 = eia[(eia["type"] == "D") & (eia["respondent"] == "US48")].copy()
        d0["value"] = _num(d0["value"])
        d0 = d0.dropna(subset=["value"]).sort_values("period")
        if not d0.empty:
            d0 = d0.drop_duplicates("period").tail(72)  # last ~3 days, readable trend
            e_labels = d0["period"].astype(str).tolist()
            e_vals = [_clean(v) for v in d0["value"].tolist()]
    charts["pow_grid"] = {"labels": e_labels, "values": e_vals}

    # (f) POWER — hardware energy efficiency (FLOP/s per watt) by release date.
    hw = t["hardware_capacity"]
    f = []
    if not hw.empty and "flops_per_watt" in hw:
        dcol = _col(hw, "release", "date") or _col(hw, "release")
        if dcol:
            g = hw[[dcol, "flops_per_watt"]].copy()
            g["_yr"] = g[dcol].map(_decimal_year)
            g["_e"] = _num(g["flops_per_watt"])
            g = g.dropna(subset=["_yr", "_e"])
            g = g[g["_e"] > 0]
            f = _xy(g["_yr"], g["_e"])
    charts["pow_efficiency"] = f

    return charts


def _bridge(t: dict[str, pd.DataFrame]) -> dict:
    """Flatten the one-row power bridge for the callout band."""
    br = t["power_bridge"]
    if br.empty:
        return {}
    r = br.iloc[0].to_dict()
    return {k: _clean(v) if isinstance(v, float) else v for k, v in r.items()}


def _provenance(t: dict[str, pd.DataFrame]) -> list[dict]:
    """Row/col counts per table so the page shows exactly what loaded (Rule Zero)."""
    return [{"table": n, "rows": int(len(df)), "cols": int(df.shape[1]), "empty": bool(df.empty)}
            for n, df in t.items()]


# ===========================================================================
# HTML RENDER
# ===========================================================================
def write_dashboard(tables: dict[str, pd.DataFrame]) -> Path:
    DOCS.mkdir(exist_ok=True)
    payload = {
        "scoreboard": _scoreboard(tables),
        "charts": _charts(tables),
        "bridge": _bridge(tables),
        "provenance": _provenance(tables),
        # built_at is passed by ingest via the tables' freshness; we read the
        # newest EIA period if present, else leave blank (no fabricated clock).
        "as_of": _as_of(tables),
    }
    blob = json.dumps(payload, allow_nan=False)
    html = _HTML_TEMPLATE.replace("/*__DATA__*/", blob)
    out = DOCS / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def _as_of(t: dict[str, pd.DataFrame]) -> str:
    """Newest real timestamp we can cite (EIA latest period), else empty."""
    br = t["power_bridge"]
    if not br.empty:
        p = br.iloc[0].get("grid_latest_period")
        if isinstance(p, str) and p:
            return f"grid data through {p} (UTC hour)"
    return ""


_HTML_TEMPLATE = r"""<title>AI Compute & Power Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#f6f7f9; --card:#fff; --ink:#12141a; --muted:#5b6270; --line:#e4e7ec;
    --capacity:#2563eb; --demand:#d97706; --power:#059669; --dead:#9aa3af;
  }
  @media (prefers-color-scheme: dark){
    :root{ --bg:#0d1117; --card:#161b22; --ink:#e6edf3; --muted:#9198a1; --line:#2a313c;
           --capacity:#60a5fa; --demand:#fbbf24; --power:#34d399; --dead:#6b7280; }
  }
  :root[data-theme="light"]{ --bg:#f6f7f9; --card:#fff; --ink:#12141a; --muted:#5b6270; --line:#e4e7ec;
           --capacity:#2563eb; --demand:#d97706; --power:#059669; --dead:#9aa3af; }
  :root[data-theme="dark"]{ --bg:#0d1117; --card:#161b22; --ink:#e6edf3; --muted:#9198a1; --line:#2a313c;
           --capacity:#60a5fa; --demand:#fbbf24; --power:#34d399; --dead:#6b7280; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
  header h1{margin:0 0 4px;font-size:26px;letter-spacing:-.01em}
  header p{margin:0;color:var(--muted);font-size:13px}
  h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
     margin:34px 0 12px;display:flex;align-items:center;gap:8px}
  h2 .dot{width:10px;height:10px;border-radius:50%}
  .grid{display:grid;gap:14px}
  .kpis{grid-template-columns:repeat(auto-fit,minmax(180px,1fr))}
  .charts{grid-template-columns:repeat(auto-fit,minmax(340px,1fr))}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
  .kpi{border-left:4px solid var(--dead)}
  .kpi.capacity{border-left-color:var(--capacity)}
  .kpi.demand{border-left-color:var(--demand)}
  .kpi.power{border-left-color:var(--power)}
  .kpi .label{font-size:12px;color:var(--muted)}
  .kpi .value{font-size:24px;font-weight:650;margin:6px 0 2px;letter-spacing:-.02em}
  .kpi .sub{font-size:11.5px;color:var(--muted)}
  .kpi.nodata .value{color:var(--dead);font-style:italic;font-weight:500;font-size:18px}
  .chart-card h3{margin:0 0 2px;font-size:14px}
  .chart-card .cap{margin:0 0 10px;font-size:12px;color:var(--muted)}
  .canvas-box{position:relative;height:260px}
  .nodata{display:flex;align-items:center;justify-content:center;height:260px;
          color:var(--dead);font-style:italic;border:1px dashed var(--line);border-radius:8px}
  .bridge{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-top:8px}
  .bridge .big{font-size:22px;font-weight:650}
  .bridge .caveat{margin-top:8px;font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:8px}
  table.prov{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:8px}
  table.prov th,table.prov td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}
  table.prov td.empty{color:#d14343;font-weight:600}
  footer{margin-top:40px;font-size:12px;color:var(--muted)}
  a{color:var(--capacity)}
  .themetoggle{float:right;font-size:12px;color:var(--muted);cursor:pointer;
               border:1px solid var(--line);border-radius:6px;padding:4px 8px;background:none}
</style>

<div class="wrap">
  <header>
    <button class="themetoggle" onclick="toggleTheme()">◐ theme</button>
    <h1>AI Compute &amp; Power Dashboard</h1>
    <p id="asof">Three lenses — compute capacity, compute demand, and power — over public data from Epoch AI (CC-BY) and the U.S. EIA.</p>
  </header>

  <section>
    <div class="grid kpis" id="scoreboard"></div>
  </section>

  <h2><span class="dot" style="background:var(--capacity)"></span>① Compute capacity</h2>
  <div class="grid charts">
    <div class="card chart-card"><h3>Cumulative cluster power capacity</h3>
      <p class="cap">Σ measured-preferred MW by first-operational date. gpu_clusters.</p>
      <div id="w_cap_cum_mw" class="canvas-box"><canvas id="c_cap_cum_mw"></canvas></div></div>
    <div class="card chart-card"><h3>Cumulative AI-chip installed base</h3>
      <p class="cap">Running total of units shipped (median). ai_chip_sales.</p>
      <div id="w_cap_cum_units" class="canvas-box"><canvas id="c_cap_cum_units"></canvas></div></div>
  </div>

  <h2><span class="dot" style="background:var(--demand)"></span>② Compute demand</h2>
  <div class="grid charts">
    <div class="card chart-card"><h3>Training compute frontier</h3>
      <p class="cap">Training compute (FLOP, log) vs publication date. ai_models.</p>
      <div id="w_dem_compute" class="canvas-box"><canvas id="c_dem_compute"></canvas></div></div>
    <div class="card chart-card"><h3>AI company annualized revenue</h3>
      <p class="cap">Annualized revenue (USD, log) over time. ai_companies.</p>
      <div id="w_dem_revenue" class="canvas-box"><canvas id="c_dem_revenue"></canvas></div></div>
  </div>

  <h2><span class="dot" style="background:var(--power)"></span>③ Power</h2>
  <div class="grid charts">
    <div class="card chart-card"><h3>US actual grid demand (recent)</h3>
      <p class="cap">Actual demand (type D) summed across balancing authorities, per hour. EIA.</p>
      <div id="w_pow_grid" class="canvas-box"><canvas id="c_pow_grid"></canvas></div></div>
    <div class="card chart-card"><h3>Hardware energy efficiency</h3>
      <p class="cap">FLOP/s per watt vs release date (log). ml_hardware — the decoupling lever.</p>
      <div id="w_pow_efficiency" class="canvas-box"><canvas id="c_pow_efficiency"></canvas></div></div>
  </div>

  <h2>Power bridge — AI vs. grid</h2>
  <div class="bridge" id="bridge"></div>

  <h2>Data provenance (what actually loaded)</h2>
  <div class="card"><table class="prov" id="prov"></table></div>

  <footer>
    Sources: <a href="https://epoch.ai/data">Epoch AI</a> (CC-BY) and
    <a href="https://www.eia.gov/opendata/">U.S. EIA Open Data</a>. Rebuilt daily by GitHub Actions.
    No values are fabricated — empty feeds render as "no data".
  </footer>
</div>

<script>
const DATA = /*__DATA__*/;

/* ---- theme toggle (stamps data-theme so it wins over prefers-color-scheme) */
function toggleTheme(){
  const r=document.documentElement;
  const dark=matchMedia('(prefers-color-scheme: dark)').matches;
  const cur=r.getAttribute('data-theme')||(dark?'dark':'light');
  r.setAttribute('data-theme', cur==='dark'?'light':'dark');
  location.reload(); /* simplest: redraw charts with new axis colors */
}
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const INK=css('--ink'), MUTED=css('--muted'), LINE=css('--line');

/* ---- scoreboard ---- */
const sb=document.getElementById('scoreboard');
DATA.scoreboard.forEach(t=>{
  const d=document.createElement('div');
  d.className='card kpi '+t.lens+(t.nodata?' nodata':'');
  d.innerHTML=`<div class="label">${t.label}</div><div class="value">${t.value}</div><div class="sub">${t.sub}</div>`;
  sb.appendChild(d);
});
if(DATA.as_of){document.getElementById('asof').textContent+=' · '+DATA.as_of;}

/* ---- chart helpers ---- */
Chart.defaults.color=MUTED; Chart.defaults.borderColor=LINE; Chart.defaults.font.family='inherit';
const yearAxis={type:'linear',title:{display:true,text:'year'},
  ticks:{callback:v=>Number.isInteger(v)?v:''}, grid:{color:LINE}};
function logY(text){return {type:'logarithmic',title:{display:true,text},grid:{color:LINE}};}
function linY(text){return {type:'linear',title:{display:true,text},grid:{color:LINE},beginAtZero:true};}

function empty(id){document.getElementById('w_'+id).innerHTML=
  '<div class="nodata">no data — feed unreachable or empty</div>';}

function scatter(id,pts,color,yaxis,label){
  if(!pts||!pts.length){return empty(id);}
  new Chart(document.getElementById('c_'+id),{type:'scatter',
    data:{datasets:[{label,data:pts,backgroundColor:color,pointRadius:3,pointHoverRadius:5}]},
    options:{maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:yearAxis,y:yaxis}}});
}
function lineXY(id,pts,color,yaxis){
  if(!pts||!pts.length){return empty(id);}
  new Chart(document.getElementById('c_'+id),{type:'line',
    data:{datasets:[{data:pts,borderColor:color,backgroundColor:color,
      pointRadius:0,borderWidth:2,tension:.15,fill:false}]},
    options:{maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:yearAxis,y:yaxis}}});
}
function lineCat(id,labels,vals,color,yaxis){
  if(!vals||!vals.length){return empty(id);}
  new Chart(document.getElementById('c_'+id),{type:'line',
    data:{labels,datasets:[{data:vals,borderColor:color,backgroundColor:color,
      pointRadius:0,borderWidth:2,tension:.15,fill:false}]},
    options:{maintainAspectRatio:false,plugins:{legend:{display:false}},
      scales:{x:{ticks:{maxTicksLimit:8,maxRotation:0},grid:{display:false}},y:yaxis}}});
}

const C=DATA.charts, CAP=css('--capacity'), DEM=css('--demand'), POW=css('--power');
lineXY('cap_cum_mw',      C.cap_cum_mw,     CAP, linY('MW (cumulative)'));
lineXY('cap_cum_units',   C.cap_cum_units,  CAP, linY('units (cumulative)'));
scatter('dem_compute',    C.dem_compute,    DEM, logY('training FLOP'),'model');
scatter('dem_revenue',    C.dem_revenue,    DEM, logY('USD / yr'),'company');
lineCat('pow_grid',       C.pow_grid.labels, C.pow_grid.values, POW, linY('MW'));
scatter('pow_efficiency', C.pow_efficiency, POW, logY('FLOP/s per W'),'chip');

/* ---- power bridge ---- */
const b=DATA.bridge, bx=document.getElementById('bridge');
if(b && b.grid_actual_demand_mw!=null && b.ai_cluster_power_mw!=null){
  const share=(b.ai_share_of_grid!=null)?(b.ai_share_of_grid*100).toFixed(1)+'%':'—';
  const f=n=>n==null?'—':Math.round(n).toLocaleString();
  bx.innerHTML=`<div class="big">${f(b.ai_cluster_power_mw)} MW known AI cluster capacity `
    +`≈ ${share} of ${f(b.grid_actual_demand_mw)} MW US actual demand</div>`
    +`<div class="caveat"><b>Scope caveat:</b> numerator is <i>global</i> AI cluster nameplate `
    +`(sum of measured-preferred MW); denominator is a snapshot of U.S. actual demand `
    +`(EIA ${b.grid_region||'US48'} lower-48 total`
    +(b.grid_latest_period?`, ${b.grid_latest_period} UTC`:'')
    +`). This is an order-of-magnitude sanity check, not a national load-share figure.</div>`;
}else{
  bx.innerHTML='<div class="nodata" style="height:auto;padding:20px">no data — power bridge unavailable</div>';
}

/* ---- provenance ---- */
const pv=document.getElementById('prov');
pv.innerHTML='<tr><th>table</th><th>rows</th><th>cols</th></tr>'+
  DATA.provenance.map(r=>`<tr><td>${r.table}</td>`+
    `<td class="${r.empty?'empty':''}">${r.empty?'0 (no data)':r.rows.toLocaleString()}</td>`+
    `<td>${r.cols}</td></tr>`).join('');
</script>
"""
