#!/usr/bin/env python3
"""
plot_bgc_monitor.py
North Atlantic BGC monitor — 2 rows × 4 cols.

    Col 0       Col 1       Col 2       Col 3
    Mod TChl    Mod NO3     Mod PO4     Mod PPINT
    Obs TChl    Obs NO3*    Obs PO4*    —

    * Obs NO3 / PO4 to be added when obs files are available.

Usage:
    python plot_bgc_monitor.py --model TOM12_TJ_HA00
    python plot_bgc_monitor.py --model TOM12_TJ_HA00 \\
        --year-start 1955 --year-end 1960 --month 3
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean.cm as cmo

sys.path.insert(0, "/gpfs/home/mep22dku/scratch/SOZONE/UTILS")
from plot_style import set_presentation_style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NEMO5_DIR  = Path("/gpfs/data/greenocean/software/runs/NEMO5")
CLIMS_DIR  = Path("/gpfs/data/greenocean/users/mep22dku/clims")

OBS_BGC_DIR        = Path("/gpfs/data/greenocean/users/mep22dku/clims/obs_bgc")
OBS_TCHL_FILE      = OBS_BGC_DIR / "bgc_obs_tchl_annual_r360x180.nc"
OBS_NO3_FILE       = OBS_BGC_DIR / "bgc_obs_no3_r360x180.nc"
OBS_PO4_FILE       = OBS_BGC_DIR / "bgc_obs_po4_r360x180.nc"

OUT_DIR_DEFAULT = Path(
    "/gpfs/home/mep22dku/scratch/NATL_MONITOR/plots"
)

SCRIPT_PATH = (
    "/gpfs/home/mep22dku/scratch/NATL_MONITOR/plot_bgc_monitor.py"
)

# ---------------------------------------------------------------------------
# Colour limits — override via CLI
# ---------------------------------------------------------------------------
TCHL_VMIN,  TCHL_VMAX  = 0.0,  2.0    # mg Chl m-3  (log scale recommended)
NO3_VMIN,   NO3_VMAX   = 0.0, 30.0    # mmol m-3
PO4_VMIN,   PO4_VMAX   = 0.0,  2.0    # mmol m-3
PPINT_VMIN, PPINT_VMAX = 0.0, 200.0   # mgC m-2 d-1

# ---------------------------------------------------------------------------
# Map settings
# ---------------------------------------------------------------------------
NA_EXTENT = [-80, 15, 25, 80]

MAP_PROJ = ccrs.LambertConformal(
    central_longitude=-30,
    central_latitude=50,
    standard_parallels=(35, 65),
)

LAND_COLOR = "#CCCCCC"
COAST_LW   = 0.5

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_panel(ax, lon, lat, data, cmap, vmin, vmax, title, extent=NA_EXTENT):
    im = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", rasterized=True,
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=COAST_LW, color="#555555")
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=2)
    ax.set_title(title, pad=4)
    return im


def add_gridlines(ax):
    ax.gridlines(
        draw_labels=False,
        linewidth=0.3, color="#AAAAAA", linestyle="--", alpha=0.7,
    )


def blank_panel(ax):
    """Hide a panel that has no data."""
    ax.set_visible(False)


def col_colorbar(fig, im, ax_list, label):
    cbar = fig.colorbar(im, ax=ax_list, orientation="vertical",
                        shrink=0.85, pad=0.01, aspect=25)
    cbar.set_label(label, labelpad=4)
    return cbar


def load_obs_chl():
    """Load annual-mean TChl from pre-processed OC-CCI file."""
    ds  = xr.open_dataset(OBS_TCHL_FILE, decode_times=False)
    chl = ds["TChl"].squeeze()
    lon = ds["lon"].values
    lat = ds["lat"].values
    ds.close()
    return lon, lat, chl.values


def load_obs_nutrients():
    """Load pre-processed WOA18 NO3 and PO4 from standalone obs dir."""
    lon = lat = no3 = po4 = None
    for varname, fpath in [("NO3", OBS_NO3_FILE), ("PO4", OBS_PO4_FILE)]:
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found — run prep_bgc_obs.py first.")
            continue
        ds = xr.open_dataset(fpath, decode_times=False)
        if lon is None:
            lon = ds["lon"].values
            lat = ds["lat"].values
        if varname == "NO3":
            no3 = ds["NO3"].squeeze().values
        else:
            po4 = ds["PO4"].squeeze().values
        ds.close()
    return lon, lat, no3, po4


def load_model_bgc(maps_file: Path):
    ds = xr.open_dataset(maps_file, decode_times=False)
    out = {}
    lon = ds["lon"].values
    lat = ds["lat"].values
    for v in ["TChl", "NO3", "PO4", "PPINT"]:
        if v in ds:
            out[v] = ds[v].squeeze().values
        else:
            print(f"  WARNING: {v} not found in {maps_file.name}")
            out[v] = None
    ds.close()
    return lon, lat, out


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(
    model: str,
    maps_file: Path,
    year_start: int,
    year_end: int,
    out_path: Path,
    extent: list = NA_EXTENT,
    font_size: int = 13,
):
    set_presentation_style(base_size=font_size)

    print("Loading model BGC maps…")
    mod_lon, mod_lat, mod = load_model_bgc(maps_file)

    print("Loading obs TChl (OCCCI)…")
    try:
        obs_chl_lon, obs_chl_lat, obs_chl = load_obs_chl()
        has_obs_chl = True
    except Exception as e:
        print(f"  WARNING: could not load obs TChl: {e}")
        has_obs_chl = False

    print("Loading obs nutrients (WOA18)…")
    obs_nut_lon, obs_nut_lat, obs_no3, obs_po4 = load_obs_nutrients()
    if obs_no3 is None and obs_po4 is None:
        print("  No obs nutrient file found — run prep_bgc_monitor.py first.")

    # ---- Derive colour limits from model data ----
    def _lims(arr):
        """Robust vmin/vmax: 2nd–98th percentile of valid data."""
        if arr is None:
            return 0, 1
        v = arr[np.isfinite(arr)]
        if len(v) == 0:
            return 0, 1
        return float(np.percentile(v, 2)), float(np.percentile(v, 98))

    tchl_vmin,  tchl_vmax  = _lims(mod["TChl"])
    no3_vmin,   no3_vmax   = _lims(mod["NO3"])
    po4_vmin,   po4_vmax   = _lims(mod["PO4"])
    ppint_vmin, ppint_vmax = _lims(mod["PPINT"])

    # ---- Layout: 3 rows × 4 cols
    # Row 0: model maps, Row 1: obs maps, Row 2: horizontal colorbars (thin)
    fig = plt.figure(figsize=(16, 8))

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        height_ratios=[1, 1, 0.06],
        hspace=0.18,
        wspace=0.06,
    )

    def map_ax(row, col):
        return fig.add_subplot(gs[row, col], projection=MAP_PROJ)

    # Map axes
    ax_tchl_mod  = map_ax(0, 0)
    ax_no3_mod   = map_ax(0, 1)
    ax_po4_mod   = map_ax(0, 2)
    ax_ppint_mod = map_ax(0, 3)

    ax_tchl_obs  = map_ax(1, 0)
    ax_no3_obs   = map_ax(1, 1)
    ax_po4_obs   = map_ax(1, 2)
    ax_ppint_obs = map_ax(1, 3)

    # Colorbar axes in row 2
    cax_tchl  = fig.add_subplot(gs[2, 0])
    cax_no3   = fig.add_subplot(gs[2, 1])
    cax_po4   = fig.add_subplot(gs[2, 2])
    cax_ppint = fig.add_subplot(gs[2, 3])

    # ---- TChl ----
    if mod["TChl"] is not None:
        im_tchl = map_panel(ax_tchl_mod, mod_lon, mod_lat, mod["TChl"],
                            cmo.algae, tchl_vmin, tchl_vmax,
                            "Model TChl", extent=extent)
        add_gridlines(ax_tchl_mod)
    else:
        blank_panel(ax_tchl_mod)
        im_tchl = None

    if has_obs_chl:
        im_tchl_obs = map_panel(ax_tchl_obs, obs_chl_lon, obs_chl_lat, obs_chl,
                                cmo.algae, tchl_vmin, tchl_vmax,
                                "Obs TChl", extent=extent)
        add_gridlines(ax_tchl_obs)
        ax_tchl_obs.text(0.99, 0.02, "OC-CCI v5  (1998–2020)",
                         transform=ax_tchl_obs.transAxes,
                         ha="right", va="bottom", fontsize=7, color="#555555",
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5))
    else:
        blank_panel(ax_tchl_obs)

    ref_tchl = im_tchl_obs if has_obs_chl else im_tchl
    if ref_tchl is not None:
        cb = fig.colorbar(ref_tchl, cax=cax_tchl, orientation="horizontal")
        cb.set_label("TChl (mg m-3)", labelpad=2)

    # ---- NO3 ----
    if mod["NO3"] is not None:
        im_no3 = map_panel(ax_no3_mod, mod_lon, mod_lat, mod["NO3"],
                           cmo.matter, no3_vmin, no3_vmax,
                           "Model NO3", extent=extent)
        add_gridlines(ax_no3_mod)
        cb = fig.colorbar(im_no3, cax=cax_no3, orientation="horizontal")
        cb.set_label("NO3 (mmol m-3)", labelpad=2)
    else:
        blank_panel(ax_no3_mod)

    if obs_no3 is not None:
        im_no3_obs = map_panel(ax_no3_obs, obs_nut_lon, obs_nut_lat, obs_no3,
                               cmo.matter, no3_vmin, no3_vmax,
                               "Obs NO3  (WOA18)", extent=extent)
        add_gridlines(ax_no3_obs)
    else:
        blank_panel(ax_no3_obs)

    # ---- PO4 ----
    if mod["PO4"] is not None:
        im_po4 = map_panel(ax_po4_mod, mod_lon, mod_lat, mod["PO4"],
                           cmo.matter, po4_vmin, po4_vmax,
                           "Model PO4", extent=extent)
        add_gridlines(ax_po4_mod)
        cb = fig.colorbar(im_po4, cax=cax_po4, orientation="horizontal")
        cb.set_label("PO4 (mmol m-3)", labelpad=2)
    else:
        blank_panel(ax_po4_mod)

    if obs_po4 is not None:
        im_po4_obs = map_panel(ax_po4_obs, obs_nut_lon, obs_nut_lat, obs_po4,
                               cmo.matter, po4_vmin, po4_vmax,
                               "Obs PO4  (WOA18)", extent=extent)
        add_gridlines(ax_po4_obs)
    else:
        blank_panel(ax_po4_obs)

    # ---- PPINT ----
    if mod["PPINT"] is not None:
        im_ppint = map_panel(ax_ppint_mod, mod_lon, mod_lat, mod["PPINT"],
                             cmo.tempo, ppint_vmin, ppint_vmax,
                             "Model PP int", extent=extent)
        add_gridlines(ax_ppint_mod)
        cb = fig.colorbar(im_ppint, cax=cax_ppint, orientation="horizontal")
        cb.set_label("PP int (mgC m-2 d-1)", labelpad=2)
    else:
        blank_panel(ax_ppint_mod)

    blank_panel(ax_ppint_obs)
    cax_ppint.set_visible(False) if mod["PPINT"] is None else None

    # ---- Supertitle ----
    fig.suptitle(
        f"North Atlantic BGC monitor  ·  {model}  ·  {year_start}–{year_end}, annual mean",
        y=0.995, fontsize=font_size + 2, fontweight="medium",
    )

    # ---- Provenance ----
    try:
        from plot_style import add_provenance
        add_provenance(fig, SCRIPT_PATH, fontsize=9, x=0.5, y=0.01, ha="center")
    except (ImportError, TypeError):
        fig.text(0.5, 0.01, SCRIPT_PATH, ha="center", fontsize=9, color="#AAAAAA")

    # ---- Save ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_last_n_years(model: str, n: int) -> tuple:
    import glob as _glob
    pattern = str(NEMO5_DIR / model / "ORCA2_7d_*_ptrc_T.nc")
    files = _glob.glob(pattern)
    if not files:
        pattern = str(NEMO5_DIR / model / "ORCA2_7d_*_diad_T.nc")
        files = _glob.glob(pattern)
    years = sorted(set(
        int(m.group(1))
        for f in files
        for m in [re.search(r"_7d_(\d{4})0101_", Path(f).name)]
        if m
    ))
    if not years:
        raise RuntimeError(f"No ptrc_T/diad_T files found for {model}.")
    year_end   = years[-1]
    year_start = years[max(0, len(years) - n)]
    return year_start, year_end


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-years", type=int, default=5)
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end",   type=int, default=None)

    parser.add_argument("--maps-file",  type=Path, default=None,
                        help="Override path to BGC maps NC.")

    parser.add_argument("--tchl-vmin",  type=float, default=TCHL_VMIN)
    parser.add_argument("--tchl-vmax",  type=float, default=TCHL_VMAX)
    parser.add_argument("--no3-vmin",   type=float, default=NO3_VMIN)
    parser.add_argument("--no3-vmax",   type=float, default=NO3_VMAX)
    parser.add_argument("--po4-vmin",   type=float, default=PO4_VMIN)
    parser.add_argument("--po4-vmax",   type=float, default=PO4_VMAX)
    parser.add_argument("--ppint-vmin", type=float, default=PPINT_VMIN)
    parser.add_argument("--ppint-vmax", type=float, default=PPINT_VMAX)
    parser.add_argument("--extent",     type=float, nargs=4,
                        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                        default=NA_EXTENT)
    parser.add_argument("--out-dir",    type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--font-size",  type=int, default=13)

    args = parser.parse_args()

    # Year range
    if args.year_start is not None and args.year_end is not None:
        year_start, year_end = args.year_start, args.year_end
    else:
        year_start, year_end = find_last_n_years(args.model, args.n_years)
        print(f"  Using last {args.n_years} years: {year_start}–{year_end}")

    # Maps file
    clims_model = CLIMS_DIR / args.model
    if args.maps_file:
        maps_file = args.maps_file
        if not maps_file.exists():
            print(f"ERROR: --maps-file not found: {maps_file}")
            sys.exit(1)
    else:
        # Use whatever bgc_monitor_maps file exists — most recent by mtime
        candidates = sorted(clims_model.glob(f"{args.model}_bgc_monitor_maps_*_annual.nc"),
                            key=lambda f: f.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"ERROR: no BGC maps file found in {clims_model}")
            print("Run prep_bgc_monitor.py first.")
            sys.exit(1)
        maps_file = candidates[0]
        print(f"  Using maps file: {maps_file.name}")
        # Override year range from filename so labels match the actual data
        import re as _re
        m = _re.search(r"_maps_(\d{4})_(\d{4})_annual", maps_file.name)
        if m:
            year_start, year_end = int(m.group(1)), int(m.group(2))

    out_path = (
        args.out_dir
        / f"{args.model}_bgc_monitor_annual_{year_start}_{year_end}.png"
    )

    make_figure(
        model=args.model,
        maps_file=maps_file,
        year_start=year_start,
        year_end=year_end,
        out_path=out_path,
        extent=args.extent,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
