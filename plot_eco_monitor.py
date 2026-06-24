#!/usr/bin/env python3
"""
plot_eco_monitor.py
North Atlantic ecosystem monitor — 2 rows × 2 cols.

Row 0: Dominant phyto map | Dominant zoo map  (Lambert Conformal, with box)
Row 1: Phyto annual cycles | Zoo annual cycles (weekly climatology timeseries)

Colour palettes:
  Phyto: TOL Bright  (#4477AA #EE6677 #228833 #CCBB44 #66CCEE #AA3377)
  Zoo:   TOL Vibrant (#EE7733 #0077BB #33BBEE #EE3377 #CC3311 #009988)

Usage:
    python plot_eco_monitor.py --model TOM12_TJ_HA00
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
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, "/gpfs/home/mep22dku/scratch/SOZONE/UTILS")
from plot_style import set_presentation_style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLIMS_DIR   = Path("/gpfs/data/greenocean/users/mep22dku/clims")
NEMO5_DIR   = Path("/gpfs/data/greenocean/software/runs/NEMO5")
OUT_DIR_DEFAULT = Path(
    "/gpfs/home/mep22dku/scratch/NATL_MONITOR/plots"
)
SCRIPT_PATH = (
    "/gpfs/home/mep22dku/scratch/NATL_MONITOR/plot_eco_monitor.py"
)

# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
PHYTO_VARS = ['DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
ZOO_VARS   = ['BAC', 'PRO', 'PTE', 'MES', 'GEL', 'CRU']

# TOL Bright (phyto) — cool blues, greens, pinks
# Phyto — TOL Bright with green hardened to dark teal for CVD safety
TOL_BRIGHT  = ['#4477AA', '#EE6677', '#006D77', '#CCBB44', '#66CCEE', '#AA3377']

# Zoo — Okabe-Ito inspired, fully CVD-safe, no overlap with phyto
TOL_VIBRANT = ['#E69F00', '#332288', '#F0E442', '#D55E00', '#00C9A7', '#7B2D8B']

PHYTO_COLORS = {v: TOL_BRIGHT[i]  for i, v in enumerate(PHYTO_VARS)}
ZOO_COLORS   = {v: TOL_VIBRANT[i] for i, v in enumerate(ZOO_VARS)}

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

# Regional box lon/lat bounds
BOX_LON0, BOX_LON1 = -64.34, -16.29
BOX_LAT0, BOX_LAT1 =  47.17,  71.11

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_discrete_cmap(colours):
    """Build a ListedColormap + BoundaryNorm for N discrete colours."""
    cmap = mcolors.ListedColormap(colours)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(colours)), len(colours))
    return cmap, norm


def dominant_map_panel(ax, lon, lat, dom_idx, colours, group_names,
                       title, extent=NA_EXTENT):
    """Plot a dominant-group map with nearest-neighbour (no interpolation artefacts)."""
    cmap, norm = make_discrete_cmap(colours)

    # Mask NaN
    data = np.where(np.isfinite(dom_idx), dom_idx, np.nan)

    im = ax.pcolormesh(
        lon, lat, data,
        transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm,
        shading="nearest",   # nearest-neighbour — avoids colour blending artefacts
        rasterized=True,
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=COAST_LW, color="#555555")
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=2)
    ax.set_title(title, pad=4)

    # Gridlines
    ax.gridlines(draw_labels=False, linewidth=0.3, color="#AAAAAA",
                 linestyle="--", alpha=0.7)

    return im


def draw_box(ax):
    """Draw the regional box on a LambertConformal axes."""
    # Build the box as a polygon in PlateCarree, then transform corner by corner
    # so it follows the projection correctly.
    import cartopy.crs as _ccrs
    lons = [BOX_LON0, BOX_LON1, BOX_LON1, BOX_LON0, BOX_LON0]
    lats = [BOX_LAT0, BOX_LAT0, BOX_LAT1, BOX_LAT1, BOX_LAT0]
    ax.plot(lons, lats,
            transform=_ccrs.PlateCarree(),
            color="#1A1A1A", lw=1.5, ls="--", zorder=5)


def add_legend(ax, colours, names, title=None):
    """Add a compact legend of coloured patches."""
    patches = [mpatches.Patch(facecolor=c, edgecolor="#555555", linewidth=0.5,
                               label=n)
               for c, n in zip(colours, names)]
    ax.legend(handles=patches, title=title,
              loc="lower left", fontsize=9,
              title_fontsize=9.5,
              framealpha=0.85, edgecolor="#CCCCCC",
              ncol=2, columnspacing=0.8, handlelength=1.2)


def load_maps(maps_file: Path):
    ds = xr.open_dataset(maps_file, decode_times=False)
    lon = ds["lon"].values
    lat = ds["lat"].values
    out = {}
    for v in list(PHYTO_VARS) + list(ZOO_VARS) + ["DOMINANT_PHYTO", "DOMINANT_ZOO"]:
        if v in ds:
            out[v] = ds[v].squeeze().values
    phyto_groups = ds.attrs.get("phyto_groups", " ".join(PHYTO_VARS)).split()
    zoo_groups   = ds.attrs.get("zoo_groups",   " ".join(ZOO_VARS)).split()
    ds.close()
    return lon, lat, out, phyto_groups, zoo_groups


def load_timeseries(ts_file: Path):
    ds = xr.open_dataset(ts_file, decode_times=False)
    out = {}
    for v in list(PHYTO_VARS) + list(ZOO_VARS):
        if v in ds:
            out[v] = ds[v].values  # (52,)
    ds.close()
    return out


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def make_figure(
    model: str,
    maps_file: Path,
    ts_file: Path,
    year_start: int,
    year_end: int,
    out_path: Path,
    extent: list = NA_EXTENT,
    font_size: int = 13,
):
    set_presentation_style(base_size=font_size)

    print("Loading maps…")
    lon, lat, maps, phyto_groups, zoo_groups = load_maps(maps_file)

    print("Loading timeseries…")
    ts = load_timeseries(ts_file)

    # ---- Layout ----
    # Row 0: maps (height 1.0), Row 1: timeseries (height 0.65)
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.0, 0.65],
        hspace=0.30,
        wspace=0.10,
    )

    ax_phyto_map = fig.add_subplot(gs[0, 0], projection=MAP_PROJ)
    ax_zoo_map   = fig.add_subplot(gs[0, 1], projection=MAP_PROJ)
    ax_phyto_ts  = fig.add_subplot(gs[1, 0])
    ax_zoo_ts    = fig.add_subplot(gs[1, 1])

    phyto_colours = [PHYTO_COLORS[v] for v in phyto_groups if v in PHYTO_COLORS]
    zoo_colours   = [ZOO_COLORS[v]   for v in zoo_groups   if v in ZOO_COLORS]

    # ---- Dominant phyto map ----
    if "DOMINANT_PHYTO" in maps:
        dominant_map_panel(
            ax_phyto_map, lon, lat, maps["DOMINANT_PHYTO"],
            phyto_colours, phyto_groups,
            f"Dominant phytoplankton  ({year_start}–{year_end})",
            extent=extent,
        )
        draw_box(ax_phyto_map)
        add_legend(ax_phyto_map, phyto_colours, phyto_groups, title="Phyto")
    else:
        ax_phyto_map.set_visible(False)

    # ---- Dominant zoo map ----
    if "DOMINANT_ZOO" in maps:
        dominant_map_panel(
            ax_zoo_map, lon, lat, maps["DOMINANT_ZOO"],
            zoo_colours, zoo_groups,
            f"Dominant zooplankton  ({year_start}–{year_end})",
            extent=extent,
        )
        draw_box(ax_zoo_map)
        add_legend(ax_zoo_map, zoo_colours, zoo_groups, title="Zoo")
    else:
        ax_zoo_map.set_visible(False)

    # ---- Phyto timeseries ----
    weeks = np.arange(52)
    for v in phyto_groups:
        if v in ts:
            ax_phyto_ts.plot(weeks, ts[v], color=PHYTO_COLORS[v],
                             lw=1.8, label=v)
    ax_phyto_ts.set_xlabel("Week")
    ax_phyto_ts.set_ylabel("Depth-integrated biomass (mmol m$^{-2}$)")
    ax_phyto_ts.set_title("Phytoplankton annual cycle  (regional box)")
    ax_phyto_ts.legend(loc="upper right", fontsize=9, ncol=2,
                       framealpha=0.85, edgecolor="#CCCCCC")
    ax_phyto_ts.spines["top"].set_visible(False)
    ax_phyto_ts.spines["right"].set_visible(False)
    # Month tick marks
    _add_month_ticks(ax_phyto_ts)

    # ---- Zoo timeseries ----
    for v in zoo_groups:
        if v in ts:
            ax_zoo_ts.plot(weeks, ts[v], color=ZOO_COLORS[v],
                           lw=1.8, marker='', label=v)
    ax_zoo_ts.set_xlabel("Week")
    ax_zoo_ts.set_ylabel("Depth-integrated biomass (mmol m$^{-2}$)")
    ax_zoo_ts.set_title("Zooplankton annual cycle  (regional box)")
    ax_zoo_ts.legend(loc="upper right", fontsize=9, ncol=2,
                     framealpha=0.85, edgecolor="#CCCCCC")
    ax_zoo_ts.spines["top"].set_visible(False)
    ax_zoo_ts.spines["right"].set_visible(False)
    ax_zoo_ts.set_ylabel("")
    ax_zoo_ts.tick_params(axis="y", left=False, labelleft=False)
    _add_month_ticks(ax_zoo_ts)

    # Shared y-axis range across both timeseries panels
    all_vals = []
    for v in list(phyto_groups) + list(zoo_groups):
        if v in ts and np.any(np.isfinite(ts[v])):
            all_vals.extend(ts[v][np.isfinite(ts[v])].tolist())
    if all_vals:
        ymin = min(0, np.min(all_vals))
        ymax = np.max(all_vals) * 1.08
        ax_phyto_ts.set_ylim(ymin, ymax)
        ax_zoo_ts.set_ylim(ymin, ymax)

    # ---- Supertitle ----
    fig.suptitle(
        f"North Atlantic ecosystem monitor  ·  {model}  ·  {year_start}–{year_end}",
        y=0.995, fontsize=font_size + 2, fontweight="medium",
    )

    # ---- Provenance ----
    try:
        from plot_style import add_provenance
        add_provenance(fig, SCRIPT_PATH, fontsize=9, x=0.5, y=0.005, ha="center")
    except (ImportError, TypeError):
        fig.text(0.5, 0.005, SCRIPT_PATH, ha="center", fontsize=9, color="#AAAAAA")

    # ---- Save ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def _add_month_ticks(ax):
    """Add month-name minor ticks at approximate week positions."""
    # Approximate first week of each month (non-leap year, 7-day steps)
    month_weeks = [0, 4, 9, 13, 17, 22, 26, 30, 35, 39, 43, 48]
    ax.set_xticks(month_weeks, minor=True)
    ax.set_xticklabels(
        [m[:1] for m in MONTH_NAMES],  # single letter: J F M A M J J A S O N D
        minor=True, fontsize=6.5, color="#888888",
    )
    ax.set_xticks(np.arange(0, 52, 13))   # major ticks every quarter
    ax.set_xlim(0, 51)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_eco_files(clims_model: Path, model: str):
    """Return (maps_file, ts_file) — most recent of each by mtime."""
    maps = sorted(clims_model.glob(f"{model}_eco_monitor_maps_*.nc"),
                  key=lambda f: f.stat().st_mtime, reverse=True)
    ts   = sorted(clims_model.glob(f"{model}_eco_box_timeseries_*.nc"),
                  key=lambda f: f.stat().st_mtime, reverse=True)
    return (maps[0] if maps else None), (ts[0] if ts else None)


def parse_years_from_filename(fpath: Path):
    m = re.search(r"_(\d{4})_(\d{4})\.nc$", fpath.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",     required=True)
    parser.add_argument("--maps-file", type=Path, default=None,
                        help="Override eco monitor maps NC.")
    parser.add_argument("--ts-file",   type=Path, default=None,
                        help="Override eco box timeseries NC.")
    parser.add_argument("--extent",    type=float, nargs=4,
                        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                        default=NA_EXTENT)
    parser.add_argument("--out-dir",   type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--font-size", type=int,  default=13)
    args = parser.parse_args()

    clims_model = CLIMS_DIR / args.model

    maps_file, ts_file = find_eco_files(clims_model, args.model)
    if args.maps_file:
        maps_file = args.maps_file
    if args.ts_file:
        ts_file = args.ts_file

    if maps_file is None or not maps_file.exists():
        print(f"ERROR: no eco monitor maps file found in {clims_model}.")
        print("Run prep_eco_monitor.py first.")
        sys.exit(1)
    if ts_file is None or not ts_file.exists():
        print(f"ERROR: no eco box timeseries file found in {clims_model}.")
        print("Run prep_eco_monitor.py first.")
        sys.exit(1)

    print(f"  Maps file:       {maps_file.name}")
    print(f"  Timeseries file: {ts_file.name}")

    year_start, year_end = parse_years_from_filename(maps_file)

    out_path = (
        args.out_dir
        / f"{args.model}_eco_monitor_{year_start}_{year_end}.png"
    )

    make_figure(
        model=args.model,
        maps_file=maps_file,
        ts_file=ts_file,
        year_start=year_start,
        year_end=year_end,
        out_path=out_path,
        extent=args.extent,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
