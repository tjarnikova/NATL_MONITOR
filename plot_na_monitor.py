#!/usr/bin/env python3
"""
plot_na_monitor.py
North Atlantic physical monitor — one figure, five panels.

Layout (3 rows × 2 cols, row 0 spans both columns):

    ┌──────────────────────────────────────────────────────┐  row 0 (thin)
    │            AMOC timeseries                           │
    └──────────────────────────────────────────────────────┘
    ┌───────────────────────┐  ┌───────────────────────┐      row 1
    │  Obs SSS              │  │  Model SSS            │  │ cbar
    └───────────────────────┘  └───────────────────────┘
    ┌───────────────────────┐  ┌───────────────────────┐      row 2
    │  Obs MLD              │  │  Model MLD            │  │ cbar
    └───────────────────────┘  └───────────────────────┘

Usage (after running prep_na_monitor.py):

    python plot_na_monitor.py \\
        --model TOM12_TJ_SM01 \\
        --maps-file /path/to/MODEL_na_monitor_maps_1920_1925_mon03.nc \\
        --amoc-file /path/to/MODEL_na_monitor_amoc_timeseries.nc \\
        --year-start 1920 --year-end 1925 --month 3

Obs files (WOA salinity, MLD climatology) default to the same paths
used in plot_obs_model_comparison.py — override with --obs-sal-file
and --obs-mld-file.
"""

import argparse
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
# Defaults — override via CLI
# ---------------------------------------------------------------------------

OBS_MLD_FILE_DEFAULT = Path(
    "/gpfs/data/greenocean/observations/MLD/mld_dr003_ref10m_v2023.nc"
)
OBS_SAL_DIR_DEFAULT  = Path(
    "/gpfs/data/greenocean/observations/WOA/WOA2023/salinity/5564"
)
OBS_SAL_PATTERN      = "woa23_5564_s{month:02d}_01.nc"

CLIMS_DIR = Path("/gpfs/data/greenocean/users/mep22dku/clims")

OUT_DIR_DEFAULT = Path(
    "/gpfs/home/mep22dku/scratch/NATL_MONITOR/plots"
)

# Script path for provenance
SCRIPT_PATH = "/gpfs/home/mep22dku/scratch/NATL_MONITOR/plot_na_monitor.py"

# ---------------------------------------------------------------------------
# Colour limits — settable here or overridden on the command line
# ---------------------------------------------------------------------------

MLD_VMIN, MLD_VMAX = 0, 800      # m
SAL_VMIN, SAL_VMAX = 32, 36      # psu

# ---------------------------------------------------------------------------
# Map extent [lon_min, lon_max, lat_min, lat_max]
# ---------------------------------------------------------------------------

NA_EXTENT = [-80, 15, 25, 80]

# ---------------------------------------------------------------------------
# Projection — Lambert Conformal avoids the inflated Greenland of PlateCarree
# and gives a sensible North-Atlantic shape.
# ---------------------------------------------------------------------------

MAP_PROJ = ccrs.LambertConformal(
    central_longitude=-30,   # centred on the Atlantic
    central_latitude=50,
    standard_parallels=(35, 65),
)

LAND_COLOR  = "#CCCCCC"
COAST_LW    = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def map_panel(ax, lon, lat, data, cmap, vmin, vmax, title, extent=NA_EXTENT):
    """Draw a pcolormesh map panel and return the mappable."""
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
    gl = ax.gridlines(
        draw_labels=False,
        linewidth=0.3, color="#AAAAAA", linestyle="--", alpha=0.7,
    )
    gl.top_labels   = False
    gl.right_labels = False


def shared_colorbar(fig, im, ax_list, label, shrink=0.85, pad=0.01):
    """Add a single vertical colorbar spanning ax_list, on the right."""
    cbar = fig.colorbar(
        im, ax=ax_list,
        orientation="vertical",
        shrink=shrink, pad=pad,
        aspect=25,
    )
    cbar.set_label(label, labelpad=6)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"])
    return cbar


def load_obs_mld(obs_mld_file: Path, month: int):
    ds = xr.open_dataset(obs_mld_file, decode_times=False)
    # Prefer dr003 variable, fall back to first data var
    var = "mld_dr003" if "mld_dr003" in ds else list(ds.data_vars)[0]
    mld = ds[var].isel(time=month - 1).squeeze()
    lon = ds["lon"].values
    lat = ds["lat"].values
    ds.close()
    return lon, lat, mld.values


def load_obs_sal(obs_sal_file: Path, month: int):
    ds = xr.open_dataset(obs_sal_file, decode_times=False)
    var = "s_an" if "s_an" in ds else list(ds.data_vars)[0]
    sos = ds[var].isel(time=0, depth=0).squeeze()
    lon = ds["lon"].values
    lat = ds["lat"].values
    ds.close()
    return lon, lat, sos.values


def load_model_maps(maps_file: Path):
    ds = xr.open_dataset(maps_file, decode_times=False)
    mld = ds["somxl030"].squeeze().values
    sos = ds["sos"].squeeze().values
    lon = ds["lon"].values
    lat = ds["lat"].values
    ds.close()
    return lon, lat, mld, sos


def load_amoc(amoc_file: Path):
    """Return (time_years, amoc_sv) as plain numpy arrays.
    Handles both the consolidated na_monitor_amoc_timeseries.nc (var: amoc_max,
    dim: time) and the raw get_AMOC_deep_v5 output (var: AMOC_26N, dim: time_counter).
    """
    ds = xr.open_dataset(amoc_file, decode_times=False)

    # Variable name
    if "amoc_max" in ds:
        amoc = ds["amoc_max"].values
    elif "AMOC_26N" in ds:
        amoc = ds["AMOC_26N"].values
    else:
        var = list(ds.data_vars)[0]
        print(f"  WARNING: guessing AMOC variable as '{var}'")
        amoc = ds[var].values

    # Time coordinate
    if "time" in ds:
        t = ds["time"].values
    elif "time_counter" in ds:
        t = ds["time_counter"].values
    else:
        t = np.arange(len(amoc))

    # Convert time to decimal years.
    # Time is typically "days since YYYY-MM-DD" with a noleap calendar,
    # so decode via cftime then convert to decimal year.
    try:
        if hasattr(t[0], "year"):
            # Already cftime objects (xarray decoded them)
            years = np.array([v.year + (v.dayofyr - 1) / 365.0 for v in t])
        else:
            # Raw numeric — decode manually using the units attribute
            units = None
            for var in ["time", "time_counter"]:
                if var in ds and hasattr(ds[var], "attrs"):
                    units = ds[var].attrs.get("units", None)
                    calendar = ds[var].attrs.get("calendar", "noleap")
                    if units:
                        break
            if units:
                import cftime
                decoded = cftime.num2date(t, units=units, calendar=calendar)
                years = np.array([v.year + (v.dayofyr - 1) / 365.0 for v in decoded])
            else:
                years = np.arange(len(amoc))
    except Exception as e:
        print(f"  WARNING: time conversion failed ({e}), using index.")
        years = np.arange(len(amoc))

    ds.close()
    return years, amoc


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def make_figure(
    model: str,
    maps_file: Path,
    amoc_file: Optional[Path],
    obs_mld_file: Path,
    obs_sal_file: Path,
    month: int,
    year_start: int,
    year_end: int,
    out_path: Path,
    mld_vmin: float = MLD_VMIN,
    mld_vmax: float = MLD_VMAX,
    sal_vmin: float = SAL_VMIN,
    sal_vmax: float = SAL_VMAX,
    extent: list = NA_EXTENT,
    font_size: int = 13,
):
    set_presentation_style(base_size=font_size)

    month_name = MONTH_NAMES[month - 1]

    # ---- Load data --------------------------------------------------------

    print("Loading model maps…")
    mod_lon, mod_lat, mod_mld, mod_sos = load_model_maps(maps_file)

    print("Loading obs MLD…")
    obs_mld_lon, obs_mld_lat, obs_mld = load_obs_mld(obs_mld_file, month)

    print("Loading obs salinity…")
    obs_sal_lon, obs_sal_lat, obs_sos = load_obs_sal(obs_sal_file, month)

    amoc_years = amoc_sv = None
    if amoc_file is not None and amoc_file.exists():
        print("Loading AMOC timeseries…")
        amoc_years, amoc_sv = load_amoc(amoc_file)

    # ---- Figure layout ----------------------------------------------------
    #
    # GridSpec: 3 rows, 3 cols.
    #   Col 0, 1: map panels.  Col 2: colorbars (narrow).
    # Row heights: AMOC row thin (1), map rows equal (2.2 each).
    #
    # We merge col 0+1 for row 0 (AMOC spans full width minus colorbar col).
    # Colorbars are placed in col 2 for rows 1 and 2.

    fig = plt.figure(figsize=(13, 10))

    gs = gridspec.GridSpec(
        3, 3,
        figure=fig,
        height_ratios=[1.2, 2.2, 2.2],
        width_ratios=[1, 1, 0.055],   # col 2 is the colorbar strip
        hspace=0.30,
        wspace=0.08,
    )

    # Row 0: AMOC, spans cols 0–1
    ax_amoc = fig.add_subplot(gs[0, 0:2])

    # Rows 1–2: maps in cols 0 and 1
    ax_sal_obs = fig.add_subplot(gs[1, 0], projection=MAP_PROJ)
    ax_sal_mod = fig.add_subplot(gs[1, 1], projection=MAP_PROJ)
    ax_mld_obs = fig.add_subplot(gs[2, 0], projection=MAP_PROJ)
    ax_mld_mod = fig.add_subplot(gs[2, 1], projection=MAP_PROJ)

    # Colorbar axes in col 2 for rows 1 and 2
    cax_sal = fig.add_subplot(gs[1, 2])
    cax_mld = fig.add_subplot(gs[2, 2])

    # ---- AMOC panel -------------------------------------------------------

    if amoc_years is not None:
        ax_amoc.plot(amoc_years, amoc_sv, lw=1.5, color="#0050FF")
        # Annual smoothed overlay
        if len(amoc_sv) >= 11:
            from scipy.ndimage import uniform_filter1d
            smooth = uniform_filter1d(amoc_sv, size=11)
            ax_amoc.plot(amoc_years, smooth, lw=2.5, color="#1A1A1A", zorder=3)
        ax_amoc.axhline(0, color="#AAAAAA", lw=0.6, ls="--")
        ax_amoc.axhline(10, color="#CCCCCC", lw=0.8, ls="-", zorder=0)
        ax_amoc.set_ylabel("AMOC (Sv)")
        ax_amoc.set_xlabel("Year")
        # Tick every 20 years, aligned to a round multiple
        yr_min, yr_max = int(np.floor(amoc_years.min())), int(np.ceil(amoc_years.max()))
        tick_start = int(np.ceil(yr_min / 20.0)) * 20
        major_ticks = np.arange(tick_start, yr_max + 1, 20)
        ax_amoc.set_xticks(major_ticks)
        ax_amoc.set_xticklabels([str(y) for y in major_ticks])
        ax_amoc.set_xlim(yr_min, yr_max)
    else:
        ax_amoc.text(0.5, 0.5, "AMOC data not available",
                     ha="center", va="center", transform=ax_amoc.transAxes,
                     color="#888888")
        ax_amoc.set_xlabel("Year")
        ax_amoc.set_ylabel("AMOC (Sv)")

    ax_amoc.set_title(
        f"{model}  —  AMOC streamfunction maximum at 26.5°N (below 500 m)",
        loc="left", fontsize=font_size,
    )

    # Remove top/right spines (already done by plot_style, but be explicit)
    ax_amoc.spines["top"].set_visible(False)
    ax_amoc.spines["right"].set_visible(False)

    # ---- Salinity maps (row 1) --------------------------------------------

    im_sal_obs = map_panel(
        ax_sal_obs, obs_sal_lon, obs_sal_lat, obs_sos,
        cmo.haline, sal_vmin, sal_vmax,
        f"Obs SSS  ({month_name})", extent=extent,
    )
    add_gridlines(ax_sal_obs)
    ax_sal_obs.text(0.99, 0.02, "World Ocean Atlas 2023",
                    transform=ax_sal_obs.transAxes,
                    ha="right", va="bottom", fontsize=7, color="#555555",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5))

    im_sal_mod = map_panel(
        ax_sal_mod, mod_lon, mod_lat, mod_sos,
        cmo.haline, sal_vmin, sal_vmax,
        f"{model}  ({month_name}, {year_start}–{year_end})", extent=extent,
    )
    add_gridlines(ax_sal_mod)

    cb_sal = fig.colorbar(im_sal_mod, cax=cax_sal, orientation="vertical")
    cb_sal.set_label("SSS (psu)", labelpad=4)

    # ---- MLD maps (row 2) -------------------------------------------------

    im_mld_obs = map_panel(
        ax_mld_obs, obs_mld_lon, obs_mld_lat, obs_mld,
        cmo.deep, mld_vmin, mld_vmax,
        f"Obs MLD  ({month_name})", extent=extent,
    )
    add_gridlines(ax_mld_obs)
    ax_mld_obs.text(0.99, 0.02, "de Boyer Montégut et al. (2004)",
                    transform=ax_mld_obs.transAxes,
                    ha="right", va="bottom", fontsize=7, color="#555555",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=1.5))

    im_mld_mod = map_panel(
        ax_mld_mod, mod_lon, mod_lat, mod_mld,
        cmo.deep, mld_vmin, mld_vmax,
        f"{model}  ({month_name}, {year_start}–{year_end})", extent=extent,
    )
    add_gridlines(ax_mld_mod)

    cb_mld = fig.colorbar(im_mld_mod, cax=cax_mld, orientation="vertical")
    cb_mld.set_label("MLD (m)", labelpad=4)

    # ---- Supertitle -------------------------------------------------------

    fig.suptitle(
        f"North Atlantic physical monitor  ·  {model}  ·  {month_name}",
        y=0.995, fontsize=font_size + 2, fontweight="medium",
    )

    # ---- Provenance -------------------------------------------------------

    try:
        from plot_style import add_provenance
        add_provenance(fig, SCRIPT_PATH, fontsize=9, x=0.5, y=0.01, ha="center")
    except (ImportError, TypeError):
        fig.text(0.5, 0.01, SCRIPT_PATH,
                 ha="center", fontsize=9, color="#AAAAAA")

    # ---- Save -------------------------------------------------------------

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end",   type=int, default=None)
    parser.add_argument("--n-years",    type=int, default=5,
                        help="Most-recent years to use if --year-start/--year-end "
                             "not given (default 5)")
    parser.add_argument("--month",      type=int, default=3)

    parser.add_argument("--maps-file", type=Path, default=None,
                        help="Override the maps NetCDF path. Defaults to "
                             "<CLIMS_DIR>/<MODEL>/<MODEL>_na_monitor_maps_"
                             "<YEAR_START>_<YEAR_END>_mon<MM>.nc")
    parser.add_argument("--amoc-file", type=Path, default=None,
                        help="Override the AMOC timeseries NetCDF path. Defaults to "
                             "<CLIMS_DIR>/<MODEL>/<MODEL>_na_monitor_amoc_timeseries.nc. "
                             "Panel is left empty if the file does not exist.")

    parser.add_argument("--obs-mld-file", type=Path, default=OBS_MLD_FILE_DEFAULT)
    parser.add_argument("--obs-sal-file", type=Path, default=None,
                        help="WOA salinity file for --month. Defaults to auto-detected "
                             "file in the WOA dir.")

    parser.add_argument("--mld-vmin", type=float, default=MLD_VMIN)
    parser.add_argument("--mld-vmax", type=float, default=MLD_VMAX)
    parser.add_argument("--sal-vmin", type=float, default=SAL_VMIN)
    parser.add_argument("--sal-vmax", type=float, default=SAL_VMAX)

    parser.add_argument("--extent", type=float, nargs=4,
                        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                        default=NA_EXTENT,
                        help="Map extent in PlateCarree degrees (default: %(default)s)")

    parser.add_argument("--out-dir",  type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--font-size", type=int, default=13)

    args = parser.parse_args()

    month_name = MONTH_NAMES[args.month - 1]

    # Resolve year range
    if args.year_start is not None and args.year_end is not None:
        year_start, year_end = args.year_start, args.year_end
    else:
        import glob as _glob, re as _re
        pattern = str(NEMO5_DIR / args.model / "ORCA2_7d_*_grid_T.nc")
        years = sorted(set(
            int(m.group(1))
            for f in _glob.glob(pattern)
            for m in [_re.search(r"_7d_(\d{4})0101_", Path(f).name)]
            if m
        ))
        if not years:
            print(f"ERROR: no grid_T files found for {args.model} in {NEMO5_DIR}.")
            sys.exit(1)
        year_end   = years[-1]
        year_start = years[max(0, len(years) - args.n_years)]
        print(f"  Auto-detected years: using last {args.n_years}: {year_start}–{year_end}")

    # Auto-derive input file paths from model/year/month if not given explicitly
    clims_model = CLIMS_DIR / args.model
    maps_file = args.maps_file or (
        clims_model / f"{args.model}_na_monitor_maps_"
                      f"{year_start}_{year_end}_mon{args.month:02d}.nc"
    )
    if args.amoc_file:
        amoc_file = args.amoc_file
    else:
        # Prefer the consolidated file written by prep_na_monitor.py
        consolidated = clims_model / f"{args.model}_na_monitor_amoc_timeseries.nc"
        if consolidated.exists():
            amoc_file = consolidated
        else:
            # Fall back to the raw get_AMOC_deep_v5 output.
            # Format: MODEL_AMOC_26N_YRST_YREND.nc — pick the file with the
            # largest (YREND - YRST) span.
            import re as _re
            raw_files = sorted(clims_model.glob(f"{args.model}_AMOC_26N_*.nc"))
            best_file, best_span = None, -1
            for f in raw_files:
                m = _re.search(r"_AMOC_26N_(\d{4})_(\d{4})\.nc$", f.name)
                if m:
                    span = int(m.group(2)) - int(m.group(1))
                    if span > best_span:
                        best_span, best_file = span, f
            amoc_file = best_file
            if amoc_file:
                print(f"  Using raw AMOC file: {amoc_file.name} (span {best_span} yrs)")
            else:
                amoc_file = consolidated  # will be missing; panel left empty

    if not maps_file.exists():
        print(f"ERROR: maps file not found: {maps_file}")
        print("Run prep_na_monitor.py first, or pass --maps-file explicitly.")
        sys.exit(1)

    obs_sal_file = args.obs_sal_file or (
        OBS_SAL_DIR_DEFAULT / OBS_SAL_PATTERN.format(month=args.month)
    )

    out_path = (
        args.out_dir
        / f"{args.model}_na_monitor_{month_name.lower()}"
          f"_{year_start}_{year_end}.png"
    )

    make_figure(
        model=args.model,
        maps_file=maps_file,
        amoc_file=amoc_file,
        obs_mld_file=args.obs_mld_file,
        obs_sal_file=obs_sal_file,
        month=args.month,
        year_start=year_start,
        year_end=year_end,
        out_path=out_path,
        mld_vmin=args.mld_vmin,
        mld_vmax=args.mld_vmax,
        sal_vmin=args.sal_vmin,
        sal_vmax=args.sal_vmax,
        extent=args.extent,
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
