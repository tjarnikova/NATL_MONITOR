#!/usr/bin/env python3
"""
plot_region_timeseries.py
Plot annual-mean regional timeseries from prep_region_timeseries.py output.

Two figures:

  Figure 1 — Physical & nutrients
    Col 0: TOS | SOS | MLD
    Col 1: NO3 | PO4 | Si

  Figure 2 — BGC & plankton
    Col 0: TChl | Cflx | PPINT
    Col 1: Phyto stacked | Zoo stacked | Legend

Usage:
    python plot_region_timeseries.py --model TOM12_TJ_HA00
    python plot_region_timeseries.py --model TOM12_TJ_HA00 --year-start 1930 --year-end 1970
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, "/gpfs/home/mep22dku/scratch/SOZONE/UTILS")
from plot_style import set_presentation_style

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLIMS_DIR = Path("/gpfs/data/greenocean/users/mep22dku/clims")
OUT_DIR_DEFAULT = Path(
    "/gpfs/home/mep22dku/scratch/AMOC-PLANKTOM/hosing/AMOCtun/MONITOR/plots"
)
SCRIPT_PATH = (
    "/gpfs/home/mep22dku/scratch/AMOC-PLANKTOM/hosing/AMOCtun/MONITOR/"
    "plot_region_timeseries.py"
)

BOX_DESC = "Wider SPG region, approx. −64° to −16°E, 47° to 71°N"

# ---------------------------------------------------------------------------
# Colours — match ecosystem monitor
# ---------------------------------------------------------------------------
PHYTO_VARS   = ['DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
ZOO_VARS     = ['BAC', 'PRO', 'PTE', 'MES', 'GEL', 'CRU']
PHYTO_COLORS = dict(zip(PHYTO_VARS,
                        ['#4477AA', '#EE6677', '#006D77',
                         '#CCBB44', '#66CCEE', '#AA3377']))
ZOO_COLORS   = dict(zip(ZOO_VARS,
                        ['#E69F00', '#332288', '#F0E442',
                         '#D55E00', '#00C9A7', '#7B2D8B']))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annual_means(ts_dir: Path, model: str,
                       year_start: int, year_end: int) -> dict:
    """
    Load all timeseries files, compute annual means, return dict of
    {varname: (years_array, values_array)}.
    """
    data = {}   # varname → list of (year, annual_mean)

    for ftype in ["grid_T", "ptrc_T", "diad_T"]:
        pattern = str(ts_dir / f"{model}_ts_{ftype}_*.nc")
        import glob as _glob
        files = sorted(_glob.glob(pattern))
        for fpath in files:
            m = re.search(r"_(\d{4})\.nc$", Path(fpath).name)
            if not m:
                continue
            year = int(m.group(1))
            if not (year_start <= year <= year_end):
                continue
            try:
                ds = xr.open_dataset(fpath)   # datetime64 time axis — opens cleanly
                for v in ds.data_vars:
                    val = float(ds[v].mean().values)
                    if v not in data:
                        data[v] = []
                    data[v].append((year, val))
                ds.close()
            except Exception as e:
                print(f"  WARNING: could not load {Path(fpath).name}: {e}")

    # Convert to sorted numpy arrays
    out = {}
    for v, pairs in data.items():
        pairs.sort(key=lambda x: x[0])
        yrs  = np.array([p[0] for p in pairs])
        vals = np.array([p[1] for p in pairs])
        out[v] = (yrs, vals)

    return out


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def ts_panel(ax, years, values, ylabel, color="#0050FF", lw=1.8):
    """Simple line panel."""
    ax.plot(years, values, color=color, lw=lw)
    ax.set_ylabel(ylabel, fontsize=plt.rcParams["axes.labelsize"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(years[0], years[-1])


def stacked_panel(ax, data: dict, var_list: list, colors: dict, ylabel: str):
    """
    True stacked fill_between plot.
    Each group's fill starts at the cumulative sum of those below.
    """
    # Find common year axis
    all_years = None
    for v in var_list:
        if v in data:
            yrs = data[v][0]
            if all_years is None:
                all_years = yrs
            else:
                # Intersect
                all_years = np.intersect1d(all_years, yrs)

    if all_years is None or len(all_years) == 0:
        ax.set_visible(False)
        return

    # Build value matrix (n_vars, n_years) aligned to common years
    vals = []
    present = []
    for v in var_list:
        if v not in data:
            continue
        yrs, v_vals = data[v]
        # Align to all_years
        idx = np.isin(yrs, all_years)
        vals.append(v_vals[idx])
        present.append(v)

    if not vals:
        ax.set_visible(False)
        return

    vals = np.array(vals)  # (n_present, n_years)
    vals = np.where(np.isfinite(vals), vals, 0.0)

    cumsum = np.zeros(len(all_years))
    for i, v in enumerate(present):
        bottom = cumsum.copy()
        top    = cumsum + vals[i]
        ax.fill_between(all_years, bottom, top,
                        color=colors[v], alpha=0.85, label=v,
                        linewidth=0.3, edgecolor="white")
        ax.plot(all_years, top, color=colors[v], lw=0.8)
        cumsum = top

    ax.set_ylabel(ylabel)
    ax.set_xlim(all_years[0], all_years[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)


def add_stacked_legend(ax, phyto_vars, phyto_colors, zoo_vars, zoo_colors):
    """Fill a blank axes with two-column patch legend."""
    ax.set_visible(True)
    ax.axis("off")

    phyto_patches = [mpatches.Patch(facecolor=phyto_colors[v],
                                     edgecolor="#555555", lw=0.5, label=v)
                     for v in phyto_vars if v in phyto_colors]
    zoo_patches   = [mpatches.Patch(facecolor=zoo_colors[v],
                                     edgecolor="#555555", lw=0.5, label=v)
                     for v in zoo_vars if v in zoo_colors]

    legend1 = ax.legend(handles=phyto_patches, title="Phytoplankton",
                         loc="upper left",
                         fontsize=9, title_fontsize=9.5,
                         framealpha=0.0, edgecolor="none",
                         ncol=2, columnspacing=0.8, handlelength=1.2)
    ax.add_artist(legend1)
    ax.legend(handles=zoo_patches, title="Zooplankton / Heterotrophs",
              loc="lower left",
              fontsize=9, title_fontsize=9.5,
              framealpha=0.0, edgecolor="none",
              ncol=2, columnspacing=0.8, handlelength=1.2)


def label_x(ax, years):
    """Add year x-label only to bottom panels."""
    ax.set_xlabel("Year")
    # Ticks every 10 or 20 years depending on range
    span = years[-1] - years[0]
    step = 10 if span <= 60 else 20
    ticks = np.arange(
        int(np.ceil(years[0] / step)) * step,
        years[-1] + 1, step
    )
    ax.set_xticks(ticks)


def provenance(fig):
    try:
        from plot_style import add_provenance
        add_provenance(fig, SCRIPT_PATH, fontsize=9, x=0.5, y=0.005, ha="center")
    except Exception:
        fig.text(0.5, 0.005, SCRIPT_PATH, ha="center", fontsize=9, color="#AAAAAA")


# ---------------------------------------------------------------------------
# Figure 1: physical & nutrients
# ---------------------------------------------------------------------------

def plot_physical(data: dict, model: str, year_start: int, year_end: int,
                   out_path: Path, font_size: int = 13):
    set_presentation_style(base_size=font_size)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    fig.suptitle(
        f"{model}  ·  {year_start}–{year_end}\n{BOX_DESC}",
        y=0.98, fontsize=font_size + 1, fontweight="medium",
    )

    panels = [
        # (row, col, varname, ylabel, color)
        (0, 0, "tos",      "SST (°C)",         "#0050FF"),
        (1, 0, "sos",      "SSS (psu)",         "#1A1A1A"),
        (2, 0, "mldr10_1", "MLD (m)",           "#DE2E25"),
        (0, 1, "NO3",      "NO3 (mol C L⁻¹)",  "#4477AA"),
        (1, 1, "PO4",      "PO4 (mol C L⁻¹)",  "#228833"),
        (2, 1, "Si",       "Si (mol C L⁻¹)",   "#CCBB44"),
    ]

    for row, col, v, ylabel, color in panels:
        ax = axes[row, col]
        if v in data:
            yrs, vals = data[v]
            ts_panel(ax, yrs, vals, ylabel, color=color)
            if row == 2:
                label_x(ax, yrs)
        else:
            ax.set_visible(False)

    # Invert MLD (deeper = down)
    if "mldr10_1" in data:
        axes[2, 0].invert_yaxis()

    provenance(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: BGC & plankton
# ---------------------------------------------------------------------------

def plot_bgc(data: dict, model: str, year_start: int, year_end: int,
              out_path: Path, font_size: int = 13):
    set_presentation_style(base_size=font_size)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9),
                              gridspec_kw={"hspace": 0.35, "wspace": 0.30})

    fig.suptitle(
        f"{model}  ·  {year_start}–{year_end}\n{BOX_DESC}",
        y=0.98, fontsize=font_size + 1, fontweight="medium",
    )

    # Col 0: scalar BGC panels
    scalar_panels = [
        (0, 0, "TChl",  "TChl (mg m⁻³)",         "#228833"),
        (1, 0, "Cflx",  "CO₂ flux (model units)", "#0050FF"),
        (2, 0, "PPINT", "PP int (model units)",   "#DE2E25"),
    ]
    for row, col, v, ylabel, color in scalar_panels:
        ax = axes[row, col]
        if v in data:
            yrs, vals = data[v]
            ts_panel(ax, yrs, vals, ylabel, color=color)
            if row == 2:
                label_x(ax, yrs)
        else:
            ax.set_visible(False)

    # Col 1 row 0: phyto stacked
    stacked_panel(
        axes[0, 1], data, PHYTO_VARS, PHYTO_COLORS,
        "Phyto biomass (mmol m⁻²)",
    )
    axes[0, 1].set_title("Phytoplankton", fontsize=font_size - 1)

    # Col 1 row 1: zoo stacked
    stacked_panel(
        axes[1, 1], data, ZOO_VARS, ZOO_COLORS,
        "Zoo biomass (mmol m⁻²)",
    )
    axes[1, 1].set_title("Zooplankton / Heterotrophs", fontsize=font_size - 1)

    # Add x labels to stacked panels
    for v_list in [PHYTO_VARS, ZOO_VARS]:
        for v in v_list:
            if v in data:
                yrs = data[v][0]
                break
        else:
            continue
        break
    if 'yrs' in dir():
        label_x(axes[1, 1], yrs)

    # Col 1 row 2: legend
    add_stacked_legend(axes[2, 1],
                        PHYTO_VARS, PHYTO_COLORS,
                        ZOO_VARS,   ZOO_COLORS)
    axes[2, 0].set_xlabel("Year")   # bottom-left already has x label from scalar

    provenance(fig)
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
    parser.add_argument("--model",      required=True)
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end",   type=int, default=None)
    parser.add_argument("--out-dir",    type=Path, default=OUT_DIR_DEFAULT)
    parser.add_argument("--font-size",  type=int,  default=13)
    args = parser.parse_args()

    ts_dir = CLIMS_DIR / args.model / "timeseries"
    if not ts_dir.exists():
        print(f"ERROR: timeseries directory not found: {ts_dir}")
        print("Run prep_region_timeseries.py first.")
        sys.exit(1)

    # Detect available years from filenames
    import glob as _glob
    all_files = _glob.glob(str(ts_dir / f"{args.model}_ts_grid_T_*.nc"))
    all_years = sorted(set(
        int(m.group(1))
        for f in all_files
        for m in [re.search(r"_(\d{4})\.nc$", Path(f).name)]
        if m
    ))
    if not all_years:
        print(f"ERROR: no timeseries files found in {ts_dir}")
        sys.exit(1)

    year_start = args.year_start or all_years[0]
    year_end   = args.year_end   or all_years[-1]
    print(f"  Model: {args.model}  Years: {year_start}–{year_end}")

    print("  Loading annual means…")
    data = load_annual_means(ts_dir, args.model, year_start, year_end)
    print(f"  Variables loaded: {sorted(data.keys())}")

    yr_tag = f"{year_start}_{year_end}"

    print("\n  Plotting Figure 1 (physical & nutrients)…")
    plot_physical(
        data, args.model, year_start, year_end,
        out_path=args.out_dir / f"{args.model}_ts_physical_{yr_tag}.png",
        font_size=args.font_size,
    )

    print("  Plotting Figure 2 (BGC & plankton)…")
    plot_bgc(
        data, args.model, year_start, year_end,
        out_path=args.out_dir / f"{args.model}_ts_bgc_{yr_tag}.png",
        font_size=args.font_size,
    )


if __name__ == "__main__":
    main()
