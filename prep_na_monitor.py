#!/usr/bin/env python3
"""
prep_na_monitor.py
Preprocessing step for the North Atlantic physical monitor.

For a given model and year range this script:
  1. Reads per-year grid_T files, selects the requested month, applies the
     MLD criterion conversion (mldr10_1 → somxl030), averages over years,
     regrids to r360x180.
  2. Reads the AMOC timeseries produced by get_AMOC_deep_v5.py / the
     postprocess pipeline, extracts the 26.5°N streamfunction maximum.
  3. Writes two output files:
       <OUT_DIR>/<MODEL>_na_monitor_maps_<YEAR_START>_<YEAR_END>_mon<MM>.nc
       <OUT_DIR>/<MODEL>_na_monitor_amoc_timeseries.nc

Usage:
    python prep_na_monitor.py --model TOM12_TJ_SM01 \\
        --year-start 1920 --year-end 1925 --month 3

    # Override output directory:
    python prep_na_monitor.py --model TOM12_TJ_SM01 \\
        --year-start 1920 --year-end 1925 --out-dir /my/output/dir
"""

import argparse
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Paths — edit to match your environment
# ---------------------------------------------------------------------------
NEMO5_DIR  = Path("/gpfs/data/greenocean/software/runs/NEMO5")
CLIMS_DIR  = Path("/gpfs/data/greenocean/users/mep22dku/clims")
CDFTOOLS_DIR = Path("/gpfs/data/greenocean/software/resources/CDFTOOLS")

# ---------------------------------------------------------------------------
# MLD criterion conversion: mldr10_1 (model native) → somxl030 (obs-comparable)
# ---------------------------------------------------------------------------
MLD_SLOPE     = 1.3607
MLD_INTERCEPT = -1.2358

# ---------------------------------------------------------------------------
# AMOC latitude to extract (degrees N). 26.5°N matches RAPID array.
# ---------------------------------------------------------------------------
AMOC_LAT = 26.5

# ---------------------------------------------------------------------------
# Target grid for regridding (CDO)
# ---------------------------------------------------------------------------
CDO_TARGET_GRID = "r360x180"


# ---------------------------------------------------------------------------

def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, check=True, **kw)
    return result


def regrid_file(src: Path, dst: Path, target_grid: str = CDO_TARGET_GRID) -> Path:
    """Regrid src → dst using cdo remapbil. Returns dst."""
    run(["cdo", f"remapbil,{target_grid}", str(src), str(dst)])
    return dst


def prep_maps(model: str, year_start: int, year_end: int, month: int,
              out_dir: Path) -> Path:
    """
    For each year: select the target month from the raw grid_T file using CDO
    (which can read NEMO's curvilinear coordinates natively), regrid to
    r360x180, then average in Python and apply the MLD criterion conversion.
    This mirrors what run_obs_model_compare.bsub does before calling the plot
    script — CDO needs the original file with nav_lon/nav_lat intact.
    """
    model_dir = NEMO5_DIR / model
    out_path = out_dir / f"{model}_na_monitor_maps_{year_start}_{year_end}_mon{month:02d}.nc"

    if out_path.exists():
        print(f"  Maps file already exists: {out_path} — skipping.")
        return out_path

    tmp_files = []
    n = 0

    for year in range(year_start, year_end + 1):
        tfile = model_dir / f"ORCA2_7d_{year}0101_{year}1231_grid_T.nc"
        if not tfile.exists():
            print(f"  WARNING: {tfile} not found — skipping year {year}.")
            continue

        # CDO pipeline on the original NEMO file (curvilinear nav_lon/nav_lat intact):
        #   selvar  — keep only the two fields we need (faster I/O)
        #   selmon  — select the target month
        #   timmean — average over those timesteps
        #   remapbil — bilinear regrid to regular 1° grid
        # The chained operators run right-to-left, so CDO reads the raw
        # curvilinear file at the selvar stage before any coordinate issues arise.
        tmp = Path(tempfile.mktemp(suffix=f"_{year}.nc"))
        run([
            "cdo",
            f"remapbil,{CDO_TARGET_GRID}",
            "-timmean",
            f"-selmon,{month}",
            "-selvar,mldr10_1,sos",
            str(tfile),
            str(tmp),
        ])
        tmp_files.append(tmp)
        n += 1

    if n == 0:
        raise RuntimeError(f"No grid_T files found for {model} in {year_start}–{year_end}.")

    # Average regridded years in Python
    print(f"  Averaging {n} years…")
    mld_accum = None
    sos_accum = None
    lon = lat = None

    for tmp in tmp_files:
        ds = xr.open_dataset(tmp, decode_times=False)
        mld = ds["mldr10_1"].squeeze().values.astype(float)
        sos = ds["sos"].squeeze().values.astype(float)
        if mld_accum is None:
            mld_accum = mld
            sos_accum = sos
            lon = ds["lon"].values
            lat = ds["lat"].values
        else:
            mld_accum += mld
            sos_accum += sos
        ds.close()
        tmp.unlink(missing_ok=True)

    mld_mean = mld_accum / n
    sos_mean = sos_accum / n

    # Apply MLD criterion conversion
    mld_mean = mld_mean * MLD_SLOPE + MLD_INTERCEPT
    mld_mean[mld_mean <= 0] = np.nan

    ds_out = xr.Dataset({
        "somxl030": xr.DataArray(mld_mean, dims=["lat", "lon"],
                                  coords={"lat": lat, "lon": lon},
                                  attrs={"long_name": "MLD (dr003 equivalent)", "units": "m"}),
        "sos":      xr.DataArray(sos_mean, dims=["lat", "lon"],
                                  coords={"lat": lat, "lon": lon},
                                  attrs={"long_name": "Sea surface salinity", "units": "psu"}),
    })
    ds_out.to_netcdf(out_path)

    print(f"  Maps written: {out_path}")
    return out_path


def prep_amoc(model: str, out_dir: Path) -> Path:
    """
    Read AMOC_26N files produced by get_AMOC_deep_v5.py (run via the bsub)
    and write a single consolidated timeseries NC for the monitor.

    get_AMOC_deep_v5.py writes files named:
        <CLIMS_DIR>/<MODEL>/<MODEL>_AMOC_26N_<yrst>_<yrend>.nc
    for each period (pre1940, post1940, full). We prefer the "full" file to
    avoid duplicating timesteps; fall back to concatenating period files.
    """
    out_path = out_dir / f"{model}_na_monitor_amoc_timeseries.nc"
    if out_path.exists():
        print(f"  AMOC file already exists: {out_path} — skipping.")
        return out_path

    clims_model = CLIMS_DIR / model
    amoc_files = sorted(clims_model.glob(f"{model}_AMOC_26N_*.nc"))

    if not amoc_files:
        raise FileNotFoundError(
            f"No {model}_AMOC_26N_*.nc files found in {clims_model}.\n"
            f"Run the bsub first:  sbatch prep_na_monitor.bsub --model {model}"
        )

    # Prefer the "full" period file; fall back to all period files
    full_files = [f for f in amoc_files if "full" in f.stem.lower()]
    load_files = full_files if full_files else amoc_files
    print(f"  Loading AMOC from: {[f.name for f in load_files]}")

    ds = xr.open_mfdataset(load_files, combine="by_coords")
    amoc = ds["AMOC_26N"].squeeze()
    time_vals = ds["time_counter"].values

    ds_out = xr.Dataset({
        "amoc_max": xr.DataArray(
            amoc.values,
            coords={"time": time_vals},
            dims=["time"],
            attrs={"long_name": "AMOC max streamfunction at 26.5°N, below 500 m",
                   "units": "Sv"},
        )
    }, attrs={"model": model, "amoc_lat": 26.5})

    ds_out.to_netcdf(out_path)
    ds.close()
    print(f"  AMOC written: {out_path}")
    return out_path


def find_last_n_years(model: str, n: int) -> tuple:
    """Return (year_start, year_end) for the last n available grid_T years."""
    import glob as _glob, re as _re
    pattern = str(NEMO5_DIR / model / "ORCA2_7d_*_grid_T.nc")
    years = sorted(set(
        int(m.group(1))
        for f in _glob.glob(pattern)
        for m in [_re.search(r"_7d_(\d{4})0101_", Path(f).name)]
        if m
    ))
    if not years:
        raise RuntimeError(f"No grid_T files found for {model} in {NEMO5_DIR}.")
    year_end   = years[-1]
    year_start = years[max(0, len(years) - n)]
    return year_start, year_end


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--year-start", type=int, default=None,
                        help="Map average start year (default: auto last --n-years)")
    parser.add_argument("--year-end",   type=int, default=None,
                        help="Map average end year (default: auto last --n-years)")
    parser.add_argument("--n-years",    type=int, default=5,
                        help="Number of most-recent years to average (default 5). "
                             "Ignored if --year-start/--year-end are given.")
    parser.add_argument("--month",      type=int, default=3,
                        help="Month to average for maps (1-12, default 3 = March)")
    parser.add_argument("--out-dir",    type=Path, default=None,
                        help="Output directory (default: <CLIMS_DIR>/<MODEL>)")
    parser.add_argument("--skip-amoc",  action="store_true",
                        help="Skip AMOC extraction (useful if not yet postprocessed)")
    parser.add_argument("--skip-maps",  action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir or (CLIMS_DIR / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.year_start is not None and args.year_end is not None:
        year_start, year_end = args.year_start, args.year_end
    else:
        year_start, year_end = find_last_n_years(args.model, args.n_years)
        print(f"  Auto-detected years: using last {args.n_years}: {year_start}–{year_end}")

    if not args.skip_maps:
        print(f"\n[1/2] Preparing map fields ({year_start}–{year_end}, "
              f"month {args.month})…")
        prep_maps(args.model, year_start, year_end, args.month, out_dir)

    if not args.skip_amoc:
        print(f"\n[2/2] Extracting AMOC timeseries…")
        try:
            prep_amoc(args.model, out_dir)
        except Exception as e:
            print(f"  WARNING: AMOC pipeline failed: {e}")
            print("  The AMOC panel will be empty. Run the bsub first or use --skip-amoc.")

    print("\nDone.")


if __name__ == "__main__":
    main()
