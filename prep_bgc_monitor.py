#!/usr/bin/env python3
"""
prep_bgc_monitor.py
Preprocessing for the North Atlantic BGC monitor.

Finds the last N years of available model output (default 5), regrids
surface PO4 and NO3 from *ptrc_T* files, and TChl + PPINT from *diad_T*
files, averaging over those years for the target month.

Usage:
    python prep_bgc_monitor.py --model TOM12_TJ_HA00
    python prep_bgc_monitor.py --model TOM12_TJ_HA00 --n-years 5 --month 3
    python prep_bgc_monitor.py --model TOM12_TJ_HA00 --year-start 1950 --year-end 1960
"""

import argparse
import glob
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NEMO5_DIR       = Path("/gpfs/data/greenocean/software/runs/NEMO5")
CLIMS_DIR       = Path("/gpfs/data/greenocean/users/mep22dku/clims")
CDO_TARGET_GRID = "r360x180"


# BGC variables: (file_type, variable_name)
BGC_VARS = {
    "PO4":   ("ptrc_T", "PO4"),
    "NO3":   ("ptrc_T", "NO3"),
    "TChl":  ("diad_T", "TChl"),
    "PPINT": ("diad_T", "PPINT"),
}


# ---------------------------------------------------------------------------

def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def find_available_years(model: str, file_type: str) -> list:
    """Return sorted list of years for which ORCA2_7d_YYYY*_{file_type}.nc exists."""
    pattern = str(NEMO5_DIR / model / f"ORCA2_7d_*_{file_type}.nc")
    files = glob.glob(pattern)
    years = []
    for f in files:
        m = re.search(r"_7d_(\d{4})0101_", Path(f).name)
        if m:
            years.append(int(m.group(1)))
    return sorted(set(years))


def prep_bgc_maps(model: str, year_start: int, year_end: int,
                  out_dir: Path) -> Path:
    """
    For each year in [year_start, year_end]: CDO-regrid ptrc_T and diad_T
    files, then average in Python.
    """
    out_path = out_dir / f"{model}_bgc_monitor_maps_{year_start}_{year_end}_annual.nc"

    # Delete any stale maps files so the plot always gets the freshest data
    for old_f in out_dir.glob(f"{model}_bgc_monitor_maps_*_annual.nc"):
        print(f"  Removing stale maps file: {old_f.name}")
        old_f.unlink()

    model_dir = NEMO5_DIR / model

    # Accumulate per file-type separately (not all years may have both)
    accum = {v: None for v in BGC_VARS}
    counts = {v: 0 for v in BGC_VARS}
    lon = lat = None

    for year in range(year_start, year_end + 1):
        for varname, (ftype, ncvar) in BGC_VARS.items():
            src = model_dir / f"ORCA2_7d_{year}0101_{year}1231_{ftype}.nc"
            if not src.exists():
                print(f"  WARNING: {src.name} not found — skipping {varname} {year}.")
                continue

            tmp = Path(tempfile.mktemp(suffix=f"_{varname}_{year}.nc"))
            try:
                # sellevidx,1 picks the top level before regridding — works for
                # both 3D (PO4, NO3, TChl) and 2D (PPINT) variables.
                run([
                    "cdo",
                    f"remapbil,{CDO_TARGET_GRID}",
                    "-timmean",
                    "-sellevidx,1",
                    f"-selvar,{ncvar}",
                    str(src), str(tmp),
                ])
            except subprocess.CalledProcessError:
                print(f"  ERROR: CDO failed for {varname} {year} — skipping.")
                tmp.unlink(missing_ok=True)
                continue

            ds = xr.open_dataset(tmp, decode_times=False)
            data = ds[ncvar].squeeze().values.astype(float)
            if data.ndim != 2:
                raise RuntimeError(
                    f"{ncvar}: expected 2D after CDO sellevidx,1 but got "
                    f"shape {data.shape} — check file."
                )
            if accum[varname] is None:
                accum[varname] = data
                if lon is None:
                    lon = ds["lon"].values
                    lat = ds["lat"].values
            else:
                accum[varname] += data
            counts[varname] += 1
            ds.close()
            tmp.unlink(missing_ok=True)

    # Build output dataset
    data_vars = {}
    for varname in BGC_VARS:
        if accum[varname] is not None and counts[varname] > 0:
            mean = accum[varname] / counts[varname]
            # Unit conversions to standard observable units:
            #   TChl: g Chl/L  → mg Chl m-3  (× 1e6)
            #     1 g/L = 1e6 mg/m3
            #   NO3, PO4: already in mmol m-3 (model C units, comparable to WOA18)
            #   PPINT: keep native units
            if varname == "TChl":
                mean = mean * 1e6
            mean[mean <= 0] = np.nan
            attrs = {"long_name": varname,
                     "units": "mg m-3" if varname == "TChl" else "native",
                     "n_years_averaged": counts[varname]}
            data_vars[varname] = xr.DataArray(
                mean, dims=["lat", "lon"],
                coords={"lat": lat, "lon": lon},
                attrs=attrs,
            )
        else:
            print(f"  WARNING: no data accumulated for {varname}.")

    if not data_vars:
        raise RuntimeError(f"No BGC data found for {model} {year_start}–{year_end}.")

    xr.Dataset(data_vars).to_netcdf(out_path)
    print(f"  BGC maps written: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-years", type=int, default=5,
                        help="Number of most-recent years to average (default 5). "
                             "Ignored if --year-start/--year-end are given explicitly.")
    parser.add_argument("--year-start", type=int, default=None,
                        help="Override: explicit start year.")
    parser.add_argument("--year-end", type=int, default=None,
                        help="Override: explicit end year.")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or (CLIMS_DIR / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine year range
    if args.year_start is not None and args.year_end is not None:
        year_start, year_end = args.year_start, args.year_end
    else:
        # Auto-detect from ptrc_T files (fall back to diad_T)
        years = find_available_years(args.model, "ptrc_T")
        if not years:
            years = find_available_years(args.model, "diad_T")
        if not years:
            raise RuntimeError(
                f"No ptrc_T or diad_T files found for {args.model} in {NEMO5_DIR}."
            )
        year_end   = years[-1]
        year_start = years[max(0, len(years) - args.n_years)]
        print(f"  Auto-detected years: {years[0]}–{years[-1]}, "
              f"using last {args.n_years}: {year_start}–{year_end}")

    print(f"\n[1/1] Preparing BGC maps ({year_start}–{year_end}, annual mean)…")
    prep_bgc_maps(args.model, year_start, year_end, out_dir)


    print("\nDone.")


if __name__ == "__main__":
    main()
