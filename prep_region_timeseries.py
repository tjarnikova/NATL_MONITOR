#!/usr/bin/env python3
"""
prep_region_timeseries.py
Extract area-weighted regional timeseries for a fixed box from NEMO5 output.

For each available year, reads grid_T, ptrc_T, and diad_T files, extracts
variables over the regional box, applies area weighting (and depth-integration
where appropriate), and saves one NetCDF per year per file type to:

    clims/<MODEL>/timeseries/
        <MODEL>_ts_grid_T_YYYY.nc
        <MODEL>_ts_ptrc_T_YYYY.nc
        <MODEL>_ts_diad_T_YYYY.nc

Time axis is converted from cftime noleap to numpy datetime64 so files open
cleanly with xarray (no decode_times=False needed).

Variables extracted:
    grid_T : tos (SST), sos (SSS), mldr10_1 → somxl030 (MLD)
    ptrc_T : NO3, PO4, Si (surface); BAC PRO PTE MES GEL DIA MIX COC PIC PHA FIX CRU (depth-integrated)
    diad_T : TChl (surface), Cflx, PPINT

Usage:
    python prep_region_timeseries.py --model TOM12_TJ_HA00
    python prep_region_timeseries.py --model TOM12_TJ_HA00 --year-start 1960 --year-end 1970
"""

import argparse
import glob
import re
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NEMO5_DIR  = Path("/gpfs/data/greenocean/software/runs/NEMO5")
CLIMS_DIR  = Path("/gpfs/data/greenocean/users/mep22dku/clims")
MESH_MASK  = Path("/gpfs/data/greenocean/software/resources/CDFTOOLS/mesh_mask_v5.nc")

# ---------------------------------------------------------------------------
# Regional box (native ORCA2 grid indices)
# ---------------------------------------------------------------------------
BOX_X0, BOX_Y0, BOX_W, BOX_H = 112, 108, 20, 20

# ---------------------------------------------------------------------------
# Variable lists
# ---------------------------------------------------------------------------
GRID_T_VARS  = ["tos", "sos", "mldr10_1"]

PTRC_SURFACE = ["NO3", "PO4", "Si"]
PTRC_DEPTHINT = ["BAC", "PRO", "PTE", "MES", "GEL",
                  "DIA", "MIX", "COC", "PIC", "PHA", "FIX", "CRU"]
PTRC_VARS    = PTRC_SURFACE + PTRC_DEPTHINT

DIAD_SURFACE = ["TChl", "Cflx"]
DIAD_DEPTHINT = ["PPINT"]          # PPINT is already column-integrated
DIAD_VARS    = DIAD_SURFACE + DIAD_DEPTHINT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_area_weights():
    """Return 2D area weight array (h, w) for the regional box."""
    ds = xr.open_dataset(MESH_MASK, decode_times=False)
    csize = (ds["e1t"][0, :, :] *
             ds["e2t"][0, :, :] *
             ds["tmask"][0, 0, :, :].astype(float))
    ds.close()
    weights = csize.values[BOX_Y0:BOX_Y0+BOX_H, BOX_X0:BOX_X0+BOX_W]
    return weights


def load_e3t():
    """Return masked cell thickness (z, y, x) for depth integration."""
    ds = xr.open_dataset(MESH_MASK, decode_times=False)
    e3t   = ds["e3t_0"].squeeze("t", drop=True)
    tmask = ds["tmask"].squeeze("t", drop=True).astype(float)
    ds.close()
    return (e3t * tmask).values  # (z, y, x)


def cftime_to_datetime64(times):
    """
    Convert cftime noleap array to numpy datetime64, mapping Feb 29 → Feb 28
    for the rare case it appears, and returning a proper DatetimeIndex.
    """
    timestamps = []
    for t in times:
        try:
            timestamps.append(pd.Timestamp(t.year, t.month, t.day,
                                           t.hour, t.minute, t.second))
        except ValueError:
            # Feb 29 in noleap → Feb 28
            timestamps.append(pd.Timestamp(t.year, 2, 28,
                                           t.hour, t.minute, t.second))
    return np.array(timestamps, dtype="datetime64[ns]")


def decode_times_from_ds(ds):
    """Extract and convert the time axis from a NEMO5 dataset."""
    for tvar in ["time_counter", "time"]:
        if tvar not in ds:
            continue
        raw   = ds[tvar].values
        units = ds[tvar].attrs.get("units", "")
        cal   = ds[tvar].attrs.get("calendar", "noleap")
        try:
            decoded = cftime.num2date(raw, units=units, calendar=cal)
            return cftime_to_datetime64(decoded)
        except Exception:
            pass
    # Fallback: just use indices
    n = ds.dims.get("time_counter", ds.dims.get("time", 1))
    return np.arange(n).astype("datetime64[ns]")


def area_weighted_mean(data_box, weights):
    """
    Area-weighted mean over box spatial dims.
    data_box: (..., h, w)
    weights:  (h, w)
    Returns:  (...,)
    """
    w = weights[np.newaxis] if data_box.ndim == 3 else weights
    # Broadcast weights to data shape
    finite = np.isfinite(data_box)
    w_valid = np.where(finite, weights, 0.0)
    total_w = w_valid.sum(axis=(-2, -1))
    total_w = np.where(total_w > 0, total_w, np.nan)
    weighted = np.where(finite, data_box * weights, 0.0).sum(axis=(-2, -1))
    return weighted / total_w


def depth_integrate_box(data, e3t, y0, x0, h, w):
    """
    Depth-integrate a 4D field (time, z, y, x) over the box.
    Returns (time,) timeseries.
    """
    data_box = data[:, :, y0:y0+h, x0:x0+w]   # (time, z, h, w)
    e3t_box  = e3t[:,    y0:y0+h, x0:x0+w]    # (z, h, w)

    # Mask fill values
    data_box = np.where(np.abs(data_box) < 1e19, data_box, np.nan)

    # depth-integral at each (time, h, w): sum over z
    depth_int = np.nansum(data_box * e3t_box[np.newaxis], axis=1)  # (time, h, w)
    depth_int = np.where(depth_int == 0, np.nan, depth_int)

    return depth_int   # return full (time, h, w) — caller does area weighting


def get_surface(ds, varname, depth_dim):
    """Select surface level of a 4D variable."""
    if varname not in ds:
        return None
    da = ds[varname]
    if depth_dim and depth_dim in da.dims:
        da = da.isel({depth_dim: 0})
    return da.squeeze().values  # (time, y, x) or (time, y, x)


def find_dims(ds, var_list):
    """Identify time, depth, y, x dim names from the first available variable."""
    for v in var_list:
        if v in ds:
            dims = ds[v].dims
            if len(dims) == 4:
                return dims[0], dims[1], dims[2], dims[3]
            elif len(dims) == 3:
                return dims[0], None, dims[1], dims[2]
    return "time_counter", None, "y", "x"


def find_years(model: str) -> list:
    pattern = str(NEMO5_DIR / model / "ORCA2_7d_*_grid_T.nc")
    years = sorted(set(
        int(m.group(1))
        for f in glob.glob(pattern)
        for m in [re.search(r"_7d_(\d{4})0101_", Path(f).name)]
        if m
    ))
    return years


# ---------------------------------------------------------------------------
# Per-year extraction functions
# ---------------------------------------------------------------------------

def process_grid_T(model: str, year: int, out_dir: Path,
                    weights: np.ndarray) -> bool:
    out_path = out_dir / f"{model}_ts_grid_T_{year}.nc"
    if out_path.exists():
        print(f"    grid_T {year}: already exists — skipping.")
        return True

    fpath = NEMO5_DIR / model / f"ORCA2_7d_{year}0101_{year}1231_grid_T.nc"
    if not fpath.exists():
        print(f"    grid_T {year}: file not found — skipping.")
        return False

    print(f"    grid_T {year}…")
    ds = xr.open_dataset(fpath, decode_times=False)
    times = decode_times_from_ds(ds)
    time_dim, depth_dim, y_dim, x_dim = find_dims(ds, GRID_T_VARS)

    out_vars = {}

    for v in ["tos", "sos"]:
        if v not in ds:
            continue
        data = ds[v].values  # (time, y, x) — 2D fields
        if data.ndim == 4:   # just in case
            data = data[:, 0, :, :]
        box = data[:, BOX_Y0:BOX_Y0+BOX_H, BOX_X0:BOX_X0+BOX_W]
        box = np.where(np.abs(box) < 1e19, box, np.nan)
        ts  = area_weighted_mean(box, weights)
        out_vars[v] = xr.DataArray(ts, dims=["time"],
                                    attrs=ds[v].attrs)

    # MLD — native model variable, no criterion conversion
    if "mldr10_1" in ds:
        data = ds["mldr10_1"].values
        if data.ndim == 4:
            data = data[:, 0, :, :]
        box = data[:, BOX_Y0:BOX_Y0+BOX_H, BOX_X0:BOX_X0+BOX_W]
        box = np.where(np.abs(box) < 1e19, box, np.nan)
        ts  = area_weighted_mean(box, weights)
        out_vars["mldr10_1"] = xr.DataArray(
            ts, dims=["time"],
            attrs={"long_name": "Mixed layer depth (10m ref, density criterion)",
                   "units": "m"})

    ds.close()
    _save(out_vars, times, out_path, model, year, "grid_T")
    return True


def process_ptrc_T(model: str, year: int, out_dir: Path,
                    weights: np.ndarray, e3t: np.ndarray) -> bool:
    out_path = out_dir / f"{model}_ts_ptrc_T_{year}.nc"
    if out_path.exists():
        print(f"    ptrc_T {year}: already exists — skipping.")
        return True

    fpath = NEMO5_DIR / model / f"ORCA2_7d_{year}0101_{year}1231_ptrc_T.nc"
    if not fpath.exists():
        print(f"    ptrc_T {year}: file not found — skipping.")
        return False

    print(f"    ptrc_T {year}…")
    ds = xr.open_dataset(fpath, decode_times=False)
    times = decode_times_from_ds(ds)
    time_dim, depth_dim, y_dim, x_dim = find_dims(ds, PTRC_VARS)

    out_vars = {}

    # Surface scalars
    for v in PTRC_SURFACE:
        if v not in ds:
            continue
        data = ds[v].values
        if data.ndim == 4 and depth_dim:
            data = data[:, 0, :, :]
        box = data[:, BOX_Y0:BOX_Y0+BOX_H, BOX_X0:BOX_X0+BOX_W]
        box = np.where(np.abs(box) < 1e19, box, np.nan)
        ts  = area_weighted_mean(box, weights)
        out_vars[v] = xr.DataArray(ts, dims=["time"], attrs=ds[v].attrs)

    # Depth-integrated plankton
    for v in PTRC_DEPTHINT:
        if v not in ds:
            continue
        data = ds[v].values  # (time, z, y, x)
        if data.ndim != 4:
            continue
        data = np.where(np.abs(data) < 1e19, data, np.nan)
        box_data = data[:, :, BOX_Y0:BOX_Y0+BOX_H, BOX_X0:BOX_X0+BOX_W]
        e3t_box  = e3t[:,    BOX_Y0:BOX_Y0+BOX_H, BOX_X0:BOX_X0+BOX_W]
        # Depth integral at each (time, h, w)
        depth_int = np.nansum(box_data * e3t_box[np.newaxis], axis=1)  # (t, h, w)
        depth_int = np.where(depth_int == 0, np.nan, depth_int)
        ts = area_weighted_mean(depth_int, weights)
        attrs = dict(ds[v].attrs)
        attrs["units"] = "mmol m-2"
        attrs["long_name"] = f"{v} depth-integrated, area-weighted box mean"
        out_vars[v] = xr.DataArray(ts, dims=["time"], attrs=attrs)

    ds.close()
    _save(out_vars, times, out_path, model, year, "ptrc_T")
    return True


def process_diad_T(model: str, year: int, out_dir: Path,
                    weights: np.ndarray, e3t: np.ndarray) -> bool:
    out_path = out_dir / f"{model}_ts_diad_T_{year}.nc"
    if out_path.exists():
        print(f"    diad_T {year}: already exists — skipping.")
        return True

    fpath = NEMO5_DIR / model / f"ORCA2_7d_{year}0101_{year}1231_diad_T.nc"
    if not fpath.exists():
        print(f"    diad_T {year}: file not found — skipping.")
        return False

    print(f"    diad_T {year}…")
    ds = xr.open_dataset(fpath, decode_times=False)
    times = decode_times_from_ds(ds)
    time_dim, depth_dim, y_dim, x_dim = find_dims(ds, DIAD_VARS)

    out_vars = {}

    # Surface fields (TChl needs unit conversion: g/L → mg m-3)
    for v in DIAD_SURFACE:
        if v not in ds:
            continue
        data = ds[v].values
        if data.ndim == 4 and depth_dim:
            data = data[:, 0, :, :]
        data = np.where(np.abs(data) < 1e19, data, np.nan)
        box = data[:, BOX_Y0:BOX_Y0+BOX_H, BOX_X0:BOX_X0+BOX_W]
        ts  = area_weighted_mean(box, weights)
        attrs = dict(ds[v].attrs)
        if v == "TChl":
            ts = ts * 1e6   # g Chl/L → mg Chl m-3
            attrs["units"] = "mg m-3"
        out_vars[v] = xr.DataArray(ts, dims=["time"], attrs=attrs)

    # PPINT — already depth-integrated in the model, just area-weight
    for v in DIAD_DEPTHINT:
        if v not in ds:
            continue
        data = ds[v].values
        if data.ndim == 4 and depth_dim:
            # depth-integrate if somehow 4D
            data = np.where(np.abs(data) < 1e19, data, np.nan)
            data = np.nansum(data * e3t[np.newaxis], axis=1)
        else:
            if data.ndim == 3:
                data = np.where(np.abs(data) < 1e19, data, np.nan)
        box = data[:, BOX_Y0:BOX_Y0+BOX_H, BOX_X0:BOX_X0+BOX_W]
        ts  = area_weighted_mean(box, weights)
        out_vars[v] = xr.DataArray(ts, dims=["time"], attrs=ds[v].attrs)

    ds.close()
    _save(out_vars, times, out_path, model, year, "diad_T")
    return True


def _save(out_vars: dict, times: np.ndarray, out_path: Path,
          model: str, year: int, ftype: str):
    """Build and write the output dataset with a clean datetime64 time axis."""
    if not out_vars:
        print(f"      No variables extracted for {ftype} {year} — skipping save.")
        return

    ds_out = xr.Dataset(out_vars)
    ds_out = ds_out.assign_coords({"time": times})
    ds_out.attrs = {
        "model":       model,
        "year":        year,
        "source_type": ftype,
        "box":         f"x={BOX_X0}:{BOX_X0+BOX_W}, y={BOX_Y0}:{BOX_Y0+BOX_H}",
        "description": "Area-weighted regional box timeseries",
    }
    ds_out.to_netcdf(out_path)
    print(f"      Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",      required=True)
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end",   type=int, default=None)
    parser.add_argument("--out-dir",    type=Path, default=None)
    parser.add_argument("--skip-grid-t",  action="store_true")
    parser.add_argument("--skip-ptrc-t",  action="store_true")
    parser.add_argument("--skip-diad-t",  action="store_true")
    args = parser.parse_args()

    out_dir = (args.out_dir or CLIMS_DIR / args.model) / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve years
    all_years = find_years(args.model)
    if not all_years:
        raise RuntimeError(f"No grid_T files found for {args.model} in {NEMO5_DIR}.")

    year_start = args.year_start or all_years[0]
    year_end   = args.year_end   or all_years[-1]
    years = [y for y in all_years if year_start <= y <= year_end]
    print(f"\nModel: {args.model}")
    print(f"Years: {year_start}–{year_end} ({len(years)} years found)")
    print(f"Output: {out_dir}")

    print("\nLoading mesh mask…")
    weights = load_area_weights()
    e3t     = load_e3t()

    for year in years:
        print(f"\n  Year {year}:")
        if not args.skip_grid_t:
            process_grid_T(args.model, year, out_dir, weights)
        if not args.skip_ptrc_t:
            process_ptrc_T(args.model, year, out_dir, weights, e3t)
        if not args.skip_diad_t:
            process_diad_T(args.model, year, out_dir, weights, e3t)

    print(f"\nDone. Files in {out_dir}")


if __name__ == "__main__":
    main()
