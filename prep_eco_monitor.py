#!/usr/bin/env python3
"""
prep_eco_monitor.py
Preprocessing for the North Atlantic ecosystem monitor.

Pipeline:
  Step 1 — Build 52-slot weekly climatology from last N years of ptrc_T
            files, depth-integrate using mesh_mask_v5.nc, save:
            clims/<MODEL>/<MODEL>_eco_clim_depthint_YYYY_YYYY.nc

  Step 2 — Load depth-integrated climatology, compute annual means,
            compute DOMINANT_PHYTO and DOMINANT_ZOO per gridcell,
            regrid everything to r360x180 with CDO, save:
            clims/<MODEL>/<MODEL>_eco_monitor_maps_YYYY_YYYY.nc

Usage:
    python prep_eco_monitor.py --model TOM12_TJ_HA00
    python prep_eco_monitor.py --model TOM12_TJ_HA00 --n-years 5
    python prep_eco_monitor.py --model TOM12_TJ_HA00 --skip-step1
    python prep_eco_monitor.py --model TOM12_TJ_HA00 --year-start 1960 --year-end 1965
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
NEMO5_DIR  = Path("/gpfs/data/greenocean/software/runs/NEMO5")
CLIMS_DIR  = Path("/gpfs/data/greenocean/users/mep22dku/clims")
MESH_MASK  = Path("/gpfs/data/greenocean/software/resources/CDFTOOLS/mesh_mask_v5.nc")

CDO_TARGET_GRID = "r360x180"

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
PHYTO_VARS = ['DIA', 'MIX', 'COC', 'PIC', 'PHA', 'FIX']
ZOO_VARS   = ['BAC', 'PRO', 'PTE', 'MES', 'GEL', 'CRU']
ALL_VARS   = ['BAC', 'PRO', 'PTE', 'MES', 'GEL', 'DIA', 'MIX', 'COC',
              'PIC', 'PHA', 'FIX', 'CRU']

WEEKS_PER_YEAR = 52

# ---------------------------------------------------------------------------
# Regional box for timeseries (indices into the native ORCA2 grid)
# ---------------------------------------------------------------------------
BOX_X0, BOX_Y0, BOX_W, BOX_H = 112, 108, 20, 20  # x0, y0, width, height

# ---------------------------------------------------------------------------

def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def find_years(model: str) -> list:
    pattern = str(NEMO5_DIR / model / "ORCA2_7d_*_ptrc_T.nc")
    years = sorted(set(
        int(m.group(1))
        for f in glob.glob(pattern)
        for m in [re.search(r"_7d_(\d{4})0101_", Path(f).name)]
        if m
    ))
    return years


# ---------------------------------------------------------------------------
# Step 1: climatology + depth integration
# ---------------------------------------------------------------------------

def load_mesh_mask():
    """Load e3t_0 and tmask from NEMO5 mesh mask, squeeze time dim."""
    ds = xr.open_dataset(MESH_MASK, decode_times=False)
    # e3t_0: (t, z, y, x) — squeeze t (singleton)
    e3t  = ds["e3t_0"].squeeze("t", drop=True)   # (z, y, x)
    tmask = ds["tmask"].squeeze("t", drop=True).astype(float)  # (z, y, x)
    ds.close()
    return e3t * tmask   # cell thickness, masked to zero on land


def compute_box_timeseries(depthint_file: Path, out_dir: Path) -> Path:
    """
    From the depth-integrated climatology (week, y, x), extract the
    area-weighted mean over the regional box and save as a timeseries
    (week, n_vars).

    Area weights: e1t * e2t * tmask[0, 0, :, :] from mesh_mask_v5.nc,
    sliced to the box.
    """
    m = re.search(r"_(\d{4})_(\d{4})\.nc$", depthint_file.name)
    year_start, year_end = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    out_path = out_dir / (depthint_file.stem.replace("_eco_clim_depthint_",
                                                      "_eco_box_timeseries_") + ".nc")
    if out_path.exists():
        print(f"  Box timeseries already exists: {out_path.name} — skipping.")
        return out_path

    print(f"  Loading mesh mask for area weights…")
    ds_mesh = xr.open_dataset(MESH_MASK, decode_times=False)
    csize = (ds_mesh["e1t"][0, :, :] *
             ds_mesh["e2t"][0, :, :] *
             ds_mesh["tmask"][0, 0, :, :].astype(float))  # (y, x)
    ds_mesh.close()

    # Slice to box
    x0, y0, w, h = BOX_X0, BOX_Y0, BOX_W, BOX_H
    csize_box = csize.values[y0:y0+h, x0:x0+w]  # (h, w)
    total_weight = csize_box.sum()

    print(f"  Computing area-weighted box timeseries "
          f"(x={x0}:{x0+w}, y={y0}:{y0+h})…")
    ds = xr.open_dataset(depthint_file, decode_times=False)

    out_vars = {}
    for v in ALL_VARS:
        if v not in ds:
            continue
        data = ds[v].values  # (52, ny, nx)
        box  = data[:, y0:y0+h, x0:x0+w]  # (52, h, w)
        # Area-weighted mean over box, ignoring NaN
        weighted = np.where(np.isfinite(box), box * csize_box[np.newaxis, :, :], 0.0)
        weight_valid = np.where(np.isfinite(box), csize_box[np.newaxis, :, :], 0.0)
        ts = weighted.sum(axis=(1, 2)) / np.maximum(weight_valid.sum(axis=(1, 2)), 1e-30)
        ts[weight_valid.sum(axis=(1, 2)) == 0] = np.nan

        out_vars[v] = xr.DataArray(
            ts,
            dims=["week"],
            coords={"week": np.arange(WEEKS_PER_YEAR)},
            attrs={"long_name": f"{v} area-weighted box mean",
                   "units": "mmol m-2",
                   "box_x0": x0, "box_y0": y0,
                   "box_width": w, "box_height": h},
        )
    ds.close()

    ds_out = xr.Dataset(out_vars,
                        attrs={"model": depthint_file.name,
                               "year_start": year_start, "year_end": year_end,
                               "box": f"x={x0}:{x0+w}, y={y0}:{y0+h}",
                               "description": "Area-weighted weekly climatology, "
                                              "depth-integrated, regional box"})
    ds_out.to_netcdf(out_path)
    print(f"  Saved: {out_path.name}")
    return out_path


def build_climatology_and_integrate(model: str, year_start: int, year_end: int,
                                     out_dir: Path) -> Path:
    """
    For each of the 52 weekly slots, average across years, then depth-integrate.
    Output has dims (week=52, y, x) for each variable.
    """
    out_path = out_dir / f"{model}_eco_clim_depthint_{year_start}_{year_end}.nc"
    if out_path.exists():
        print(f"  Depth-integrated climatology already exists: {out_path.name} — skipping.")
        return out_path

    model_dir = NEMO5_DIR / model
    years = list(range(year_start, year_end + 1))

    print(f"  Loading mesh mask…")
    e3t_masked = load_mesh_mask()   # (z, y, x)

    # Accumulate: for each week slot, sum across years
    # We'll store (n_weeks, n_z, n_y, n_x) per variable then sum/count
    accum  = None   # dict varname → (n_weeks, n_z, n_y, n_x) array
    counts = None   # dict varname → (n_weeks,) int array
    n_z = e3t_masked.shape[0]
    nav_lon = nav_lat = depth_dim = None

    for year in years:
        fpath = model_dir / f"ORCA2_7d_{year}0101_{year}1231_ptrc_T.nc"
        if not fpath.exists():
            print(f"  WARNING: {fpath.name} not found — skipping year {year}.")
            continue

        print(f"  Loading {fpath.name}…")
        ds = xr.open_dataset(fpath, decode_times=False)

        # Identify dims — NEMO5 uses VERY long names
        # Find time dim (first dim of first available var)
        sample_var = next((v for v in ALL_VARS if v in ds), None)
        if sample_var is None:
            print(f"  WARNING: no ecosystem vars found in {fpath.name} — skipping.")
            ds.close()
            continue

        dims = ds[sample_var].dims
        time_dim  = dims[0]
        depth_dim_name = dims[1]
        y_dim     = dims[2]
        x_dim     = dims[3]

        n_t = ds.dims[time_dim]

        # Infer week index for each timestep (0-based, wrap at WEEKS_PER_YEAR)
        week_indices = np.arange(n_t) % WEEKS_PER_YEAR

        # Save nav coords from first year
        if nav_lon is None:
            # Look for nav_lon/lat with any suffix
            for v in ds.coords:
                if "nav_lon" in v:
                    nav_lon = ds[v].values
                if "nav_lat" in v:
                    nav_lat = ds[v].values
            depth_dim = depth_dim_name

        # Initialise accumulators from first year's shape
        if accum is None:
            ny = ds.dims[y_dim]
            nx = ds.dims[x_dim]
            accum  = {v: np.zeros((WEEKS_PER_YEAR, n_z, ny, nx), dtype=np.float64)
                      for v in ALL_VARS if v in ds}
            counts = {v: np.zeros(WEEKS_PER_YEAR, dtype=np.int32)
                      for v in ALL_VARS if v in ds}

        for v in list(accum.keys()):
            if v not in ds:
                continue
            data = ds[v].values.astype(np.float64)  # (n_t, n_z, ny, nx)
            data = np.where(np.isfinite(data), data, 0.0)
            for w in range(WEEKS_PER_YEAR):
                mask_w = (week_indices == w)
                if mask_w.any():
                    accum[v][w]  += data[mask_w].mean(axis=0)
                    counts[v][w] += 1

        ds.close()

    if accum is None:
        raise RuntimeError(f"No data loaded for {model} {year_start}–{year_end}.")

    # Divide by counts to get climatological mean per week slot
    print(f"  Computing climatological means and depth-integrating…")
    e3t_np = e3t_masked.values  # (z, y, x)

    out_vars = {}
    for v, arr in accum.items():
        # arr: (52, n_z, ny, nx); counts[v]: (52,)
        c = counts[v][:, np.newaxis, np.newaxis, np.newaxis]
        c = np.where(c > 0, c, 1)
        clim = arr / c   # (52, n_z, ny, nx)

        # Depth-integrate: sum over z of (clim * e3t)
        integrated = (clim * e3t_np[np.newaxis, :, :, :]).sum(axis=1)  # (52, ny, nx)
        integrated[integrated == 0] = np.nan

        out_vars[v] = xr.DataArray(
            integrated,
            dims=["week", "y", "x"],
            attrs={"long_name": f"{v} depth-integrated climatology",
                   "units": "mmol m-2",
                   "years_used": f"{year_start}-{year_end}"},
        )

    ds_out = xr.Dataset(out_vars)
    if nav_lon is not None:
        ds_out["nav_lon"] = xr.DataArray(nav_lon, dims=["y", "x"],
                                          attrs={"units": "degrees_east",
                                                 "standard_name": "longitude"})
        ds_out["nav_lat"] = xr.DataArray(nav_lat, dims=["y", "x"],
                                          attrs={"units": "degrees_north",
                                                 "standard_name": "latitude"})
    ds_out.attrs = {"model": model, "year_start": year_start, "year_end": year_end,
                    "description": "52-slot weekly climatology, depth-integrated"}
    ds_out.to_netcdf(out_path)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Step 2: annual means, dominants, regrid
# ---------------------------------------------------------------------------

def compute_dominants_and_regrid(model: str, depthint_file: Path,
                                  out_dir: Path) -> Path:
    """
    Load the depth-integrated climatology, compute annual means, find
    DOMINANT_PHYTO and DOMINANT_ZOO per gridcell, regrid to r360x180.
    """
    m = re.search(r"_(\d{4})_(\d{4})\.nc$", depthint_file.name)
    year_start, year_end = (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    out_path = out_dir / f"{model}_eco_monitor_maps_{year_start}_{year_end}.nc"
    if out_path.exists():
        print(f"  Eco monitor maps already exist: {out_path.name} — skipping.")
        return out_path

    print(f"  Loading depth-integrated climatology…")
    ds = xr.open_dataset(depthint_file, decode_times=False)

    # Annual means (mean over 52 weeks)
    annual = {}
    for v in ALL_VARS:
        if v in ds:
            annual[v] = ds[v].mean(dim="week").values  # (ny, nx)

    nav_lon = ds["nav_lon"].values if "nav_lon" in ds else None
    nav_lat = ds["nav_lat"].values if "nav_lat" in ds else None
    ds.close()

    ny, nx = next(iter(annual.values())).shape

    # ---- Dominant phyto ----
    phyto_present = [v for v in PHYTO_VARS if v in annual]
    if phyto_present:
        phyto_stack = np.stack([annual[v] for v in phyto_present], axis=0)  # (n, ny, nx)
        phyto_stack = np.where(np.isfinite(phyto_stack), phyto_stack, -np.inf)
        dom_phyto_idx = np.argmax(phyto_stack, axis=0).astype(np.float32)  # (ny, nx)
        # Mask where all values are -inf (land/missing)
        all_missing = np.all(~np.isfinite(np.stack([annual[v] for v in phyto_present])), axis=0)
        dom_phyto_idx[all_missing] = np.nan
    else:
        dom_phyto_idx = np.full((ny, nx), np.nan)
        phyto_present = []

    # ---- Dominant zoo ----
    zoo_present = [v for v in ZOO_VARS if v in annual]
    if zoo_present:
        zoo_stack = np.stack([annual[v] for v in zoo_present], axis=0)
        zoo_stack = np.where(np.isfinite(zoo_stack), zoo_stack, -np.inf)
        dom_zoo_idx = np.argmax(zoo_stack, axis=0).astype(np.float32)
        all_missing = np.all(~np.isfinite(np.stack([annual[v] for v in zoo_present])), axis=0)
        dom_zoo_idx[all_missing] = np.nan
    else:
        dom_zoo_idx = np.full((ny, nx), np.nan)
        zoo_present = []

    # ---- Write temporary curvilinear file for CDO ----
    print(f"  Writing temporary file for CDO regridding…")
    tmp_in  = Path(tempfile.mktemp(suffix="_eco_curv.nc"))
    tmp_out = Path(tempfile.mktemp(suffix="_eco_rg.nc"))

    import netCDF4 as nc4
    with nc4.Dataset(tmp_in, "w") as nc:
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)
        nc.createDimension("time", 1)

        v_lon = nc.createVariable("lon", "f4", ("y", "x"))
        v_lon.units         = "degrees_east"
        v_lon.standard_name = "longitude"
        v_lon[:]            = nav_lon

        v_lat = nc.createVariable("lat", "f4", ("y", "x"))
        v_lat.units         = "degrees_north"
        v_lat.standard_name = "latitude"
        v_lat[:]            = nav_lat

        def write_var(name, data, units, long_name, extra_attrs=None):
            vv = nc.createVariable(name, "f4", ("time", "y", "x"),
                                   fill_value=1e20)
            vv.coordinates   = "lon lat"
            vv.units         = units
            vv.long_name     = long_name
            vv.missing_value = 1e20
            if extra_attrs:
                for k, val in extra_attrs.items():
                    setattr(vv, k, val)
            arr = np.where(np.isfinite(data), data, 1e20)
            vv[0, :, :]  = arr

        for v in ALL_VARS:
            if v in annual:
                write_var(v, annual[v], "mmol m-2",
                          f"{v} depth-integrated annual mean")

        write_var("DOMINANT_PHYTO", dom_phyto_idx, "1",
                  "Index of dominant phytoplankton group",
                  extra_attrs={
                      "index_labels": ", ".join(
                          f"{i}={n}" for i, n in enumerate(phyto_present)),
                      "group_names": " ".join(phyto_present),
                  })

        write_var("DOMINANT_ZOO", dom_zoo_idx, "1",
                  "Index of dominant zooplankton group",
                  extra_attrs={
                      "index_labels": ", ".join(
                          f"{i}={n}" for i, n in enumerate(zoo_present)),
                      "group_names": " ".join(zoo_present),
                  })

    # ---- CDO regrid ----
    # Use nearest-neighbour for all fields: bilinear would interpolate between
    # integer dominant indices producing meaningless fractional values, and
    # can also fail on the curvilinear → regular grid transition.
    print(f"  Regridding to {CDO_TARGET_GRID} (nearest-neighbour)…")
    run(["cdo", f"remapnn,{CDO_TARGET_GRID}", str(tmp_in), str(tmp_out)])
    tmp_in.unlink(missing_ok=True)

    if not tmp_out.exists():
        raise RuntimeError(
            f"CDO regridding failed — output file not created: {tmp_out}. "
            f"Check CDO output above for details."
        )

    # ---- Clean up and write final file ----
    # Load everything into memory before closing so xarray doesn't try to
    # read lazily from a deleted temp file.
    ds_rg = xr.open_dataset(tmp_out, decode_times=False)
    ds_final = xr.Dataset()
    for v in list(ALL_VARS) + ["DOMINANT_PHYTO", "DOMINANT_ZOO"]:
        if v in ds_rg:
            ds_final[v] = ds_rg[v].squeeze(drop=True).load()
    ds_rg.close()
    tmp_out.unlink(missing_ok=True)

    ds_final.attrs = {"model": model, "year_start": year_start,
                      "year_end": year_end,
                      "phyto_groups": " ".join(phyto_present),
                      "zoo_groups":   " ".join(zoo_present)}
    ds_final.to_netcdf(out_path)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",      required=True)
    parser.add_argument("--n-years",    type=int, default=5)
    parser.add_argument("--year-start", type=int, default=None)
    parser.add_argument("--year-end",   type=int, default=None)
    parser.add_argument("--out-dir",    type=Path, default=None)
    parser.add_argument("--skip-step1", action="store_true",
                        help="Skip climatology/depth-integration (use existing file)")
    parser.add_argument("--skip-step2", action="store_true",
                        help="Skip dominant computation and regridding")
    args = parser.parse_args()

    out_dir = args.out_dir or (CLIMS_DIR / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve year range
    if args.year_start is not None and args.year_end is not None:
        year_start, year_end = args.year_start, args.year_end
    else:
        years = find_years(args.model)
        if not years:
            raise RuntimeError(f"No ptrc_T files found for {args.model}.")
        year_end   = years[-1]
        year_start = years[max(0, len(years) - args.n_years)]
        print(f"  Auto-detected years: {years[0]}–{years[-1]}, "
              f"using last {args.n_years}: {year_start}–{year_end}")

    # Step 1
    if not args.skip_step1:
        print(f"\n[Step 1] Building climatology + depth-integrating "
              f"({year_start}–{year_end})…")
        depthint_file = build_climatology_and_integrate(
            args.model, year_start, year_end, out_dir)
    else:
        # Find existing depth-int file
        candidates = sorted(out_dir.glob(
            f"{args.model}_eco_clim_depthint_*.nc"),
            key=lambda f: f.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(
                f"No depth-integrated climatology found in {out_dir}. "
                f"Run without --skip-step1 first.")
        depthint_file = candidates[0]
        print(f"\n[Step 1] Skipped — using {depthint_file.name}")

    # Step 1b: box timeseries
    print(f"\n[Step 1b] Computing regional box timeseries…")
    compute_box_timeseries(depthint_file, out_dir)

    # Step 2
    if not args.skip_step2:
        print(f"\n[Step 2] Computing dominants and regridding…")
        compute_dominants_and_regrid(args.model, depthint_file, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
