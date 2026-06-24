#!/usr/bin/env python3
"""
prep_bgc_obs.py
One-off preprocessing of BGC observational climatologies for the NA monitor.

Run this once; the output files are model-independent and are read by
plot_bgc_monitor.py for all models.

Processes:
  - TChl : OC-CCI v5 monthly climatology  → annual mean, r360x180
  - NO3  : WOA18 (already on ORCA2 grid)  → attach nav_lon/lat, regrid r360x180
  - PO4  : WOA18 (already on ORCA2 grid)  → attach nav_lon/lat, regrid r360x180

Output (all to OBS_OUT_DIR):
  bgc_obs_tchl_annual_r360x180.nc
  bgc_obs_no3_r360x180.nc
  bgc_obs_po4_r360x180.nc

Usage:
    python prep_bgc_obs.py
    python prep_bgc_obs.py --out-dir /my/obs/dir
    python prep_bgc_obs.py --skip-tchl   # if TChl already done
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Paths — edit if files move
# ---------------------------------------------------------------------------

OBS_OUT_DIR = Path("/gpfs/data/greenocean/users/mep22dku/clims/obs_bgc")

OBS_CHL_FILE = Path(
    "/gpfs/data/greenocean/observations/CHL/OCCCI/climatology_OCCCIv5_monthly.nc"
)

WOA_DIR      = Path("/gpfs/home/avd22gnu/scratch/WOA/scripts/unitsConv")
WOA_NO3_FILE = WOA_DIR / "woa18_all_n00_01_regridORCA_converted.nc"
WOA_PO4_FILE = WOA_DIR / "woa18_all_p00_01_regridORCA_converted.nc"

MESH_MASK = Path(
    "/gpfs/data/greenocean/software/resources/CDFTOOLS/mesh_mask3_6.nc"
)

CDO_TARGET_GRID = "r360x180"

# ---------------------------------------------------------------------------

def run(cmd):
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def prep_tchl(out_dir: Path) -> Path:
    """
    OC-CCI v5: compute annual mean across the 12 monthly climatology fields,
    regrid to r360x180. The file is already on a regular 1° grid so CDO
    can handle it directly without curvilinear coordinate tricks.
    """
    out_path = out_dir / "bgc_obs_tchl_annual_r360x180.nc"
    if out_path.exists():
        print(f"  TChl obs already exists: {out_path} — skipping.")
        return out_path

    print("  Processing TChl (OC-CCI v5)…")

    ds = xr.open_dataset(OBS_CHL_FILE, decode_times=False)
    chl = ds["TCHL_CLIM"]   # (TMONTH, AY, AX)
    lon = ds["AX"].values
    lat = ds["AY"].values

    annual_mean = chl.mean(dim="TMONTH").values
    annual_mean[annual_mean <= 0] = np.nan

    # Already on a regular grid — write with standard lon/lat dims so CDO
    # (if needed downstream) and xarray can both read it cleanly.
    ds_out = xr.Dataset({
        "TChl": xr.DataArray(
            annual_mean, dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
            attrs={"long_name": "Total chlorophyll, annual mean",
                   "units": "mg m-3",
                   "source": "OC-CCI v5 (1998-2020)"},
        )
    })
    ds.close()
    ds_out.to_netcdf(out_path)
    print(f"  TChl written: {out_path}")
    return out_path


def _prep_woa_nutrient(label: str, src_file: Path, src_var: str,
                       nav_lon, nav_lat, out_dir: Path) -> Path:
    """
    Attach curvilinear nav_lon/nav_lat to a WOA18 nutrient file that is
    already on the ORCA2 grid, then regrid to r360x180 with CDO.
    """
    out_path = out_dir / f"bgc_obs_{label.lower()}_r360x180.nc"
    if out_path.exists():
        print(f"  {label} obs already exists: {out_path} — skipping.")
        return out_path

    if not src_file.exists():
        print(f"  WARNING: {src_file} not found — skipping {label}.")
        return None

    print(f"  Processing {label} (WOA18)…")

    ds = xr.open_dataset(src_file, decode_times=False)

    # Find the variable — skip 1D coordinate variables (X, Y, Z, TIME_COUNTER)
    if src_var in ds:
        data = ds[src_var]
    else:
        # Pick the first data var that is at least 2D
        candidates = [v for v in ds.data_vars if ds[v].ndim >= 2]
        if not candidates:
            raise RuntimeError(f"No 2D+ variables found in {src_file}")
        src_var = candidates[0]
        print(f"    Using variable '{src_var}'")
        data = ds[src_var]

    # Surface level (z=0), single time record.
    # Force selection on every possible depth/time dim name, then squeeze hard.
    for dim in list(data.dims):
        if dim.lower() in ("time_counter", "time", "t"):
            data = data.isel({dim: 0})
        elif dim.lower() in ("z", "depth", "deptht", "z_t", "lev"):
            data = data.isel({dim: 0})
    data = data.squeeze()   # should now be (y, x)
    if data.ndim != 2:
        raise RuntimeError(
            f"Expected 2D surface slice for {label} but got shape {data.shape} "
            f"with dims {data.dims}. Check the file structure."
        )
    ds.close()

    # Build a temporary file with curvilinear coords attached.
    # CDO requires the 2D coordinate variables to be named "lon"/"lat" with
    # standard units and the "coordinates" attribute set on the data variable
    # — otherwise it reports "Unsupported generic coordinates".
    tmp_in  = Path(tempfile.mktemp(suffix=f"_woa_{label}_curv.nc"))
    tmp_out = Path(tempfile.mktemp(suffix=f"_woa_{label}_rg.nc"))

    import netCDF4 as nc4
    with nc4.Dataset(tmp_in, "w") as ds_nc:
        ds_nc.createDimension("y", nav_lat.shape[0])
        ds_nc.createDimension("x", nav_lat.shape[1])
        ds_nc.createDimension("time", 1)

        v_lon = ds_nc.createVariable("lon", "f4", ("y", "x"))
        v_lon.units         = "degrees_east"
        v_lon.standard_name = "longitude"
        v_lon[:]            = nav_lon

        v_lat = ds_nc.createVariable("lat", "f4", ("y", "x"))
        v_lat.units         = "degrees_north"
        v_lat.standard_name = "latitude"
        v_lat[:]            = nav_lat

        v_data = ds_nc.createVariable(label, "f4", ("time", "y", "x"),
                                      fill_value=1e20)
        v_data.coordinates  = "lon lat"
        v_data.long_name    = f"{label} surface concentration"
        v_data.missing_value = 1e20
        v_data[0, :, :]     = data.values

    run([
        "cdo", f"remapbil,{CDO_TARGET_GRID}",
        str(tmp_in), str(tmp_out),
    ])
    tmp_in.unlink(missing_ok=True)

    # Load regridded result
    ds_rg = xr.open_dataset(tmp_out, decode_times=False)
    rg_var = label if label in ds_rg else list(ds_rg.data_vars)[0]
    arr   = ds_rg[rg_var].squeeze().values
    lon   = ds_rg["lon"].values
    lat   = ds_rg["lat"].values
    ds_rg.close()
    tmp_out.unlink(missing_ok=True)

    arr[arr <= 0] = np.nan
    ds_clean = xr.Dataset({
        label: xr.DataArray(
            arr, dims=["lat", "lon"],
            coords={"lat": lat, "lon": lon},
            attrs={"long_name": f"{label} surface concentration, annual mean",
                   "units": "mmol m-3",
                   "source": "WOA18 (annual mean, converted to model units)"},
        )
    })
    ds_clean.to_netcdf(out_path)
    print(f"  {label} written: {out_path}")
    return out_path


def prep_nutrients(out_dir: Path):
    """Load mesh mask coords and regrid both WOA18 nutrient files."""
    print("  Loading nav_lon/nav_lat from mesh mask…")
    ds_mesh  = xr.open_dataset(MESH_MASK, decode_times=False)
    nav_lon  = ds_mesh["nav_lon"].values
    nav_lat  = ds_mesh["nav_lat"].values
    ds_mesh.close()

    _prep_woa_nutrient("NO3", WOA_NO3_FILE, "NO3", nav_lon, nav_lat, out_dir)
    _prep_woa_nutrient("PO4", WOA_PO4_FILE, "PO4", nav_lon, nav_lat, out_dir)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-dir",    type=Path, default=OBS_OUT_DIR)
    parser.add_argument("--skip-tchl", action="store_true")
    parser.add_argument("--skip-nuts", action="store_true",
                        help="Skip NO3/PO4 regridding")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_tchl:
        print("\n[1/2] TChl (OC-CCI v5)…")
        prep_tchl(args.out_dir)

    if not args.skip_nuts:
        print("\n[2/2] Nutrients (WOA18)…")
        prep_nutrients(args.out_dir)

    print("\nDone. Obs files in:", args.out_dir)


if __name__ == "__main__":
    main()
