"""
Rename grid_V dimension names for CDFTOOLS compatibility.
Skips years where the output file already exists.

Usage: python rename_v5_dims_arg.py <MODEL_NAME>
"""

import sys
import os
import xarray as xr
import glob

MODELRUNS_DIR = "/gpfs/home/mep22dku/scratch/ModelRuns"
MODELRUNS_DIR = "/gpfs/data/greenocean/software/runs/NEMO5"
CLIMS_DIR     = "/gpfs/data/greenocean/users/mep22dku/clims"

def process_grid_V(model_name):
    model_path = os.path.join(MODELRUNS_DIR, model_name)
    out_base   = os.path.join(CLIMS_DIR, model_name)
    os.makedirs(out_base, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0

    for year in range(1000, 2101):
        pattern  = os.path.join(model_path, f"ORCA2_*_{year}0101_{year}1231_grid_V.nc")
        matches  = glob.glob(pattern)
        if not matches:
            continue

        fpath    = matches[0]
        fname    = os.path.basename(fpath)
        out_path = os.path.join(out_base, fname)

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            ds = xr.open_dataset(fpath)

            drop_vars = [v for v in ["x_grid_T", "y_grid_T", "nav_lat_grid_T",
                                      "nav_lon_grid_T", "vos", "tauvo"] if v in ds]
            ds = ds.drop_vars(drop_vars)

            rename_map = {}
            if "x_grid_V"       in ds.dims: rename_map["x_grid_V"]       = "x"
            if "y_grid_V"       in ds.dims: rename_map["y_grid_V"]       = "y"
            if "nav_lat_grid_V" in ds:      rename_map["nav_lat_grid_V"] = "nav_lat"
            if "nav_lon_grid_V" in ds:      rename_map["nav_lon_grid_V"] = "nav_lon"
            ds = ds.rename(rename_map)

            ds.attrs["history"] = "dims renamed for CDFTOOLS compatibility"
            ds.to_netcdf(out_path)
            ds.close()
            print(f"  saved {fname}")
            done += 1

        except Exception as e:
            print(f"  ERROR on {fname}: {e}")
            failed += 1

    print(f"  Rename complete: {done} new, {skipped} already done, {failed} failed")
    # Return counts for the calling script to parse
    print(f"  RENAME_COUNTS:{done}:{skipped}:{failed}")
    return failed == 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python rename_v5_dims_arg.py <MODEL_NAME>")
        sys.exit(1)
    success = process_grid_V(sys.argv[1])
    sys.exit(0 if success else 1)
