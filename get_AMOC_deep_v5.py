import sys
import xarray as xr
import numpy as np
import pandas as pd
import glob
import re
from pathlib import Path

# ===== INPUTS =====

baseDir   = '/gpfs/data/greenocean/software/resources/CDFTOOLS/MOCresults/'
clims_dir = '/gpfs/data/greenocean/users/mep22dku/clims/'

# AMOC output periods. A file's year decides which (if any) period it
# falls into; "end=None" means "no upper bound". Periods may overlap —
# each one is written out as its own independent pair of files.
PERIODS = [
    {'label': 'pre1940',  'start': 1750, 'end': 1940},  # 1750 <= year < 1940
    {'label': 'post1940', 'start': 1940, 'end': None},  # year >= 1940
    {'label': 'full',     'start': 1000, 'end': None},  # everything from 1000 onward
]

# ===== FUNCTIONS =====

def compute_amoc_one_file(fpath, nav_lat, depth_vals, deep_idx, north_idx):
    """Process a single MOC file, return (time, amoc_26n, amoc_north, depth_at_max, lat_at_max)."""
    ds = xr.open_dataset(fpath)
    da = ds.zomsfatl.squeeze()  # (time_counter, depthw, y)

    y_26n = int(np.abs(nav_lat - 26.5).argmin())

    atl_26n_deep = da.isel(y=y_26n, depthw=deep_idx).where(lambda x: x != 0)
    amoc_26n     = atl_26n_deep.max(dim='depthw').values

    atl_north  = da.isel(depthw=deep_idx, y=north_idx).where(lambda x: x != 0)
    amoc_north = atl_north.max(dim=['depthw', 'y']).values

    vals        = atl_north.values               # (time, n_deep, n_north)
    n_time, n_deep, n_north = vals.shape
    vals_masked = np.ma.masked_invalid(vals)
    flat_idx    = np.ma.argmax(vals_masked.reshape(n_time, -1), axis=1)
    deep_pos, north_pos = np.unravel_index(flat_idx, (n_deep, n_north))
    depth_at_max = depth_vals[deep_idx[deep_pos]]
    lat_at_max   = nav_lat[north_idx[north_pos]]

    time_pd = pd.to_datetime(
        [pd.Timestamp(t.isoformat()) for t in ds.time_counter.values]
    )

    ds.close()
    return time_pd, amoc_26n, amoc_north, depth_at_max, lat_at_max


def get_available_files(model, baseDir):
    """
    Glob all MOC files for this model and return them sorted by year, along
    with the year extracted from each filename.
    Filename format: {model}_7d_{year}0101_{year}1231_MOC.nc

    Returns: list of (fpath, year) tuples sorted by year, overall yrst, yrend
    """
    pattern   = f'{baseDir}{model}_7d_*MOC.nc'
    all_files = sorted(glob.glob(pattern))

    file_years = []
    for f in all_files:
        m = re.search(r'_7d_(\d{4})0101_', Path(f).name)
        year = int(m.group(1)) if m else None
        file_years.append((f, year))

    # Sort by year (files without a parseable year go last)
    file_years.sort(key=lambda fy: (fy[1] is None, fy[1]))

    years = [y for _, y in file_years if y is not None]
    yrst  = min(years) if years else None
    yrend = max(years) if years else None

    return file_years, yrst, yrend


def select_period_files(file_years, start, end):
    """Filter (fpath, year) tuples to those with start <= year < end (end=None means no upper bound)."""
    selected = []
    for fpath, year in file_years:
        if year is None:
            continue
        if year < start:
            continue
        if end is not None and year >= end:
            continue
        selected.append((fpath, year))
    return selected


def process_period(model, period_label, period_files, output_dir, nav_lat, depth_vals, deep_idx, north_idx):
    """Run the AMOC computation over one period's files and write (a) and (b) outputs.
    Returns True if outputs were written, False if there was nothing to do."""
    if not period_files:
        print(f"  [{period_label}] No files available for this period — skipping.")
        return False

    files  = [f for f, _ in period_files]
    pyears = [y for _, y in period_files]
    yrst, yrend = min(pyears), max(pyears)
    print(f"  [{period_label}] {len(files)} files ({yrst}-{yrend})")

    all_time, all_26n, all_north, all_depth, all_lat = [], [], [], [], []
    for fpath in files:
        try:
            time_pd, amoc_26n, amoc_north, depth_at_max, lat_at_max = \
                compute_amoc_one_file(fpath, nav_lat, depth_vals, deep_idx, north_idx)
            all_time.append(time_pd)
            all_26n.append(amoc_26n)
            all_north.append(amoc_north)
            all_depth.append(depth_at_max)
            all_lat.append(lat_at_max)
            print(f"  [{period_label}] processed {Path(fpath).name}")
        except Exception as e:
            print(f"  [{period_label}] ERROR on {Path(fpath).name}: {e}")

    if not all_time:
        print(f"  [{period_label}] No data processed.")
        return False

    time_pd        = np.concatenate(all_time)
    amoc_26n_arr   = np.concatenate(all_26n)
    amoc_north_arr = np.concatenate(all_north)
    depth_arr      = np.concatenate(all_depth)
    lat_arr        = np.concatenate(all_lat)

    # ================================================================
    # Save (a): AMOC at 26.5°N
    # ================================================================
    amoc_26n_da = xr.DataArray(
        amoc_26n_arr, coords={'time_counter': time_pd}, dims='time_counter', name='AMOC_26N'
    )
    ds_26n = amoc_26n_da.to_dataset()
    ds_26n.attrs.update({
        'made_in':       '/gpfs/home/mep22dku/scratch/AMOC-PLANKTOM/hosing/AMOC-NEMOv5.py',
        'source_years':  f'{yrst}-{yrend}',
        'source_model':  model,
        'period':        period_label,
        'description':   f'Maximum Atlantic overturning at 26.5N below 500 m depth ({yrst}-{yrend})',
    })
    out_26n = output_dir / f'{model}_AMOC_26N_{yrst}_{yrend}.nc'
    ds_26n.to_netcdf(out_26n)
    print(f"  [{period_label}] (a) Saved to {out_26n}")

    # ================================================================
    # Save (b): Basin-wide AMOC north of 0°N + location
    # ================================================================
    ds_north = xr.Dataset({
        'AMOC_north':       xr.DataArray(amoc_north_arr, coords={'time_counter': time_pd}, dims='time_counter'),
        'AMOC_north_depth': xr.DataArray(depth_arr, coords={'time_counter': time_pd}, dims='time_counter',
                                         attrs={'units': 'm', 'long_name': 'Depth of maximum AMOC north of 0N'}),
        'AMOC_north_lat':   xr.DataArray(lat_arr,   coords={'time_counter': time_pd}, dims='time_counter',
                                         attrs={'units': 'degrees_north', 'long_name': 'Latitude of maximum AMOC north of 0N'}),
    })
    ds_north.attrs.update({
        'made_in':      '/gpfs/home/mep22dku/scratch/AMOC-PLANKTOM/hosing/AMOC-NEMOv5.py',
        'source_years': f'{yrst}-{yrend}',
        'source_model': model,
        'period':       period_label,
        'description':  f'Maximum Atlantic overturning north of 0N and below 500 m depth, with location per timestep ({yrst}-{yrend}).',
    })
    out_north = output_dir / f'{model}_AMOC_north_{yrst}_{yrend}.nc'
    ds_north.to_netcdf(out_north)
    print(f"  [{period_label}] (b) Saved to {out_north}")

    return True


def compute_amoc_timeseries(model, baseDir, clims_dir):
    output_dir = Path(clims_dir) / model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available files
    file_years, yrst, yrend = get_available_files(model, baseDir)

    if not file_years:
        print(f"  No MOC files found for {model} in {baseDir}")
        return None

    print(f"  Found {len(file_years)} files  ({yrst}-{yrend})")

    # Get coordinate arrays from the earliest available file
    first_file = file_years[0][0]
    ds0        = xr.open_dataset(first_file)
    da0        = ds0.zomsfatl.squeeze()
    nav_lat    = ds0['nav_lat'].values.flatten()
    depth_vals = da0['depthw'].values
    deep_idx   = np.where(depth_vals < -500)[0]
    north_idx  = np.where(nav_lat > 0)[0]
    y_26n      = int(np.abs(nav_lat - 26.5).argmin())
    lat_actual = nav_lat[y_26n]
    print(f"  Nearest lat to 26.5N: {lat_actual:.3f} (y={y_26n})")
    ds0.close()

    # Process each period independently, writing its own pair of output files
    any_written = False
    for period in PERIODS:
        period_files = select_period_files(file_years, period['start'], period['end'])
        written = process_period(
            model, period['label'], period_files, output_dir,
            nav_lat, depth_vals, deep_idx, north_idx,
        )
        any_written = any_written or written

    if not any_written:
        print("  No AMOC output produced for any period.")
        return None

    return True


# ===== RUN =====

if len(sys.argv) != 2:
    print("Usage: python get_AMOC_deep_v5.py <MODEL_NAME>")
    print("  e.g. python get_AMOC_deep_v5.py TOM12_TJ_PF04")
    sys.exit(1)

model = sys.argv[1]

print(f"\n{'='*60}")
print(f"Processing AMOC for model: {model}")
print(f"{'='*60}")

try:
    result = compute_amoc_timeseries(model, baseDir, clims_dir)
    if result is None:
        sys.exit(1)
except Exception as e:
    print(f"ERROR processing {model}: {e}")
    sys.exit(1)

print(f"\n{'='*60}")
print("Done.")
print(f"{'='*60}")
