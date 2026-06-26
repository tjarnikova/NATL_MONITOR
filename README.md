# NA Monitor Suite

Scripts for generating physical and biogeochemical diagnostic monitors for
PlankTOM/NEMO5 runs, focusing on the Subpolar Gyre region. All scripts live in:

    /gpfs/home/mep22dku/scratch/AMOC-PLANKTOM/hosing/AMOCtun/MONITOR/

Outputs go to `MONITOR/plots/`. Preprocessed model fields go to
`/gpfs/data/greenocean/users/mep22dku/clims/<MODEL>/`.

## Example output

[v0.1 monitor (TOM12_TJ_HA00, 1995–2000)](https://tjarnikova.github.io/sklad/TOM12_TJ_HA00_monitor_1995_2000/TOM12_TJ_HA00_monitor_1995_2000.html)

---

## Quick start

**One-button submission (physical + BGC):**

```bash
sbatch run_monitor.bsub --model TOM12_TJ_HA00
```

**Run only part of the pipeline:**

```bash
sbatch run_monitor.bsub --model TOM12_TJ_HA00 --skip-bgc      # physical only
sbatch run_monitor.bsub --model TOM12_TJ_HA00 --skip-physical  # BGC only
sbatch run_monitor.bsub --model TOM12_TJ_HA00 --skip-amoc      # skip AMOC pipeline (maps + BGC only)
```

**BGC obs preprocessing (run once, ever — not per model):**

```bash
python prep_bgc_obs.py
```

---

## File inventory

### Submission scripts

#### `run_monitor.bsub`
The single entry point. Submits a SLURM job that runs the full pipeline in
order: AMOC preprocessing → physical maps → physical plot → BGC maps → BGC
plot. Loads the correct gcc/netcdf modules for cdfmoc.

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | required | Model name, e.g. `TOM12_TJ_HA00` |
| `--month` | 3 | Month for physical maps (March) |
| `--year-start` | auto | Physical map start year |
| `--year-end` | auto | Physical map end year |
| `--n-years` | 5 | Years to average if start/end not given |
| `--skip-amoc` | off | Skip AMOC pipeline (steps 1–3) |
| `--skip-physical` | off | Skip physical monitor entirely |
| `--skip-bgc` | off | Skip BGC monitor entirely |

#### `prep_na_monitor.bsub`
Older standalone bsub for the physical monitor only. Superseded by
`run_monitor.bsub` but kept for reference.

---

### Physical monitor

#### `prep_na_monitor.py`
Preprocesses model fields for the physical monitor.

- Finds the last `--n-years` of `grid_T` files (or uses explicit
  `--year-start`/`--year-end`)
- For each year: CDO `selmon` → `timmean` → `remapbil,r360x180` for
  `mldr10_1` and `sos`
- Applies MLD criterion conversion: `somxl030 = mldr10_1 × 1.3607 − 1.2358`
- Averages years in Python and writes:
  `clims/<MODEL>/<MODEL>_na_monitor_maps_YYYY_YYYY_monMM.nc`
- Also consolidates AMOC timeseries from `AMOC_26N_*.nc` into:
  `clims/<MODEL>/<MODEL>_na_monitor_amoc_timeseries.nc`

Called by `run_monitor.bsub` with `--skip-amoc`. Can be run standalone
(interactive, no module load needed):

```bash
python prep_na_monitor.py --model TOM12_TJ_HA00 --skip-amoc
```

**AMOC pipeline** (steps 1–3, handled by `run_monitor.bsub`):

1. `rename_v5_dims_arg.py MODEL` — renames grid_V dims for CDFTOOLS
   compatibility, writes to `clims/<MODEL>/`
2. `cdfmoc` — runs on each renamed grid_V, writes to
   `CDFTOOLS/MOCresults/<MODEL>_7d_YYYY*_MOC.nc`
3. `get_AMOC_deep_v5.py MODEL` — extracts AMOC timeseries from MOC files,
   writes `clims/<MODEL>/<MODEL>_AMOC_26N_YYYY_YYYY.nc`

#### `plot_na_monitor.py`
Plots the physical monitor figure (3 rows × 2 cols):

- **Row 0 (thin):** AMOC at 26.5°N timeseries (below 500 m), all available
  years. Raw series in blue, 11-point running mean in black. Grey gridline
  at 10 Sv.
- **Row 1:** Obs SSS (WOA 2023) | Model SSS — shared colorbar (`cmo.haline`)
- **Row 2:** Obs MLD (de Boyer Montégut 2004) | Model MLD — shared colorbar
  (`cmo.deep`)

Projection: Lambert Conformal centred on the North Atlantic (avoids inflated
Greenland). Obs dataset labels inset on obs panels. Provenance stamp at bottom.

Auto-detects both the maps file and AMOC file from `clims/<MODEL>/` by name
pattern; picks the AMOC file with the largest year span.

```bash
python plot_na_monitor.py --model TOM12_TJ_HA00
```

**Key optional arguments:** `--month`, `--n-years`, `--year-start`,
`--year-end`, `--mld-vmin/vmax`, `--sal-vmin/vmax`, `--extent`.

**Obs data used:**
- SSS: WOA 2023 (`/gpfs/data/greenocean/observations/WOA/WOA2023/salinity/5564/`)
- MLD: de Boyer Montégut et al. (2004)
  (`/gpfs/data/greenocean/observations/MLD/mld_dr003_ref10m_v2023.nc`)

---

### BGC monitor

#### `prep_bgc_obs.py`
**Run once, ever — not per model.** Preprocesses observational BGC
climatologies and writes to `clims/obs_bgc/`.

- **TChl:** OC-CCI v5 monthly climatology → annual mean → written as
  `bgc_obs_tchl_annual_r360x180.nc` (units: mg Chl m⁻³)
- **NO3:** WOA18 `n_an` on ORCA2 grid → attaches `nav_lon`/`nav_lat` from
  mesh mask → CDO `remapbil,r360x180` → surface level → written as
  `bgc_obs_no3_r360x180.nc`
- **PO4:** Same pipeline as NO3 from WOA18 `p_an` →
  `bgc_obs_po4_r360x180.nc`

```bash
python prep_bgc_obs.py
# If TChl already done:
python prep_bgc_obs.py --skip-tchl
```

**Source files:**
- OC-CCI: `/gpfs/data/greenocean/observations/CHL/OCCCI/climatology_OCCCIv5_monthly.nc`
- WOA18: `/gpfs/home/avd22gnu/scratch/WOA/scripts/unitsConv/woa18_all_n00_01_regridORCA_converted.nc`
- WOA18: `/gpfs/home/avd22gnu/scratch/WOA/scripts/unitsConv/woa18_all_p00_01_regridORCA_converted.nc`
- Mesh mask: `/gpfs/data/greenocean/software/resources/CDFTOOLS/mesh_mask3_6.nc`

#### `prep_bgc_monitor.py`
Preprocesses model BGC fields for the BGC monitor.

- Auto-detects last `--n-years` of `ptrc_T`/`diad_T` files
- For each year and variable: CDO `sellevidx,1` (surface) → `timmean` →
  `remapbil,r360x180`
- Averages years in Python
- Unit conversion: TChl `g Chl/L` → `mg Chl m⁻³` (× 10⁶)
- Deletes any stale maps file before writing fresh output:
  `clims/<MODEL>/<MODEL>_bgc_monitor_maps_YYYY_YYYY_annual.nc`

**Variables processed:**

| Variable | Source file | Notes |
|---|---|---|
| `TChl` | `diad_T` | Total chlorophyll, surface, annual mean |
| `NO3` | `ptrc_T` | Nitrate, surface, annual mean |
| `PO4` | `ptrc_T` | Phosphate, surface, annual mean |
| `PPINT` | `diad_T` | Integrated primary production, annual mean |

```bash
python prep_bgc_monitor.py --model TOM12_TJ_HA00
python prep_bgc_monitor.py --model TOM12_TJ_HA00 --year-start 1950 --year-end 1960
```

#### `plot_bgc_monitor.py`
Plots the BGC monitor figure (2 rows × 4 cols + horizontal colorbars).

- **Row 0:** Model TChl | Model NO3 | Model PO4 | Model PPINT
- **Row 1:** Obs TChl | Obs NO3 | Obs PO4 | (empty — no PPINT obs)
- **Row 2:** Horizontal colorbars, one per column

Colour limits are derived automatically from the 2nd–98th percentile of the
model data, so they are always in the correct model units. The shared
colorscale means obs and model are directly comparable.

Finds the maps file automatically by glob (most recent `*_bgc_monitor_maps_*_annual.nc`),
and reads the year range from the filename so labels are always correct.

```bash
python plot_bgc_monitor.py --model TOM12_TJ_HA00
```

**Obs data used:**
- TChl: OC-CCI v5 (1998–2020), from `clims/obs_bgc/`
- NO3/PO4: WOA18, from `clims/obs_bgc/`

---

## Dependencies

All Python scripts require the `ocean` conda environment:

```bash
source activate ocean
```

Key packages: `xarray`, `numpy`, `matplotlib`, `cartopy`, `cmocean`,
`netCDF4`, `cftime`, `scipy`.

CDO must be available (`module add netcdf/4.7.4/gcc gcc/11.1.0`) — this is
handled automatically by `run_monitor.bsub`.

The `plot_style.py` module (Gill Sans, colour palettes, `set_presentation_style()`)
must be on the path at:

    /gpfs/home/mep22dku/scratch/SOZONE/UTILS/plot_style.py

---

## Directory structure

```
MONITOR/
├── run_monitor.bsub              # One-button submission (physical + BGC)
├── prep_na_monitor.bsub          # Standalone physical-only bsub (legacy)
├── prep_na_monitor.py            # Physical map preprocessing
├── plot_na_monitor.py            # Physical monitor figure
├── prep_bgc_obs.py               # Obs BGC preprocessing (run once)
├── prep_bgc_monitor.py           # BGC map preprocessing (per model)
├── plot_bgc_monitor.py           # BGC monitor figure
├── rename_v5_dims_arg.py         # NEMO5 grid_V dim renaming (copy here)
├── get_AMOC_deep_v5.py           # AMOC timeseries extraction (copy here)
└── plots/                        # Output figures
```

**Note:** `rename_v5_dims_arg.py` and `get_AMOC_deep_v5.py` must be copied
into `MONITOR/` alongside the other scripts — `run_monitor.bsub` calls them
from `${SLURM_SUBMIT_DIR}`.

---

## Adding a new model

```bash
# 1. Run the full pipeline (auto-detects available years)
sbatch run_monitor.bsub --model TOM12_TJ_NEWMODEL

# 2. If the model run is still in progress and you want the current latest years:
sbatch run_monitor.bsub --model TOM12_TJ_NEWMODEL --skip-amoc
# (then resubmit without --skip-amoc when the run finishes)
```

---

## Planned additions

- **Ecosystem monitor** — zooplankton, export, food web diagnostics
- Quantitative model performance metrics
- Multi-model comparison mode




---

## License

MIT License

Copyright (c) 2026 tjarnikova

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
