# North Atlantic Monitor Suite

A set of diagnostic scripts for PlankTOM/NEMO5 runs, producing physical,
biogeochemical, and ecosystem monitors for the North Atlantic.

All scripts live in:

    /gpfs/home/mep22dku/scratch/AMOC-PLANKTOM/hosing/AMOCtun/MONITOR/

---

## Quick start

```bash
sbatch run_monitor.bsub --model TOM12_TJ_HA00 --year-start 1995 --year-end 2000
```

This runs the full pipeline and produces a self-contained output folder:

    plots/TOM12_TJ_HA00_monitor_1995_2000/
        TOM12_TJ_HA00_na_monitor_mar_1995_2000.png      ← physical maps
        TOM12_TJ_HA00_bgc_monitor_annual_1995_2000.png  ← BGC maps
        TOM12_TJ_HA00_eco_monitor_1995_2000.png         ← ecosystem maps
        TOM12_TJ_HA00_ts_physical_1995_2000.png         ← physical timeseries
        TOM12_TJ_HA00_ts_bgc_1995_2000.png              ← BGC timeseries
        TOM12_TJ_HA00_monitor_1995_2000.html            ← tabbed browser report
        TOM12_TJ_HA00_monitor_1995_2000.pdf             ← portable hardcopy

Copy the entire folder to your local machine to view the HTML report.

---

## Toggle flags

Edit these at the top of `run_monitor.bsub` before submitting:

```bash
RUN_PHYSICAL=1   # physical maps (SSS, MLD) + AMOC timeseries
RUN_BGC=1        # BGC maps (TChl, NO3, PO4, PPINT)
RUN_AMOC=1       # AMOC pipeline (rename → cdfmoc → timeseries extraction)
RUN_ECO=1        # ecosystem maps (dominant phyto/zoo + annual cycles)
RUN_TS=1         # regional timeseries (physical, nutrients, plankton)
RUN_REPORT=1     # HTML + PDF report
```

Or pass flags on the command line:

```bash
sbatch run_monitor.bsub --model TOM12_TJ_HA00 --skip-amoc --skip-eco
```

---

## What each monitor shows

### Physical monitor
Maps of March mean SSS and MLD for the last 5 years of the run, compared
against observations (WOA 2023 for SSS, de Boyer Montégut 2004 for MLD).
Full AMOC timeseries at 26.5°N below 500 m.

### BGC monitor
Annual mean surface maps of TChl, NO3, PO4, and integrated primary
production, compared against observations (OC-CCI v5 for TChl, WOA18
for nutrients).

### Ecosystem monitor
Maps of dominant phytoplankton and zooplankton group (by depth-integrated
biomass) across the North Atlantic, plus annual cycle timeseries for each
group from a subpolar gyre box (approx. −64° to −16°E, 47° to 71°N).

### Regional timeseries
Long annual-mean timeseries from the same subpolar box: SST, SSS, MLD,
surface nutrients, TChl, CO2 flux, primary production, and depth-integrated
biomass for all 12 plankton groups.

---

## One-off steps

These only need to be run once, not per model:

```bash
# Regrid BGC obs climatologies (TChl, NO3, PO4)
python prep_bgc_obs.py
```

---

## Year range

By default the monitor uses the last 5 years of available model output.
Override with:

```bash
sbatch run_monitor.bsub --model TOM12_TJ_HA00 --year-start 1960 --year-end 1970
```

---

## Adding a new model

```bash
sbatch run_monitor.bsub --model TOM12_TJ_NEWMODEL
```

If the AMOC files are already up to date from a previous run:

```bash
sbatch run_monitor.bsub --model TOM12_TJ_NEWMODEL --skip-amoc
```

---

## Dependencies

- `ocean` conda environment (`source activate ocean`)
- CDO (`module add cdo/2.5.0`)
- gcc/netcdf for cdfmoc (`module add gcc/11.1.0 netcdf/4.7.4/gcc`)
- `rename_v5_dims_arg.py` and `get_AMOC_deep_v5.py` must be in the MONITOR/ directory
