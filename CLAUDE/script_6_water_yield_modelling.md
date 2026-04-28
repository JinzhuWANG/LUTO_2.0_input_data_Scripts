# script_6_water_yield_modelling.py

## Purpose

Runs the InVEST water yield model to quantify water provisioning under baseline and future climate scenarios. Downloads required soil data from FTP, reprojects all inputs to Albers equal-area (EPSG:3577) as required by InVEST, and appends results to the biophysical dataframe.

## Key Inputs

- FTP: `qld.auscover.org.au/tern-soils/` — Soil Landscape Grid Australia (downloaded at runtime)
- `N:/Data-Master/WorldClim_CMIP6/Australia/` — precipitation and PET rasters under CMIP6 scenarios
- NLUM and land use rasters (reprojected to Albers for InVEST)
- `cell_biophysical_df.h5` — existing biophysical snapshot (extended in place)

## Key Outputs

### Appended to `cell_biophysical_df.h5`

| Column | Description |
|---|---|
| `WATER_YIELD_HIST_BASELINE_ML_HA` | Historical baseline water yield (ML/ha/yr) |

Individual water yield GeoTiffs are also written for QA purposes.

## Design Notes

- InVEST requires Albers equal-area projection — all inputs are reprojected before the model run and results are reprojected back to GDA94.
- Unit conversions applied: evapotranspiration ×1000, some soil properties ×0.01 to match InVEST's expected input ranges.
- This script must run after script 4 (which creates `cell_biophysical_df.h5`) but its outputs are logically part of the same biophysical dataset.
