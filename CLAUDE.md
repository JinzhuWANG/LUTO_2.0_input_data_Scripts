# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Repository Overview

Sequential data processing pipeline that assembles all input data for LUTO 2.0 (Land Use Trade-Offs model). Scripts process geospatial, agricultural, biodiversity, and biophysical data for land use modelling across Australia at 1 km resolution (~7 million cells).

Individual script documentation is in the `CLAUDE/` directory — read the relevant file before modifying any script.

## Pipeline Execution Order

Scripts must be run in order. Later scripts depend on outputs from earlier ones.

| Script | Topic | Key output |
|---|---|---|
| `script_1_assemble_zones_data.py` | Spatial framework | `cell_zones_df.h5` |
| `script_2_assemble_agricultural_data.py` | Agricultural economics | `cell_LU_mapping.h5`, SA2 ag data HDF5s |
| `script_3_agriculture_climate_damage.py` | Climate yield damage | `SA2_climate_damage_mult.h5` |
| `script_4_assemble_biophysical_data.py` | Biophysical variables | `cell_biophysical_df.h5` |
| `script_5_0_SNES_ECNES_selected.py` | SNES/ECNES target species lists | imported by script 5_2 |
| `script_5_1_assemble_biodiversity_data.py` | Biodiversity scores (national) | NVIS `.nc`, SNES/ECNES `.nc` + target CSVs |
| `script_5_2_get_NVIS_SNES_ECNES_targets_by_regions.py` | Biodiversity scores (regional) | NRM/IBRA target Excel/CSV files |
| `script_6_water_yield_modelling.py` | Water yield | Appended into `cell_biophysical_df.h5` |
| `script_7_assemble_additional_land_use_sieve_data.py` | Land use constraints | `cell_lu_sieve_df.pkl` |
| `script_8_assemble_ag_yield_gap_data.py` | Yield gap | `SA2_yield_gap_mult.h5` |
| `script_9_reforestation_carbon_data.py` | Carbon sequestration | `tCO2_ha_*.nc` NetCDF files |

## Spatial Framework

All scripts share a consistent spatial framework:

- **NLUM mask**: `N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif`  
  The 1 km raster mask of Australia. Non-zero pixels define the study area (~7 M cells).
- **Cell dataframe**: produced by script 1, used by all others.  
  Path: `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5`
- **1D ↔ 2D convention**: `NLUM.values` gives a 2D boolean mask; `np.nonzero(NLUM.values)` gives the 1D cell index. All cell-level arrays are 1D with length = number of NLUM cells.

## Common Helper Patterns

Most scripts define:
- `conv_1D_to_2D(arr)` — places a 1D cell array back into the 2D NLUM grid for plotting
- `map_in_2D(arr, title)` — quick matplotlib visualisation of spatial data
- `downcast(df)` — reduces int64/float64 columns to smallest fitting dtype (memory optimisation)

## Output Directories

- `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/` — static cell-level snapshots (HDF5, pickle)
- `N:/Data-Master/LUTO_2.0_input_data/Input_data/3D_Spatial_Timeseries/` — time × cell NetCDF arrays (carbon)
- `Intermediate_data_outputs/` — scratch outputs used for validation and debugging

## Critical Design Notes

### Biodiversity weighting (scripts 5_1 and 5_2)
Two distinct weights are used in the biodiversity pipeline — they must not be confused:

- **`biodiv_degrade_ly`** — the land-condition degradation weight. Represents the fraction of pre-1750 biodiversity remaining in each cell given current land use (HCAS PERCENTILE_50, normalised to unallocated natural = 1). Used for **area-weighted score computation**.
- **`bio_presence_weight = {'LIKELY': 0.8, 'MAYBE': 0.3}`** — presence uncertainty weights. Used **only for zonation** to combine LIKELY and MAYBE presence layers. Never applied when computing area scores.

See `CLAUDE/5_1_assemble_biodiversity_data.md` and `CLAUDE/5_2_get_NVIS_SNES_ECNES_targets_by_regions.md` for full details.

### Memory management
- Pandas HDF5 (`.h5`) for large tabular data with fast keyed access
- xarray NetCDF for multi-dimensional arrays (species × cell, age × cell)
- `downcast()` used throughout to halve memory footprint

### Pipeline dependencies
`cell_zones_df.h5` is foundational — all other scripts read it for cell coordinates, area (`CELL_HA`), and administrative boundaries. Script 5_2 also depends on all outputs from script 5_1.
