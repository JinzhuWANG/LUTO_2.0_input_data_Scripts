# script_7_assemble_additional_land_use_sieve_data.py

## Purpose

Creates a spatial constraint ("sieve") dataframe that flags cells where land use conversion is restricted. Restrictions arise from Indigenous land tenure, protected areas, and biodiversity priority zones. The LUTO optimisation model reads this to exclude constrained cells from certain land use transitions.

## Key Inputs

- `cell_zones_df.h5` — spatial framework
- `N:/Data-Master/Indigenous_lands/Native_title_determinations/NNTT2020_*.tif` — Native Title determination rasters 1992–2020
- `N:/Data-Master/Protected_areas/CAPAD2020/` — CAPAD 2020 protected area rasters (all categories and Indigenous Protected Areas)
- `N:/Data-Master/LUF-Modelling/LUTO2.0_Reporting/Data/` — biodiversity priority zone grids

## Key Outputs

### `cell_lu_sieve_df.pkl`

Path: `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_lu_sieve_df.pkl`

One row per cell. Binary (0/1) columns:

| Column | Description |
|---|---|
| `NNTT_EXCLUSIVE_TITLE_2020` | 1 = cell has exclusive Native Title determination |
| `CAPAD_IPAs_2020` | 1 = cell is an Indigenous Protected Area |
| `PROTECTED_AREAS_2020` | 1 = cell is within any CAPAD protected area category |
| Biodiversity priority flags | Various binary flags for biodiversity priority zones |

## Design Notes

- Raster layers are reprojected using nearest-neighbour resampling to preserve crisp binary boundaries (no anti-aliasing of constraint polygons).
- Stored as pickle (not HDF5) to preserve exact dtypes and allow fast loading without schema declaration.
- The sieve is applied in LUTO's optimisation to prevent constrained cells from being assigned to agricultural or plantation land uses.
