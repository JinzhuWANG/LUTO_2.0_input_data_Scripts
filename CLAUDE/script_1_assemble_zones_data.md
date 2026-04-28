# script_1_assemble_zones_data.py

## Purpose

Creates the foundational spatial framework for the entire pipeline. Converts the NLUM raster to a flat 1D cell-indexed dataframe that every downstream script reads. All subsequent scripts index their data by the same cell ordering defined here.

## Key Inputs

- `N:/Data-Master/National_Landuse_Map/NLUM_2010-11_clip.tif` — primary 1 km land use raster
- `N:/Data-Master/Australian_administrative_boundaries/` — SA2, SA4, state boundaries (PSMA)
- ABS Census and ABARES spatial data for administrative zone joins

## Key Outputs

### `cell_zones_df.h5` (key: `cell_zones_df`)

Path: `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5`

The most fundamental file in the pipeline. One row per NLUM cell (~7 M rows). Every downstream script reads some columns from this file.

Key columns:

| Column | Description |
|---|---|
| `CELL_ID` | Sequential integer index aligned to the NLUM non-zero pixels |
| `X`, `Y` | Centroid coordinates (GDA94 / EPSG:4283) |
| `CELL_HA` | Cell area in hectares (varies slightly due to projection) |
| `STE_NAME11` | State/territory name |
| `SA2_MAINCODE_2016`, `SA4_CODE_2016` | ABS Statistical Area codes |
| `NRM_CODE`, `NRM_NAME` | NRM (Natural Resource Management) region code and name |
| `IBRA_REG_NAME_7`, `IBRA_SUB_NAME_7` | IBRA bioregion and subregion names (89 regions, 419 subregions) |
| `LU_ID`, `LU_DESC` | Primary land use ID and description (NLUM classification) |
| `SPREAD_ID`, `SPREAD_DESC` | SPREAD commodity ID linked to agricultural commodity system |
| `IRRIGATION` | Binary: 1 = irrigated, 0 = dryland |

## Design Notes

- The NLUM raster is read with rasterio; `np.nonzero(NLUM.values)` gives the 1D cell ordering used throughout the pipeline.
- Administrative boundaries are joined via geopandas spatial join (point-in-polygon).
- All columns are downcast to the smallest fitting dtype to minimise HDF5 file size.
- `CELL_HA` varies because the 1 km equal-angle grid cells have different true areas at different latitudes; always use `CELL_HA` rather than assuming constant area.
- `NRM_NAME` and `IBRA_REG_NAME_7` cover all NLUM cells — no NaN values in these columns.
