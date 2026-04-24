# 9_reforestation_carbon_data.py

## Purpose

Processes FullCAM carbon stock projections into cell × age NetCDF arrays for five reforestation scenario types. These represent the carbon sequestration trajectory of a cell if it is planted in year 0 and measured at each subsequent year.

## Key Inputs

- `N:/Data-Master/FullCAM/FullCAM_REST_API_GET_DATA_2025/data/processed/Output_GeoTIFFs/` — FullCAM 4D outputs (tree carbon, debris carbon, soil carbon) by geography and age
- `cell_biophysical_df.h5` — cell area (`CELL_HA`), rainfall, riparian length (used for riparian planting extent)

## Key Outputs

All written to `N:/Data-Master/LUTO_2.0_input_data/Input_data/3D_Spatial_Timeseries/`.

| File | Scenario |
|---|---|
| `tCO2_ha_ep_block.nc` | Environmental plantings — block |
| `tCO2_ha_ep_rip.nc` | Environmental plantings — riparian |
| `tCO2_ha_ep_belt.nc` | Environmental plantings — belt/agroforestry |
| `tCO2_ha_cp_block.nc` | Carbon plantings — block |
| `tCO2_ha_cp_belt.nc` | Carbon plantings — belt |
| `tCO2_ha_hir_block.nc` | High-intensity reforestation — block |
| `tCO2_ha_hir_rip.nc` | High-intensity reforestation — riparian |

Each NetCDF has dimensions `(age: 0–90 years, cell: all NLUM cells)` and stores tCO2-eq/ha for tree, debris, and soil carbon components.

## Design Notes

- **Soil carbon is reported as a delta**: the FullCAM output gives cumulative soil carbon stock. The script subtracts the year-0 value so that the stored value represents the additional carbon sequestered since planting, not the total stock.
- NaN cells (where FullCAM has no output) are filled using `scipy.ndimage.distance_transform_edt` nearest-neighbour fill to maintain spatial continuity.
- NetCDF is used (not HDF5) because it preserves named dimensions (`age`, `cell`) that the LUTO model references directly.
- Compression: zlib level 5 with 4096-cell chunks, reducing file size substantially for large arrays.
