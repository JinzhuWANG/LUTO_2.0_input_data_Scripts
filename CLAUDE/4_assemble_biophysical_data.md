# 4_assemble_biophysical_data.py

## Purpose

Integrates diverse biophysical rasters (climate, soil, water, habitat) into a single cell-level dataframe. Provides the environmental context for land use modelling — rainfall, soil properties, habitat condition, and carbon stocks.

## Key Inputs

- `N:/Data-Master/ANUCLIM_climate_data/` — gridded rainfall and evapotranspiration
- `N:/Data-Master/Soil_Landscape_Grid_Australia/` — soil AWC, pH, erosion risk (250 m, reprojected to 1 km)
- `N:/Data-Master/Emissions_Reduction_Fund/Maximum_aboveground_biomass_M/` — Roxburgh biomass raster
- `N:/Data-Master/HCAS/` — Habitat Condition Assessment System raster (biodiversity condition score)
- `N:/Data-Master/Water/Water_account/` — agricultural and domestic water use by sector

## Key Outputs

### `cell_biophysical_df.h5` (key: `cell_biophysical_df`)

Path: `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5`

One row per cell. Extended further by script 6 (water yield). Key columns:

| Column | Description |
|---|---|
| `AVG_AN_PREC_MM_YR` | Mean annual precipitation (mm/yr) |
| `AVG_AN_EVAP_MM_YR` | Mean annual evapotranspiration (mm/yr) |
| `AWC_MM` | Plant-available water capacity (mm) |
| `EROSITY_RISK` | Soil erosion risk index |
| `NATURAL_AREA_INC_WATER` | Binary: 0 = natural land (including water bodies), 1 = non-natural |
| `DCCEEW_NCI` | HCAS nature condition index (0–1 percentile of habitat condition) |
| `NATURAL_AREA_CONNECTIVITY` | Landscape connectivity metric |
| `MAX_ABOVEGROUND_BIOMASS_T_HA` | Roxburgh maximum above-ground biomass (t/ha) |
| `RIPARIAN_LENGTH_M` | Total riparian channel length per cell (m) |

## Design Notes

- All rasters are reprojected to match the NLUM 1 km grid using rasterio with bilinear resampling (continuous data) or nearest-neighbour (categorical data).
- Gaps from nodata pixels are filled using rasterio `fillnodata` (inverse-distance weighted interpolation).
- `NATURAL_AREA_INC_WATER` is the primary flag used by scripts 5_1 and 5_2 to distinguish natural from non-natural land outside the LUTO study area.
- Script 6 appends water yield columns to this same file rather than creating a separate output.
