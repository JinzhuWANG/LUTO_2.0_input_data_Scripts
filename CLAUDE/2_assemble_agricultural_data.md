# 2_assemble_agricultural_data.py

## Purpose

Assembles cell-level and SA2-level agricultural economic data by merging CSIRO PROFIT MAP outputs with ABS census and production data. Produces the datasets that define agricultural commodity profitability, yield, water use, and GHG emissions for each land use and location.

## Key Inputs

- `cell_zones_df.h5` — spatial framework (SA2 codes, irrigation flags, LU_IDs)
- `N:/Data-Master/Profit_map/From_CSIRO/pfe_table_*.csv` — CSIRO PROFIT MAP: crop economics by SA2 and irrigation
- `N:/Data-Master/Profit_map/From_CSIRO/lmap.h5` — livestock density by X/Y coordinate
- ABS agricultural census data (crop areas, livestock numbers, production volumes)
- ABARE data (commodity prices, input costs)

## Key Outputs

All written to `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/`.

### `cell_LU_mapping.h5`

One row per cell. Maps each cell to its LUTO land use ID.

| Column | Description |
|---|---|
| `LU_ID_LUTO` | Integer land use ID in LUTO's commodity system |
| `LU_DESC` | Land use description string |
| `IRRIGATION` | Binary irrigated flag |

### `cell_livestock_data.h5`

Livestock dry matter production and stocking rates per cell, used for grazing land calculations.

### `SA2_crop_data.h5`

SA2 × crop × irrigation indexed. Core agricultural economics table.

Key columns: `YIELD_POT_DRY_T_HA`, `P_EA_KG`, `QC_AUD_HA`, `AC_AUD_HA`, `FDC_AUD_HA`, `FLC_AUD_HA`, `FOC_AUD_HA`, `WR_ML_HA`, `NPK_*`

### `SA2_off_land_commodity_data.h5`

Chickens (LU_ID 40), eggs (LU_ID 41), pigs (LU_ID 42) — production economics not tied to land cells directly.

### GHG data files

- `SA2_crop_GHG_data.h5`
- `SA2_livestock_GHG_data.h5`
- `SA2_irrigated_pasture_GHG_data.h5`

Greenhouse gas emissions (tCO2-eq/ha) by commodity, SA2, and irrigation status.

### `NLUM_SPREAD_LU_ID_Mapped_Concordance.h5`

Lookup table linking NLUM land use codes → SPREAD commodity IDs → LUTO LU_IDs. Used by scripts 3 and 8 as a template for building SA2-indexed data structures.

## Design Notes

- Livestock maps are joined to cells using X/Y coordinate rounding (×100 precision) rather than spatial join.
- LU_ID ranges: crops 5–25, pasture/grazing 30–35, off-land 40–42.
- Economics columns follow the PROFIT MAP naming convention: QC (quantity cost), AC (area cost), FDC (fixed direct cost), FLC (fixed labour cost), FOC (fixed overhead cost).
- All SA2-indexed data uses a MultiIndex of `(SA2_ID, LU_ID, IRRIGATION)`.
