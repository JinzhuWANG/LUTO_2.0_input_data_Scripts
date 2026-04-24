# 3_agriculture_climate_damage.py

## Purpose

Calculates yield damage multipliers representing how climate change reduces agricultural productivity relative to the 2010 baseline. Covers four RCP scenarios and four time periods. These multipliers are applied in LUTO to scale baseline yields under future climate scenarios.

## Key Inputs

- `NLUM_SPREAD_LU_ID_Mapped_Concordance.h5` — land use template for building output index
- `N:/Data-Master/Climate_damage/GAEZ_approach/From_Michalis/LUTO_CC_yield_impacts_SA2_*.csv` — GAEZ-derived yield impact projections by crop, SA2, RCP scenario, and year

## Key Outputs

### `SA2_climate_damage_mult.h5`

Path: `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_climate_damage_mult.h5`

MultiIndex DataFrame: `(SA2_ID, LU_ID, IRRIGATION)` × columns for each `(RCP, Year)` combination.

Values are multipliers in the range ~0–2 (typically <1 indicating damage). A value of 0.85 means yield is 85% of baseline.

RCPs covered: 2.6, 4.5, 6.0, 8.5  
Years covered: 2010, 2020, 2050, 2080

## Design Notes

- Extreme outliers (outside 5th–95th percentile, or >3.0) are filtered before aggregation.
- Hierarchical gap-filling when GAEZ data is sparse: SA2 → SA4 → State → National average.
- The 2010 multiplier is 1.0 (baseline); all other years express relative change.
- Forward-filling (time interpolation) is applied where intermediate years are missing.
