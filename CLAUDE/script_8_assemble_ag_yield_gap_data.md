# script_8_assemble_ag_yield_gap_data.py

## Purpose

Quantifies the yield gap — the difference between current yields and the attainable yield under sustainable intensification — for each crop, SA2, and irrigation regime. Produces multipliers used in LUTO to model intensification scenarios.

## Key Inputs

- `NLUM_SPREAD_LU_ID_Mapped_Concordance.h5` — land use template (same structure as script 3)
- `N:/Data-Master/Sustainable_intensification/From_Michalis/LUTO_Current+attainable_yields_SA2_*.csv` — GAEZ attainable yield projections by crop, SA2, and irrigation

## Key Outputs

### `SA2_yield_gap_mult.h5`

Path: `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_yield_gap_mult.h5`

MultiIndex DataFrame: `(SA2_ID, LU_ID, IRRIGATION)`.

| Column | Description |
|---|---|
| `YIELD_CURR` | Current yield (t/ha) |
| `YIELD_ATT` | Attainable yield under sustainable intensification (t/ha) |
| `YIELD_MULT` | Ratio `YIELD_ATT / YIELD_CURR` |
| `YIELD_GAP_SOURCE` | Data provenance: SA2-specific, SA4 fallback, State fallback, or National average |

## Design Notes

- Hierarchical gap-filling mirrors script 3: SA2 → SA4 → State → National when crop-specific data is unavailable for a given SA2.
- `YIELD_GAP_SOURCE` records which level the multiplier came from to allow auditing of data quality.
- Multipliers are applied in LUTO to scale baseline yields when intensification scenarios are active.
