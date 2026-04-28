# script_5_1_assemble_biodiversity_data.py

## Purpose

Computes national-level biodiversity area-weighted scores for three data types:

1. **NVIS** — pre-1750 vegetation (MVG and MVS classifications)
2. **SNES** — threatened species (DCCEEW Species of National Environmental Significance)
3. **ECNES** — threatened ecological communities

For each data type, outputs a processed NetCDF (for use in LUTO's biodiversity constraint) and a target CSV (for setting GBF3/GBF4 conservation targets). Script 5_2 reads these outputs and decomposes them by NRM and IBRA regions.

---

## Biodiversity Weighting — Critical Design

Two distinct weights exist in the pipeline. They serve completely different purposes and must not be confused.

### Weight 1: `biodiv_degrade_ly` — land condition degradation (used for area scoring)

```python
biodiv_degrade_ly = HCAS_PERCENTILE_50 / HCAS_PERCENTILE_50[unallocated_natural_code]
```

Represents the **fraction of pre-1750 biodiversity remaining** in each cell given current land use. Derived from the Habitat Condition Assessment System (HCAS) PERCENTILE_50, normalised so that unallocated natural land = 1.0.

Applied in the area score formula:
- **Inside LUTO**: `weighted_area = presence_fraction × biodiv_degrade_ly × cell_ha`
- **Outside LUTO, natural**: `weighted_area = presence_fraction × 1 × cell_ha` (no degradation, weight = 1)
- **Outside LUTO, non-natural**: excluded from restorable area (weight effectively = 0 for `ATTAINABLE_LEVEL`)
- **ALL_HA (pre-1750 baseline)**: `presence_fraction × 1 × cell_ha` (weight = 1 everywhere)

### Weight 2: `bio_presence_weight = {'LIKELY': 0.8, 'MAYBE': 0.3}` — presence uncertainty (zonation only)

Used **only** when building `bio_DCCEEW_SNES_weighted.nc` and `bio_DCCEEW_ECNES_weighted.nc` for Zonation spatial prioritisation. These weights combine LIKELY and MAYBE presence layers:

```python
SNES_likely_and_maybe = np.maximum(SNES_likely * 0.8, SNES_maybe * 0.3)
```

**These weights are never applied when computing area-weighted scores.** The weighted NetCDF files are only consumed by the Zonation workflow. Script 5_2 reads the raw `bio_DCCEEW_SNES.nc` file, not the weighted one.

---

## Score Column Definitions

All score columns follow the same formula across NVIS, SNES, and ECNES:

| Column | Formula | Meaning |
|---|---|---|
| `ALL_HA` / `AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA` | `presence × cell_ha` over all cells | Pre-1750 potential area (baseline denominator) |
| `IN_LUTO_HA` / `AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_HA` | `presence × biodiv_degrade_ly × cell_ha` over inside-LUTO cells | Current biodiversity condition within the study area |
| `NATURAL_OUT_LUTO_HA` / `AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA` | `presence × cell_ha` over outside-LUTO natural cells | Biodiversity in protected/natural land outside LUTO |
| `NON_NATURAL_OUT_LUTO_HA` | `presence × cell_ha` over outside-LUTO non-natural cells | Irreversibly lost area (urban, roads) |
| `BASEYEAR_SCORE` | `IN_LUTO_HA + NATURAL_OUT_LUTO_HA` | Total current biodiversity score |
| `BASEYEAR_LEVEL` | `BASEYEAR_SCORE / ALL_HA × 100` | Current level as % of pre-1750 |
| `ATTAINABLE_LEVEL` | `(1 − NON_NATURAL_OUT_LUTO_HA / ALL_HA) × 100` | Maximum achievable % if all restorable land is recovered |

The `BASEYEAR_SCORE` split into `IN_LUTO_HA` (LUTO can improve) and `NATURAL_OUT_LUTO_HA` (already protected, assumed maintained) is essential for the LUTO constraint formulation.

---

## NVIS Processing

**Source**: `N:/Data-Master/NVIS/Processed/NVIS7_0_AUST_PRE_MVG.nc` and `NVIS7_0_AUST_PRE_MVS.nc`

NVIS arrays store **uint8 percentage** (0–100) of each vegetation group per cell. Dividing by 100 gives the presence fraction [0–1] used in all score computations:

```python
xr_pre = xr.load_dataarray(nc_path).astype(np.float32) / 100
```

Two classifications:
- **MVG** (Major Vegetation Group): ~30 coarser groups
- **MVS** (Major Vegetation Subgroup): ~90 finer groups

**Output files:**
- `N:/Data-Master/NVIS/Processed/NVIS7_0_AUST_PRE_MVG.nc` — (group, cell) float32, presence fraction
- `N:/Data-Master/NVIS/Processed/NVIS7_0_AUST_PRE_MVS.nc` — (group, cell) float32, presence fraction
- `N:/Data-Master/NVIS/Processed/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.xlsx` — national scores and user-defined target columns; sheets `NVIS_MVG` and `NVIS_MVS`

Target Excel column layout (per vegetation group):

| Column | Description |
|---|---|
| `group` | Vegetation group name |
| `BASE_YR_PERCENT` | `BASEYEAR_LEVEL` — current % of pre-1750 |
| `USER_DEFINED_TARGET_PERCENT_2030/2050/2100` | User-editable target % (NaN by default) |
| `AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA` | `ALL_HA` |
| `AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA` | `NATURAL_OUT_LUTO_HA` |
| `AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_HA` | `IN_LUTO_HA` |
| `ATTAINABLE_LEVEL` | Max achievable % |

---

## SNES Processing

**Source raw**: `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_SNES.nc`
Dimensions: `(species, presence, cell)` where presence ∈ `{'LIKELY', 'MAYBE'}`.

**Important**: the raw file uses `'MAYBE'` for the combined presence category, not `'LIKELY_AND_MAYBE'`. The label `'LIKELY_AND_MAYBE'` only appears in the weighted file (zonation use only).

Area scores are computed separately for LIKELY and MAYBE presence, then split and merged:

```python
SNES_df_LIKELY       = SNES_df.query('PRESENCE_RANK == "LIKELY"')
SNES_df_LIKELY_MAYBE = SNES_df.query('PRESENCE_RANK == "MAYBE"')   # NOT "LIKELY_AND_MAYBE"
```

After suffix-renaming and outer merge, the final CSV has columns suffixed `_LIKELY` and `_LIKELY_MAYBE` for each score component.

**Output:**
- `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_SNES_target.csv`
- `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_SNES_weighted.nc` — zonation only, not used for area scoring

---

## ECNES Processing

Identical structure to SNES. Key difference: species ID column is `COMMUNITY` (not `SCIENTIFIC_NAME`) and presence rank column is `PRES_RANK` (not `PRESENCE_RANK`).

**Output:**
- `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_ECNES_target.csv`
- `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_ECNES_weighted.nc` — zonation only

---

## Global Variables Used by Script 5_2

These variables are set up in this script's header and re-established independently in 5_2:

| Variable | Description |
|---|---|
| `NLUM` | rioxarray DataArray of the NLUM mask (2D) |
| `zones` | `cell_zones_df.h5` columns subset |
| `bioph` | `cell_biophysical_df.h5` columns subset |
| `lumap` | `cell_LU_mapping.h5` columns subset |
| `biodiv_degrade_ly` | 1D float32 array (n_cells,) — degradation weight |
| `idx_in_LUTO` | 1D bool (n_cells,) — True for cells inside LUTO study area |
| `idx_out_LUTO_natural` | 1D bool — outside LUTO and natural |
| `idx_out_LUTO_non_natural` | 1D bool — outside LUTO and non-natural |

`idx_in_LUTO` is derived as: cells whose `LU_DESC != 'Non-agricultural land'`.
