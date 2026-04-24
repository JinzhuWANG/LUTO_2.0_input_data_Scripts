# 5_2_get_NVIS_SNES_ECNES_targets_by_regions.py

## Purpose

Decomposes the national-level biodiversity scores from script 5_1 into regional scores for two geographic frameworks:

- **NRM** (Natural Resource Management) regions — 56 regions across Australia
- **IBRA** (Interim Biogeographic Regionalisation for Australia) — 89 regions and 419 subregions

For each framework, three data products are produced:

1. **NVIS** Pre-1750 MVG and MVS weighted area scores and targets (GBF3)
2. **SNES** species weighted area scores and targets (GBF4)
3. **ECNES** ecological community weighted area scores and targets (GBF4)

Column naming follows the national 5_1 convention with an additional `region` column, so national and regional outputs can be directly compared.

---

## Core Computation Pattern

All score columns are computed by a single shared helper `compute_region_scores()` using xarray groupby:

```python
def compute_region_scores(arr: xr.DataArray, region_labels: np.ndarray) -> xr.Dataset:
    arr = arr.assign_coords({'region': ('cell', region_labels)})
    return xr.Dataset({
        'ALL_HA':                  (arr * cell_ha).groupby('region').sum('cell'),
        'IN_LUTO_HA':              (arr * cell_ha * degrade_in_xr).groupby('region').sum('cell'),
        'NATURAL_OUT_LUTO_HA':     (arr * cell_ha * nat_out_xr).groupby('region').sum('cell'),
        'NON_NATURAL_OUT_LUTO_HA': (arr * cell_ha * nnat_out_xr).groupby('region').sum('cell'),
    }).compute()
```

`region_labels` is a 1D numpy object-dtype array (length = n_cells) assigning each cell to a region name. The groupby assigns region as a non-dimension coordinate on the `cell` dimension, then reduces it in one vectorised pass — no per-region loops.

**Important**: `region_labels` must be `np.asarray(..., dtype=object)`, not `.values` from a pandas Series. If the pandas column uses `StringDtype`, `.values` returns a `pandas.arrays.StringArray` which causes infinite recursion in xarray's `PandasExtensionArray.__getattr__`. Always convert explicitly:

```python
nrm_region_per_cell = np.asarray(zones['NRM_NAME'], dtype=object)
```

---

## Pre-computed Cell Weights

Four 1D xarray DataArrays encode the scoring partitions. They are multiplied into `arr * cell_ha` inside `compute_region_scores`:

```python
cell_ha       = zones['CELL_HA'].values.astype(np.float32)    # plain numpy, not xr
degrade_in_xr = xr.DataArray((biodiv_degrade_ly * idx_in_LUTO).astype(np.float32), dims=['cell'])
nat_out_xr    = xr.DataArray(idx_out_LUTO_natural.astype(np.float32),               dims=['cell'])
nnat_out_xr   = xr.DataArray(idx_out_LUTO_non_natural.astype(np.float32),           dims=['cell'])
```

Note that `degrade_in_xr`, `nat_out_xr`, `nnat_out_xr` do **not** include `cell_ha`. The `cell_ha` factor is provided separately in `compute_region_scores` so that `ALL_HA = arr * cell_ha` is applied uniformly to all four columns. Baking `cell_ha` into the pre-computed arrays would double-count it.

---

## Biodiversity Weighting — Same Design as 5_1

See `CLAUDE/5_1_assemble_biodiversity_data.md` for the full explanation. The key points:

- **`biodiv_degrade_ly`** is the land-condition weight for area scoring. It is the fraction of pre-1750 biodiversity remaining given current land use.
- **`bio_presence_weight = {'LIKELY': 0.8, 'MAYBE': 0.3}`** is for Zonation only. Script 5_2 reads the **raw** `bio_DCCEEW_SNES.nc` file (presence ∈ `{'LIKELY', 'MAYBE'}`), not the weighted file.

---

## Score Column Definitions (same as 5_1)

| Internal column | Output column name (after rename) | Meaning |
|---|---|---|
| `ALL_HA` | `AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA` | Pre-1750 baseline (weight = 1) |
| `IN_LUTO_HA` | `AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_HA` | Degraded current score inside LUTO |
| `NATURAL_OUT_LUTO_HA` | `AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA` | Natural land outside LUTO (weight = 1) |
| `BASEYEAR_LEVEL` | `BASE_YR_PERCENT` | `(IN_LUTO_HA + NATURAL_OUT_LUTO_HA) / ALL_HA × 100` |
| `ATTAINABLE_LEVEL` | `ATTAINABLE_LEVEL` | `(1 − NON_NATURAL_OUT_LUTO_HA / ALL_HA) × 100` |

---

## NVIS by NRM and IBRA

**Source**: `NVIS7_0_AUST_PRE_MVG.nc` and `NVIS7_0_AUST_PRE_MVS.nc` (same as 5_1). NVIS arrays are uint8 percentage — divide by 100 before passing to `compute_region_scores`.

**NRM output**: `N:/Data-Master/NVIS/Processed/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS_NRM.xlsx`
- Sheets: `NVIS_MVG`, `NVIS_MVS`
- Columns: `group`, `region`, then score and target columns
- Default targets for NECMA regions (North East, Goulburn Broken): 30% by 2030, 50% by 2050/2100

**IBRA output**: `N:/Data-Master/NVIS/Processed/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS_IBRA.xlsx`
- Sheets: `NVIS_MVG` (IBRA regions, 89), `NVIS_MVS` (IBRA subregions, 419)
- Computed using `ones_arr` (all 1s) — each cell contributes its full area to its IBRA region with no NVIS fraction applied. `ALL_HA` sums to total Australia cell area.
- Default targets: 30% by 2030, 50% by 2050/2100 for all regions

The IBRA file is consumed by `luto/data.py` via:
```python
sheet_name=f'NVIS_{settings.GBF3_IBRA_TARGET_CLASS}'   # e.g. 'NVIS_MVG' or 'NVIS_MVS'
```
where `GBF3_IBRA_TARGET_CLASS` ∈ `{'MVG', 'MVS'}` (note: the `NVIS_` prefix is added by data.py, not stored in the setting).

---

## SNES and ECNES by NRM and IBRA

**Source**: raw `bio_DCCEEW_SNES.nc` and `bio_DCCEEW_ECNES.nc` — presence ∈ `{'LIKELY', 'MAYBE'}`.

```python
SNES_raw        = xr.open_dataarray('.../bio_DCCEEW_SNES.nc', chunks={'species': 50, 'presence': 1})
SNES_likely_arr = SNES_raw.sel(presence='LIKELY').astype(np.float32)
SNES_lm_arr     = SNES_raw.sel(presence='MAYBE').astype(np.float32)
```

Both LIKELY and LIKELY_MAYBE are computed using `compute_region_scores` with the same degradation weights — the presence arrays differ but the area weighting formula is identical.

Column naming after rename:
- LIKELY columns: `BASEYEAR_LEVEL_LIKELY`, `BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY`, `BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY`, `BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY`, target columns `_LIKELY`
- LIKELY_MAYBE columns: same names with `_LIKELY_MAYBE` suffix

**NRM outputs:**
- `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_SNES_target_NRM.csv`
- `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_ECNES_target_NRM.csv`

**IBRA outputs:**
- `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_SNES_target_IBRA.csv`
- `N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES/Processed/bio_DCCEEW_ECNES_target_IBRA.csv`

---

## NECMA Targets (NRM only)

NECMA (North East CMA and Goulburn Broken CMA) specific targets are written into the NRM SNES/ECNES CSVs for listed species:

```python
NECMA_NRM_NAMES = ['North East', 'Goulburn Broken']
```

For species/communities contracted under the NECMA agreement:
- SNES: target ≥50% by 2030, ≥70% by 2050/2100 (LIKELY only)
- ECNES: same targets

The full species/community lists (`NECMA_SNES`, `GBCMA_SNES`, `NECMA_ECNES`, `GBCMA_ECNES`) are hardcoded in the script.

---

## Sanity Check

A standalone validation script compares regional sums against national totals:

`N:/Data-Master/LUTO_2.0_input_data/Scripts/work_in_progress/sanity_check_5_1_vs_5_2_nvis.py`

Expected results:
- NVIS NRM regional sum vs national: <0.1% (float32 accumulation noise on `OUTSIDE_LUTO_NATURAL`)
- IBRA total area vs `sum(cell_ha)`: <0.0001%
- SNES/ECNES LIKELY NRM and IBRA regional sum vs national: <0.00001%
