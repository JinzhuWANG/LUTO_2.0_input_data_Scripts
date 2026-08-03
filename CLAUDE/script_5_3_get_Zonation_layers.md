# script_5_3_get_Zonation_layers.py

## Purpose

Runs Zonation 5 spatial prioritisation over the SNES/ECNES threatened species layers. **Layer generation only** — the rank-to-area performance curves derived from the resulting rankmaps live in `script_5_4_get_Zonation_performance_curves.py`.

Split out of `script_5_1_assemble_biodiversity_data.py`, which writes the weighted NetCDFs this script consumes. The split is clean in one direction: 5_1 no longer reads anything 5_3 produces, so the pipeline stays strictly ordered 5_1 → 5_2 → 5_3 → 5_4.

---

## Inputs

| Input | Produced by |
|---|---|
| `{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES_weighted.nc` | script 5_1 |
| `{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES_weighted.nc` | script 5_1 |
| `cell_LU_mapping.h5` | script 2 |

`SNES_ECNES_dir = N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES`

Requires the Zonation 5 executable at `C:/Program Files (x86)/Zonation5/z5.exe`.

---

## Presence weighting

The weighted NetCDFs read here already have `bio_presence_weight = {'LIKELY': 0.8, 'MAYBE': 0.3}` applied by script 5_1:

```python
likely_and_maybe = np.maximum(likely * 0.8, maybe * 0.3)
```

These presence-uncertainty weights exist **only** for zonation. They are never applied when computing area-weighted scores — that is `biodiv_degrade_ly`'s job in scripts 5_1 and 5_2. The `presence` dimension of the weighted files is therefore `{'LIKELY', 'LIKELY_AND_MAYBE'}`, whereas the raw files use `{'LIKELY', 'MAYBE'}`.

---

## Processing steps

1. **Write per-species GeoTIFFs** — each `(species, presence)` slice of the two weighted NetCDFs is expanded back into the 2D NLUM grid (NaN outside the mask) and written to `Processed/SNES_ECNES_WEIGHTED/{SNES,ECNES}/{safe_name}_{presence}.tif`. Parallelised with joblib.

2. **Write feature list files** — the TIF paths are globbed and written one-per-line (quoted, with a `filename` header) to six txt files under `Processed/Zonation/`: SNES, ECNES, and MNES (= SNES + ECNES concatenated), each in `_likely` and `_likely_may` variants.

3. **Write mask and hierarchy rasters** —
   - `zone_mask.tif` = the NLUM mask, defining the analysis area.
   - `zone_hierarchy.tif` = NLUM cells that are **outside** the LUTO study area. Zonation removes hierarchic-mask cells last, so already-protected/non-agricultural land is retained preferentially.

4. **Write six settings files** — each names its feature list plus the shared mask and hierarchy layers.

5. **Run Zonation** — all six runs launched concurrently via `subprocess.Popen` with `--mode=CAZMAX -ah`, then joined. A non-zero exit code prints a warning but does not abort the script.

6. **Merge rankmaps to NetCDF** — the six `rankmap.tif` outputs are masked to NLUM cells and stacked into a `(layer, cell)` array saved as `bio_NES_Zonation.nc`.

---

## RHI (Relative Habitat Importance)

A separate, much smaller section at the end. RHI is a **finished DCCEEW product** — they ran the Zonation 5 CAZMAX prioritisation over the SNES 'likely to occur' SDMs and published the ranked raster. No Zonation run happens here; the section only normalises it to the LUTO spatial format.

Source: `{RHI_dir}/Relative Habitat Importance/relative_habitat_importance.tif`
`RHI_dir = N:/Data-Master/Biodiversity/DCCEEW/RHI (Relative Habitat Importance)`

Two decisions worth knowing, both driven by what the raster actually contains:

- **`Resampling.nearest`, not bilinear.** RHI already matches NLUM's CRS (EPSG:4283) and 0.01° resolution, so this is not a rescale — but the grid origin is offset by a fraction of a cell (−0.35 in x, 0.46 in y), so `reproject_match` is still required. The values are a true uniform rank: percentiles 1/25/50/75/99 of the non-zero cells fall exactly on 1/25/50/75/99. Bilinear would blend neighbouring ranks into values that are no longer ranks.

- **Zeros are preserved, via `to_fill=-1`.** Normalisation goes through `tools.raster.reproject_and_fill`, whose `fill_with_nearest` step treats `to_fill` as a gap alongside NaN. The default `to_fill=0` would overwrite the zeros; `-1` never occurs in a 0–100 rank layer, so the fill is a deliberate no-op.

  This matters because DCCEEW uses 0 for "not ranked", not "lowest priority" — 78 704 NLUM cells (1.13%), mostly islands outside the GEODATA COAST 100K extent their analysis was clipped to. Filling them would hand those cells a rank DCCEEW never assigned, which would be wrong if RHI is ever used to *exclude* land rather than merely rank it.

  Verified output: 6 956 407 cells, range 0–100, 78 704 zeros, no NaNs. Reprojection after `reproject_match` introduces no NaN at all, since RHI's bounding box fully contains NLUM's.

---

## Outputs

| Path | Description |
|---|---|
| `{SNES_ECNES_dir}/Processed/bio_NES_Zonation.nc` | `(layer, cell)` float32; 6 layers |
| `{RHI_dir}/bio_DCCEEW_RHI.nc` | `(cell,)` float32; ranks 1–100, ~30 MB |
| `{SNES_ECNES_dir}/Processed/Zonation/{layer}_Priority/rankmap.tif` | consumed by script 5_4 |
| `{SNES_ECNES_dir}/Processed/SNES_ECNES_WEIGHTED/**/*.tif` | per-species zonation inputs |
| `{SNES_ECNES_dir}/Processed/Zonation/*` | feature lists, settings, masks, and Zonation run directories |

Layer names: `SNES_likely`, `SNES_likely_may`, `ECNES_likely`, `ECNES_likely_may`, `MNES_likely`, `MNES_likely_may`.

---

## Notes

- This script re-establishes `NLUM`, `lumap`, `idx_in_LUTO`, and `ref_meta_float` independently rather than importing from 5_1 — it does not need `zones`, `real_area_ha`, `bioph`, HCAS, or `biodiv_degrade_ly`.
- Step 1 writes tens of thousands of GeoTIFFs and step 5 is long-running. Both are safe to re-run: outputs are overwritten in place.
