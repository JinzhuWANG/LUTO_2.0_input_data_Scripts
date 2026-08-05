# script_5_4_get_Zonation_performance_curves.py

## Purpose

Computes rank-to-area performance curves for every zonation rank layer in the pipeline, and writes them to a single Excel workbook. The NES curves feed the GBF2 conservation-priority targets, but the method is generic — any zonation rankmap can be added to either loop.

Pure consumer. Every input is a rank layer already on disk (GeoTIFF for SSP and NES, NetCDF for RHI) — this script generates no spatial layers of its own, which is why it sits last and can be re-run freely without touching any producer.

---

## Inputs

| Input | Produced by |
|---|---|
| `{bio_Carla_EnviroSuit_dir}/Zonation/{ssp}/{ssp}_zonation_rank_1km.tif` | script 4 |
| `{SNES_ECNES_dir}/Processed/Zonation/{nes}_Priority/rankmap.tif` | script 5_3 |
| `{RHI_dir}/bio_DCCEEW_RHI.nc` | script 5_3 |
| `cell_zones_df.h5` | script 1 |
| `cell_LU_mapping.h5` | script 2 |

`SNES_ECNES_dir = N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES`
`bio_Carla_EnviroSuit_dir = N:/Data-Master/Biodiversity/Environmental-suitability`
`RHI_dir = N:/Data-Master/Biodiversity/DCCEEW/RHI (Relative Habitat Importance)`

Eleven sources in total: four SSPs (`ssp126`, `ssp245`, `ssp370`, `ssp585`), six NES layers (`SNES_likely`, `SNES_likely_may`, `ECNES_likely`, `ECNES_likely_may`, `MNES_likely`, `MNES_likely_may`), and `RHI`.

---

## Method

The SSP and NES loops are identical apart from the raster path and the `source` label.

1. Open the rank raster and sample it at every cell's `(X, Y)` from `cell_zones_df.h5` with `method='nearest'`, attaching `CELL_HA` as an `area` coordinate.
2. Subset to `idx_in_LUTO` (cells whose `LU_DESC != 'Non-agricultural land'`) and sort by descending `PRIORITY_RANK`.
3. Accumulate two curves:
   - `AREA_COVERAGE_PERCENT` — cumulative area / total area × 100
   - `PRIORITY_RANK_CUMSUM_CONTRIBUTION` — cumulative (rank × area) / total (rank × area) × 100
4. Thin to the 101 rows whose area coverage is closest to each integer 0–100, then overwrite the column with exactly `0…100`.

`area` is cast to float before `cumsum()` — with ~7 M cells the integer accumulation loses precision otherwise.

### Every sheet is a threshold lookup for its own layer

Read a row as: *to cover this percentage of in-LUTO area under this layer, threshold it at this `PRIORITY_RANK`.* The value is in whatever units the layer itself uses. LUTO consumes the layer and its sheet as a pair, so **no layer is rescaled to match any other** — cross-layer comparability of the `PRIORITY_RANK` column is not a goal, and chasing it would put the numbers on an invented scale.

That is why RHI keeps its native 0–100. It skips step 1 (already a 1D cell array on the NLUM grid) and is otherwise handled exactly like the rasters. Its values rank each cell against all of Australia rather than against the study area, but the curve is built from in-LUTO cells only, so the rank reported at a given coverage is already the raw value that cuts the study area there — which is all LUTO needs to threshold on. Re-ranking within in-LUTO would select **exactly the same cells**, since ranking is monotonic; it would only change the units.

RHI is also re-emitted here as `bio_RHI_Zonation.nc`, matching the `bio_NES_Zonation.nc` name LUTO's dataprep copies. Same values as `bio_DCCEEW_RHI.nc` from script 5_3 — the point is that the layer LUTO reads and the `RHI` curve built from it always ship from one script, so they cannot drift onto different scales.

---

## Output

| Path | Contents |
|---|---|
| `{SNES_ECNES_dir}/Processed/Biodiversity_conserve_performance.xlsx` | one sheet per source, each with `AREA_COVERAGE_PERCENT`, `PRIORITY_RANK`, `PRIORITY_RANK_CUMSUM_CONTRIBUTION` |
| `{RHI_dir}/bio_RHI_Zonation.nc` | the RHI layer under the name LUTO's dataprep copies |

**All eleven sources must be computed in one run.** The single `pd.ExcelWriter` uses pandas' default `mode='w'`, which recreates the workbook from the in-memory DataFrame. Splitting the SSP and NES loops across two scripts would make whichever ran second truncate the other's sheets — the file would look healthy while silently missing half its tabs. If the halves ever do need to run independently, give them separate output files rather than reaching for `mode='a'`.

Sheet order is alphabetical regardless of loop order, because `groupby('source')` sorts its keys.

---

## Caveat — the layers are not ranked over the same domain

The NES rankmaps come from Zonation runs in script 5_3 that use a **hierarchic mask** (`zone_hierarchy.tif` = cells inside Australia but outside LUTO). Zonation removes hierarchic-mask cells last, so outside-LUTO cells occupy the top of the rank range and the in-LUTO cells are ranked *among themselves*. The arithmetic is exact — outside-LUTO is 39.4% of cells, and in the rankmaps:

| | in-LUTO mean | out-LUTO mean | top decile outside LUTO |
|---|---|---|---|
| `MNES_likely` / `SNES_likely` | 0.303 (= 0.606/2) | 0.803 (= (0.606+1)/2) | **100%** |
| `ssp245` | 0.329 | 0.758 | 92.3% |
| RHI, before re-rank | 52.3 | 46.2 | 42.1% |

RHI's 42.1% against a 39.4% base rate means **no LUTO conditioning at all** — DCCEEW ranked all of Australia with no knowledge of the study area. Its in-LUTO values are a truncated national ranking, not a within-LUTO one.

This does not need correcting for the sheet's purpose. Thresholding RHI at the value its own sheet reports selects the right cells regardless of what scale those values sit on. It does mean the `PRIORITY_RANK` **column** is not comparable across sheets, and the `PRIORITY_RANK_CUMSUM_CONTRIBUTION` column shifts a little with the scale — RHI raw reads 18.3 / 42.2 / 73.6 at 10 / 25 / 50% coverage where a within-LUTO re-rank would read 19.1 / 43.9 / 75.1, a maximum of 2.0 pp across all 101 rows. Neither column changes *which* cells a given coverage selects.

**A deeper limit no rescaling can reach.** The hierarchic mask does not merely push outside-LUTO cells to the top — it makes Zonation's iterative removal treat them as already secured, so the in-LUTO ordering accounts for what is protected elsewhere. RHI arrives as a finished raster with no SDMs to re-run, so that complementarity conditioning is unrecoverable. Two layers can therefore agree on this curve while disagreeing about *which* cells are important; measured directly, RHI and `SNES_likely` correlate at 0.933 but their top deciles overlap at only Jaccard 0.646.

**The SSP sheets use a hierarchic mask too, drawn slightly differently.** Their `PRIORITY_RANK` spans 0–0.9995 within in-LUTO cells rather than stopping near 0.606, which looks at first like no hierarchy was applied. It was. `analysis.log` in each SSP run directory records the same invocation this pipeline uses:

```
Arguments given: --gui --mode=CAZMAX -ah ssp126.z5 ...
Using setting: hierarchic mask layer    = .../hierarchic-mask/hierarchic_mask.tif
Using setting: analysis area mask layer = .../area-mask/area_mask.tif
...
Hierarchic mask level 1 minimum rank: 0.615131
```

That threshold recovers the partition even though `hierarchic_mask.tif` is no longer on disk — cells at or above rank 0.615131 are level 1. Comparing them to LUTO's outside-study-area cells:

| | cells | % of NLUM |
|---|---|---|
| Carla's hierarchy level 1 | 2,705,231 | 38.89% |
| LUTO outside study area | 2,737,674 | 39.35% |

**93.0% agreement, Jaccard 0.835.** The two definitions disagree on 228,479 cells that are level 1 but inside the LUTO study area (5.4% of in-LUTO land, treated by her as already secured), and 260,922 the other way. The surviving `ag_natural_area_mask_LUid.tif` in that folder suggests hers was built from an ag/natural land-use split rather than this pipeline's `LU_DESC != 'Non-agricultural land'` rule.

Those 228,479 cells are the entire reason the SSP in-LUTO range runs past 0.606, and they explain the residual gap in the curve (21.0 vs ~19.2 at top 10%). So the SSP layers are conditioned the same way in kind, just against a slightly different notion of "already protected". No correction is applied — the difference is a boundary disagreement of about 7%, and re-running would mean repeating a four-day Zonation job.

---

## Reading these curves

For a uniform rank over roughly equal-area cells the curve has a closed form, `contribution(p) = 1 − (1−p)²`, giving **19.0 / 43.8 / 75.0**. Every sheet in the workbook is within about a point of that.

These curves are therefore close to degenerate: they mostly confirm that Zonation emits a uniform rank, and carry little information about the underlying biodiversity. What actually distinguishes the layers is *where* the high ranks fall spatially, which no rank-to-area curve can show. Use them for area-target arithmetic, not for judging which layer is "better".

---

## Notes

- Needs no NLUM mask, no HCAS, and no `biodiv_degrade_ly` — cell selection is by `(X, Y)` lookup and the `idx_in_LUTO` boolean only.
- Cheap and idempotent relative to the rest of the biodiversity pipeline: re-running re-reads ten rasters plus one NetCDF and rewrites one workbook. Takes about 2.5 minutes.
