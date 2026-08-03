# script_5_4_get_Zonation_performance_curves.py

## Purpose

Computes rank-to-area performance curves for every zonation rank layer in the pipeline, and writes them to a single Excel workbook. The NES curves feed the GBF2 conservation-priority targets, but the method is generic — any zonation rankmap can be added to either loop.

Pure consumer. Every input is a rank raster already on disk — this script generates no spatial layers of its own, which is why it sits last and can be re-run freely without touching any producer.

---

## Inputs

| Input | Produced by |
|---|---|
| `{bio_Carla_EnviroSuit_dir}/Zonation/{ssp}/{ssp}_zonation_rank_1km.tif` | script 4 |
| `{SNES_ECNES_dir}/Processed/Zonation/{nes}_Priority/rankmap.tif` | script 5_3 |
| `cell_zones_df.h5` | script 1 |
| `cell_LU_mapping.h5` | script 2 |

`SNES_ECNES_dir = N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES`
`bio_Carla_EnviroSuit_dir = N:/Data-Master/Biodiversity/Environmental-suitability`

Ten sources in total: four SSPs (`ssp126`, `ssp245`, `ssp370`, `ssp585`) and six NES layers (`SNES_likely`, `SNES_likely_may`, `ECNES_likely`, `ECNES_likely_may`, `MNES_likely`, `MNES_likely_may`).

---

## Method

Both loops are identical apart from the raster path and the `source` label.

1. Open the rank raster and sample it at every cell's `(X, Y)` from `cell_zones_df.h5` with `method='nearest'`, attaching `CELL_HA` as an `area` coordinate.
2. Subset to `idx_in_LUTO` (cells whose `LU_DESC != 'Non-agricultural land'`) and sort by descending `PRIORITY_RANK`.
3. Accumulate two curves:
   - `AREA_COVERAGE_PERCENT` — cumulative area / total area × 100
   - `PRIORITY_RANK_CUMSUM_CONTRIBUTION` — cumulative (rank × area) / total (rank × area) × 100
4. Thin to the 101 rows whose area coverage is closest to each integer 0–100, then overwrite the column with exactly `0…100`.

`area` is cast to float before `cumsum()` — with ~7 M cells the integer accumulation loses precision otherwise.

---

## Output

`{SNES_ECNES_dir}/Processed/Biodiversity_conserve_performance.xlsx` — one sheet per source, each with `AREA_COVERAGE_PERCENT`, `PRIORITY_RANK`, `PRIORITY_RANK_CUMSUM_CONTRIBUTION`.

**All ten sources must be computed in one run.** The single `pd.ExcelWriter` uses pandas' default `mode='w'`, which recreates the workbook from the in-memory DataFrame. Splitting the SSP and NES loops across two scripts would make whichever ran second truncate the other's sheets — the file would look healthy while silently missing half its tabs. If the halves ever do need to run independently, give them separate output files rather than reaching for `mode='a'`.

Sheet order is alphabetical regardless of loop order, because `groupby('source')` sorts its keys.

---

## Notes

- Needs no NLUM mask, no HCAS, and no `biodiv_degrade_ly` — cell selection is by `(X, Y)` lookup and the `idx_in_LUTO` boolean only.
- Cheap and idempotent relative to the rest of the biodiversity pipeline: re-running only re-reads ten rasters and rewrites one workbook.
