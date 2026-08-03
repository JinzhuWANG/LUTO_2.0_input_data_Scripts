# script_10_1_REM_get_tables_inputs.py

## Purpose

Reformats the renewable energy input tables into the shapes LUTO expects, and simplifies the AEMO REZ boundary polygons.

Ported from `N:/Data-Master/Renewable Energy/code/formatting_tables.py` plus `convert_shp_to_geojson.py`, which were merged here at the user's request.

Non-spatial apart from the final REZ section — no NLUM mask, no cell arrays. Everything reads from and writes to `N:/Data-Master/Renewable Energy/`.

---

## Sections

### 1. Electricity price tables

`{data_root}/20260210/{solar,wind}_elec_price_AUD_MWh.csv` → long format with `Year`, `State`, `Price_AUD_per_MWh`.

Two fixups, applied identically to solar and wind:
- **Back-fill 2010–2020** with the 2021 price, since the forecast series starts at 2021 but LUTO's base year is 2010.
- **ACT takes NSW prices.** ACT is treated as part of NSW throughout the LUTO renewable pipeline.

### 2. Renewable targets

`{data_root}/20260408_RE_Targets/renewable_targets-input-20260408.xlsx` (sheet `in`) → long format with `scen`, `state`, `tech`, `Year`, `Renewable_Target_TWh`.

- State abbreviations mapped to full names via `state_rename`.
- **ACT targets are summed into NSW**, then the ACT rows dropped — note this differs from the price handling, where ACT *copies* NSW. Targets are additive quantities, prices are not.
- Years are linearly interpolated (`method='index'`) to fill the 5-year gaps that appear after 2030.

### 3. Wind / solar bundle

Concatenates per-land-use sheets from `20260105_Bundle_Wind.xlsx` and `20260105_Bundle_SPV.xlsx`.

Wind has four sheets (`cropping`, `horticulture`, `livestock`, `unallocated`); **solar has only three** — there is no `Solar PV (horticulture)` sheet. `REQUIRED_COLUMNS` is checked per sheet and any absence printed as a warning, not raised.

### 4. REZ boundaries

Reads the AEMO 2025 Indicative REZ polygon shapefile, keeps rows where `descriptio == "REZ"`, and simplifies geometries with a 0.1° tolerance.

Despite the original filename (`convert_shp_to_geojson.py`), this writes an **ESRI Shapefile**, not GeoJSON.

---

## Outputs

All under `N:/Data-Master/Renewable Energy/processed/`:

| File | Contents |
|---|---|
| `renewable_price_AUD_MWh_solar.csv` | Year × State solar price |
| `renewable_price_AUD_MWh_wind.csv` | Year × State wind price |
| `renewable_targets.csv` | scen × state × tech × Year target TWh |
| `renewable_energy_bundle.csv` | Combined wind + solar land-use bundle |
| `REZ_boundary/aemo_rez_boundaries_2025.shp` | Simplified REZ polygons |

---

## Notes

- Independent of scripts 10_2 and 10_3 — shares no inputs or outputs with either, so it can run in any order relative to them.
- The merged import block drops `import json` and `from itertools import product`, both unused in the original files.
