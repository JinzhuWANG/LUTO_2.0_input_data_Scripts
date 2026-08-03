# script_10_2_REM_get_existing_capacity.py

## Purpose

Turns the AEMO Network Map renewables point/polygon data into per-cell installed capacity (MW) and per-cell area fraction, by commission year, for existing wind and solar plants.

Ported from `N:/Data-Master/Renewable Energy/code/get_existing_renewable_plants.py`. One line was removed — see below.

---

## Independent of 10_3

The original loaded `processed/renewable_energy_layers_2D.nc` into a `capacity_factor` variable that was never referenced again. Since script 10_3 writes that file, the dead load made 10_2 crash on a clean run unless 10_3 had gone first — a backwards dependency the numbering did not suggest.

That line is deleted. Scripts 10_1, 10_2, and 10_3 now share no inputs or outputs and can run in any order.

Three other loads remain unused: `idx_out_LUTO`, `states`, and `ref_shape` / `ref_meta_float`. Those read files that scripts 1 and 2 produce, so they create no backwards dependency — just wasted I/O — and were left alone.

---

## Spatial framework

Uses `N:/Data-Master/National_Landuse_Map/lumap.tif` as the reference grid, **not** the NLUM mask used by scripts 1–9. `build_luto_cells()` materialises one `shapely.box` per valid cell, in parallel via joblib, so polygon overlays can be done in vector space.

Areas are computed in `EPSG:3577` (Australian Albers) and divided by 10 000 for hectares.

---

## Capacity field selection

Neither technology has one clean capacity column, so each has an explicit priority list and the first non-null value wins (`pick_capacity`). The orders differ between solar and wind, and the source comments record how many rows each tier actually resolves:

- **Solar** — `Agg Nameplate Capacity (MW AC)` first (aggregate + AC side, closest to grid-metered generation), then `Aggregated Upper Nameplate Capacity`, then site-level and unit-level fallbacks.
- **Wind** — same first choice, but `Max Site Capacity (AC)` outranks the aggregated upper bound.

Rows are filtered through a cascade that is printed as a table at the end: raw → drop `Anticipated` → drop `Publicly Announced` → capacity-priority picks → drop no-capacity → drop no-commission-year.

---

## Commission years

`Commision Year` (sic — the misspelling is in the data and the code) is **manually populated via Google Earth**, not derived. The script bootstraps this:

- If `processed/renewable_existing_capacity_commision_year_{solar,wind}_points.csv` exists, it is read and rows with a blank year are dropped.
- If it does not exist, the script **writes the site list out as a stub CSV for hand-filling** and keeps all rows.

On a first run the stub branch fires, and `commision_year_solar` / `commision_year_wind` are then referenced further down at line 219 — so a genuine first run fails until the CSVs are filled in and the script re-run.

---

## Polygon dissolve

Overlapping or touching project polygons are dissolved into unified groups before intersection with LUTO cells, so that staged developments (Stage 1 + Stage 2 sharing land) are not double-counted. Within a group, capacity is **summed** and commission year is the **earliest** — the cell becomes active when the first stage comes online.

Capacity is then distributed by area: `capacity_MW_insec = area_insec × Capacity per ha (MW/ha)`.

---

## Outputs

All under `N:/Data-Master/Renewable Energy/processed/`:

| File | Dims |
|---|---|
| `renewable_existing_capacity_MW_2D.nc` | tech_name × year × y × x |
| `renewable_existing_capacity_MW_1D.nc` | tech_name × year × cell |
| `renewable_existing_capacity_area_fraction_2D.nc` | tech_name × year × y × x |
| `renewable_existing_capacity_area_fraction_1D.nc` | tech_name × year × cell |

`tech_name` ∈ `{'Onshore Wind', 'Utility Solar PV'}`. Filtered point layers are also written back beside the source data as `*_points_{solar,wind}_filtered.gpkg`.

The `area_insec_frac` column is a sanity check — it should be ≤ 1 for every row.
