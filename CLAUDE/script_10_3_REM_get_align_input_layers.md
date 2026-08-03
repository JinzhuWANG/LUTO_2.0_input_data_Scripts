# script_10_3_REM_get_align_input_layers.py

## Purpose

Reprojects every renewable energy raster onto the LUTO grid and merges them into NetCDF, in both 2D (y, x) and 1D (cell) form.

Ported from `N:/Data-Master/Renewable Energy/code/reproject_match_rasters_LUTO.py`. The only change from the original is that `fill_with_nearest` and `reproject_and_fill` were lifted into `tools/raster.py` for reuse; their bodies are unchanged.

---

## Spatial framework

Template is `N:/Data-Master/National_Landuse_Map/lumap.tif`, **not** the NLUM mask used by scripts 1–9. The valid-cell mask is `luto_template >= -1`, stacked to `cell=('y','x')` and reused for every 1D conversion so all outputs share one cell index.

`reproject_and_fill()` and `fill_with_nearest()` now live in `tools/raster.py` and are imported, not defined here. `reproject_and_fill()` does three things in order:
1. `rio.reproject_match` onto the template (nearest by default).
2. `fill_with_nearest` — nearest-neighbour fill via `scipy.ndimage.distance_transform_edt`, treating **both NaN and 0 as gaps**. Worth knowing: a legitimate zero in a source raster is treated as missing and overwritten, and the input DataArray is mutated in place.
3. Mask back to the LUTO valid area.

---

## Layers processed

### Capacity factor
`20260127/capacity_factor/capacity_factor_{solar,wind}.tif` → stacked on a `tech_name` dimension. Also written back out as standalone GeoTIFFs (`capacity_factor_*_reproject_match.tif`) for reuse.

### Scenario rasters — capex, opex, distribution loss

Five scenarios: `step_change`, `accelerated_transition`, and three ANU transmission variants (`T3`, `T5`, `T10`, mapped to `Top3_2050` / `Top5_2050` / `Top10_2050` directories).

Year is parsed from the last four characters of each filename stem. Distribution loss has **no** `tech_name` dimension; capex and opex do.

**The ANU scenarios only supply 2050 data**, so they are broadcast across the full year range with `reindex(year=full_years, method='nearest')` — every year in those scenarios holds the same 2050 values.

### QLD EPBC MNES exclusion

`20260317/QLD_EPBC_MNES_prioritization.tif`, reprojected and NaN-filled to 0. A rank-to-area performance curve is then built from it, using the same method as `script_5_4_get_Zonation_performance_curves.py`: sort by descending priority, accumulate area and rank×area, thin to 101 integer percentiles.

---

## Outputs

All under `N:/Data-Master/Renewable Energy/processed/`:

| File | Contents |
|---|---|
| `renewable_energy_layers_2D.nc` | 4 vars: capacity factor, capex, distribution loss, opex |
| `renewable_energy_layers_1D.nc` | Same, stacked to `cell` |
| `capacity_factor_{solar,wind}_reproject_match.tif` | Standalone capacity factor rasters |
| `renewable_QLD_EPBC_MNES_prioritization.nc` | 1D exclusion layer |
| `renewable_QLD_EPBC_MNES_prioritization_performance.csv` | 101-row rank-to-area curve |

Variable names in the layer NetCDFs: `capacity_factor_multiplier`, `Cost_of_install_AUD_kw`, `distribution_loss_factor_multiplier`, `Cost_of_operation_AUD_kw`.

---

## Notes

- No other script in the 10 series reads these outputs. Script 10_2 used to load `renewable_energy_layers_2D.nc` into an unused variable; that dead line has been removed, so 10_1, 10_2, and 10_3 are mutually independent and can run in any order.
- `import pandas as pd` is unused in the original and was kept as-is by the port.
