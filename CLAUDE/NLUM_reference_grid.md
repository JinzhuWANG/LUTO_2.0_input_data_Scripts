# NLUM reference grid

Not a script doc — this describes the spatial reference frame that scripts 1–9 all share, and where it came from. Read this before changing anything that reprojects, masks, or indexes cell-level data.

## Summary

Every LUTO input layer is aligned to a **4071 × 3364 grid at 0.01° in GDA94 (EPSG:4283)** containing **6,956,407 valid cells**. Both numbers are inherited from the ABARES *Land Use of Australia 2010-11* release; neither was chosen by this pipeline.

### Geospatial reference

| Property | Value |
|---|---|
| CRS | `EPSG:4283` — GDA94 geographic (Geocentric Datum of Australia 1994) |
| Size | 4071 cols × 3364 rows = 13,694,844 pixels |
| Resolution | 0.01° × 0.01° (~1 km) |
| Origin | 112.925 °E, −10.015 °S |
| Bounds | W 112.925, E 153.635, S −43.655, N −10.015 |
| Transform | `Affine(0.01, 0.0, 112.925, 0.0, -0.01, -10.015)` |

### Valid cells

| Property | Value |
|---|---|
| **Valid cells** | **6,956,407** |
| Total pixels | 13,694,844 |
| Valid fraction | 50.80 % |
| Cell area | 89.64 min / 110.52 mean / 121.26 max ha |
| Total area | 768.8 Mha |

The ~49% that is nodata is ocean and the corners of the bounding rectangle — the grid is an axis-aligned box around a continent, so half of it is sea.

Note the cell area spread: 89.64 to 121.26 ha, a 35% range. That is the equal-angle grid — cells narrow toward the south (Tasmania ~90 ha, Cape York ~121 ha). The mean of 110.52 ha is **not** a safe substitute for `CELL_HA` in any area-weighted calculation.

## Provenance chain

### 1. Source — ABARES *Land Use of Australia 2010-11*

ESRI Grid `lu10v5ug` at `N:/Data-Master/National_Landuse_Map/Land-use_of_Australia_2010-11/lu10v5ug/` (`w001001.adf`, `vat.adf`, …). Native size `4150 × 3401`, already 0.01° GDA94, already containing exactly 6,956,407 non-NoData pixels.

- **Custodian**: Australian Bureau of Agricultural and Resource Economics and Sciences (ABARES)
- **Version**: definitive categorical summary grid, public release 4 July 2016 (supersedes the July 2014 Interim Version 5)
- **Currency**: 1 April 2010 – 31 March 2011
- **Classification**: Australian Land Use and Management Classification (ALUM) Version 7
- **Status**: ANZLIC Foundation Spatial Data; ABS Essential Statistical Asset
- **Licence**: CC BY 3.0 AU, © Commonwealth of Australia 2016
- **Citation**: ABARES 2016, *Land Use of Australia 2010-11*, ABARES, Canberra

How the values were produced (per `Raw_data_20200122_2010-11/metadata_0.txt`):

- **Non-agricultural land uses** — overlay of seven 0.01° thematic rasters: topographic features (GEODATA TOPO 250K series 3), catchment-scale land use (CLUM Jan 2014), protected areas (CAPAD 2010 + declared IPAs), World Heritage areas, tenure, forest cover (Forests of Australia 2013), vegetation condition (VAST).
- **Agricultural land uses** — ABS 2010-11 Agricultural Census areas at SLA level, spatially disaggregated by **SPREAD II** (Bayesian MCMC) driven by monthly AVHRR NDVI with NLWRA training data. Grazing on forested land was allocated separately, outside SPREAD II, prioritised by crown cover then slope.
- The result is described by ABARES as "an approximation to a maximum likelihood map", not an observed map.

Local documentation: `NLUM_UserGuide_2010-11_v1.0.0.pdf` (Smart 2016), `ALUM_Classification_V7_May_2010_*.pdf`, `metadata_0.txt`.

### 2. Clip — `NLUM_preprocessing.ipynb`

Run once, by hand. The **second** cell is the one that produced the current files:

```python
src    = rasterio.open(r'...\lu10v5ug\w001001.adf')
window = get_data_window(src.read(1, masked=True))   # -> Window(col_off=42, row_off=2, width=4071, height=3364)
kwargs.update({'driver':'GTiff', 'dtype':'int16', 'compress':'lzw',
               'height': window.height, 'width': window.width,
               # 'transform': rasterio.windows.transform(window, src.transform)  # imprecise
               'transform': Affine(0.01, 0.0, 112.925, 0.0, -0.01, -10.015)})    # set manually
dst.write(src.read(window=window))                    # -> NLUM_2010-11_clip.tif

clipmask = src.read(1, window=window, masked=True).mask
msk      = np.where(clipmask, 0, 1).astype(np.uint8)  # -> NLUM_2010-11_mask.tif
```

`get_data_window` returns the tightest rectangle enclosing all non-NoData pixels. Trimming was 42 columns left + 37 right, 2 rows top + 35 bottom. Those rows and columns were **entirely NoData**, so the clip changed the rectangle but not the valid-cell set — 6,956,407 in, 6,956,407 out.

The transform is hardcoded rather than derived because `112.505 + 42 × 0.01` does not land exactly on 112.925 in floating point. Geometrically identical, just clean.

The notebook's **first** cell is superseded. It clipped with a bounding box "established manually from QGIS" — `112.925, 153.635, −43.655, −10.015` — which is byte-for-byte the extent `get_data_window` later derived. The switch to `get_data_window` changed nothing about the grid.

**Critically: this is a pure clip.** No reprojection, no resampling, no reclassification. Pixel values in `clip.tif` are the source VAT `VALUE` codes (1–5135) untouched. The LUTO grid *is* the ABARES grid, trimmed.

### 3. Cell ordering — `script_1_assemble_zones_data.py`

A raster gives a footprint, not an index. The ordering is created at `script_1_assemble_zones_data.py:52`:

```python
xy = np.nonzero(NLUM_mask)   # row-major over the 3364 × 4071 grid
```

`CELL_ID = index + 1`, running 1 … 6,956,407. This is why **any 1D array of length 6,956,407 in this project aligns positionally with any other** — nothing carries a spatial key. Ordering is also unchanged from the source grid: deleting wholly-empty rows and columns preserves row-major order of the survivors, so cell *n* is the *n*-th valid pixel of `lu10v5ug` read left-to-right, top-to-bottom.

`script_1` also derives, from the same `meta`:
- `CELL_HA` — each cell vectorised, reprojected to EPSG:3577 (Australian Albers Equal Area), area / 10000. Written back out as `NLUM_2010-11_cell_ha.tif`.
- All administrative joins (SA2, SA4, state, NRM, IBRA) — rasterised onto this grid and nearest-neighbour gap-filled so no NLUM cell is left unlabelled.

## Files in `N:/Data-Master/National_Landuse_Map/`

| File | Grid | dtype / nodata | Contents |
|---|---|---|---|
| `NLUM_2010-11_clip.tif` | 4071 × 3364, EPSG:4283 | int16 / −32768 | VAT `VALUE` codes 1–5135 |
| `NLUM_2010-11_mask.tif` | same | uint8 / 0 | binary 1 = valid |
| `NLUM_2010-11_cell_ha.tif` | same | float32 / 0 | cell area in ha |
| `NLUM_2010-11_cell_id.tif` | same | float32 / 0 | `CELL_ID` |
| `NLUM_2010-11_clip.tif.csv` | — | — | VAT exported to CSV, 28 columns, join on `VALUE` |
| `NLUM_2010-11_clip.gpkg` | — | — | vectorised NLUM zones + VAT (written by script 1) |
| `NLUM_table.csv` | — | — | VAT + derived `SPREAD_ID` / `SPREAD_DESC` (written by script 1) |
| `NLUM_2010-11_mask_ALBERS.tif` | **3910 × 3647, EPSG:3577** | uint8 / 0 | **not an alignment target** — see below |

`clip.tif` and `mask.tif` have been verified pixel-for-pixel identical in footprint, and `np.nonzero()` returns identical index arrays for both. They were written by the same notebook cell from the same `src.read()`, so they cannot drift.

`NLUM_2010-11_clip.tif.csv` is the source VAT verbatim: `VALUE`, `LU_CODEV7N`, `PRIMARY/SECONDARY/TERTIARY_V7`, `CLASSES_18`, `COMMODITIES`, `IRRIGATION`, plus all seven input themes and their descriptions.

## Which file each script opens

| Script | Opens |
|---|---|
| `script_1_assemble_zones_data.py:22` | `NLUM_2010-11_clip.tif` — needs the VAT codes and defines the ordering |
| `script_5_1_assemble_biodiversity_data.py:864` | `NLUM_2010-11_clip.tif` — needs `meta` for the NVIS reprojection block |
| scripts 2, 3, 4, 6, 7, 8, 9, 5_1 (main), 5_2, 5_3 | `NLUM_2010-11_mask.tif` |
| script 10 series (REM) | neither — uses `National_Landuse_Map/lumap.tif` |

Most scripts only need the geometry and the valid-cell set, so they take `mask.tif` (~350 KB uint8) rather than `clip.tif` (3.3 MB int16). Same grid either way.

## Gotchas

- **Never assume 100 ha per cell.** The grid is equal-*angle*, so true cell area runs 89.64–121.26 ha with latitude. Always use `CELL_HA` from `cell_zones_df.h5`.
- **`NLUM_2010-11_mask_ALBERS.tif` is not an alignment target.** It is EPSG:3577 at 1025.5 m with 7,310,879 valid cells — a reprojected convenience layer for area work. Cell *n* in it is **not** cell *n* in the pipeline.
- **Path history.** The share has been renamed `N:\Brett-lab\Data-Master` → `N:\Planet-A\Data-Master` → `N:\Data-Master`. `NLUM_preprocessing.ipynb` and the folder `README.txt` still show the older roots, as do all of `Superseded/`. `script_7_assemble_additional_land_use_sieve_data.py:14` still points at `N:\Planet-A\...` and will fail unless that path still resolves — every other current script uses `N:/Data-Master/...`.
- **`README.txt`** in `National_Landuse_Map/` says the mask was "created via NLUM_2010-11_mask.tif" — a typo. It was created from the ESRI grid's own NoData mask, in the same notebook cell as `clip.tif`.
- **Do not regenerate `clip.tif`.** The cell ordering in `cell_zones_df.h5` and every downstream HDF5/NetCDF depends on this exact footprint. Regenerating would silently invalidate the positional alignment the whole pipeline assumes.

## Reproducing the checks

```python
import rasterio, numpy as np
from rasterio.windows import get_data_window

NLUM = 'N:/Data-Master/National_Landuse_Map/'

# The two tables above
with rasterio.open(NLUM + 'NLUM_2010-11_mask.tif') as r:
    print(r.crs, r.width, r.height, r.res, r.bounds)    # EPSG:4283 4071 3364 (0.01, 0.01) ...
    a = r.read(1)
    print((a != 0).sum(), a.size)                       # 6956407 13694844
with rasterio.open(NLUM + 'NLUM_2010-11_cell_ha.tif') as r:
    ha = r.read(1)[a != 0]; ha = ha[np.isfinite(ha)]
    print(ha.min(), ha.mean(), ha.max(), ha.sum()/1e6)  # 89.64 110.52 121.26 768.8

# The clip is lossless — source already has 6,956,407
with rasterio.open(NLUM + 'Land-use_of_Australia_2010-11/lu10v5ug/w001001.adf') as src:
    print(src.width, src.height)                        # 4150 3401
    print(get_data_window(src.read(1, masked=True)))    # Window(col_off=42, row_off=2, width=4071, height=3364)
    print((~np.ma.getmaskarray(src.read(1, masked=True))).sum())   # 6956407
```
