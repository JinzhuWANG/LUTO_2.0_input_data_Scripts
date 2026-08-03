
from rasterio.enums import Resampling
from scipy.ndimage import distance_transform_edt


def fill_with_nearest(data_2d, to_fill=0):
    """Fill gaps in a 2D DataArray with the value of the nearest valid cell.

    NaN and `to_fill` are both treated as gaps. The default `to_fill=0` therefore
    overwrites legitimate zeros — fine when zero means 'no data', wrong when zero is
    a real value. To fill only NaN, pass a sentinel that cannot occur in the data
    (e.g. `to_fill=-1` for a 0-100 layer); the fill then becomes a no-op for zeros.

    Mutates `data_2d` in place (and also returns it).
    """
    mask = data_2d.isnull() | (data_2d == to_fill)
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    data_2d.values = data_2d.values[tuple(indices)]
    return data_2d


def reproject_and_fill(raw_raster, template, mask, to_fill=0, resampling=Resampling.nearest):
    """Reproject a raster onto `template`, fill its gaps, and mask to valid cells.

    `raw_raster` is expected to still carry a `band` dimension; band 1 is selected
    after the reproject. `to_fill` is passed to `fill_with_nearest` — see its docstring
    for how to disable the fill when zero is a meaningful value. `mask` must be boolean.

    Returns a 2D DataArray aligned to `template`.
    """
    # Reproject and match to LUTO template
    matched = raw_raster.rio.reproject_match(template, resampling=resampling).sel(band=1, drop=True)
    # Fill nan values with nearest neighbor interpolation
    filled = fill_with_nearest(matched, to_fill=to_fill)
    # Mask to LUTO valid areas
    masked = filled.where(mask)
    return masked
