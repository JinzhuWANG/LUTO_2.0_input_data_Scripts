import os
import h5py
import numpy as np
import numpy.ma as ma
import geopandas as gpd
import rasterio
import xarray as xr
import fiona

from scipy import ndimage as nd
from rasterio.warp import reproject
from joblib import Parallel, delayed
from pyproj import CRS
from affine import Affine



'''
Reproject NVIS Extant + Pre-European Major Vegetation Groups and Subgroups rasters, match NLUM, save GeoTiff
'''

ref_GEOTIFF = 'N:/Data-Master/National_Landuse_Map/NLUM_2010-11_clip.tif'

# Open NLUM_ID as mask raster and get metadata
with rasterio.open(ref_GEOTIFF) as rst:
    # Load a 2D masked array with nodata masked out
    NLUM_ID_raster = rst.read(1, masked=True) 
    NLUM_mask = NLUM_ID_raster.mask == False
    # Get metadata and update parameters
    NLUM_transform = rst.transform
    NLUM_crs = rst.crs
    meta = rst.meta.copy()
    meta.update(compress='lzw', driver='GTiff') # , dtype='int32', nodata='0')
    [meta.pop(key) for key in ['dtype', 'nodata', 'count', 'driver']] # Need to add dtype and nodata manually when exporting GeoTiffs



############## Functions

def reproject_and_average(val:int, src_arr:np.ndarray, src_trans:Affine, src_crs:CRS, target_meta:dict):
    zero_arr = np.zeros((meta.get('height'), meta.get('width')), np.float32)
    band_arr = (src_arr == val).astype(np.uint8)
    reproject(
        band_arr, 
        zero_arr, 
        resampling=rasterio.enums.Resampling.average, 
        src_transform=src_trans, 
        src_crs=src_crs, 
        dst_transform=target_meta.get('transform'), 
        dst_crs=target_meta.get('crs')
    )
    return np.round(zero_arr * 100, 0).astype(np.uint8)


############## List raster layers in NVIS geoDatabase folder

# Present Major Vegetation Groups and Subgroups
fiona.listlayers('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb')
     
# Pre1750 Major Vegetation Groups and Subgroups
fiona.listlayers('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb')


# Set paths and layer names
files = [
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb','NVIS7_0_AUST_EXT_MVG_ALB', 'VAT_NVIS7_0_AUST_EXT_MVG_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb','NVIS7_0_AUST_EXT_MVS_ALB', 'VAT_NVIS7_0_AUST_EXT_MVS_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb','NVIS7_0_AUST_PRE_MVG_ALB', 'VAT_NVIS7_0_AUST_PRE_MVG_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb','NVIS7_0_AUST_PRE_MVS_ALB', 'VAT_NVIS7_0_AUST_PRE_MVS_ALB')
]

# Set number of workers for parallel processing
n_workers = 10

# Loop through each raster layer and reproject to match NLUM
for gdb_path, layer_raster, layer_attribute in files:
    
    # Read NVIS raster and reproject to match NLUM
    with rasterio.open(f'OpenFileGDB:{gdb_path}:{layer_raster}') as src: 
        src_arr = src.read(1)
    # Load in look-up tables of MVG and MVS names
    src_att = gpd.read_file(gdb_path, layer=layer_attribute)
    # Rename column that contains 'NAME' to 'NAME'
    src_att = src_att.rename(columns={src_att.filter(like='NAME').columns[0]: 'NAME'})
    src_att = src_att[['Value', 'NAME']]
    src_att.to_csv(f'{os.path.dirname(gdb_path)}/{layer_raster}_lookup.csv', index=False)

    # Create a list of delayed jobs, so we can reproject and average rasters in parallel with `n_workers`
    jobs = [delayed(reproject_and_average)(val, src_arr, src.transform, src.crs, meta) for val in src_att['Value']]
    dst_array = np.stack(Parallel(n_jobs=n_workers)(jobs), axis=0)
    
    # Save reprojected raster to GeoTiff
    save_path = f'{os.path.dirname(gdb_path)}/{layer_raster}1.tif'
    with rasterio.open(save_path, 'w', **meta, PROFILE='GEOTIFF', count=dst_array.shape[0], dtype=dst_array.dtype) as dst:
        # Write each band to the raster
        for i in range(dst_array.shape[0]):
            dst.write(dst_array[i], i+1)
            dst.set_band_description(i+1, src_att['Value'][i])
        
        

    # # Get the cells based on NLUM mask
    # dst_array_flat = dst_array[:,NLUM_mask]
    # # Create xarray DataArray with group and cell dimensions
    # dst_array_xr = xr.DataArray(
    #     dst_array_flat, 
    #     dims=['group', 'cell'], 
    #     coords={'group':src_att['NAME'], 'cell':np.arange(dst_array_flat.shape[1])}
    # )
    

    # # Save xarray DataArray to NetCDF
    # save_path = f'{os.path.dirname(gdb_path)}/{layer_raster}.nc'
    # encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}} 
    # dst_array_xr.name = 'data'
    # dst_array_xr.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')






'''
Below is the original code for reference
'''

# ############## NVIS Pre-European Major Vegetation Groups

# with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_PRE_MVG/aus6_0p_mvg/w001000.adf') as src:
#     dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
#     reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))

# # Mask out nodata cells
# dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)

# # Fill nodata in raster using value of nearest cell to match NLUM mask
# ind = nd.distance_transform_edt(dst_array.mask, return_distances = False, return_indices = True)
# NVIS_raster_filled = dst_array[tuple(ind)]
# NVIS_raster_clipped = NVIS_raster_filled * NLUM_mask
    
# # Save as geoTiff
# with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_PRE_MVG/aus6_0p_mvg.tif', 'w+', nodata = 0, dtype = 'uint8', **meta) as dst:
#     dst.write_band(1, NVIS_raster_clipped)

# # Flatten 2D array to 1D array of valid values only, add NVIS to cell_df dataframe
# cell_df['NVIS_PRE_EURO_MVG_ID'] = NVIS_raster_clipped[NLUM_mask]

# # Join the lookup table to the cell_df DataFrame
# cell_df = cell_df.merge(NVIS_MVG_LUT, left_on = 'NVIS_PRE_EURO_MVG_ID', right_on = 'MVG_ID', how = 'left')
# cell_df.rename(columns = {'Major Vegetation Group':'NVIS_PRE_EURO_MVG_NAME'}, inplace = True)
# cell_df = cell_df.drop(columns = ['MVG_ID'])
