import os
import netCDF4
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import xarray as xr
import fiona

from rasterio.warp import reproject
from joblib import Parallel, delayed
from pyproj import CRS
from affine import Affine



'''
Reproject NVIS Extant + Pre-European Major Vegetation Groups and Subgroups rasters to match NLUM, save to GeoTiff and NetCDF
'''

mask_GEOTIFF = 'N:/Data-Master/National_Landuse_Map/NLUM_2010-11_clip.tif'
area_ha_GEOTIFF = 'N:/Data-Master/National_Landuse_Map/NLUM_2010-11_cell_ha.tif'

# Get metadata from mask_GEOTIFF
with rasterio.open(mask_GEOTIFF) as rst:
    # Load a 2D masked array with nodata masked out
    NLUM_ID_raster = rst.read(1, masked=True) 
    NLUM_mask = NLUM_ID_raster.mask == False
    # Get metadata and update parameters
    NLUM_transform = rst.transform
    NLUM_crs = rst.crs
    meta = rst.meta.copy()
    meta.update(compress='lzw', driver='GTiff') # , dtype='int32', nodata='0')
    [meta.pop(key) for key in ['dtype', 'nodata', 'count', 'driver']] # Need to add dtype and nodata manually when exporting GeoTiffs

# Get real area in hectares from area_ha_GEOTIFF
with rasterio.open(area_ha_GEOTIFF) as rst:
    NLUM_area = rst.read(1)
    NLUM_area_mask = NLUM_area[NLUM_mask]
    

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

# Present Vegetation Groups and Subgroups
fiona.listlayers('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb')
     
# Pre1750 Vegetation Groups and Subgroups
fiona.listlayers('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb')


# Set paths and layer names
files = [
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb','NVIS7_0_AUST_PRE_MVG_ALB', 'VAT_NVIS7_0_AUST_PRE_MVG_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb','NVIS7_0_AUST_PRE_MVS_ALB', 'VAT_NVIS7_0_AUST_PRE_MVS_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb','NVIS7_0_AUST_EXT_MVG_ALB', 'VAT_NVIS7_0_AUST_EXT_MVG_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb','NVIS7_0_AUST_EXT_MVS_ALB', 'VAT_NVIS7_0_AUST_EXT_MVS_ALB')
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
    save_path = f'{os.path.dirname(gdb_path)}/{layer_raster}.tif'
    with rasterio.open(save_path, 'w', **meta, PROFILE='GEOTIFF', count=dst_array.shape[0], dtype=dst_array.dtype) as dst:
        # Write each band to the raster
        for i in range(dst_array.shape[0]):
            dst.write(dst_array[i], i+1)
        # Set band descriptions
        dst.descriptions = tuple(src_att['Value'].astype(str).str.zfill(2).values)
        

    # Get the cells based on NLUM mask
    dst_array_flat = dst_array[:,NLUM_mask]
    # Create xarray DataArray with group and cell dimensions
    dst_array_xr = xr.DataArray(
        dst_array_flat, 
        dims=['group', 'cell'], 
        coords={'group':src_att['NAME'], 'cell':np.arange(dst_array_flat.shape[1])}
    )
    
    
    # Save xarray DataArray to NetCDF
    save_path = f'{os.path.dirname(gdb_path)}/{layer_raster}.nc'
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}} 
    dst_array_xr.name = 'data'
    dst_array_xr.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')




# --------------- Remove invalid vegetation class; Apply spatial mask ---------------

# Get group names for both pre-European and extant vegetation
PRE_mvg_groups = pd.read_csv('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS7_0_AUST_PRE_MVG_ALB_lookup.csv')['NAME'].tolist()
PRE_mvs_groups = pd.read_csv('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS7_0_AUST_PRE_MVS_ALB_lookup.csv')['NAME'].tolist()
EXT_mvg_groups = pd.read_csv('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS7_0_AUST_EXT_MVG_ALB_lookup.csv')['NAME'].tolist()
EXT_mvs_groups = pd.read_csv('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS7_0_AUST_EXT_MVS_ALB_lookup.csv')['NAME'].tolist()


# # Some groups are undertmined and should be exclude from the analysis, such as 'Other ...', 'Unknown/no data', and 'Unclassified'.
# MVG_rm_names = [i for i in PRE_mvg_groups if 'Other' in i or 'Unknown' in i or 'Unclassified' in i]     # 7 groups
# MVS_rm_names = [i for i in PRE_mvs_groups if 'Other' in i or 'Unknown' in i or 'Unclassified' in i]     # 11 groups

# # Names appread in extant but not in pre-European should be removed
# MVG_rm_names += [i for i in EXT_mvg_groups if i not in PRE_mvg_groups]  # 11 groups
# MVS_rm_names += [i for i in EXT_mvs_groups if i not in PRE_mvs_groups]  # 16 groups

# # Remove duplicates
# rm_names = list(set(MVG_rm_names + MVS_rm_names))


rm_names = ['Unknown/no data', 'Unknown/No data']


# Calculate the sum of all groups for each raster
for gdb_path, layer_raster, layer_attribute in files:
    
    # Read NVIS raster and filter out the groups that should be removed
    dst_array_xr = xr.load_dataarray(f'{os.path.dirname(gdb_path)}/{layer_raster}.nc')
    dst_array_xr = dst_array_xr.sel(group=~dst_array_xr.group.isin(rm_names))
    
    # Reorder the groups lexicographically
    dst_array_xr = dst_array_xr.sortby('group')
    
    # Optioin-1: Use the index of the largest group value to represent the cell
    dst_array_xr_argmax = dst_array_xr.argmax(dim='group')     
    dst_array_xr_argmax_area_ha = np.bincount(dst_array_xr_argmax.values, weights=NLUM_area_mask, minlength=dst_array_xr_argmax.max().values+1)
    dst_array_xr_argmax_area_ha = pd.DataFrame({'group':dst_array_xr.coords['group'], 'AREA_HA':dst_array_xr_argmax_area_ha})
    
    # Option-2: Split each group as a separate layer, which is the percentage [0-100] of the group in each cell
    dst_array_xr_area_ha = dst_array_xr * NLUM_area_mask[None,:]
    dst_array_xr_group_area_ha = dst_array_xr_area_ha.sum(dim='cell').compute().to_dataframe(name='AREA_HA').reset_index()
    
    # Save xarray DataArray to NetCDF
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}}
    output_layer_name = layer_raster.replace('_ALB', '')
    
    save_path = f'{os.path.dirname(gdb_path)}/{output_layer_name}_LOW_SPATIAL_DETAIL.nc'
    dst_array_xr_argmax.name = 'data'
    dst_array_xr_argmax.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')
    
    save_path = f'{os.path.dirname(gdb_path)}/{output_layer_name}_HIGH_SPATIAL_DETAIL.nc'
    dst_array_xr.name = 'data'
    dst_array_xr.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')
    
 




# --------------- Get the sum of areas (ha) for pre-1750 ---------------

# Read raw zones and lumap database
zones = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5', key='cell_zones_df', columns=['CELL_HA'])
bioph = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5', key = 'cell_biophysical_df', columns=['NATURAL_AREA_INC_WATER'])
lumap = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', key = 'cell_LU_mapping', columns=['LU_DESC'])

natural_cells = np.logical_not(bioph['NATURAL_AREA_INC_WATER'].values) # 0 is natural, 1 is non-natural; so we flip the values to make 1 natural

# Get the index of cells that are outside the LUTO study area, AND, also in natural state
idx_out_LUTO = np.isin(lumap['LU_DESC'], ['Non-agricultural land'])     # shape=6956407, sum=2737674
idx_out_LUTO_natural = idx_out_LUTO & natural_cells

# Get the index of cells that are inside the LUTO study area, AND, also in natural state
idx_in_LUTO_natural = np.isin(lumap['LU_DESC'], ['Beef - natural land', 'Dairy - natural land', 'Sheep - natural land', 'Unallocated - natural land'])



# Read NVIS data
PRE1750_path = 'N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL'

NVIS_pre_mvg_xr_low_spatial_detail = xr.load_dataarray(f'{PRE1750_path}/NVIS7_0_AUST_PRE_MVG_LOW_SPATIAL_DETAIL.nc')
NVIS_pre_mvs_xr_low_spatial_detail = xr.load_dataarray(f'{PRE1750_path}/NVIS7_0_AUST_PRE_MVS_LOW_SPATIAL_DETAIL.nc')
NVIS_pre_mvg_xr_high_spatial_detail = xr.load_dataarray(f'{PRE1750_path}/NVIS7_0_AUST_PRE_MVG_HIGH_SPATIAL_DETAIL.nc') / 100  # Convert percentage to fraction
NVIS_pre_mvs_xr_high_spatial_detail = xr.load_dataarray(f'{PRE1750_path}/NVIS7_0_AUST_PRE_MVS_HIGH_SPATIAL_DETAIL.nc') / 100  # Convert percentage to fraction

# Get the NVIS names
NVIS_pre_mvg_names = NVIS_pre_mvg_xr_high_spatial_detail.coords['group'].values.tolist()
NVIS_pre_mvs_names = NVIS_pre_mvs_xr_high_spatial_detail.coords['group'].values.tolist()


# --------------- Total vegataion area (ha) pre-1750 ---------------

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------
NVIS_pre_mvg_total_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvg_xr_low_spatial_detail.values,
    weights = zones['CELL_HA'].values,
    minlength = NVIS_pre_mvg_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvs_total_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvs_xr_low_spatial_detail.values,
    weights = zones['CELL_HA'].values,
    minlength = NVIS_pre_mvs_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvg_total_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvg_names, 'TOTAL_AREA_HA': NVIS_pre_mvg_total_ha_low_spatial_detail})
NVIS_pre_mvs_total_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvs_names, 'TOTAL_AREA_HA': NVIS_pre_mvs_total_ha_low_spatial_detail})




# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------
NVIS_pre_mvg_total_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail * zones['CELL_HA'].values[None, :]
NVIS_pre_mvs_total_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail * zones['CELL_HA'].values[None, :]
NVIS_pre_mvg_total_ha_df_high_spatial_detail = NVIS_pre_mvg_total_ha_high_spatial_detail.sum(dim='cell').to_dataframe('TOTAL_AREA_HA').reset_index()
NVIS_pre_mvs_total_ha_df_high_spatial_detail = NVIS_pre_mvs_total_ha_high_spatial_detail.sum(dim='cell').to_dataframe('TOTAL_AREA_HA').reset_index()





# --------------- Vegataion area outside the LUTO study area ---------------

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------
NVIS_pre_mvg_outside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvg_xr_low_spatial_detail.sel(cell=idx_out_LUTO_natural).values, 
    weights = zones['CELL_HA'].values[idx_out_LUTO_natural],
    minlength = NVIS_pre_mvg_xr_low_spatial_detail.max().values + 1
)


NVIS_pre_mvs_outside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvs_xr_low_spatial_detail.sel(cell=idx_out_LUTO_natural).values,
    weights = zones['CELL_HA'].values[idx_out_LUTO_natural],
    minlength = NVIS_pre_mvs_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvg_outside_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvg_names,'OUTSIDE_LUTO_AREA_HA': NVIS_pre_mvg_outside_ha_low_spatial_detail})
NVIS_pre_mvs_outside_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvs_names,'OUTSIDE_LUTO_AREA_HA': NVIS_pre_mvs_outside_ha_low_spatial_detail})


# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------

NVIS_pre_mvg_outside_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail.sel(cell=idx_out_LUTO_natural) * zones['CELL_HA'].values[None, idx_out_LUTO_natural]     
NVIS_pre_mvs_outside_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail.sel(cell=idx_out_LUTO_natural) * zones['CELL_HA'].values[None, idx_out_LUTO_natural]     
NVIS_pre_mvg_outside_ha_df_high_spatial_detail = NVIS_pre_mvg_outside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('OUTSIDE_LUTO_AREA_HA').reset_index()
NVIS_pre_mvs_outside_ha_df_high_spatial_detail = NVIS_pre_mvs_outside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('OUTSIDE_LUTO_AREA_HA').reset_index()





# --------------- Vegataion area inside the LUTO study area ---------------

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------

NVIS_pre_mvg_inside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvg_xr_low_spatial_detail.sel(cell=idx_in_LUTO_natural).values, 
    weights = zones['CELL_HA'].values[idx_in_LUTO_natural],
    minlength = NVIS_pre_mvg_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvs_inside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvs_xr_low_spatial_detail.sel(cell=idx_in_LUTO_natural).values,
    weights = zones['CELL_HA'].values[idx_in_LUTO_natural],
    minlength = NVIS_pre_mvs_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvg_inside_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvg_names,'INSIDE_LUTO_AREA_HA': NVIS_pre_mvg_inside_ha_low_spatial_detail})
NVIS_pre_mvs_inside_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvs_names,'INSIDE_LUTO_AREA_HA': NVIS_pre_mvs_inside_ha_low_spatial_detail})


# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------
NVIS_pre_mvg_inside_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail.sel(cell=idx_in_LUTO_natural) * zones['CELL_HA'].values[None, idx_in_LUTO_natural]
NVIS_pre_mvs_inside_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail.sel(cell=idx_in_LUTO_natural) * zones['CELL_HA'].values[None, idx_in_LUTO_natural]
NVIS_pre_mvg_inside_ha_df_high_spatial_detail = NVIS_pre_mvg_inside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('INSIDE_LUTO_AREA_HA').reset_index()
NVIS_pre_mvs_inside_ha_df_high_spatial_detail = NVIS_pre_mvs_inside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('INSIDE_LUTO_AREA_HA').reset_index()









# ------------- Combine 'HIGH' and 'LOW' -------------

# Concatenate the two dataframes
NVIS_pre_mvg_low_spatial_detail = NVIS_pre_mvg_total_ha_df_low_spatial_detail.merge(NVIS_pre_mvg_outside_ha_df_low_spatial_detail, on='group')
NVIS_pre_mvg_high_spatial_detail = NVIS_pre_mvg_total_ha_df_high_spatial_detail.merge(NVIS_pre_mvg_outside_ha_df_high_spatial_detail, on='group')

NVIS_pre_mvs_low_spatial_detail = NVIS_pre_mvs_total_ha_df_low_spatial_detail.merge(NVIS_pre_mvs_outside_ha_df_low_spatial_detail, on='group')
NVIS_pre_mvs_high_spatial_detail = NVIS_pre_mvs_total_ha_df_high_spatial_detail.merge(NVIS_pre_mvs_outside_ha_df_high_spatial_detail, on='group')

# Append a user-defined target column
NVIS_pre_mvg_low_spatial_detail['CONSERVATION_TARGET_PCT'] = 30
NVIS_pre_mvg_high_spatial_detail['CONSERVATION_TARGET_PCT'] = 30
NVIS_pre_mvs_low_spatial_detail['CONSERVATION_TARGET_PCT'] = 30
NVIS_pre_mvs_high_spatial_detail['CONSERVATION_TARGET_PCT'] = 30

# Save to CSV
NVIS_pre_mvg_low_spatial_detail.to_csv(PRE1750_path + '/NVIS_MVG_LOW_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvg_high_spatial_detail.to_csv(PRE1750_path + '/NVIS_MVG_HIGH_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvs_low_spatial_detail.to_csv(PRE1750_path + '/NVIS_MVS_LOW_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvs_high_spatial_detail.to_csv(PRE1750_path + '/NVIS_MVS_HIGH_SPATIAL_DETAIL.csv', index=False)




# ------------------------- TMP -------------------------

save_path = 'N:/LUF-Modelling/LUTO2_JZ/TEMP/vegetation_pre_calc_area_ha'

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------

NVIS_pre_mvg_low_spatial_detail_area_ha = pd.concat([
    NVIS_pre_mvg_total_ha_df_low_spatial_detail.set_index('group'), 
    NVIS_pre_mvg_outside_ha_df_low_spatial_detail.set_index('group'),
    NVIS_pre_mvg_inside_ha_df_low_spatial_detail.set_index('group')], axis=1).reset_index()

NVIS_pre_mvg_low_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_AREA_HA'] = NVIS_pre_mvg_low_spatial_detail_area_ha.eval('OUTSIDE_LUTO_AREA_HA	+ INSIDE_LUTO_AREA_HA')
NVIS_pre_mvg_low_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_to_TOTAL'] = NVIS_pre_mvg_low_spatial_detail_area_ha.eval('INSIDE_OUTSIDE_SUM_AREA_HA / TOTAL_AREA_HA')
NVIS_pre_mvg_low_spatial_detail_area_ha.to_csv(f'{save_path}/NVIS_pre_mvg_low_spatial_detail_area_ha.csv', index=False)

NVIS_pre_mvg_high_spatial_detail_area_ha = pd.concat([
    NVIS_pre_mvg_total_ha_df_high_spatial_detail.set_index('group'), 
    NVIS_pre_mvg_outside_ha_df_high_spatial_detail.set_index('group'),
    NVIS_pre_mvg_inside_ha_df_high_spatial_detail.set_index('group')], axis=1).reset_index()

NVIS_pre_mvg_high_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_AREA_HA'] = NVIS_pre_mvg_high_spatial_detail_area_ha.eval('OUTSIDE_LUTO_AREA_HA	+ INSIDE_LUTO_AREA_HA')
NVIS_pre_mvg_high_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_to_TOTAL'] = NVIS_pre_mvg_high_spatial_detail_area_ha.eval('INSIDE_OUTSIDE_SUM_AREA_HA / TOTAL_AREA_HA')
NVIS_pre_mvg_high_spatial_detail_area_ha.to_csv(f'{save_path}//NVIS_pre_mvg_high_spatial_detail_area_ha.csv', index=False)



# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------

NVIS_pre_mvs_low_spatial_detail_area_ha = pd.concat([
    NVIS_pre_mvs_total_ha_df_low_spatial_detail.set_index('group'), 
    NVIS_pre_mvs_outside_ha_df_low_spatial_detail.set_index('group'),
    NVIS_pre_mvs_inside_ha_df_low_spatial_detail.set_index('group')], axis=1).reset_index()

NVIS_pre_mvs_low_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_AREA_HA'] = NVIS_pre_mvs_low_spatial_detail_area_ha.eval('OUTSIDE_LUTO_AREA_HA	+ INSIDE_LUTO_AREA_HA')
NVIS_pre_mvs_low_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_to_TOTAL'] = NVIS_pre_mvs_low_spatial_detail_area_ha.eval('INSIDE_OUTSIDE_SUM_AREA_HA / TOTAL_AREA_HA')
NVIS_pre_mvs_low_spatial_detail_area_ha.to_csv(f'{save_path}//NVIS_pre_mvs_low_spatial_detail_area_ha.csv', index=False)


NVIS_pre_mvs_high_spatial_detail_area_ha = pd.concat([
    NVIS_pre_mvs_total_ha_df_high_spatial_detail.set_index('group'), 
    NVIS_pre_mvs_outside_ha_df_high_spatial_detail.set_index('group'),
    NVIS_pre_mvs_inside_ha_df_high_spatial_detail.set_index('group')], axis=1).reset_index()

NVIS_pre_mvs_high_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_AREA_HA'] = NVIS_pre_mvs_high_spatial_detail_area_ha.eval('OUTSIDE_LUTO_AREA_HA	+ INSIDE_LUTO_AREA_HA')
NVIS_pre_mvs_high_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_to_TOTAL'] = NVIS_pre_mvs_high_spatial_detail_area_ha.eval('INSIDE_OUTSIDE_SUM_AREA_HA / TOTAL_AREA_HA')
NVIS_pre_mvs_high_spatial_detail_area_ha.to_csv(f'{save_path}/NVIS_pre_mvs_high_spatial_detail_area_ha.csv', index=False)

# --------------------------  TMP END --------------------------