import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy import ndimage as nd
import numpy.ma as ma
from dbfread import DBF
import rasterio, matplotlib, h5py
from rasterio import features
from rasterio.fill import fillnodata
from rasterio.warp import reproject
from rasterio.enums import Resampling


# Set some options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.float_format', '{:,.4f}'.format)


# Read cell_df from disk, just grab the CELL_ID column
cell_df = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5')[['CELL_ID', 
                                                                                                             'CELL_HA', 
                                                                                                             'PRIMARY_V7', 
                                                                                                             'HR_DRAINDIV_NAME', 
                                                                                                             'NVIS_PRE_EURO_MVG_ID', 
                                                                                                             'NVIS_PRE_EURO_MVG_NAME']]


################################ Create some helper functions

# Open NLUM mask raster and get metadata
with rasterio.open('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as rst:

    # Read geotiff to numpy array
    NLUM_mask = rst.read(1) # Loads a 2D masked array with nodata masked out
    
    # Get transform and metadata and update parameters
    NLUM_transform = rst.transform
    NLUM_height = rst.height
    NLUM_width = rst.width
    NLUM_crs = rst.crs
    meta = rst.meta.copy()
    meta.update(compress='lzw', driver='GTiff') # dtype='int32', nodata='-99')
    [meta.pop(key) for key in ['dtype', 'nodata']] # Need to add dtype and nodata manually when exporting GeoTiffs
        
    # Set some data structures to enable conversion on 1D arrays to 2D
    array_2D = np.zeros(NLUM_mask.shape) - 9999
    xy = np.nonzero(NLUM_mask)
    
    
# Convert 1D column to 2D spatial array
def conv_1D_to_2D(in_1D_array):
    array_2D[xy] = np.array(in_1D_array)
    return array_2D.astype(in_1D_array.dtype)


# Print array stats
def desc(inarray):
    print('Shape =', inarray.shape, ' Mean =', inarray.mean(), ' Max =', inarray.max(), ' Min =', inarray.min(), ' NaNs =', np.sum(np.isnan(inarray)))


# Convert 1D column to 2D spatial array and plot map
def map_in_2D(col, data): # data = 'continuous' or 'categorical'
    a2D = conv_1D_to_2D(col)
    if data == 'categorical':
        n = col.nunique()
        cmap = matplotlib.colors.ListedColormap(np.random.rand(n,3))
        plt.imshow(a2D, cmap=cmap, resample=False)
    elif data == 'continuous':
        plt.imshow(a2D, cmap='pink', resample=False)
    plt.show()


# Convert object columns to categories and downcast int64 columns to save memory and space
def downcast(dframe):
    obj_cols = dframe.select_dtypes(include = ['object']).columns
    dframe[obj_cols] = dframe[obj_cols].astype('category')
    int_cols = dframe.select_dtypes(include = ['integer']).columns
    dframe[int_cols] = dframe[int_cols].apply(pd.to_numeric, downcast = 'integer')
    fcols = dframe.select_dtypes('float').columns
    dframe[fcols] = dframe[fcols].apply(pd.to_numeric, downcast = 'float')





########### Avoided clearance of native vegetation (based on Roxburgh's M [maximum aboveground biomass])

with rasterio.open('N:/Data-Master/Emissions_Reduction_Fund/Maximum_aboveground_biomass_M/Version_2/New_M_2019.tif') as src:
    
    # Create a destination array with nodata = -9999
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32) - 9999
    
    # Reproject/resample input raster to match NLUM mask (meta)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.bilinear)
    
# Mask out nodata cells
dst_array = ma.masked_where(dst_array <= -9999, dst_array)

# Fill nodata in raster using value of nearest cell to match NLUM mask
ind = nd.distance_transform_edt(dst_array.mask, return_distances = False, return_indices = True)
dst_array_filled = dst_array[tuple(ind)]
dst_array_masked = np.where(NLUM_mask == 0, -9999, dst_array_filled)

# Save the output to GeoTiff
with rasterio.open('N:/Data-Master/Emissions_Reduction_Fund/Maximum_aboveground_biomass_M/Version_2/ROXBURGHS_M_T_DM_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, dst_array_masked)

# Flatten 2D array to 1D array of valid values only
rox_M_tDM_ha = dst_array_masked[NLUM_mask == 1]

# Load aboveground and belowground biomass from FullCAM estimates. Grab year 91 AGB and BGB. Shape = (6956407, 3)
ep_C_AGB_BGB = np.load('N:/Data-Master/FullCAM/Output_layers/ep_block_AGB_BGB.npy')[..., 94]

# Calculate the proportion of biomass that is carbon. Should be around 0.492
pct_carbon = ep_C_AGB_BGB[:, 0] / (ep_C_AGB_BGB[:, 1] + ep_C_AGB_BGB[:, 2])

# Calculate total CO2 in maximum AGB and BGB and add data to cell_df dataframe
cell_df['REMNANT_VEG_T_CO2_HA'] = rox_M_tDM_ha * pct_carbon * (44 / 12) / (ep_C_AGB_BGB[:, 1] / (ep_C_AGB_BGB[:, 1] + ep_C_AGB_BGB[:, 2]))

# Save the output to GeoTiff
with rasterio.open('N:/Data-Master/Emissions_Reduction_Fund/Maximum_aboveground_biomass_M/Version_2/REMNANT_VEG_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['REMNANT_VEG_T_CO2_HA']))




############## Average annual carbon sequestration by reforestation land uses

path = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/3D_Spatial_Timeseries/'
gpath = 'N:/Data-Master/FullCAM/Output_AnnAvg_GeoTiffs/'

with h5py.File(path + 'tCO2_ha_ep_block.h5', 'r') as h5f:
    cell_df['EP_BLOCK_TREES_AVG_T_CO2_HA_YR'] = np.mean(h5f['Trees_tCO2_ha'], axis = 0) / 91
    cell_df['EP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR'] = np.mean(h5f['Debris_tCO2_ha'], axis = 0) / 91
    cell_df['EP_BLOCK_SOIL_AVG_T_CO2_HA_YR'] = np.mean(h5f['Soil_tCO2_ha'], axis = 0) / 91

with h5py.File(path + 'tCO2_ha_ep_rip.h5', 'r') as h5f:
    cell_df['EP_RIP_TREES_AVG_T_CO2_HA_YR'] = np.mean(h5f['Trees_tCO2_ha'], axis = 0) / 91
    cell_df['EP_RIP_DEBRIS_AVG_T_CO2_HA_YR'] = np.mean(h5f['Debris_tCO2_ha'], axis = 0) / 91
    cell_df['EP_RIP_SOIL_AVG_T_CO2_HA_YR'] = np.mean(h5f['Soil_tCO2_ha'], axis = 0) / 91

with h5py.File(path + 'tCO2_ha_ep_belt.h5', 'r') as h5f:
    cell_df['EP_BELT_TREES_AVG_T_CO2_HA_YR'] = np.mean(h5f['Trees_tCO2_ha'], axis = 0) / 91
    cell_df['EP_BELT_DEBRIS_AVG_T_CO2_HA_YR'] = np.mean(h5f['Debris_tCO2_ha'], axis = 0) / 91
    cell_df['EP_BELT_SOIL_AVG_T_CO2_HA_YR'] = np.mean(h5f['Soil_tCO2_ha'], axis = 0) / 91


with h5py.File(path + 'tCO2_ha_cp_block.h5', 'r') as h5f:
    cell_df['CP_BLOCK_TREES_AVG_T_CO2_HA_YR'] = np.mean(h5f['Trees_tCO2_ha'], axis = 0) / 91
    cell_df['CP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR'] = np.mean(h5f['Debris_tCO2_ha'], axis = 0) / 91
    cell_df['CP_BLOCK_SOIL_AVG_T_CO2_HA_YR'] = np.mean(h5f['Soil_tCO2_ha'], axis = 0) / 91

with h5py.File(path + 'tCO2_ha_cp_belt.h5', 'r') as h5f:
    cell_df['CP_BELT_TREES_AVG_T_CO2_HA_YR'] = np.mean(h5f['Trees_tCO2_ha'], axis = 0) / 91
    cell_df['CP_BELT_DEBRIS_AVG_T_CO2_HA_YR'] = np.mean(h5f['Debris_tCO2_ha'], axis = 0) / 91
    cell_df['CP_BELT_SOIL_AVG_T_CO2_HA_YR'] = np.mean(h5f['Soil_tCO2_ha'], axis = 0) / 91

    
with h5py.File(path + 'tCO2_ha_hir_block.h5', 'r') as h5f:
    cell_df['HIR_BLOCK_TREES_AVG_T_CO2_HA_YR'] = np.mean(h5f['Trees_tCO2_ha'], axis = 0) / 91
    cell_df['HIR_BLOCK_DEBRIS_AVG_T_CO2_HA_YR'] = np.mean(h5f['Debris_tCO2_ha'], axis = 0) / 91
    cell_df['HIR_BLOCK_SOIL_AVG_T_CO2_HA_YR'] = np.mean(h5f['Soil_tCO2_ha'], axis = 0) / 91

with h5py.File(path + 'tCO2_ha_hir_rip.h5', 'r') as h5f:
    cell_df['HIR_RIP_TREES_AVG_T_CO2_HA_YR'] = np.mean(h5f['Trees_tCO2_ha'], axis = 0) / 91
    cell_df['HIR_RIP_DEBRIS_AVG_T_CO2_HA_YR'] = np.mean(h5f['Debris_tCO2_ha'], axis = 0) / 91
    cell_df['HIR_RIP_SOIL_AVG_T_CO2_HA_YR'] = np.mean(h5f['Soil_tCO2_ha'], axis = 0) / 91


# Save the output to GeoTiff - TOTAL CO2 sequestration over 91 years

with rasterio.open(gpath + 'EP_BLOCK_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_TREES_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'EP_BLOCK_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'EP_BLOCK_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_SOIL_AVG_T_CO2_HA_YR'] * 91))

with rasterio.open(gpath + 'EP_RIP_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_TREES_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'EP_RIP_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_DEBRIS_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'EP_RIP_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_SOIL_AVG_T_CO2_HA_YR'] * 91))

with rasterio.open(gpath + 'EP_BELT_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_TREES_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'EP_BELT_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_DEBRIS_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'EP_BELT_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_SOIL_AVG_T_CO2_HA_YR'] * 91))


with rasterio.open(gpath + 'CP_BLOCK_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_TREES_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'CP_BLOCK_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'CP_BLOCK_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_SOIL_AVG_T_CO2_HA_YR'] * 91))
    
with rasterio.open(gpath + 'CP_BELT_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_TREES_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'CP_BELT_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_DEBRIS_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'CP_BELT_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_SOIL_AVG_T_CO2_HA_YR'] * 91))


with rasterio.open(gpath + 'HIR_BLOCK_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_TREES_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'HIR_BLOCK_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_DEBRIS_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'HIR_BLOCK_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_SOIL_AVG_T_CO2_HA_YR'] * 91))
    
with rasterio.open(gpath + 'HIR_RIP_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_RIP_TREES_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'HIR_RIP_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_RIP_DEBRIS_AVG_T_CO2_HA_YR'] * 91))
with rasterio.open(gpath + 'HIR_RIP_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_RIP_SOIL_AVG_T_CO2_HA_YR'] * 91))


# Save the output to GeoTiff - ANNUAL AVERAGE CO2 sequestration over 91 years

with rasterio.open(gpath + 'EP_BLOCK_TREES_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_TREES_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'EP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'EP_BLOCK_SOIL_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_SOIL_AVG_T_CO2_HA_YR']))

with rasterio.open(gpath + 'EP_RIP_TREES_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_TREES_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'EP_RIP_DEBRIS_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_DEBRIS_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'EP_RIP_SOIL_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_SOIL_AVG_T_CO2_HA_YR']))

with rasterio.open(gpath + 'EP_BELT_TREES_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_TREES_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'EP_BELT_DEBRIS_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_DEBRIS_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'EP_BELT_SOIL_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_SOIL_AVG_T_CO2_HA_YR']))


with rasterio.open(gpath + 'CP_BLOCK_TREES_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_TREES_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'CP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_DEBRIS_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'CP_BLOCK_SOIL_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_SOIL_AVG_T_CO2_HA_YR']))
    
with rasterio.open(gpath + 'CP_BELT_TREES_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_TREES_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'CP_BELT_DEBRIS_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_DEBRIS_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'CP_BELT_SOIL_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_SOIL_AVG_T_CO2_HA_YR']))


with rasterio.open(gpath + 'HIR_BLOCK_TREES_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_TREES_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'HIR_BLOCK_DEBRIS_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_DEBRIS_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'HIR_BLOCK_SOIL_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_SOIL_AVG_T_CO2_HA_YR']))
    
with rasterio.open(gpath + 'HIR_RIP_TREES_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_RIP_TREES_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'HIR_RIP_DEBRIS_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_RIP_DEBRIS_AVG_T_CO2_HA_YR']))
with rasterio.open(gpath + 'HIR_RIP_SOIL_AVG_T_CO2_HA_YR.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_RIP_SOIL_AVG_T_CO2_HA_YR']))
    


    
############## Length of watercourse occurring on each cell for riparian restoration

with rasterio.open('N:/Data-Master/Riparian_areas/Data/Riparian_areas/riverAtlas_riparianLength_30mBuff_18112021.tif') as src:
    cell_df['RIP_LENGTH_M_CELL'] = src.read(1)[NLUM_mask == 1].astype(np.float32)    





############## Total mass of soil organic carbon in top 30cm of soil

with rasterio.open('N:/Data-Master/Soils/Soil_Landscape_Grid_Australia/SOC_T_HA_30cm_AUS_1km.tif') as src:
    cell_df['SOC_T_HA_TOP_30CM'] = src.read(1)[NLUM_mask == 1].astype(np.float32)    




############## Establishment costs for reforestation

# Environmental plantings
with rasterio.open('N:/Data-Master/Establishment_costs/costs_tif/estabCostEnvPlant_ave.tif') as src:
    cell_df['EP_EST_COST_HA'] = src.read(1)[NLUM_mask == 1].astype(np.float32)    

# Carbon plantings
with rasterio.open('N:/Data-Master/Establishment_costs/costs_tif/estabCostCarbon_ave.tif') as src:
    cell_df['CP_EST_COST_HA'] = src.read(1)[NLUM_mask == 1].astype(np.float32)    

# Biomass plantings
with rasterio.open('N:/Data-Master/Establishment_costs/costs_tif/estabCostBiomass_ave.tif') as src:
    cell_df['BP_EST_COST_HA'] = src.read(1)[NLUM_mask == 1].astype(np.float32)    


    

############## Mean annual rainfall (1975 - 2005) from ANUCLIM modelled using Australian 9 second DEM

with rasterio.open('N:/Data-Master/ANUCLIM_climate_data/AUS_9sec_climate_data_2021/dem-9s_p12.tif') as src:
    
    # Create an empty destination array 
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32)
    
    # Reproject/resample input raster to match NLUM mask (meta)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.bilinear)
    
    # Create mask for filling cells
    fill_mask = np.where(dst_array > 0, 1, 0)
    
    # Fill nodata using inverse distance weighted averaging and mask to NLUM
    dst_array_filled = fillnodata(dst_array, fill_mask, max_search_distance = 100.0) * NLUM_mask
    
    # Save the output to GeoTiff
    with rasterio.open('N:/Data-Master/ANUCLIM_climate_data/AUS_9sec_climate_data_2021/AVG_AN_PREC_MM_YR.tif', 'w+', dtype = 'float32', nodata = 0, **meta) as dst:        
        dst.write_band(1, dst_array_filled)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array_filled[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_df['AVG_AN_PREC_MM_YR'] = np.round(dataFlat).astype(np.uint16)




############## Growing season rainfall (1975 - 2005) from WORLDCLIM monthly rainfall data

with rasterio.open('N:/Data-Master/WorldClim_CMIP6/Australia/Australia_1km/Monthly_20-year_snapshots/Historical_1970-2000/wc2.1_2.5m_prec_Historical_1970-2000_AUS_1km_Monthly.tif') as src:
    arr = src.read()
    
    # Sum growing season rainfall (April - October)
    grow_seas_prec = np.round(np.sum(arr[3:11, ...], axis = 0)).astype(np.int16)
    grow_seas_prec = np.where(grow_seas_prec > -9999, grow_seas_prec, -9999)
    
    # Save the output to GeoTiff
    with rasterio.open('N:/Data-Master/ANUCLIM_climate_data/AUS_9sec_climate_data_2021/AVG_GROW_SEAS_PREC_MM_YR.tif', 'w+', dtype = 'int16', nodata = -9999, **meta) as dst:        
        dst.write_band(1, grow_seas_prec)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = grow_seas_prec[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_df['AVG_GROW_SEAS_PREC_MM_YR'] = np.round(dataFlat).astype(np.int16)
    cell_df['AVG_GROW_SEAS_PREC_GE_175_MM_YR'] = cell_df.eval('AVG_GROW_SEAS_PREC_MM_YR >= 175')




############## Water use by trees from AWRA-L  ***DEPRECATED***  Now using the INVEST modelling

# with rasterio.open('N:/Data-Master/Water/water_use_by_trees/wrimpact.tif') as src:
    
#     # Create an empty destination array 
#     dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32)
    
#     # Reproject/resample input raster to match NLUM mask (meta)
#     reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.bilinear)
    
#     # Create mask for filling cells
#     fill_mask = np.where(dst_array > 0, 1, 0)
    
#     # Fill nodata using inverse distance weighted averaging and mask to NLUM
#     dst_array_filled = fillnodata(dst_array, fill_mask, max_search_distance = 100.0) * NLUM_mask
    
#     # Save the output to GeoTiff
#     with rasterio.open('N:/Data-Master/Water/water_use_by_trees/WATER_USE_TREES_ML_HA.tif', 'w+', dtype = 'float32', nodata = 0, **meta) as dst:        
#         dst.write_band(1, dst_array_filled)
    
#     # Flatten 2D array to 1D array of valid values only
#     dataFlat = dst_array_filled[NLUM_mask == 1]
        
#     # Add data to cell_df dataframe
#     cell_df['WATER_USE_TREES_KL_HA'] = np.round(dataFlat * 1000).astype(np.uint16)


    
############## Water use by SHALLOW-ROOTED and DEEP_ROOTED plants from INVEST modelling

pth = 'N:/Data-Master/Water/Water_yield_modelling/Water_yield_projections/HDF5/'
fn = 'Water_yield_GCM-Ensemble_ssp245_1970-2100_DR_ML_HA_mean'

with h5py.File(pth + fn + '.h5', 'r') as h5:
    cell_df['WATER_YIELD_HIST_DR_ML_HA'] = h5[fn][15, :] # Column 15 is 1985 which is the historical mean 1970 - 2000
    
fn = 'Water_yield_GCM-Ensemble_ssp245_1970-2100_SR_ML_HA_mean'
with h5py.File(pth + fn + '.h5', 'r') as h5:
    cell_df['WATER_YIELD_HIST_SR_ML_HA'] = h5[fn][15, :]

    
# % Pre-European Deep-Rooted Vegetation - NVIS Pre-European Major Vegetation Groups  
"""
Calculate the proportion of each cell which was originally covered by deep-rooted vegetation as a basis for calculating baseline water yield based on NVIS Pre-European Major Vegetation Groups.

NVIS Technical Working Group (2017) Australian Vegetation Attribute Manual: National
Vegetation Information System, Version 7.0. Department of the Environment and Energy,
Canberra. Prep by Bolton, M.P., deLacey, C. and Bossard, K.B. (Eds) 

Table 7 NVIS Structural Formation Terminology https://www.dcceew.gov.au/sites/default/files/documents/australian-vegetation-attribute-manual-v70.pdf

       closed forest open forest woodland open woodland isolated trees isolated clumps of trees
%Cover >80           50-80       20-50    0.25-20       <0.25          0-5

NVIS_PRE_EURO_MVG_ID                            NVIS_PRE_EURO_MVG_NAME
                   1                     Rainforests and Vine Thickets 
                   2                        Eucalypt Tall Open Forests 
                   3                             Eucalypt Open Forests 
                   4                         Eucalypt Low Open Forests 
                   5                                Eucalypt Woodlands 
                   6                      Acacia Forests and Woodlands 
                   7                   Callitris Forests and Woodlands 
                   8                   Casuarina Forests and Woodlands 
                   9                   Melaleuca Forests and Woodlands 
                  10                       Other Forests and Woodlands 
                  11                           Eucalypt Open Woodlands 
                  12            Tropical Eucalypt Woodlands/Grasslands 
                  13                             Acacia Open Woodlands 
                  14                   Mallee Woodlands and Shrublands 
                  15     Low Closed Forests and Tall Closed Shrublands 
                  16                                 Acacia Shrublands 
                  17                                  Other Shrublands 
                  18                                        Heathlands 
                  19                                Tussock Grasslands 
                  20                                Hummock Grasslands 
                  21  Other Grasslands, Herblands, Sedgelands and Ru...
                  22  Chenopod Shrublands, Samphire Shrublands and F...
                  23                                         Mangroves 
                  24  Inland Aquatic - freshwater, salt lakes, lagoons 
                  26                    Unclassified native vegetation 
                  27     Naturally bare - sand, rock, claypan, mudflat 
                  28                                 Sea and estuaries 
                  30                               Unclassified forest 
                  31                              Other Open Woodlands 
                  32  Mallee Open Woodlands and Sparse Mallee Shrubl...

"""

# Create a new field to contain the information
cell_df['DEEP_ROOTED_PROPORTION'] = 0.0

cell_df.loc[cell_df.query('NVIS_PRE_EURO_MVG_ID in [1, 15, 23]').index, 'DEEP_ROOTED_PROPORTION'] = 0.8               # Closed forest
cell_df.loc[cell_df.query('NVIS_PRE_EURO_MVG_ID in [2, 3, 4, 30]').index, 'DEEP_ROOTED_PROPORTION'] = 0.65             # Open forest
cell_df.loc[cell_df.query('NVIS_PRE_EURO_MVG_ID in [6, 7, 8, 9, 10]').index, 'DEEP_ROOTED_PROPORTION'] = 0.35          # Forest/woodlands
cell_df.loc[cell_df.query('NVIS_PRE_EURO_MVG_ID in [5, 12, 14, 16, 17]').index, 'DEEP_ROOTED_PROPORTION'] = 0.35       # Woodlands/shrublands
cell_df.loc[cell_df.query('NVIS_PRE_EURO_MVG_ID in [11, 13, 18, 31, 32]').index, 'DEEP_ROOTED_PROPORTION'] = 0.2      # Open woodlands/heathlands
cell_df.loc[cell_df.query('NVIS_PRE_EURO_MVG_ID in [19, 20, 21, 22, 26]').index, 'DEEP_ROOTED_PROPORTION'] = 0.0        # Grasslands


# Print a summary of the classification table
sum_df = cell_df.groupby(['NVIS_PRE_EURO_MVG_ID'], as_index = False).agg(NVIS_PRE_EURO_MVG_NAME = ('NVIS_PRE_EURO_MVG_NAME', 'first'),
                                                                         DEEP_ROOTED_PROPORTION = ('DEEP_ROOTED_PROPORTION', 'mean')
                                                                        ).sort_values(by = ['NVIS_PRE_EURO_MVG_ID'])
print(sum_df)

cell_df['WATER_YIELD_HIST_BASELINE_ML_HA'] = cell_df.eval('WATER_YIELD_HIST_DR_ML_HA * DEEP_ROOTED_PROPORTION + \
                                                           WATER_YIELD_HIST_SR_ML_HA * (1 - DEEP_ROOTED_PROPORTION)')


    
    
############## Water license cost from BOM water trade data

# Load water trade data
wt_df = pd.read_csv('N:/Data-Master/Water/water_license_cost/Entitlements_Trades_downloaded_20210408.csv') 

# Remove rows with zero in price_per_ML column
wt_df = wt_df[wt_df['price_per_ML'] != 0]

# Calculate some stats on price_per_ML. Data is highly variable as a result of thin markets in many areas. Median is best metric to use.

def perc_25(g):
    return np.percentile(g, 5)

def perc_75(g):
    return np.percentile(g, 95)

pvt = pd.pivot_table(wt_df, values = 'price_per_ML', index = 'drainage_division', aggfunc = [perc_25, 'median', 'mean', perc_75, len])

# Reduce dataframe to a single level
pvt.columns = pvt.columns.get_level_values(0)

# Merge pivot table to cell_df dataframe
cell_df = cell_df.merge(pvt['median'], how = 'left', left_on = 'HR_DRAINDIV_NAME', right_on = 'drainage_division')

# Drop and rename columns
cell_df = cell_df.drop(columns = 'HR_DRAINDIV_NAME')
cell_df.rename(columns = {'median':'WATER_PRICE_ML_BOM'}, inplace = True)

# Replace NaNs with zeros and change datatype
cell_df['WATER_PRICE_ML_BOM'] = cell_df['WATER_PRICE_ML_BOM'].fillna(0).astype(np.int16)

water_price_2D = conv_1D_to_2D(cell_df['WATER_PRICE_ML_BOM'])

# Write out GeoTiff of BOM water price
with rasterio.open('N:/Data-Master/Water/water_license_cost/waterPrice/WATER_PRICE_ML_BOM.tif', 'w+', dtype = 'int16', nodata = '-99', **meta) as dst:
    dst.write_band(1, water_price_2D)





############## Water license cost from ABARES report: Burns, K, Hug, B, Lawson, K, Ahammad, H and Zhang, K 2011, Abatement potential from reforestation under selected carbon price scenarios, ABARES Special Report, Canberra, July. p35-36

# Import shapefile to GeoPandas DataFrame
gdf = gpd.read_file('N:/Data-Master/Water/water_license_cost/waterPrice/waterCostBasin.shp')

# Convert column data type for conversion to raster
gdf['waterCost'] = gdf['waterCost'].astype(np.int16)

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.waterCost))

# Open a new GeoTiFF file
outfile = 'N:/Data-Master/Water/water_license_cost/waterPrice/WATER_PRICE_ML_ABARES.tif'
with rasterio.open(outfile, 'w+', dtype = 'int16', nodata = -9999, **meta) as out:
    
    # Rasterise shapefile
    newrast = features.rasterize(shapes=shapes, fill=0, out=out.read(1), transform=out.transform)
    
    # Clip raster to NLUM
    raster_clipped = np.where(np.logical_and(NLUM_mask == 1, newrast == -9999), 0, newrast)
    
    # Save output to GeoTiff
    out.write_band(1, raster_clipped)

# Flatten the 2D array to 1D array of valid values only
dataFlat = raster_clipped[NLUM_mask == 1]

# Add data to cell_df dataframe
cell_df['WATER_PRICE_ML_ABARES'] = dataFlat





# ############## Natural areas and connectivity - DEPRECATED

# # Identify natural areas (i.e. native vegetation and water) vs other land cover using NLUM
# index = cell_df.query("PRIMARY_V7 in ['1 Conservation and natural environments', '2 Production from relatively natural environments', '6 Water']").index   # Natural (inc waterbodies) vs modified ecosystems

# # Set up pandas series of zeros for each valid grid cell and set natural areas cells to 1
# cell_df['NATURAL_AREA_INC_WATER'] = 0
# cell_df['NATURAL_AREA_INC_WATER'] = cell_df['NATURAL_AREA_INC_WATER'].astype('uint8')
# cell_df.loc[index, 'NATURAL_AREA_INC_WATER'] = 1

# # Convert to 2D raster 
# nat_areas_2D = conv_1D_to_2D(cell_df['NATURAL_AREA_INC_WATER'].to_numpy())

# # Change nodata cells to 0 for gaussian calculation
# nat_areas_2D = np.where(nat_areas_2D > 0, 1, 0)

# # Write out GeoTiff of natural areas
# with rasterio.open('N:/Data-Master/Natural_area_connectivity/NATURAL_AREA_INC_WATER.tif', 'w+', dtype = 'uint8', nodata = '255', **meta) as dst:
#     dst.write_band(1, nat_areas_2D)

# # Calculate the gaussian filter using parameters which give a good result selected by trial and error
# gauss_1 = nd.gaussian_filter(nat_areas_2D, sigma = 2.0, cval = 0, output = np.float32)

# # Convert to int16
# gauss_2 = np.where(NLUM_mask == 1, gauss_1, -99).astype('float32')

# # Write out GeoTiff of connectivity to natural areas
# with rasterio.open('N:/Data-Master/Natural_area_connectivity/NATURAL_AREA_CONNECTIVITY.tif', 'w+', dtype = 'float32', nodata = '-99', **meta) as dst:
#     dst.write_band(1, gauss_2)
    
# # Flatten 2D array to 1D array and add data to cell_df dataframe
# cell_df['NATURAL_AREA_CONNECTIVITY'] = gauss_2[NLUM_mask == 1]

# # Drop column
# cell_df = cell_df.drop(columns = 'PRIMARY_V7')
    




############## Natural areas and connectivity

# Identify natural areas (i.e. native vegetation and water) vs other land cover using NLUM
index = cell_df.query("PRIMARY_V7 in ['1 Conservation and natural environments', '2 Production from relatively natural environments', '6 Water']").index   # Natural (inc waterbodies) vs modified ecosystems

# Set up pandas series of ones for each valid grid cell and set natural areas cells to 0
cell_df['NATURAL_AREA_INC_WATER'] = 1
cell_df['NATURAL_AREA_INC_WATER'] = cell_df['NATURAL_AREA_INC_WATER'].astype('uint8')
cell_df.loc[index, 'NATURAL_AREA_INC_WATER'] = 0

# Convert to 2D raster. Natural land and water = 0, modified land = 1.
nat_areas_2D = conv_1D_to_2D(cell_df['NATURAL_AREA_INC_WATER'].to_numpy())
nat_areas_2D = np.where(nat_areas_2D == 241, 0, nat_areas_2D)

# Write out GeoTiff of natural areas
with rasterio.open('N:/Data-Master/Natural_area_connectivity/NATURAL_AREA_INC_WATER.tif', 'w+', dtype = 'uint8', nodata = '255', **meta) as dst:
    dst.write_band(1, nat_areas_2D)

# Calculate the distance (of modified grid cells to nearest natural cell (natural areas have a distance of 0). Units = 0.01 degrees (~1.11 km).
distance_to_natural = nd.distance_transform_edt(nat_areas_2D).astype('float32')

# Subtract 1 so that rook's case neighbour cell distance to natural == 0
distance_to_natural = (distance_to_natural - 1)

# Convert to km with a minimum distance of 0
distance_to_natural = np.where(distance_to_natural < 0, 0, distance_to_natural * 1.11)

# Set nodata areas to -99 for conversion to GeoTiFF
distance_to_natural = np.where(NLUM_mask == 1, distance_to_natural, -99)

# Write out GeoTiFF of connectivity to natural areas
with rasterio.open('N:/Data-Master/Natural_area_connectivity/NATURAL_AREA_CONNECTIVITY.tif', 'w+', dtype = 'float32', nodata = '-99', **meta) as dst:
    dst.write_band(1, distance_to_natural)

# Flatten 2D array to 1D array and add data to cell_df dataframe
cell_df['NATURAL_AREA_CONNECTIVITY'] = distance_to_natural[NLUM_mask == 1]

# Drop column
cell_df = cell_df.drop(columns = 'PRIMARY_V7')
    




############## Vegetation Assets, States and Transitions (VAST Version 2) - 2008 

with rasterio.open('N:/Data-Master/Vegetation_states_and_transitions/vastgridv2_1k.tif') as src:
    
    # Create an empty destination array 
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8) + 255
    
    # Reproject/resample input raster to match NLUM mask (meta)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.nearest)
    
    # Mask out nodata cells
    dst_array = ma.masked_where(dst_array == 255, dst_array)
    
    # Fill nodata in raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(dst_array.mask, return_distances = False, return_indices = True)
    dst_array_filled = dst_array[tuple(ind)]
    dst_array_masked = np.where(NLUM_mask == 0, 255, dst_array_filled)
    
    # Save the output to GeoTiff
    with rasterio.open('N:/Data-Master/Vegetation_states_and_transitions/VAST_CODE.tif', 'w+', dtype = 'uint8', nodata = 255, **meta) as dst:        
        dst.write_band(1, dst_array_masked)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array_masked[NLUM_mask == 1]
        
    # Add data to cell_df dataframe
    cell_df['VAST_CODE'] = dataFlat
    
    # Load the original VAST lookup table as downloaded
    dbf = DBF('N:/Data-Master/Vegetation_states_and_transitions/vastgridv2_1k.tif.vat.dbf')
    df = pd.DataFrame(iter(dbf))
    
    # Modify the VAST DBF file
    df = df.drop(columns = ['COUNT'])
    df.rename(columns = {'LANDSCAPE_':'VAST_LANDSCAPE'}, inplace = True)
    df = pd.concat([df, pd.DataFrame([[4, 'Replaced', 'Replacement']], columns = df.columns)], ignore_index = True)
    
    df.loc[0, 'VAST_CLASS'] = 'Bare'
    df.loc[1, 'VAST_CLASS'] = 'Residual'
    df.loc[2, 'VAST_CLASS'] = 'Modified'
    df.loc[3, 'VAST_CLASS'] = 'Transformed'
    df.loc[4, 'VAST_CLASS'] = 'Replaced'
    df.loc[5, 'VAST_CLASS'] = 'Removed'    
    
    # Merge dbf table to cell_df dataframe
    cell_df = cell_df.merge(df, how = 'left', left_on = 'VAST_CODE', right_on = 'VALUE')
    
    # Drop unecessary fields
    cell_df = cell_df.drop(columns = ['VAST_CODE', 'VALUE'])
    
    # Change category datatype to object to stop PyTables error "UserWarning: a closed node found in the registry"
    cell_df['VAST_LANDSCAPE'] = cell_df['VAST_LANDSCAPE'].astype('object')
    cell_df['VAST_CLASS'] = cell_df['VAST_CLASS'].astype('object')
    
    # Downcast dataframe to reduce size
    downcast(cell_df)





############## Fire and drought risk impacts (2k resolution) - reproject and resample to NLUM spatial template

# Load 1D (2km grid cell resolution) modelled fire and drought risk data
df = pd.read_csv('N:/Data-Master/Fire_drought_risk/ep_CO2_percentage.csv', header = None, names = ('X', 'Y', 'FD_RISK_PERC_5TH', 'FD_RISK_MEDIAN', 'FD_RISK_PERC_95TH'))

# Open 2k grid cell resolution mask raster
with rasterio.open('N:/Data-Master/Fire_drought_risk/mask2k_z') as src:
    
    # Get and set some metadata for the 2k mask
    meta_2k = src.meta.copy()
    meta_2k.update(compress='lzw', driver='GTiff')
    [meta_2k.pop(key) for key in ['dtype', 'nodata']] # Need to add dtype and nodata manually when exporting GeoTiffs
    
    # Set up a mask for converting 1D risk data to 2D raster
    mask_2k = np.where(src.read(1) > 0, 1, 0)
    
    # Set some data structures to enable conversion on 1D arrays to 2D, nodata = -99
    array_2D_2k = np.zeros(mask_2k.shape) - 99
    xy_2k = np.nonzero(mask_2k)
        
    # Loop through the three columns annd reproject, resample, and add to cell_df
    for col in ['FD_RISK_PERC_5TH', 'FD_RISK_MEDIAN', 'FD_RISK_PERC_95TH']:
        
        # Convert 1D array to 2D
        array_2D_2k[xy_2k] = np.array(df[col])
        
        # Set nodata values to nan
        array_2D_2k = np.where(array_2D_2k == -99, np.nan, array_2D_2k)
        
        # Create an empty destination array to match NLUM
        dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32)
            
        # Reproject/resample input raster to match NLUM mask (meta)
        reproject(array_2D_2k, dst_array, 
                  src_transform = meta_2k.get('transform'), src_crs = meta_2k.get('crs'),
                  dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), 
                  resampling = Resampling.bilinear)

        # Create mask for filling cells
        fill_mask = np.where(np.nan_to_num(dst_array) > 0, 1, 0)
        
        # Fill nodata using inverse distance weighted averaging and mask to NLUM
        dst_array_filled = fillnodata(dst_array, fill_mask, max_search_distance = 100.0)
        
        # Mask out nodata
        dst_array_masked = np.where(NLUM_mask == 1, dst_array_filled, -99)
        
        # Save the output to GeoTiff
        with rasterio.open('N:/Data-Master/Fire_drought_risk/' + col + '.tif', 'w+', dtype = 'float32', nodata = -99, **meta) as dst:        
            dst.write_band(1, dst_array_masked)
        
        # Flatten 2D array to 1D array of valid values only
        dataFlat = dst_array_masked[NLUM_mask == 1]
        
        # Round and add data to cell_df dataframe
        cell_df[col] = dataFlat




############## Soil erosion calculated using INVEST SDR model and Teng et al. 2016 http://dx.doi.org/10.1016/j.envsoft.2015.11.024

with rasterio.open('N:/Data-Master/Soils/Soil_erosion/Dataset_9s/SDR_key_outputs_1km/INVEST_RKLS.tif') as src:
    cell_df['INVEST_RKLS'] = src.read(1)[NLUM_mask == 1]

with rasterio.open('N:/Data-Master/Soils/Soil_erosion/Dataset_9s/SDR_key_outputs_1km/INVEST_SDR.tif') as src:
    cell_df['INVEST_SDR'] = src.read(1)[NLUM_mask == 1]

with rasterio.open('N:/Data-Master/Soils/Soil_erosion/Dataset_9s/SDR_key_outputs_1km/C_FACTOR_VEG.tif') as src:
    cell_df['C_FACTOR_VEG'] = src.read(1)[NLUM_mask == 1]

with rasterio.open('N:/Data-Master/Soils/Soil_erosion/Dataset_9s/SDR_key_outputs_1km/P_FACTOR_AG.tif') as src:
    cell_df['P_FACTOR_AG'] = src.read(1)[NLUM_mask == 1]

# C_FACTOR_VEG_PCT should be applied in cells supporting native vegetation, whereas for other land-uses C factors need to be applied 
# in the LUTO model based on land-use (from Teng et al. 2016)...  Cropping (dryland) = 7%, Cropping (irrigated) = 10%, Pasture = 8%, All forest = 3%   
# P_FACTOR_AG should be applied in areas of agricultural production and modified by regenerative agriculture methods, and be set to 1 elsewhere




############## Biodiversity prioritization ##############

# Function to 
# 1) reproject/resample input raster to match NLUM mask (meta), 
# 2) fill nodata using inverse distance weighted averaging and mask to NLUM, 
# 3) save the output to GeoTiff, 
# 4) flatten 2D array to 1D array of valid values only
def get_bio_priority(bio_path:str, resampling: Resampling):
    
    with rasterio.open(bio_path) as src:
        # Create an empty destination array 
        dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32)
        # Reproject/resample input raster to match NLUM mask (meta)
        reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = resampling)
        
    # Create mask for filling cells
    fill_mask = np.where(dst_array > 0, 1, 0)
    
    # Fill nodata using inverse distance weighted averaging and mask to NLUM
    dst_array_filled = fillnodata(dst_array, fill_mask, max_search_distance = 100.0) * NLUM_mask

    return dst_array_filled


# ------------ Bio prioritization by Carla Archibald using Zonation ------------

zonpath = 'N:/Data-Master/Biodiversity/Environmental-suitability/Zonation/'
    
for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585', 'HCAS']: 
    
    # Match the biodiversity prioritization to the NLUM mask
    bio_path = zonpath + '/' + ssp + '/rankmap.tif'
    if ssp == 'HCAS':
        # HCAS is 0.25km resolution, so use 'average' to dowmsample it to 1km
        dst_array_filled = get_bio_priority(bio_path, resampling = Resampling.average)  
    else:
        # Carla's bio data is 5km resolution, so use 'bilinear' to upsample it to 1km
        dst_array_filled = get_bio_priority(bio_path, resampling = Resampling.bilinear)
    
    # Save the output to GeoTiff
    with rasterio.open(zonpath + '/' + ssp + '/' + ssp + '_zonation_rank_1km.tif', 'w+', dtype = 'float32', nodata = 0, **meta) as dst:        
        dst.write_band(1, dst_array_filled)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array_filled[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_df['BIODIV_PRIORITY_' + ssp.upper()] = dataFlat



# Check that there are no NaNs in the entire dataset
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])


# Write dataframe to HDF5
cell_df.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5', key = 'cell_biophysical_df', mode = 'w', format = 'table')



