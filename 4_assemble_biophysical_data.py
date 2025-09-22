import os
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import rasterio, matplotlib, h5py

from scipy import ndimage as nd
from dbfread import DBF
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
cell_df = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5')


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




############## Average annual carbon sequestration by reforestation land uses

path = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/3D_Spatial_Timeseries/'
gpath = 'N:/Data-Master/FullCAM/Output_TOT_CO2_HA_GeoTiffs/'

# This takes the total stand forest growth by 2100 (i.e., index 90)
# Note that soil carbon is the marginal change in soil carbon resulting from tree planting from 2010 to 2100 rather than the total accumulated SOC.
# This ensures additional SOC sequestration only is considered.
with h5py.File(path + 'tCO2_ha_ep_block.h5', 'r') as h5f:
    cell_df['EP_BLOCK_TREES_T_CO2_HA'] = h5f['Trees_tCO2_ha'][-1]
    cell_df['EP_BLOCK_DEBRIS_T_CO2_HA'] = h5f['Debris_tCO2_ha'][-1]
    cell_df['EP_BLOCK_SOIL_T_CO2_HA'] = (h5f['Soil_tCO2_ha'][-1] - h5f['Soil_tCO2_ha'][0])

with h5py.File(path + 'tCO2_ha_ep_rip.h5', 'r') as h5f:
    cell_df['EP_RIP_TREES_T_CO2_HA'] = h5f['Trees_tCO2_ha'][-1]
    cell_df['EP_RIP_DEBRIS_T_CO2_HA'] = h5f['Debris_tCO2_ha'][-1]
    cell_df['EP_RIP_SOIL_T_CO2_HA'] = (h5f['Soil_tCO2_ha'][-1] - h5f['Soil_tCO2_ha'][0])

with h5py.File(path + 'tCO2_ha_ep_belt.h5', 'r') as h5f:
    cell_df['EP_BELT_TREES_T_CO2_HA'] = h5f['Trees_tCO2_ha'][-1]
    cell_df['EP_BELT_DEBRIS_T_CO2_HA'] = h5f['Debris_tCO2_ha'][-1]
    cell_df['EP_BELT_SOIL_T_CO2_HA'] = (h5f['Soil_tCO2_ha'][-1] - h5f['Soil_tCO2_ha'][0])

with h5py.File(path + 'tCO2_ha_cp_block.h5', 'r') as h5f:
    cell_df['CP_BLOCK_TREES_T_CO2_HA'] = h5f['Trees_tCO2_ha'][-1]
    cell_df['CP_BLOCK_DEBRIS_T_CO2_HA'] = h5f['Debris_tCO2_ha'][-1]
    cell_df['CP_BLOCK_SOIL_T_CO2_HA'] = (h5f['Soil_tCO2_ha'][-1] - h5f['Soil_tCO2_ha'][0])

with h5py.File(path + 'tCO2_ha_cp_belt.h5', 'r') as h5f:
    cell_df['CP_BELT_TREES_T_CO2_HA'] = h5f['Trees_tCO2_ha'][-1]
    cell_df['CP_BELT_DEBRIS_T_CO2_HA'] = h5f['Debris_tCO2_ha'][-1]
    cell_df['CP_BELT_SOIL_T_CO2_HA'] = (h5f['Soil_tCO2_ha'][-1] - h5f['Soil_tCO2_ha'][0])

    
with h5py.File(path + 'tCO2_ha_hir_block.h5', 'r') as h5f:
    cell_df['HIR_BLOCK_TREES_T_CO2_HA'] = h5f['Trees_tCO2_ha'][-1]
    cell_df['HIR_BLOCK_DEBRIS_T_CO2_HA'] = h5f['Debris_tCO2_ha'][-1]
    cell_df['HIR_BLOCK_SOIL_T_CO2_HA'] = (h5f['Soil_tCO2_ha'][-1] - h5f['Soil_tCO2_ha'][0])


# Save the output to GeoTiff - TOTAL CO2 sequestration
with rasterio.open(gpath + 'EP_BLOCK_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_TREES_T_CO2_HA']))
with rasterio.open(gpath + 'EP_BLOCK_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_DEBRIS_T_CO2_HA']))
with rasterio.open(gpath + 'EP_BLOCK_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BLOCK_SOIL_T_CO2_HA']))

with rasterio.open(gpath + 'EP_RIP_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_TREES_T_CO2_HA']))
with rasterio.open(gpath + 'EP_RIP_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_DEBRIS_T_CO2_HA']))
with rasterio.open(gpath + 'EP_RIP_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_RIP_SOIL_T_CO2_HA']))

with rasterio.open(gpath + 'EP_BELT_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_TREES_T_CO2_HA']))
with rasterio.open(gpath + 'EP_BELT_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_DEBRIS_T_CO2_HA']))
with rasterio.open(gpath + 'EP_BELT_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['EP_BELT_SOIL_T_CO2_HA']))


with rasterio.open(gpath + 'CP_BLOCK_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_TREES_T_CO2_HA']))
with rasterio.open(gpath + 'CP_BLOCK_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_DEBRIS_T_CO2_HA']))
with rasterio.open(gpath + 'CP_BLOCK_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BLOCK_SOIL_T_CO2_HA']))
    
with rasterio.open(gpath + 'CP_BELT_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_TREES_T_CO2_HA']))
with rasterio.open(gpath + 'CP_BELT_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_DEBRIS_T_CO2_HA']))
with rasterio.open(gpath + 'CP_BELT_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['CP_BELT_SOIL_T_CO2_HA']))


with rasterio.open(gpath + 'HIR_BLOCK_TREES_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_TREES_T_CO2_HA']))
with rasterio.open(gpath + 'HIR_BLOCK_DEBRIS_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_DEBRIS_T_CO2_HA']))
with rasterio.open(gpath + 'HIR_BLOCK_SOIL_TOT_T_CO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['HIR_BLOCK_SOIL_T_CO2_HA']))




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



####### Calculate CO2 in aboveground living biomass in natural land i.e., the part impacted by livestock **** 

# Load aboveground and belowground biomass from FullCAM estimates. Grab year 91 AGB and BGB. Shape = (6956407, 3)
ep_C_AGB_BGB = np.load('N:/Data-Master/FullCAM/Output_layers/ep_block_AGB_BGB.npy')[..., 94]

# Calculate the proportion of biomass that is carbon. Should be around 0.492
pct_carbon = ep_C_AGB_BGB[:, 0] / (ep_C_AGB_BGB[:, 1] + ep_C_AGB_BGB[:, 2])

# Convert biomass in tDM/ha to tCO2/ha and add data to cell_df dataframe 
cell_df['NATURAL_LAND_AGB_TCO2_HA'] = rox_M_tDM_ha * pct_carbon * (44 / 12)

# Save the output to GeoTiff
with rasterio.open('N:/Data-Master/Emissions_Reduction_Fund/Maximum_aboveground_biomass_M/Version_2/NATURAL_LAND_AGB_TCO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['NATURAL_LAND_AGB_TCO2_HA']))



####### Calculate CO2 in aboveground living biomass and debris in natural land i.e., the part impacted by fire **** 

# Calculate aboveground biomass (tDM/ha) for EP 2100 and convert to tCO2/ha
ep_AGB_TCO2 = (ep_C_AGB_BGB[:, 1] * pct_carbon * (44 / 12))

# Adjustment factor for converting aboveground tree biomass CO2 to total CO2 (i.e. including debris)  
ratio_ABG_to_AGB_DEBRIS = ep_AGB_TCO2 / (ep_AGB_TCO2 + cell_df['EP_BLOCK_DEBRIS_T_CO2_HA'])  
                      
# Calculate total CO2 in maximum AGB and debris and add data to cell_df dataframe
cell_df['NATURAL_LAND_AGB_DEBRIS_TCO2_HA'] = cell_df['NATURAL_LAND_AGB_TCO2_HA'] / ratio_ABG_to_AGB_DEBRIS

# Save the output to GeoTiff
with rasterio.open('N:/Data-Master/Emissions_Reduction_Fund/Maximum_aboveground_biomass_M/Version_2/NATURAL_LAND_AGB_DEBRIS_TCO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['NATURAL_LAND_AGB_DEBRIS_TCO2_HA']))



####### Calculate CO2 in aboveground living biomass and debris and soil in natural land i.e., the part impacted by land clearance **** 

# Adjustment factor for converting aboveground tree biomass CO2 to total CO2 (i.e. including trees, roots, debris, SOC ) 
ratio_ABG_to_TREES_DEBRIS_SOIL = (ep_AGB_TCO2 /                             # Aboveground biomass for 2100 in tCO2/ha
                                 (cell_df['EP_BLOCK_TREES_T_CO2_HA'] +      # Sum of trees, roots, debris and soil in tCO2/ha terms for 2100 
                                  cell_df['EP_BLOCK_DEBRIS_T_CO2_HA'] + 
                                  cell_df['EP_BLOCK_SOIL_T_CO2_HA'])
                                 )

# Calculate total CO2 in maximum AGB and BGB and add data to cell_df dataframe
cell_df['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA'] = cell_df['NATURAL_LAND_AGB_TCO2_HA'] / ratio_ABG_to_TREES_DEBRIS_SOIL 

# Save the output to GeoTiff
with rasterio.open('N:/Data-Master/Emissions_Reduction_Fund/Maximum_aboveground_biomass_M/Version_2/NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(cell_df['NATURAL_LAND_TREES_DEBRIS_SOIL_TCO2_HA']))



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



############## Water use by demostic and industrial sectors
'''
Data source:
 - Total water consumption: taken from (https://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/4610.02010-11?OpenDocument)
 - Total Population: taken from (https://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/3101.0Dec%202010?OpenDocument)
 - Population grid cell (persons/km2): data from (https://www.abs.gov.au/AUSSTATS/abs@.nsf/DetailsPage/1270.0.55.0072011?OpenDocument)

Logic:
 1) Water consumption per person = Total water consumption / Total Population
 2) Reproject water shed to match population grid
 3) Sum (water consumption per person * population grid cell) by watershed to get water consumption per region
'''


# Get the total population (number, Dec 2010)
population = {
    'New South Wales':7272158,
    'Victoria':  5585566,
    'Queensland':  4548661,
    'South Australia':  1650377,
    'Western Australia':  2317064,
    'Tasmania':  509292,
    'Northern Territory':  229874,
    'Australian Capital Territory':  361914
}

# Get the total water consumption (ML, 2010)
water_ag_state = {}
water_domestic_state = {}
for state in population.keys():
    df = pd.read_excel(
        f'N:/Data-Master/Water/Water_account/Water Supply and Use 2010-11 - {state}.xls', 
        sheet_name = 'Table_1',
    )[['Australian Bureau of Statistics', 'Unnamed: 9']]

    water_use_ag = df.iloc[17, 1]
    
    water_use_mining = df.iloc[22, 1]
    water_use_manufacturing = df.iloc[23, 1]
    water_use_elec = df.iloc[25, 1]
    water_use_supply_sewerage_drainage = df.iloc[26, 1]
    water_use_collection_treatment_disposal = df.iloc[27, 1]
    water_use_other_indus = df.iloc[28, 1]
    water_use_household = df.iloc[29, 1]
    
    water_ag_state[state] = water_use_ag
    water_domestic_state[state] = (
        water_use_mining 
        + water_use_manufacturing 
        + water_use_elec
        + water_use_supply_sewerage_drainage
        + water_use_collection_treatment_disposal
        + water_use_other_indus 
        + water_use_household
    )
    
ag_water_df = pd.DataFrame.from_dict(water_ag_state, orient='index', columns=['Water_Use_Agriculture_ML']).reset_index(names='State')
ag_water_df.to_csv('N:/Data-Master/Water/Water_account/Water_Use_Agriculture_ML.csv', index=False) 


# Calculate the per capita water consumption (ML/person)
water_domestic_capita = {}
for state in population.keys():
    water_domestic_capita[state] = water_domestic_state[state] / population[state]
    
    
    
# Burning the per capita water consumption data SA2
with rasterio.open('N:/Data-Master/Population/australian_population_grid_2011_tif_format/Australian_Population_Grid_2011.tif') as src_pop_per_km2,\
     rasterio.open('N:/Data-Master/Water/GeoFabric_V3.2/HR_Regions_GDB_V3_2/HR_Regions_GDB/HR_DrainDiv_raster_filled.tif') as water_DR,\
     rasterio.open('N:/Data-Master/Water/GeoFabric_V3.2/HR_Regions_GDB_V3_2/HR_Regions_GDB/HR_RivReg_raster_filled.tif') as water_RR:
         
    # Get geo reference info
    src_pop_meta = src_pop_per_km2.meta.copy()
    src_pop_meta.update({'nodata': None,'compress': 'lzw'})
    src_pop_arr = src_pop_per_km2.read(1)                                                      # persons/km2
    src_pop_arr *= np.array(list(population.values())).sum() / src_pop_arr.sum()               # Adjust population grid to match ABS report
    
    # Reproject SA2 shapefile to match population grid
    SA2_gdf = gpd.read_file('N:/Data-Master/Australian_administrative_boundaries/sa2_2011_aus/SA2_2011_AUST.shp')
    SA2_gdf = SA2_gdf.to_crs(src_pop_meta['crs'])
    SA2_gdf = SA2_gdf[SA2_gdf['geometry'].notna()].reset_index()
    SA2_gdf['SA2_MAIN11'] = SA2_gdf['SA2_MAIN11'].astype(np.int32)
    shapes = ((geom, water_domestic_capita[s]) for geom, s in zip(SA2_gdf.geometry, SA2_gdf['STE_NAME11']) if s in water_domestic_capita.keys())
     
    # Rasterise the SA2 shapefile, filling the nodata cells with the nearest neighbour
    out_arr = np.zeros((src_pop_meta['height'], src_pop_meta['width']), np.float32)
    water_per_capita = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=src_pop_meta['transform'])      # ML/person
    
    ind_tofill = nd.distance_transform_edt(water_per_capita==0, return_distances = False, return_indices = True)
    water_per_capita = out_arr[tuple(ind_tofill)]

    with rasterio.open('N:/Data-Master/Water/Water_account/WATER_USE_DOMESTIC_INDUSTRIAL_ML_PER_PERSON.tif', 'w+', **src_pop_meta) as out:
        out.write_band(1, water_per_capita)
    
    # Reproject watershed to match population grid
    water_DR_arr = np.zeros((src_pop_meta['height'], src_pop_meta['width']), np.int16)
    water_RR_arr = np.zeros((src_pop_meta['height'], src_pop_meta['width']), np.int16)
    reproject(rasterio.band(water_DR, 1), water_DR_arr, dst_transform = src_pop_meta['transform'], dst_crs = src_pop_meta['crs'], resampling = Resampling.nearest)
    reproject(rasterio.band(water_RR, 1), water_RR_arr, dst_transform = src_pop_meta['transform'], dst_crs = src_pop_meta['crs'], resampling = Resampling.nearest)
    with rasterio.open('N:/Data-Master/Water/Water_account/WATER_USE_DRAINAGE_REGIONS_MATCH_POP.tif', 'w+', **src_pop_meta) as water_DR_out,\
         rasterio.open('N:/Data-Master/Water/Water_account/WATER_USE_RIVER_REGIONS_MATCH_POP.tif', 'w+', **src_pop_meta) as water_RR_out:
            water_DR_out.write_band(1, water_DR_arr)
            water_RR_out.write_band(1, water_RR_arr)
            
    # Calculate the water consumption per watershed
    water_ues_total_DD = np.bincount(water_DR_arr[water_DR_arr > 0], weights=(water_per_capita * src_pop_arr)[water_DR_arr > 0])
    water_ues_total_RR = np.bincount(water_RR_arr[water_RR_arr > 0], weights=(water_per_capita * src_pop_arr)[water_RR_arr > 0])

    water_ues_total_DD_dict = {('Drainage Division',k):[v] for k, v in enumerate(water_ues_total_DD) if k!= 0}
    water_ues_total_RR_dict = {('River Region',k):[v] for k, v in enumerate(water_ues_total_RR) if k!= 0}
    
    out_df = pd.concat([
        pd.DataFrame(water_ues_total_DD_dict).T,
        pd.DataFrame(water_ues_total_RR_dict).T
    ], axis=0).reset_index()
    
    out_df.columns = ['REGION_TYPE', 'REGION_ID', 'DOMESTIC_INDUSTRIAL_WATER_USE_ML']
    out_df.to_csv('N:/Data-Master/Water/Water_account/Water_Use_Domestic.csv', index=False)








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
def reproj_resample(from_raster_path:str, to_raster_meta=meta, resampling: Resampling=Resampling.nearest, fill_nodata:bool=True) -> np.ndarray:
    
    with rasterio.open(from_raster_path) as src:
        # Create an empty destination array 
        dst_array = np.zeros((to_raster_meta.get('height'), to_raster_meta.get('width')), np.float32)
        # Reproject/resample input raster to match NLUM mask (to_raster_meta)
        reproject(
            rasterio.band(src, 1), 
            dst_array, 
            dst_transform = to_raster_meta.get('transform'), 
            dst_crs = to_raster_meta.get('crs'), 
            resampling = resampling)
        
    # Create mask for filling cells
    fill_mask = np.where(dst_array > 0, 1, 0)
    
    # Fill nodata using inverse distance weighted averaging
    dst_array = fillnodata(dst_array, fill_mask, max_search_distance = 100.0) if fill_nodata else dst_array

    return dst_array


# ------------ Bio prioritization by Carla Archibald using Zonation ------------

zonpath = 'N:/Data-Master/Biodiversity/Environmental-suitability/Zonation/'
    
for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']: 
    
    # Match the biodiversity prioritization to the NLUM mask
    bio_path = f"{zonpath}/{ssp}/rankmap.tif"

    # Carla's bio data is 5km resolution, so use 'bilinear' to upsample it to 1km
    dst_array_filled = reproj_resample(bio_path, resampling = Resampling.bilinear)
    
    # Save the output to GeoTiff
    with rasterio.open(f"{zonpath}/{ssp}/{ssp}_zonation_rank_1km.tif", 'w+', dtype = 'float32', nodata = 0, **meta) as dst:        
        dst.write_band(1, dst_array_filled)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array_filled[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_df['BIODIV_PRIORITY_' + ssp.upper()] = dataFlat






# ------------ Habitat Condition Assessment System [HCAS] data (https://data.csiro.au/collection/csiro:63571) ------------


HCAS_path = "N:/Data-Master/Habitat_condition_assessment_system/Data"
percentiles =[10, 25, 50, 75, 90]

with rasterio.Env(CPL_DEBUG=False):
    with rasterio.open(f'{HCAS_path}/HCAS_v3.1/1.HABITAT_CONDITION/HCAS31_HCB_1988_2022.tif') as HCAS_src:
        
        # Read the HCAS data
        HCAS_arr = HCAS_src.read(1)
        HCAS_meta = HCAS_src.meta.copy()

        # Read the LUMAP data for year 2010, then reproject and resample it to match HCAS
        lu_arr_match_HCAS = reproj_resample(
            'N:/Data-Master/National_Landuse_Map/lumap.tif',
            HCAS_meta,
            resampling = Resampling.nearest,     # use 'nearest' resampling to upsample the LUMAP data (1km) to match HCAS (90m)
            fill_nodata = False                  # do not fill nodata when reprojecting LUMAP
        )

        # Calculate the percentiles
        HCAS_lumap_percentile = {}
        for lu_code in np.unique(lu_arr_match_HCAS):
            # Skip if lu_code is NaN or negative
            if np.isnan(lu_code) or lu_code < 0:
                continue
            # Calculate the percentiles
            HCAS_lumap_percentile[lu_code] = np.nanpercentile(
                np.where(lu_arr_match_HCAS == lu_code, HCAS_arr, np.nan),
                percentiles
            )
  
        
# Save the output to CSV
HCAS_LUMAP_PERCENTILE_df = pd.DataFrame(HCAS_lumap_percentile).T
HCAS_LUMAP_PERCENTILE_df.index.name = 'lu'
HCAS_LUMAP_PERCENTILE_df.columns = percentiles
HCAS_LUMAP_PERCENTILE_df.columns = ['PERCENTILE_' + str(i) for i in HCAS_LUMAP_PERCENTILE_df.columns]
HCAS_LUMAP_PERCENTILE_df['USER_DEFINED'] = None

if os.path.exists(f"{HCAS_path}/Processed/HABITAT_CONDITION.csv"):
    os.remove(f"{HCAS_path}/Processed/HABITAT_CONDITION.csv")
    
HCAS_LUMAP_PERCENTILE_df.to_csv(f"{HCAS_path}/Processed/HABITAT_CONDITION.csv")



# ------------ National Connectivity Index [NCI] (https://data.csiro.au/collection/csiro:63571) ------------

NCI_reproj_NLUM = reproj_resample(
    f'{HCAS_path}/HCAS_v3.1/4.CONNECTIVITY.CONDITION/NCI/HCAS31_NCIB_1988_2022.tif',
    meta,
    resampling = Resampling.average,    # use 'average' resampling to downsample the NCI data (90m) to match NLUM (1km)
    fill_nodata = True
)

# Save NCI to cell_df dataframe
cell_df['DCCEEW_NCI'] = NCI_reproj_NLUM[NLUM_mask == 1]


############## Sanity Check and Write to HDF5 ##############

# Check that there are no NaNs in the entire dataset
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])


# Write dataframe to HDF5
cell_df.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5', key = 'cell_biophysical_df', mode = 'w', format = 'table')



