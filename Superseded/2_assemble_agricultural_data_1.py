import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy import ndimage as nd
import rasterio, matplotlib
from rasterio import features
from rasterio.warp import reproject
from rasterio.enums import Resampling
from shapely.geometry import box


# Set some options
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', '{:,.2f}'.format)

infile = 'N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif'

# Set file path
in_cell_df_path = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.pkl'
 
# Read cell_df file from disk to a new data frame for ag data with just the relevant columns
cell_df = pd.read_pickle(in_cell_df_path)[['CELL_ID', 'X', 'Y', 'CELL_HA', 'SA2_ID', 'PRIMARY_V7', 'SPREAD_ID', 'SPREAD_DESC', 'IRRIGATION']] # 'COMMODITIES', 'COMMODITIES_DESC', 'IRRIGATION']]

# cell_df = pd.read_csv('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.csv')

################################ Open NLUM_ID as mask raster and get metadata, create some helper functions

with rasterio.open(infile) as rst:
    NLUM_mask = rst.read(1)
    
    # Get metadata and update parameters
    NLUM_transform = rst.transform
    NLUM_bounds = rst.bounds
    NLUM_height = rst.height
    NLUM_width = rst.width
    NLUM_crs = rst.crs
    meta = rst.meta.copy()
    meta.update(compress='lzw', driver='GTiff') # dtype='int32', nodata='-99')
    [meta.pop(key) for key in ['dtype', 'nodata']] # Need to add dtype and nodata manually when exporting GeoTiffs
        
    # Set some data structures to enable conversion on 1D arrays to 2D
    array_2D = np.zeros(NLUM_mask.shape) - 99
    xy = np.nonzero(NLUM_mask)
    
# Convert 1D column to 2D spatial array
def conv_1D_to_2D(in_1D_array):
    array_2D[xy] = np.array(in_1D_array)
    return array_2D.astype(in_1D_array.dtype)

# Convert 1D column to 2D spatial array and plot map
def map_in_2D(col, data): # data = 'continuous' or 'categorical'
    a2D = conv_1D_to_2D(col)
    if data == 'categorical':
        n = col.nunique()
        cmap = matplotlib.colors.ListedColormap(np.random.rand(n,3))
        plt.imshow(a2D, cmap=cmap, resample=False)
    elif data == 'continuous':
        plt.imshow(a2D, cmap = 'pink', resample = False)
    plt.show()

# Convert object columns to categories and downcast int64 columns to save memory and space
def downcast(dframe):
    obj_cols = dframe.select_dtypes(include = ["object"]).columns
    dframe[obj_cols] = dframe[obj_cols].astype('category')
    int64_cols = dframe.select_dtypes(include = ["int64"]).columns
    dframe[int64_cols] = dframe[int64_cols].apply(pd.to_numeric, downcast = 'integer')
    f64_cols = dframe.select_dtypes(include = ["float64"]).columns
    dframe[f64_cols] = dframe[f64_cols].apply(pd.to_numeric, downcast = 'float')





################################ Join livestock mapping and yield data to the cell_df dataframe

lmap = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210630/lmap_jul30/lmap.csv', low_memory = False)
lmap = lmap[['X', 'Y', 'SA2_ID', 'irrigation', 'irrig_factor', 'kgDMhayr', 'SPREAD_original', 'SPREAD_name', 'safe_pur', 'feed_req_factor', 
             'pur', 'no_sheep', 'SPREAD_mapped','heads_mapped', 'heads_mapped_cum', 'Sheep', 'ha_pixel', 'SPREAD_id_mapped']]

# From Ray/Javi...
# a. ha_dairy: The hectares of dairy pastures within a 1km2 cell, derived from the CLUM. We only use that value for ranking purposes
# b. ha_pixel: That's the real ha that Brett provided
# c. safe_pur: Safe pasture utilisation rate (per pixel)
# d. Feed_req_factor: The ratio of heads to map to feed availability at the sa4 level
# e. pur: Effective pasture utilisation rate (safe_pur * feed_req_factor)
# f. no_sheep: This is the dingo fence, we set "Sheep" to zero where no_sheep is not null
# g. Beef_Cattle, Dairy_Cattle, Sheep: the QDAF stocking rates. We use them for ranking solely, and to mask out areas (if value is 0)
# h. heads_mapped: Total heads mapped to the pixel
# i. stocking_rate: heads_mapped / area_mapped (area mapped is the entire pixel area)

# Checking and validation code
# lmap.groupby(['SPREAD_name', 'irrigation', 'SPREAD_original'])[['SPREAD_original']].count().head(100)
# lmap.groupby(['no_sheep', 'Sheep'])[['SPREAD_original']].count().head(100)
# lmap[lmap['Sheep'] == 0]
# lmap[lmap['no_sheep'].notnull()]
# lmap[lmap['no_sheep'] == 'no sheep']

# Reality check pasture growth
lmap['Pasture_growth'] = lmap.eval('kgDMhayr * irrig_factor')
lmap.sort_values(by = ['Pasture_growth'], ascending = False).head(100)

# Calculate stocking rates
# From Javi... stocking rates (head/ha) = pur * kgDMhayr * ha_pixel * irrig_factor / (dse_per_head * 365 * grassfed_factor)
# Grassfed_factor is 0.65 for Dairy, and 0.85 for Beef and Sheep. 17 DSE/head for dairy, 8 DSE/head for beef, 1.5 DSE/head for sheep
# But they shouldn't map heads of Sheep where the field "Sheep" is 0. 
lmap['BEEF_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (8 * 365 * 0.85)')
lmap['SHEEP_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (1.5 * 365 * 0.85)')
lmap['DAIRY_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (17 * 365 * 0.65)')
lmap['DSE_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / 365')
lmap['BEEF_HEAD_PER_CELL'] = lmap.eval('pur * kgDMhayr * ha_pixel * irrig_factor / (8 * 365 * 0.85)')
lmap['SHEEP_HEAD_PER_CELL'] = lmap.eval('pur * kgDMhayr * ha_pixel * irrig_factor / (1.5 * 365 * 0.85)')
lmap['DAIRY_HEAD_PER_CELL'] = lmap.eval('pur * kgDMhayr * ha_pixel * irrig_factor / (17 * 365 * 0.65)')

# Sort and view
lmap.sort_values(by = 'DSE_PER_HA', ascending = False)
lmap.sort_values(by = 'SPREAD_original', ascending = False)

# Check number of grazing cells in NLUM, n = 3749048
cell_df[cell_df['SPREAD_DESC'].str.contains('Grazing')]
cell_df.query('(SPREAD_ID >= 1) and (SPREAD_ID <= 3)')


# Create new integer X/Y columns to join, then join the livestock map table to the cell_df dataframe and drop uneccesary columns
cell_df['X-round'] = np.round(cell_df['X'] * 100).astype(np.int16)
cell_df['Y-round'] = np.round(cell_df['Y'] * 100).astype(np.int16)
lmap['X-round'] = np.round(lmap['X'] * 100).astype(np.int16)
lmap['Y-round'] = np.round(lmap['Y'] * 100).astype(np.int16)
lmap_lite = lmap.drop(columns = ['X', 'Y', 'irrig_factor', 'kgDMhayr', 'safe_pur', 'feed_req_factor', 'pur', 'ha_pixel', 'no_sheep', 'heads_mapped_cum'])
ludf = cell_df.merge(lmap_lite, how = 'left', left_on = ['X-round', 'Y-round'], right_on = ['X-round', 'Y-round'])
ludf = ludf.drop(columns = ['X-round', 'Y-round', 'irrigation', 'SPREAD_original', 'SA2_ID_y', 'SPREAD_name', 'Pasture_growth', 'DSE_PER_HA'])

# Create LU_ID field to hold IDs for crops and livestock
ludf['LU_ID'] = ludf['SPREAD_ID']
index = ludf.query('(SPREAD_ID >= 1) and (SPREAD_ID <= 3)').index
ludf.loc[index, 'LU_ID'] = ludf['SPREAD_id_mapped']
ludf.loc[ludf['LU_ID'].isnull(), 'LU_ID'] = 0 # Unallocated agricultural land

# Create LU_DESC field to hold names of crops and livestock
ludf['LU_DESC'] = ludf['SPREAD_DESC']
ludf.loc[ludf['LU_ID'] == 0, 'LU_DESC'] = 'Unallocated agricultural land'
ludf['LU_DESC'] = ludf['LU_DESC'].cat.add_categories(['Dairy', 'Beef', 'Sheep'])
ludf.loc[ludf['LU_ID'] == 31, 'LU_DESC'] = 'Dairy'
ludf.loc[ludf['LU_ID'] == 32, 'LU_DESC'] = 'Beef'
ludf.loc[ludf['LU_ID'] == 33, 'LU_DESC'] = 'Sheep'

# Downcast data types to reduce space
downcast(ludf)
ludf['LU_ID'] = ludf['LU_ID'].astype(np.int8)


# Helper queries
ludf.groupby(['LU_ID', 'SPREAD_DESC'], observed = True)[['LU_ID']].count()
ludf.groupby(['LU_ID', 'LU_DESC'], observed = True)[['LU_ID']].count()
cell_df.groupby(['PRIMARY_V7', 'SPREAD_DESC'], observed = True)[['PRIMARY_V7']].count().sort_values(by = 'PRIMARY_V7', ascending = True)
cell_df.groupby(['PRIMARY_V7', 'SPREAD_DESC'], observed = True)[['PRIMARY_V7']].count().sort_values(by = 'PRIMARY_V7', ascending = True)

ludf.query('(SPREAD_ID >= 1) and (SPREAD_ID <= 3)').head(100)
ludf.query('SPREAD_ID >= 1').head(100)
ludf.info()
ludf['LU_ID'].isnull().sum() == lmap['SPREAD_id_mapped'].isnull().sum()
cell_df.groupby(['SPREAD_ID', 'SPREAD_DESC'], observed = True)[['SPREAD_DESC']].count().head(100)

# Use dingo fence data to set SHEEP_MASK column - Javi: "But they shouldn't map heads of Sheep where the field "Sheep" is 0."
ludf['SHEEP_MASK'] = 1
ns_filter = ludf['Sheep'] == 0
ludf.loc[ns_filter, 'SHEEP_MASK'] = 0
ludf = ludf.drop(columns = ['Sheep'])
                     
# Cross-check the mapped national herd size with ABS numbers
ludf.groupby(['SPREAD_mapped'])[['heads_mapped']].sum()
ag_df.query("SPREAD_ID >= 31 and SPREAD_ID <= 33").groupby(['SPREAD_Commodity', 'SPREAD_ID_original', 'irrigation'])[['prod_ABS']].sum()





################################ Join CROPS data from profit map table to the cell_df dataframe

# Read in the PROFIT MAP table as provided by CSIRO to dataframe
ag_df = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210621/T_pfe_per_product_21062021.csv').drop(columns = 'rev_notes')

# Select crops only
crops_df = ag_df.query("SPREAD_ID >= 5 and SPREAD_ID <= 25")

# Rename columns to avoid python built-in naming
crops_df.rename(columns = {'yield': 'Yield', 'irrigation': 'Irrigation', 'irrig_factor': 'Prod_factor'}, inplace = True) 

######## Aggregate the ABS Commodity-level data to SPREAD class

# Define a lambda function to compute the weighted mean
area_weighted_mean = lambda x: np.average(x, weights = crops_df.loc[x.index, 'area_ABS'])  
prod_weighted_mean = lambda x: np.average(x, weights = crops_df.loc[x.index, 'prod_ABS'])  

# Aggregate commodity-level to SPREAD-level using weighted mean based on area of Commodity within SA2
crops_SA2_df = crops_df.groupby(['SA2_ID', 'SPREAD_ID', 'Irrigation'], as_index = False).agg(
                    STATE_ID = ('STATE_ID', 'first'),
                    SPREAD_Name = ('SPREAD_Commodity', 'first'),
                    Area_ABS = ('area_ABS', 'sum'),
                    Prod_ABS = ('prod_ABS', 'sum'),
                    Yield = ('Yield', area_weighted_mean),
                    P1 = ('P1', prod_weighted_mean),
                    AC = ('AC_rel', area_weighted_mean),
                    QC = ('QC_rel', prod_weighted_mean),
                    FDC = ('FDC_rel', area_weighted_mean),
                    FLC = ('FLC_rel', area_weighted_mean),
                    FOC = ('FOC_rel', area_weighted_mean),
                    WR = ('WR', area_weighted_mean),
                    WP = ('WP', area_weighted_mean)
                    )

# Aggregate commodity-level to SPREAD-level using weighted mean based on area of Commodity within STATE for Stone Fruit and Vegetables

# Change ACT STATE_ID to same as NSW
crops_df.loc[crops_df['STATE_ID'] == 9, 'STATE_ID'] = 1

crops_STE_df = crops_df.groupby(['STATE_ID', 'SPREAD_ID', 'Irrigation'], as_index = False).agg(
                    SPREAD_Name_STE = ('SPREAD_Commodity', 'first'),
                    Area_ABS_STE = ('area_ABS', 'sum'),
                    Prod_ABS_STE = ('prod_ABS', 'sum'),
                    Yield_STE = ('Yield', area_weighted_mean),
                    P1_STE = ('P1', prod_weighted_mean),
                    AC_STE = ('AC_rel', area_weighted_mean),
                    QC_STE = ('QC_rel', prod_weighted_mean),
                    FDC_STE = ('FDC_rel', area_weighted_mean),
                    FLC_STE = ('FLC_rel', area_weighted_mean),
                    FOC_STE = ('FOC_rel', area_weighted_mean),
                    WR_STE = ('WR', area_weighted_mean),
                    WP_STE = ('WP', area_weighted_mean)
                    )

# Join the table to the cell_df dataframe and drop uneccesary columns
crops_sum_df = crops_SA2_df.merge(crops_STE_df, how = 'left', left_on = ['STATE_ID', 'SPREAD_ID', 'Irrigation'], right_on = ['STATE_ID', 'SPREAD_ID', 'Irrigation']) 

# Sort in place
crops_sum_df.sort_values(by = ['SA2_ID', 'SPREAD_ID', 'Irrigation'], ascending = True, inplace = True)

# Allocate state-level values for Stone Fruit and Vegetables to smooth out the variation
for col in ['Yield', 'P1', 'AC', 'QC', 'FDC', 'FLC', 'FOC', 'WR', 'WP']:
    crops_sum_df.loc[crops_sum_df['SPREAD_Name'].isin(['Stone Fruit', 'Vegetables']), col] = crops_sum_df[col + '_STE']

# Clean up dataframe
crops_sum_df.drop(columns = ['STATE_ID', 'SPREAD_Name_STE', 'Area_ABS_STE', 'Prod_ABS_STE', 'Yield_STE', 'P1_STE', 'AC_STE', 'QC_STE', 'FDC_STE', 'FLC_STE', 'FOC_STE', 'WR_STE', 'WP_STE'], inplace = True)

# Code to calculate revenue and costs
# crops_sum_df.eval('Yield_ha_TRUE = Production / Area', inplace = True)
# crops_sum_df.eval('Rev_ha = Yield * P1', inplace = True)
# crops_sum_df.eval('Rev_tot = Production * P1', inplace = True)
# crops_sum_df.eval('Costs_ha = (AC + FDC + FOC + FLC) + (QC * Yield) + (WR * WP)', inplace = True)
# crops_sum_df.eval('Costs_t = Costs_ha / Yield', inplace = True)

print('Number of NaNs =', crops_sum_df[crops_sum_df.isna().any(axis=1)].shape[0])

downcast(crops_sum_df)

# Save file
crops_sum_df.to_csv('N:/Planet-A/Data-Master/Profit_map/crop_yield_econ_water_SPREAD.csv')
crops_sum_df.to_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/crop_data_SPREAD.h5', key = 'crop_yield_econ_water_SPREAD', mode = 'w', format = 't')

# Join the table to the cell_df dataframe and drop uneccesary columns
adf = cell_df.merge(crops_sum_df, how = 'left', left_on = ['SA2_ID', 'SPREAD_ID', 'IRRIGATION'], right_on = ['SA2_ID', 'SPREAD_ID', 'Irrigation']) 

# Limit cell dataframe to crops only
# adf = adf.query("SPREAD_ID >= 5 and SPREAD_ID <= 25")

# Check for NaNs
print('Number of NaNs =', adf[adf.isna().any(axis = 1)].shape[0])

# Check the NLUM vs ABS commodity area by SA2
tmp = adf.groupby(['SA2_ID', 'SPREAD_DESC']).agg(Area_NLUM = ('CELL_HA', 'sum'), Area_ABS = ('Area_ABS', 'first'))
tmp.groupby(['SPREAD_DESC']).agg(Area_CELL = ('Area_NLUM', 'sum'), Area_ABS = ('Area_ABS', 'sum'))

# Check NLUM vs ABS total production by SA2
adf['Prod_NLUM'] = adf.eval('Yield * CELL_HA')
tmp = adf.groupby(['SA2_ID', 'SPREAD_DESC']).agg(Prod_NLUM = ('Prod_NLUM', 'sum'), Prod_ABS = ('Prod_ABS', 'first'))
tmp.groupby(['SPREAD_DESC']).agg(Prod_NLUM = ('Prod_NLUM', 'sum'), Prod_ABS = ('Prod_ABS', 'sum'))




################################ Assemble CHICKENS, EGGS, and PIGS data for post-processing

# Chickens, eggs, and pigs are 'off-land'. Demand is automatically met. There is an implication for feed pre-calculated in the demand model.
# Then the area, costs, and impacts (if any) of off-land commodities is calculated post-hoc in the reporting. These commodities do have a water requirement
# but it is minimal and safely ignored.

# Select crops only[
cep_df = ag_df.query("SPREAD_ID >= 34 and SPREAD_ID <= 36")[['STATE_ID', 'SA4_ID', 'SA2_ID', 'SA2_Name', 'SPREAD_Commodity', 'SPREAD_ID', 'prod_ABS', 'area_ABS', 'yield_ABS', 
                                                             'Q1', 'P1', 'AC_rel', 'QC_rel', 'FDC_rel', 'FLC_rel', 'FOC_rel']].reset_index()

# Rename columns to avoid python built-in naming
cep_df.rename(columns = {'SPREAD_Commodity': 'SPREAD_Name', 
                         'prod_ABS': 'Production', 
                         'area_ABS': 'Area', 
                         'yield_ABS': 'Yield',
                         'AC_rel': 'AC', 
                         'QC_rel': 'QC', 
                         'FDC_rel': 'FDC', 
                         'FLC_rel': 'FLC', 
                         'FOC_rel': 'FOC'}, inplace = True) 

# Do some test calculations of revenue and costs
# cep_df.eval('Yield_ha_TRUE = Production / Area', inplace = True)
# cep_df.eval('Rev_ha = Yield * Q1 * P1', inplace = True)
# cep_df.eval('Costs_ha = (AC + FDC + FOC + FLC) + (QC * Yield)', inplace = True)
# cep_df.eval('Costs_t = Costs_ha / Yield', inplace = True)

downcast(cep_df)

# Save file
cep_df.to_csv('N:/Planet-A/Data-Master/Profit_map/cep_yield_econ_SPREAD.csv')
cep_df.to_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cep_yield_econ_SPREAD.h5', key = 'cep_yield_econ_SPREAD', mode = 'w', format = 't')





################################ Livestock mapping

with rasterio.open('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210623/livestock_test.tif') as src:
    # rst = src.read(1, window = from_bounds(*NLUM_bounds, transform = src.transform, height = NLUM_height, width = NLUM_width)) # Clip raster
    
    # Create an empty destination array 
    dst_array = np.zeros((NLUM_height, NLUM_width), np.uint8)
    
    # Reproject/resample input raster to match NLUM mask (meta)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = NLUM_transform, dst_crs = NLUM_crs, resampling = Resampling.nearest)
    
    # Set nodata
    dst_array = np.where(NLUM_mask == 1, dst_array, 99)    
    
    # Save the output to GeoTiff
    with rasterio.open('N:/Planet-A/Data-Master/Profit_map/livestock_map.tif', 'w+', dtype = 'int8', nodata = 99, **meta) as dst:        
        dst.write_band(1, dst_array)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_df['SPREAD_ID_LS'] = dst_array[NLUM_mask == 1].astype(np.uint8)





# cell_df[cell_df['SPREAD_ID'] <= 3].groupby(['SPREAD_DESC', 'IRRIGATION', 'SPREAD_ID'])[['SPREAD_ID']].count().head(100)
# cell_df.groupby(['SPREAD_ID', 'SPREAD_DESC'])[['SPREAD_DESC']].count().head(100)




print('Number of NaNs =', lmap[lmap.isna().any(axis = 1)].shape[0])
lmap[lmap['kgDMhayr'].isna()]

index = cell_df['SPREAD_DESC'].str.contains('Grazing')
cell_df.loc[index].groupby(['SPREAD_DESC'])[['SPREAD_ID']].first().sort_values(by = 'SPREAD_ID')

cell_df.groupby(['SPREAD_DESC'])[['SPREAD_ID']].first().head(100).sort_values(by = 'SPREAD_ID')
cell_df.groupby(['SPREAD_DESC', 'IRRIGATION', 'SPREAD_ID_LS'])[['SPREAD_ID_LS']].count().head(100)

x = cell_df[cell_df['SPREAD_ID'] < 5]
x.groupby(['SPREAD_DESC', 'IRRIGATION'])[['SPREAD_ID']].first().head(100)
x.groupby(['SPREAD_DESC', 'IRRIGATION', 'SPREAD_ID_LS'])[['SPREAD_ID_LS']].count().head(100)





################################ Join EMISSIONS table

# Read in the CROPS EMISSIONS table, join to cell_df, and drop unwanted columns
CO2 = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210603/T_emissions_by_SPREAD_SA2_crops.csv')
tmp = crops_sum_df.merge(CO2, how = 'left', left_on = ['SA2_ID', 'SPREAD_ID', 'Irrigation'], right_on = ['sa2_id', 'spread_id', 'irrigation']).drop(columns = ['Area_ABS', 'Prod_ABS', 'Yield', 'P1', 'AC', 'QC', 'FDC', 'FLC', 'FOC', 'WR', 'WP'])
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])

# cell_df = cell_df.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['SLA_code_2006', 'SPREAD_ID', 'irrigation'])
# cell_df = cell_df.drop(columns=['kwhfert', 'kwhpest', 'kwhirrig', 'kwh_chemapp', 'kwh_cropmanagement', 'kwhcult', 'kwhharvest', 'kwhsowing'])
# 
# #join the CROPS EMISSIONS table to NLUM_SA2_gdf and check for NaNs 
# NLUM_SA2_gdf2 = NLUM_SA2_gdf.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['SLA_code_2006', 'SPREAD_ID', 'irrigation']) # Need to fix PFE table first
# # NLUM_SA2_gdf2 = NLUM_SA2_gdf2.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# print('Number of NaNs =', NLUM_SA2_gdf2[NLUM_SA2_gdf2.isna().any(axis=1)].shape[0])
# 
# # Read in the LIVESTOCK EMISSIONS table, join to cell_df, and drop unwanted columns
# tmp = pd.read_csv(r'N:\Planet-A\Data-Master\Profit_map\emissions-maps\T_emissions_by_SPREAD_SLA_livestock.csv')
# cell_df = cell_df.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['sla06_id', 'spread_id', 'irrigation']) # sla06_id needs to change to SA2 2011 code, no 'irrigation' code?
# # cell_df = cell_df.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# 
# #join the LIVESTOCK EMISSIONS table to NLUM_SA2_gdf and check for NaNs 
# NLUM_SA2_gdf2 = NLUM_SA2_gdf.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['sla06_id', 'SPREAD_ID', 'irrigation']) # Need to fix PFE table first
# # NLUM_SA2_gdf2 = NLUM_SA2_gdf2.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# print('Number of NaNs =', NLUM_SA2_gdf2[NLUM_SA2_gdf2.isna().any(axis=1)].shape[0])
# 
# =============================================================================



################################ Join Fertilizer table 

# Read in the Fertilizer table
NPKS = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210603/T_NPKS_by_SPREAD_SA2.csv')
tmp = crops_sum_df.merge(NPKS, how = 'left', left_on = ['SA2_ID', 'SPREAD_ID', 'Irrigation'], right_on = ['sa2_id', 'SPREAD_ID', 'irrigation']).drop(columns = ['Area_ABS', 'Prod_ABS', 'Yield', 'P1', 'AC', 'QC', 'FDC', 'FLC', 'FOC', 'WR', 'WP'])
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])


################################ Join TOXICITY table to the cell_df dataframe

# =============================================================================
# Read in the TOXICITY table
TOX = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210603/T_USETOX_CFvalue_by_SPREAD_SA2_wide.csv')
tmp = crops_sum_df.merge(TOX, how = 'left', left_on = ['SA2_ID', 'SPREAD_ID', 'Irrigation'], right_on = ['SA211_id', 'SPREAD_ID', 'irrigation']).drop(columns = ['Area_ABS', 'Prod_ABS', 'Yield', 'P1', 'AC', 'QC', 'FDC', 'FLC', 'FOC', 'WR', 'WP'])
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])


 
# # Join the table to the dataframe
# cell_df = cell_df.merge(tmp, how = 'left', left_on = ['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on = ['SA211_id', 'SPREAD_ID', 'irrigation'])
# # cell_df = cell_df.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# 
# #join the TOXICITY table to NLUM_SA2_gdf and check for NaNs 
# NLUM_SA2_gdf2 = NLUM_SA2_gdf.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['SA211_id', 'SPREAD_ID', 'irrigation']) # Need to fix PFE table first
# # NLUM_SA2_gdf2 = NLUM_SA2_gdf2.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# print('Number of NaNs =', NLUM_SA2_gdf2[NLUM_SA2_gdf2.isna().any(axis=1)].shape[0])
# 
# =============================================================================




# cell_df.query('SA2_ID' == 702011054 & 'SPREAD_ID' == 25), ('SA2_ID', 'SA2_Name', 'SPREAD_ID', 'SPREAD_Commodity', 'product', 'ha_weight_SPREAD')]


# tmp.groupby(['SPREAD'], as_index=False)[['SPREAD_ID']].first().sort_values(by=['SPREAD_ID'])
cell_df.groupby(['COMMODITIES'])[['COMMODITIES_DESC']].nunique() 
cell_df.groupby(['COMMODITIES'])[['COMMODITIES_DESC']].first()



cell_df.groupby(['PROT_AREAS'], as_index=False)[['PROT_AREAS_DESC']].first().sort_values(by=['PROT_AREAS'])

cell_df[cell_df['SA2_ID'] == 101011002].groupby(['COMMODITIES', 'IRRIGATION'])[['COMMODITIES_DESC']].size()

ag_df[(ag_df['area_ABS'] < 1) & (ag_df['SPREAD_ID'] <= 25) & (ag_df['SPREAD_ID'] >= 5)].iloc[:,3:16]

################################ Summarise Javi's profit map data table and join it to the cell_df dataframe


# Read in the profit map table to dataframe
javi = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/T_pfe_per_product_07052021.csv')
javi.rename(columns={'yield': 'Yield', 'prod': 'Production', 'irrig_factor': 'Prod_factor'}, inplace = True)

javi.info()


javi['Commodity'].unique()
javi['SPREAD_Commodity'].unique()

javi.groupby(['SPREAD_Commodity'])[['SPREAD_ID']].nunique() 
javi.groupby(['SPREAD_Commodity'])[['SPREAD_ID']].first()

crops_df.groupby(['Commodity'])[['SPREAD_Commodity']].nunique()
crops_df.groupby(['SPREAD_Commodity', 'Commodity'])[['SPREAD_ID']].count()


javi[javi['SA2_Name'] == 'Adelaide Hills'].groupby(['SPREAD_Commodity', 'irrigation', 'SA2_Name'])[['pfe']].mean()

javi.groupby(['SPREAD_Commodity', 'irrigation', 'SPREAD_ID_original'])[['pfe']].mean()

pd.pivot_table(javi, index = 'SA2_ID', columns = 'SPREAD_ID', values = 'SPREAD_Commodity', aggfunc = 'count')

# Define a lambda function to compute the weighted mean
wm = lambda x: np.average(x, weights = javi.loc[x.index, 'area'])  

# Summarise ag data by SPREAD Commodity
j1 = javi[javi['SA2_Name'].isin(['Goulburn'])]

j = javi.groupby(['SA2_ID', 'SPREAD_ID', 'SPREAD_ID_original', 'irrigation'], as_index = False).agg(
                    SA2_Name = ('SA2_Name', 'first'),
                    SPREAD_Commodity = ('SPREAD_Commodity', 'first'),
                    Production = ('Production', wm), 
                    Area_crops = ('area', 'sum'),
                    Area_livestock = ('area', 'first'),
                    Prod_factor = ('Prod_factor', wm),
                    Yield = ('Yield', wm),
                    F1 = ('F1', wm),
                    Q1 = ('Q1', wm),
                    P1 = ('P1', wm),
                    F2 = ('F2', wm),
                    Q2 = ('Q2', wm),
                    P2 = ('P2', wm),
                    F3 = ('F3', wm),
                    Q3 = ('Q3', wm),
                    P3 = ('P3', wm),
                    Rev = ('rev', wm),
                    Cost_pct = ('cost_pct', wm),
                    AC = ('AC', wm),
                    QC = ('QC', wm),
                    FDC = ('FDC', wm),
                    FLC = ('FLC', wm),
                    FOC = ('FOC', wm),
                    AC_rel = ('AC_rel', wm),
                    QC_rel = ('QC_rel', wm),
                    FDC_rel = ('FDC_rel', wm),
                    FLC_rel = ('FLC_rel', wm),
                    FOC_rel = ('FOC_rel', wm),
                    WR = ('WR', wm),
                    WP = ('WP', wm),
                    PFE = ('pfe', wm),
                    Trace = ('trace', wm)
                    )

j.sort_values(by=['SA2_ID', 'SPREAD_ID', 'SPREAD_ID_original', 'irrigation'], ascending = True, inplace = True)
cols = list(range(1, 5)) + list(range(-5, -1))
j[j.columns[cols]]

j.to_csv('N:/Planet-A/Data-Master/Profit_map/SPREAD_aggregated.csv')
j.to_pickle('N:/Planet-A/Data-Master/Profit_map/SPREAD_aggregated.pkl')


adf.loc[adf.duplicated(subset=['CELL_ID']), ('CELL_ID', 'CELL_HA', 'SA2_ID', 'COMMODITIES', 'COMMODITIES_DESC', 'IRRIGATION', 'SPREAD_ID', 'SPREAD_ID_original', 'Irrigation', 'SA2_Name', 'SPREAD_Name', 'Production', 'Area', 'Prod_factor', 'Yield')]

# Check the sum of area of commmodities ABS vs NLUM
tmp = crops_sum_df.groupby(['SPREAD_Name', 'SA2_ID'])[['Area_crops']].first()
tmp.groupby(['SPREAD_Name'])[['Area_crops']].sum()

cell_df.groupby(['COMMODITIES_DESC'])[['CELL_HA']].sum()


""" Javi's equations
( (prod/area) + ((prod/area) * (irrig_factor-1)) ) * 
(F1*Q1*P1)+(F2*Q2*P2)+(F3*Q3*P3))) - (AC_rel+FDC_rel+FOC_rel+FLC_rel) - (QC_rel * ((prod/area)+((prod/area)*(irrig_factor-1)))) - (WR*WP)
rev = prod/area * irrig_factor * ((F1 * Q1 * P1) + (F2 * Q2 * P2) + (F3 * Q3 * P3))
costs = (AC_rel + FDC_rel + FOC_rel + FLC_rel) + (QC_rel * prod/area * irrig_factor) + (WR * WP)
"""



javi[['SA2_ID', 'Commodity', 'SPREAD_Commodity', 'Yield', 'F1', 'Q1', 'P1', 'irrig_factor', 'irrigation']]

# Calculate revenue and costs
javi['Revenue_BB'] = javi.eval('Production * irrig_factor / area * ((F1 * Q1 * P1) + (F2 * Q2 * P2) + (F3 * Q3 * P3))')
javi['Costs_PCT'] = javi.eval('rev * cost_pct')
javi['Costs_RAW'] = javi.eval('(AC_rel + FDC_rel + FOC_rel + FLC_rel) + (QC_rel * Production / area * irrig_factor) + (WR * WP)')
javi[javi['SPREAD_Commodity'].isin(['Beef Cattle','Sheep'])] \
    [['SPREAD_Commodity', 'irrigation', 'irrig_factor', 'Costs_PCT', 'Costs_RAW']]

javi[javi['SPREAD_Commodity'].isin(['Beef Cattle','Sheep'])] \
    [['SA2_ID', 'Commodity', 'SPREAD_Commodity', 'irrigation', 'rev', 'Revenue_BB']] \
    .sort_values(by=['SPREAD_Commodity', 'Commodity'], ascending = False)

javi.loc[(javi['SA2_ID'] == 801041042) & (javi['SPREAD_Commodity'] == 'Beef Cattle'), ('SA2_ID', 'Commodity', 'SPREAD_Commodity', 'Production', 'area', 'irrig_factor', 'Yield', 'rev', 'Revenue_BB')].sort_values(by=['SPREAD_Commodity', 'Commodity'])

    
    
# Calculate costs
javi['Costs_BB'] = javi.eval('AC + QC + FDC + FLC + FOC')
javi['Costs_JN'] = javi.eval('gross_revenue * mean_cost_pct')
javi['Costs_DIFF_%'] = javi.eval('abs(100 * (Costs_BB - Costs_JN) / Costs_JN)')
javi[['SA2_ID', 'product', 'Commodity', 'SPREAD_Commodity', 'gross_revenue', 'Costs_BB', 'Costs_JN', 'Costs_DIFF_%']].sort_values(by=['Costs_DIFF_%', 'Commodity'], ascending=False).head(100)

# Using the mean_cost_pct method the costs per tonne should be constant for all Commodities
javi['PFE_Tonne'] = javi.eval('PFE_dry / yield_sa2_avg')
javi[javi['Commodity'] == 'Canola']['PFE_Tonne'].describe()


# Quick comparison of costs 
javi.eval('Costs_BB - Costs_JN').describe()


# Calculate PFE
javi['PFE_BB'] = javi.eval('Revenue_BB - Costs_BB') 
javi[['SA2_ID', 'yield', 'Commodity', 'SPREAD_Commodity', 'rev', 'pfe']].head(100)




# Groupby and aggregate with namedAgg
wm = lambda x: np.average(x, weights=javi.loc[x.index, "ha_weight_SPREAD"])  # Define a lambda function to compute the weighted mean
j = javi.groupby(['SA2_ID', 'SPREAD_ID'], as_index=False).agg(SPREAD_Name = ('SPREAD_Commodity', 'first'),
                                                              sum_ha_weights = ('ha_weight_SPREAD', 'sum'), 
                                                              Count = ('SPREAD_Commodity', 'size'),
                                                              PFE_Dry_WM = ('PFE_dry', wm)
                                                              )



# Join the table to the dataframe
dfm = cell_df.merge(javi, how='left', left_on=['SA2_MAIN11', 'COMMODITIES', 'IRRIGATION'], right_on=['SA2_ID', 'SPREAD_ID', 'irrigation'])




# Check weighted sum calculations
j.loc[j['sum_ha_weights'] < 0.8]
javi.loc[(javi['SA2_ID'] == 702011054) & (javi['SPREAD_ID'] == 25), ('SA2_ID', 'SA2_Name', 'SPREAD_ID', 'SPREAD_Commodity', 'product', 'ha_weight_SPREAD')]


index = cell_df.query("(C18_DESCRIPTION == 'Dryland cropping (3.3)' or \
                        C18_DESCRIPTION == 'Grazing modified pastures (3.2)' or \
                        C18_DESCRIPTION == 'Grazing native vegetation (2.1)' or \
                        C18_DESCRIPTION == 'Other minimal use (1.3)') and \
                       (FOREST_TYPE_DESC == 'Non-forest or no data')").index
                 
index = cell_df[cell_df['C18_DESCRIPTION'].isin(["Dryland cropping (3.3)", "Grazing modified pastures (3.2)", "Grazing native vegetation (2.1)", "Other minimal use (1.3)"]) & \
                (cell_df['FOREST_TYPE_DESC'] == 'Non-forest or no data')].index
    

# Select rows which satisfy the query
index = cell_df.query("C18_DESCRIPTION in ['Dryland cropping (3.3)', 'Grazing modified pastures (3.2)', 'Grazing native vegetation (2.1)', 'Other minimal use (1.3)'] and \
                      FOREST_TYPE_DESC == 'Non-forest or no data'").index

# Add a new field ad calculate values for selected rows
cell_df['MASK'] = 0
cell_df['MASK'] = cell_df['MASK'].astype(np.uint8)
cell_df.loc[index,'MASK'] = 1










# Add new column in cell_df 
pos = cell_df.columns.get_loc('VEG_COND_DESC') + 1
cell_df.insert(loc = pos, column = 'NATURAL_AREAS', value = 0, allow_duplicates = True)

    
    
    
######################## Handy crosstab queries for exploring the NLUM categorisations

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'VEG_MASK', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'C18_DESCRIPTION', columns = 'VEG_MASK', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'C18_DESCRIPTION', columns = 'COMMODITIES', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'C18_DESCRIPTION', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'PRIMARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'SECONDARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'SECONDARY_V7', columns = 'PRIMARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'SECONDARY_V7', columns = 'VEG_MASK', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'PRIMARY_V7', columns = 'VEG_MASK', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'PRIMARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'VEG_COND_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'PRIMARY_V7', columns = 'VEG_COND_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'PRIMARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'FOREST_TYPE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'SECONDARY_V7', columns = 'FOREST_TYPE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'PRIMARY_V7', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'PRIMARY_V7', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')



cell_df.groupby(['COMMODITIES'])[['COMMODITIES_DESC']].first()



# Knock out 
Mining and waste (5.8, 5.9)
No data
Plantation forestry (3.1, 4.1)
Rural residential and farm infrastructure (5.4...
Urban intensive uses (5.3, 5.4, 5.4.1, 5.5, 5.6...
Water (6.0)

Tenure - water, defence, ocean, aboriginal?


SECONDARY_V7                                                                                                                                                                                           
3.1 Plantation forestry
4.1 Irrigated plantation forestry




np.sort(cell_df['COMMODITIES'].unique())
np.sort(javi['SPREAD_ID'].unique())
np.sort(javi['Commodity'].unique())


# Print mapping of 'product' vs 'SPREAD Commodity' vs 'Commodity' classifications
javi.groupby(['product', 'Commodity', 'SPREAD_Commodity'], as_index=False)[['SPREAD_ID']].first().sort_values(by=['SPREAD_ID', 'Commodity', 'product'])


# Pivot table
pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', values = 'COMMODITIES', aggfunc='mean').sort_values(by=['COMMODITIES'])
pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', values = ['COMMODITIES', 'ALBERS_SQM'], aggfunc= ['sum', 'first']).sort_values(by=[('first', 'COMMODITIES')])
pd.pivot_table(javi, index = 'SPREAD_Commodity', values = 'SPREAD_ID', aggfunc='first').sort_values(by=['SPREAD_ID'])

gdf = gpd.read_file(r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.gpkg')
# df = pd.read_csv(r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.tif.csv')
gdf["AREA"] = gdf['geometry'].area
pd.pivot_table(gdf, index = 'COMMODITIES_DESC', columns = 'VEG_COND_DESC', values = 'AREA', aggfunc = 'sum')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'VEG_COND_DESC', values = 'CELL_HA', aggfunc = 'sum')
pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'VEG_COND_DESC', values = 'CELL_HA', aggfunc = 'count')
pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')


# Dissolve grid cells to 
attrs = ["attr0", "attr1", "attr2"]
SA2_gdf_dissolved = SA2_gdf.dissolve(by=attrs, as_index=False)



