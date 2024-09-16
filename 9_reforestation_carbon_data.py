import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
# from scipy import ndimage as nd
import rasterio, matplotlib, itertools, h5py
from rasterio import features
from rasterio.fill import fillnodata
from rasterio.warp import reproject
from rasterio.enums import Resampling


############################################################################################################################################
# Initialisation. Create some helper data and functions
############################################################################################################################################

# Set some options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.float_format', '{:,.4f}'.format)

# Open NLUM_ID as mask raster and get metadata
with rasterio.open('N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as rst:
    NLUM_mask = rst.read(1)
    
    # Get metadata and update parameters
    NLUM_transform = rst.transform
    NLUM_bounds = rst.bounds
    NLUM_height = rst.height
    NLUM_width = rst.width
    NLUM_crs = rst.crs
    meta = rst.meta.copy()
    meta.update(compress = 'lzw', driver = 'GTiff') # dtype='int32', nodata='-99')
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


inpath = 'N:/Planet-A/Data-Master/FullCAM/Output_layers/'
outpath = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/3D_Spatial_Timeseries/'

b_df = pd.read_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5')

cell_ha = np.array(b_df[['CELL_HA']]).squeeze().astype(np.float32)

rainfall = np.array(b_df[['AVG_AN_PREC_MM_YR']]).squeeze().astype(np.float32)

rip_length = np.array(b_df[['RIP_LENGTH_M_CELL']]).squeeze().astype(np.float32)

buffer_dist = 30

rip_area_prop = ((rip_length * buffer_dist) / 10000) / cell_ha



########### Environmental plantings (block, riparian, and belt planting arrangements)

# Set cap on amount of carbon in tonnes per hectare
max_tree_C = 1500
max_debris_C = 300
max_soil_C = 500

# Open the numpy array of shape 6956407, 3, 104. Axis 2 is C (t/ha) of trees, debris, soil. Axis 3 is ID, X, Y, then 101 years of simulated data, We want the first 91 years. Ultimate shape is (91, 3, 6956407)
ep_block_array = np.transpose(np.load(inpath + 'ep_block.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)
ep_rip_array = np.transpose(np.load(inpath + 'ep_rip.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)
ep_belt_array = np.transpose(np.load(inpath + 'ep_belt_hd.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)

# Burn in riparian areas
ep_block_array = (1 - rip_area_prop) * ep_block_array + rip_area_prop * ep_block_array

# Cap total carbon per hectare and convert to total CO2e per cell
ep_block_array[:, 0, :] = np.where(ep_block_array[:, 0, :] > max_tree_C, max_tree_C, ep_block_array[:, 0, :]) * 44 / 12 
ep_block_array[:, 1, :] = np.where(ep_block_array[:, 1, :] > max_debris_C, max_debris_C, ep_block_array[:, 1, :]) * 44 / 12 
ep_block_array[:, 2, :] = np.where(ep_block_array[:, 2, :] > max_soil_C, max_soil_C, ep_block_array[:, 2, :]) * 44 / 12 

ep_rip_array[:, 0, :] = np.where(ep_rip_array[:, 0, :] > max_tree_C, max_tree_C, ep_rip_array[:, 0, :]) * 44 / 12 
ep_rip_array[:, 1, :] = np.where(ep_rip_array[:, 1, :] > max_debris_C, max_debris_C, ep_rip_array[:, 1, :]) * 44 / 12 
ep_rip_array[:, 2, :] = np.where(ep_rip_array[:, 2, :] > max_soil_C, max_soil_C, ep_rip_array[:, 2, :]) * 44 / 12 

ep_belt_array[:, 0, :] = np.where(ep_belt_array[:, 0, :] > max_tree_C, max_tree_C, ep_belt_array[:, 0, :]) * 44 / 12 
ep_belt_array[:, 1, :] = np.where(ep_belt_array[:, 1, :] > max_debris_C, max_debris_C, ep_belt_array[:, 1, :]) * 44 / 12 
ep_belt_array[:, 2, :] = np.where(ep_belt_array[:, 2, :] > max_soil_C, max_soil_C, ep_belt_array[:, 2, :]) * 44 / 12 

# Export to HDF5
with h5py.File(outpath + 'tCO2_ha_ep_block.h5', 'w') as h5f:
    h5f.create_dataset('Trees_tCO2_ha', data = ep_block_array[:, 0, :], chunks = True)
    h5f.create_dataset('Debris_tCO2_ha', data = ep_block_array[:, 1, :], chunks = True)
    h5f.create_dataset('Soil_tCO2_ha', data = ep_block_array[:, 2, :], chunks = True)

with h5py.File(outpath + 'tCO2_ha_ep_rip.h5', 'w') as h5f:
    h5f.create_dataset('Trees_tCO2_ha', data = ep_rip_array[:, 0, :], chunks = True)
    h5f.create_dataset('Debris_tCO2_ha', data = ep_rip_array[:, 1, :], chunks = True)
    h5f.create_dataset('Soil_tCO2_ha', data = ep_rip_array[:, 2, :], chunks = True)
    
with h5py.File(outpath + 'tCO2_ha_ep_belt.h5', 'w') as h5f:
    h5f.create_dataset('Trees_tCO2_ha', data = ep_belt_array[:, 0, :], chunks = True)
    h5f.create_dataset('Debris_tCO2_ha', data = ep_belt_array[:, 1, :], chunks = True)
    h5f.create_dataset('Soil_tCO2_ha', data = ep_belt_array[:, 2, :], chunks = True)



########### Carbon plantings (block and belt planting arrangements)

# Open the numpy array of shape 6956407, 3, 104. Axis 2 is C (t/ha) of trees, debris, soil. Axis 3 is ID, X, Y, then 101 years of simulated data, We want the first 91 years. Ultimate shape is (91, 3, 6956407)
mal_block_array = np.transpose(np.load(inpath + 'mal_block.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)
mal_rip_array = np.transpose(np.load(inpath + 'mal_rip.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)
mal_belt_array = np.transpose(np.load(inpath + 'mal_belt_hd.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)
eglob_array = np.transpose(np.load(inpath + 'eglob_lr.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)

# Burn in riparian areas
mal_block_array = (1 - rip_area_prop) * mal_block_array + rip_area_prop * mal_block_array

# Smooth out transition from Mallee to E. globulus around 600mm rainfall
x = (rainfall - 550) / 100
p = np.where(x > 1, 1, np.where(x < 0, 0, x))
cp_block_array = (1 - p) * mal_block_array + p * eglob_array
cp_belt_array = (1 - p) * mal_belt_array + p * eglob_array

# Cap total carbon per hectare and convert to total CO2e per cell
cp_block_array[:, 0, :] = np.where(cp_block_array[:, 0, :] > max_tree_C, max_tree_C, cp_block_array[:, 0, :]) * 44 / 12 
cp_block_array[:, 1, :] = np.where(cp_block_array[:, 1, :] > max_debris_C, max_debris_C, cp_block_array[:, 1, :]) * 44 / 12 
cp_block_array[:, 2, :] = np.where(cp_block_array[:, 2, :] > max_soil_C, max_soil_C, cp_block_array[:, 2, :]) * 44 / 12 

cp_belt_array[:, 0, :] = np.where(cp_belt_array[:, 0, :] > max_tree_C, max_tree_C, cp_belt_array[:, 0, :]) * 44 / 12 
cp_belt_array[:, 1, :] = np.where(cp_belt_array[:, 1, :] > max_debris_C, max_debris_C, cp_belt_array[:, 1, :]) * 44 / 12 
cp_belt_array[:, 2, :] = np.where(cp_belt_array[:, 2, :] > max_soil_C, max_soil_C, cp_belt_array[:, 2, :]) * 44 / 12 

# Export to HDF5
with h5py.File(outpath + 'tCO2_ha_cp_block.h5', 'w') as h5f:
    h5f.create_dataset('Trees_tCO2_ha', data = cp_block_array[:, 0, :], chunks = True)
    h5f.create_dataset('Debris_tCO2_ha', data = cp_block_array[:, 1, :], chunks = True)
    h5f.create_dataset('Soil_tCO2_ha', data = cp_block_array[:, 2, :], chunks = True)
    
with h5py.File(outpath + 'tCO2_ha_cp_belt.h5', 'w') as h5f:
    h5f.create_dataset('Trees_tCO2_ha', data = cp_belt_array[:, 0, :], chunks = True)
    h5f.create_dataset('Debris_tCO2_ha', data = cp_belt_array[:, 1, :], chunks = True)
    h5f.create_dataset('Soil_tCO2_ha', data = cp_belt_array[:, 2, :], chunks = True)


########### Human-induced regeneration (block arrangement)

# Open the numpy array of shape 6956407, 3, 104. Axis 2 is C (t/ha) of trees, debris, soil. Axis 3 is ID, X, Y, then 101 years of simulated data, We want the first 91 years. Ultimate shape is (91, 3, 6956407)
hir_gt_500_array = np.transpose(np.load(inpath + 'HIR_NFMR_AC_gt_500mm.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)
hir_gt_500_rip_array = np.transpose(np.load(inpath + 'HIR_NFMR_AC_gt_500mm_rip.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)
hir_lt_500_array = np.transpose(np.load(inpath + 'HIR_NFMR_AC_lt_500mm.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)
hir_lt_500_rip_array = np.transpose(np.load(inpath + 'HIR_NFMR_AC_lt_500mm_rip.npy')[..., 3:94], axes = [2, 1, 0]).astype(np.float32)

# Burn riparian into HIR
hir_gt_500_array = (1 - rip_area_prop) * hir_gt_500_array + rip_area_prop * hir_gt_500_rip_array
hir_lt_500_array = (1 - rip_area_prop) * hir_lt_500_array + rip_area_prop * hir_lt_500_rip_array

# Smooth out transition from gt 500 to lt 500 mm rainfall
x = (rainfall - 450) / 100
p = np.where(x > 1, 1, np.where(x < 0, 0, x))
hir_array = (1 - p) * hir_lt_500_array + p * hir_gt_500_array
hir_rip_array = (1 - p) * hir_lt_500_rip_array + p * hir_gt_500_rip_array

# Cap total carbon per hectare and convert to total CO2e per cell
hir_array[:, 0, :] = np.where(hir_array[:, 0, :] > max_tree_C, max_tree_C, hir_array[:, 0, :]) * 44 / 12 
hir_array[:, 1, :] = np.where(hir_array[:, 1, :] > max_debris_C, max_debris_C, hir_array[:, 1, :]) * 44 / 12 
hir_array[:, 2, :] = np.where(hir_array[:, 2, :] > max_soil_C, max_soil_C, hir_array[:, 2, :]) * 44 / 12 

hir_rip_array[:, 0, :] = np.where(hir_rip_array[:, 0, :] > max_tree_C, max_tree_C, hir_rip_array[:, 0, :]) * 44 / 12 
hir_rip_array[:, 1, :] = np.where(hir_rip_array[:, 1, :] > max_debris_C, max_debris_C, hir_rip_array[:, 1, :]) * 44 / 12 
hir_rip_array[:, 2, :] = np.where(hir_rip_array[:, 2, :] > max_soil_C, max_soil_C, hir_rip_array[:, 2, :]) * 44 / 12 

# Export to HDF5
with h5py.File(outpath + 'tCO2_ha_hir_block.h5', 'w') as h5f:
    h5f.create_dataset('Trees_tCO2_ha', data = hir_array[:, 0, :], chunks = True)
    h5f.create_dataset('Debris_tCO2_ha', data = hir_array[:, 1, :], chunks = True)
    h5f.create_dataset('Soil_tCO2_ha', data = hir_array[:, 2, :], chunks = True)

with h5py.File(outpath + 'tCO2_ha_hir_rip.h5', 'w') as h5f:
    h5f.create_dataset('Trees_tCO2_ha', data = hir_rip_array[:, 0, :], chunks = True)
    h5f.create_dataset('Debris_tCO2_ha', data = hir_rip_array[:, 1, :], chunks = True)
    h5f.create_dataset('Soil_tCO2_ha', data = hir_rip_array[:, 2, :], chunks = True)



    



"""
# Example of loading HDF5 data
with h5py.File(outpath + 'cell_CO2_ep_block.h5', 'r') as h5f:
    brick = h5f['Trees_CO2_ha'][...]





# Open the template to crosscheck that we have all records needed
def_df = pd.read_hdf('N:/Planet-A/Data-Master/Profit_map/NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')
def_df['SA2_ID'] = pd.to_numeric(def_df['SA2_ID'], downcast = 'integer')
# def_df.rename(columns = {'SA2_MAIN11': 'SA2_ID'})

# # Build an SA2, SA4, STATE concordance file - run once then load file
# sa2 = gpd.read_file('N:/Planet-A/Data-Master/Australian_administrative_boundaries/sa2_2011_aus/SA2_2011_AUST.shp')
# sa2 = pd.DataFrame(sa2)
# sa2 = sa2[['SA2_MAIN11', 'SA2_NAME11', 'SA4_CODE11', 'SA4_NAME11', 'STE_CODE11', 'STE_NAME11']]
# cols = ['SA2_MAIN11', 'SA4_CODE11', 'STE_CODE11']
# sa2[cols] = sa2[cols].apply(pd.to_numeric, axis = 1)
# downcast(sa2)
# sa2.to_hdf('N:/Planet-A/Data-Master/Profit_map/SA2_SA4_STATE_Concordance.h5', key = 'SA2_SA4_STATE_Concordance', mode = 'w', format = 't')
sa2 = pd.read_hdf('N:/Planet-A/Data-Master/Profit_map/SA2_SA4_STATE_Concordance.h5')

# Merge SA4 and State info to LU template
def_df = def_df.merge(sa2, how = 'left', left_on = 'SA2_ID', right_on = 'SA2_MAIN11')
def_df.drop(columns = ['SA2_MAIN11'], inplace = True)
def_df['GAEZ_ID'] = def_df['LU_ID']
def_df.loc[def_df['LU_ID'] >= 30, 'GAEZ_ID'] = 13 # All pasture (even natural) attributed the same climate damage as hay

# Creat a dataframe with all possible combinations of 'SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP'
all_SA2_IDs = def_df['SA2_ID'].unique()
all_GAEZ_IDs = def_df['GAEZ_ID'].unique()
all_IRRIGATIONs = list(def_df['IRRIGATION'].unique())
all_RCPs = ['rcp2p6', 'rcp4p5', 'rcp6p0', 'rcp8p5']
combined = [all_SA2_IDs, all_GAEZ_IDs, all_IRRIGATIONs, all_RCPs]
def_df_combos = pd.DataFrame(columns = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP'], data=list(itertools.product(*combined))).sort_values(by = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP']).reset_index().drop(columns = 'index')

# Merge NLUM template to ensure all RCPs are listed for each mapped combination of 'SA2_ID', 'GAEZ_ID', 'IRRIGATION'
def_df = def_df.merge(def_df_combos, how = 'left', on = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION']).sort_values(by = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION']).sort_values(by = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP']).reset_index().drop(columns = 'index')



# Load GAEZ yield gap data from Michalis (only for crops)
gaez = pd.read_csv('N:/Planet-A/Data-Master/Climate_damage/GAEZ_approach/From_Michalis/LUTO_CC_yield_impacts_SA2_20-Aug-2021 17.59.csv')
gaez = gaez.query('CO2_fert == 1')
gaez.drop(columns = ['Unnamed: 0', 'LU_ID', 'Units', 'CO2_fert'], inplace = True)
gaez.rename(columns = {'ID': 'GAEZ_ID', 'SA2_MAIN11': 'SA2_ID', 'Irrigation': 'IRRIGATION', '2010': 'YR_2010', '2020': 'YR_2020', '2050': 'YR_2050', '2080': 'YR_2080'}, inplace = True)


# Get 95% confidence interval limits
gaez_d = gaez.loc[:, ['YR_2020', 'YR_2050', 'YR_2080']]
gaez_d.replace([np.inf, -np.inf, 0], np.nan, inplace = True)
low, high = np.nanpercentile(gaez_d[['YR_2020', 'YR_2050', 'YR_2080']], [2.5, 97.5])

# Knock out extreme values
ind = gaez.query('YR_2020 < @low or YR_2050 < @low or YR_2080 < @low or YR_2020 > @high or YR_2050 > @high or YR_2080 > @high').index
gaez.loc[ind, ['YR_2020', 'YR_2050', 'YR_2080']] = np.nan
gaez.dropna(inplace = True)

# # Enumerate all possible combinations of 'SA2_ID', 'GAEZ_ID', 'IRRIGATION', and 'RCP'
# gaez_full = pd.pivot_table(gaez, values = ['YR_2010', 'YR_2020', 'YR_2050', 'YR_2080'], index = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION'], columns = 'RCP', dropna = False, aggfunc = np.mean)
# gaez_full = gaez_full.stack(dropna = False).sort_values(by = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP'])

# Merge climate change agricultural damage with NLUM template to check for gaps
ccd = def_df.merge(gaez.drop(columns = ['LU_DESC', 'SA2_NAME11', 'STE_NAME11']), how = 'left', on = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP']).sort_values(by = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP'])

# Make a couple of small mods
ccd.loc[:, 'YR_2010'] = 1
ccd.loc[:, 'CCD_SOURCE'] = 'GAEZ'

# Summarise climate change damage by crop, irrigation status, and SA4 and merge
ccd_SA4 = ccd.groupby(['SA4_CODE11', 'GAEZ_ID', 'IRRIGATION', 'RCP'], observed = True, as_index = False).agg(YR_2020_SA4 = ('YR_2020', 'mean'), YR_2050_SA4 = ('YR_2050', 'mean'), YR_2080_SA4 = ('YR_2080', 'mean'))
ccd = ccd.merge(ccd_SA4, how = 'left', on = ['SA4_CODE11', 'GAEZ_ID', 'IRRIGATION', 'RCP'])

# Summarise climate change damage by irrigation status and SA2 and merge
ccd_SA2_allcrops = ccd.groupby(['SA2_ID', 'IRRIGATION', 'RCP'], observed = True, as_index = False).agg(YR_2020_SA2_allcrops = ('YR_2020', 'mean'), YR_2050_SA2_allcrops = ('YR_2050', 'mean'), YR_2080_SA2_allcrops = ('YR_2080', 'mean'))
ccd = ccd.merge(ccd_SA2_allcrops, how = 'left', on = ['SA2_ID', 'IRRIGATION', 'RCP'])

# Summarise climate change damage by irrigation status and SA2 and merge
ccd_SA4_allcrops = ccd.groupby(['SA4_CODE11', 'IRRIGATION', 'RCP'], observed = True, as_index = False).agg(YR_2020_SA4_allcrops = ('YR_2020', 'mean'), YR_2050_SA4_allcrops = ('YR_2050', 'mean'), YR_2080_SA4_allcrops = ('YR_2080', 'mean'))
ccd = ccd.merge(ccd_SA4_allcrops, how = 'left', on = ['SA4_CODE11', 'IRRIGATION', 'RCP'])

# Summarise climate change damage by irrigation status and state and merge
ccd_STE_allcrops = ccd.groupby(['STE_CODE11', 'IRRIGATION', 'RCP'], observed = True, as_index = False).agg(YR_2020_STE_allcrops = ('YR_2020', 'mean'), YR_2050_STE_allcrops = ('YR_2050', 'mean'), YR_2080_STE_allcrops = ('YR_2080', 'mean'))
ccd = ccd.merge(ccd_STE_allcrops, how = 'left', on = ['STE_CODE11', 'IRRIGATION', 'RCP'])


# Use average value for same crop and irrigation status at SA4 level
idx = ccd.query('(YR_2020 != YR_2020) and (YR_2020_SA4 == YR_2020_SA4)').index
ccd.loc[idx, 'YR_2020'] = ccd['YR_2020_SA4']
ccd.loc[idx, 'YR_2050'] = ccd['YR_2050_SA4']
ccd.loc[idx, 'YR_2080'] = ccd['YR_2080_SA4']
ccd.loc[idx, 'CCD_SOURCE'] = 'SA4 - mean from same crop/irr status'

# Use average value for same irrigation status at SA2 level
idx = ccd.query('(YR_2020 != YR_2020) and (YR_2020_SA2_allcrops == YR_2020_SA2_allcrops)').index
ccd.loc[idx, 'YR_2020'] = ccd['YR_2020_SA2_allcrops']
ccd.loc[idx, 'YR_2050'] = ccd['YR_2050_SA2_allcrops']
ccd.loc[idx, 'YR_2080'] = ccd['YR_2080_SA2_allcrops']
ccd.loc[idx, 'CCD_SOURCE'] = 'SA2 - mean from same irr status'

# Use average value for same irrigation status at SA4 level
idx = ccd.query('(YR_2020 != YR_2020) and (YR_2020_SA4_allcrops == YR_2020_SA4_allcrops)').index
ccd.loc[idx, 'YR_2020'] = ccd['YR_2020_SA4_allcrops']
ccd.loc[idx, 'YR_2050'] = ccd['YR_2050_SA4_allcrops']
ccd.loc[idx, 'YR_2080'] = ccd['YR_2080_SA4_allcrops']
ccd.loc[idx, 'CCD_SOURCE'] = 'SA4 - mean from same irr status'

# Use average value for same irrigation status at State level
idx = ccd.query('(YR_2020 != YR_2020) and (YR_2020_STE_allcrops == YR_2020_STE_allcrops)').index
ccd.loc[idx, 'YR_2020'] = ccd['YR_2020_STE_allcrops']
ccd.loc[idx, 'YR_2050'] = ccd['YR_2050_STE_allcrops']
ccd.loc[idx, 'YR_2080'] = ccd['YR_2080_STE_allcrops']
ccd.loc[idx, 'CCD_SOURCE'] = 'STATE - mean from same irr status'

# Clean up dataframe
ccd.drop(columns = ['YR_2020_SA4', 'YR_2050_SA4', 'YR_2080_SA4', 'YR_2020_SA2_allcrops', 'YR_2050_SA2_allcrops', 'YR_2080_SA2_allcrops', 'YR_2020_SA4_allcrops', 'YR_2050_SA4_allcrops', 'YR_2080_SA4_allcrops', 'YR_2020_STE_allcrops', 'YR_2050_STE_allcrops', 'YR_2080_STE_allcrops'], inplace = True)

# Check for nodata
print('Number of NaNs =', ccd[ccd.drop(columns = ['Crop', 'Crop_group']).isna().any(axis = 1)].shape[0])

# Create pivot table
ccp = pd.pivot_table(ccd, values = ['YR_2020', 'YR_2050', 'YR_2080'], index = ['SA2_ID', 'LU_ID', 'IRRIGATION'], columns = 'RCP', aggfunc = np.mean)
# xx = ccp['YR_2080', 'rcp8p5']
# xx[xx < 2].plot.hist(bins = 50, alpha = 0.5)
ccp.columns = ccp.columns.rename('Year', level = 0)
ccp = ccp.reorder_levels(['RCP', 'Year'], axis = 1)
ccp.sort_index(axis = 1, level = 0, inplace = True)

# Check for nodata
print('Number of NaNs =', ccp[ccp.isna().any(axis = 1)].shape[0])


# Check that we have data everywhere we need it
ludf = pd.read_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5')
ldf = ludf[['CELL_ID', 'SA2_ID', 'IRRIGATION', 'LU_ID', 'LU_DESC']]
tmp = ldf.merge(ccp, how = 'left', on = ['SA2_ID', 'LU_ID', 'IRRIGATION']).query('LU_ID >= 5')
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])


# Export to HDF5
downcast(ccp)
ccp.to_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_climate_damage_mult.h5', key = 'SA2_climate_damage_mult', mode = 'w', format = 'fixed')
"""

