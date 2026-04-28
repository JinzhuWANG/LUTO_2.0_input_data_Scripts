import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import rasterio, matplotlib, itertools
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
# 
# Open NLUM_ID as mask raster and get metadata
with rasterio.open('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as rst:
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


# Open the template to crosscheck that we have all records needed
def_df = pd.read_hdf('N:/Data-Master/Profit_map/NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')
def_df['SA2_ID'] = pd.to_numeric(def_df['SA2_ID'], downcast = 'integer')
# def_df.rename(columns = {'SA2_MAIN11': 'SA2_ID'})

# # Build an SA2, SA4, STATE concordance file - run once then load file
# sa2 = gpd.read_file('N:/Data-Master/Australian_administrative_boundaries/sa2_2011_aus/SA2_2011_AUST.shp')
# sa2 = pd.DataFrame(sa2)
# sa2 = sa2[['SA2_MAIN11', 'SA2_NAME11', 'SA4_CODE11', 'SA4_NAME11', 'STE_CODE11', 'STE_NAME11']]
# cols = ['SA2_MAIN11', 'SA4_CODE11', 'STE_CODE11']
# sa2[cols] = sa2[cols].apply(pd.to_numeric, axis = 1)
# downcast(sa2)
# sa2.to_hdf('N:/Data-Master/Profit_map/SA2_SA4_STATE_Concordance.h5', key = 'SA2_SA4_STATE_Concordance', mode = 'w', format = 't')
sa2 = pd.read_hdf('N:/Data-Master/Profit_map/SA2_SA4_STATE_Concordance.h5')

# Merge SA4 and State info to LU template
def_df = def_df.merge(sa2, how = 'left', left_on = 'SA2_ID', right_on = 'SA2_MAIN11')
def_df.drop(columns = ['SA2_MAIN11'], inplace = True)
def_df['GAEZ_ID'] = def_df['LU_ID']
def_df.loc[def_df['LU_ID'] >= 30, 'GAEZ_ID'] = 13 # All pasture (even natural) attributed the same climate damage as hay

# Create a dataframe with all possible combinations of 'SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP'
all_SA2_IDs = def_df['SA2_ID'].unique()
all_GAEZ_IDs = def_df['GAEZ_ID'].unique()
all_IRRIGATIONs = list(def_df['IRRIGATION'].unique())
all_RCPs = ['rcp2p6', 'rcp4p5', 'rcp6p0', 'rcp8p5']
combined = [all_SA2_IDs, all_GAEZ_IDs, all_IRRIGATIONs, all_RCPs]
def_df_combos = pd.DataFrame(columns = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP'], data=list(itertools.product(*combined))).sort_values(by = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP']).reset_index().drop(columns = 'index')

# Merge NLUM template to ensure all RCPs are listed for each mapped combination of 'SA2_ID', 'GAEZ_ID', 'IRRIGATION'
def_df = def_df.merge(def_df_combos, how = 'left', on = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION']).sort_values(by = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION']).sort_values(by = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION', 'RCP']).reset_index().drop(columns = 'index')


###################### Create dataframes with CO2 fertilization on/off

# Load GAEZ yield gap data from Michalis (only for crops)
gaez_orig = pd.read_csv('N:/Data-Master/Climate_damage/GAEZ_approach/From_Michalis/LUTO_CC_yield_impacts_SA2_20-Aug-2021 17.59.csv')

# Create list to hold the two dataframes
df_holder = []

for i in [0, 1]:
    
    gaez = gaez_orig.query('CO2_fert == @i')
    gaez = gaez.drop(columns = ['Unnamed: 0', 'LU_ID', 'Units', 'CO2_fert'])
    gaez.rename(columns = {'ID': 'GAEZ_ID', 'SA2_MAIN11': 'SA2_ID', 'Irrigation': 'IRRIGATION', '2010': 'YR_2010', '2020': 'YR_2020', '2050': 'YR_2050', '2080': 'YR_2080'}, inplace = True)
    
    # Knock out values that seemingly had not been converted to multipliers
    ind = gaez.query('YR_2020 > 3 or YR_2050 > 3 or YR_2080 > 3').index
    gaez.loc[ind, ['YR_2020', 'YR_2050', 'YR_2080']] = np.nan
    
    # Get 95% confidence interval limits
    gaez_d = gaez.loc[:, ['YR_2020', 'YR_2050', 'YR_2080']]
    gaez_d.replace([np.inf, -np.inf, 0], np.nan, inplace = True)
    low, high = np.nanpercentile(gaez_d[['YR_2020', 'YR_2050', 'YR_2080']], [5, 95])
    
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
    
    # Summarise climate change damage by irrigation status and national and merge
    ccd_AUS_allcrops = ccd.groupby(['IRRIGATION', 'RCP'], observed = True, as_index = False).agg(YR_2020_AUS_allcrops = ('YR_2020', 'mean'), YR_2050_AUS_allcrops = ('YR_2050', 'mean'), YR_2080_AUS_allcrops = ('YR_2080', 'mean'))
    ccd = ccd.merge(ccd_AUS_allcrops, how = 'left', on = ['IRRIGATION', 'RCP'])
    
    
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
    ccd.loc[idx, 'CCD_SOURCE'] = 'SA2 - mean from all crops with same irr status'
    
    # Use average value for same irrigation status at SA4 level
    idx = ccd.query('(YR_2020 != YR_2020) and (YR_2020_SA4_allcrops == YR_2020_SA4_allcrops)').index
    ccd.loc[idx, 'YR_2020'] = ccd['YR_2020_SA4_allcrops']
    ccd.loc[idx, 'YR_2050'] = ccd['YR_2050_SA4_allcrops']
    ccd.loc[idx, 'YR_2080'] = ccd['YR_2080_SA4_allcrops']
    ccd.loc[idx, 'CCD_SOURCE'] = 'SA4 - mean from all crops with same irr status'
    
    # Use average value for same irrigation status at State level
    idx = ccd.query('(YR_2020 != YR_2020) and (YR_2020_STE_allcrops == YR_2020_STE_allcrops)').index
    ccd.loc[idx, 'YR_2020'] = ccd['YR_2020_STE_allcrops']
    ccd.loc[idx, 'YR_2050'] = ccd['YR_2050_STE_allcrops']
    ccd.loc[idx, 'YR_2080'] = ccd['YR_2080_STE_allcrops']
    ccd.loc[idx, 'CCD_SOURCE'] = 'STATE - mean from all crops with same irr status'
    
    # Use average value for same irrigation status at State level
    idx = ccd.query('(YR_2020 != YR_2020) and (YR_2020_AUS_allcrops == YR_2020_AUS_allcrops)').index
    ccd.loc[idx, 'YR_2020'] = ccd['YR_2020_AUS_allcrops']
    ccd.loc[idx, 'YR_2050'] = ccd['YR_2050_AUS_allcrops']
    ccd.loc[idx, 'YR_2080'] = ccd['YR_2080_AUS_allcrops']
    ccd.loc[idx, 'CCD_SOURCE'] = 'AUS - mean from all crops with same irr status'

    
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
    ludf = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5')
    ldf = ludf[['CELL_ID', 'SA2_ID', 'IRRIGATION', 'LU_ID', 'LU_DESC']]
    ccp_tmp = ccp.copy()
    ccp_tmp.columns = ccp_tmp.columns.to_flat_index()
    tmp = ldf.merge(ccp_tmp, how = 'left', on = ['SA2_ID', 'LU_ID', 'IRRIGATION']).query('LU_ID >= 5')
    print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])
    
    # Downcast and append dataframe
    downcast(ccp)
    
    ccp = pd.concat([ccp], names=['CO2_Fert'], keys=[i], axis = 1)
    
    df_holder.append(ccp)

# Concatenate dataframes
result = pd.concat(df_holder, axis = 1, sort = False)

# Export to HDF5
result.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_climate_damage_mult.h5', key = 'SA2_climate_damage_mult', mode = 'w', format = 'fixed')



# Summarise averages
avg = result.groupby(level = [1, 2]).mean()

# Create a lookup table of LU_ID x LU_DESC
def_df = pd.read_hdf('N:/Data-Master/Profit_map/NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')
lut = def_df.groupby('LU_ID')['LU_DESC'].first()

# Flatten multi-level dataframe
avg.columns = avg.columns.to_flat_index()
avg = avg.reset_index()

# Merge names
avg = avg.merge(lut, how = 'left', on = 'LU_ID')

avg = avg.set_index(['LU_ID', 'LU_DESC','IRRIGATION'])

# Recreate multiindex
avg.columns = pd.MultiIndex.from_tuples(avg.columns)

# Save to CSV
avg.to_csv('N:/Data-Master/LUTO_2.0_input_data/Scripts/avg.csv')
