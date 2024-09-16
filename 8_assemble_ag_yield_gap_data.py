import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
# from scipy import ndimage as nd
import rasterio, matplotlib
from rasterio import features
from rasterio.fill import fillnodata
from rasterio.warp import reproject
from rasterio.enums import Resampling


############################################################################################################################################
# Initialisation. Create some helper data and functions
############################################################################################################################################

# Set some options
pd.set_option('display.width', 380)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.float_format', '{:,.4f}'.format)
# 
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
def_df.loc[def_df['LU_ID'] >= 34, 'GAEZ_ID'] = 13 


# Load GAEZ yield gap data from Michalis (only for crops)
gaez = pd.read_csv('N:/Planet-A/Data-Master/Sustainable_intensification/From_Michalis/LUTO_Current+attainable_yields_SA2_19-Aug-2021 15.34.csv')
gaez.drop(columns = ['Unnamed: 0', 'LU_ID', 'Units', 'RCP', 'Year', 'SA2_NAME11', 'STE_NAME11'], inplace = True)
gaez.rename(columns = {'Yield.current': 'YIELD_CURR', 'Yield.attainable': 'YIELD_ATT', 'Yield.multiplier': 'YIELD_MULT'}, inplace = True)

# Fix error with rice
idx = gaez.query('ID == 7 and Irrigation == 0').index
gaez.loc[idx, 'Irrigation'] = 1

# Fix error with olives
# gaez[gaez['Crop'] == 'Olive'].groupby(['SA2_MAIN11', 'Irrigation'], observed = True, as_index = False)['Crop'].first()


# Merge GAEZ yield gap data to NLUM/SA2 template
yg_df = def_df.merge(gaez, how = 'left', left_on = ['SA2_ID', 'GAEZ_ID', 'IRRIGATION'], right_on = ['SA2_MAIN11', 'ID', 'Irrigation'])
yg_df = yg_df[['STE_CODE11', 'STE_NAME11', 'SA4_CODE11', 'SA4_NAME11', 'SA2_ID', 'SA2_NAME11', 'LU_ID', 'GAEZ_ID', 'LU_DESC_x', 'Crop', 'Crop_group', 'IRRIGATION', 'YIELD_CURR', 'YIELD_ATT', 'YIELD_MULT']]
yg_df.rename(columns = {'LU_DESC_x': 'LU_DESC'}, inplace = True)

# Calculate summary tables by SA4 and STE to fill in gaps
yg_df_lite = yg_df.query('YIELD_CURR > 0 and YIELD_ATT > 0')

yg_SA4 = yg_df_lite.groupby(['SA4_CODE11', 'GAEZ_ID', 'IRRIGATION'], observed = True, as_index = False).agg(YM_SA4 = ('YIELD_MULT', 'mean'))
yg_STE = yg_df_lite.groupby(['STE_CODE11', 'GAEZ_ID', 'IRRIGATION'], observed = True, as_index = False).agg(YM_STE = ('YIELD_MULT', 'mean'))
yg_AUS = yg_df_lite.groupby(['GAEZ_ID', 'IRRIGATION'], observed = True, as_index = False).agg(YM_AUS = ('YIELD_MULT', 'mean'))
yg_SA2_ALLCROPS = yg_df_lite.groupby(['SA2_ID', 'IRRIGATION'], observed = True, as_index = False).agg(YM_SA2_ALLCROPS = ('YIELD_MULT', 'mean'))
yg_SA4_ALLCROPS = yg_df_lite.groupby(['SA4_CODE11', 'IRRIGATION'], observed = True, as_index = False).agg(YM_SA4_ALLCROPS = ('YIELD_MULT', 'mean'))
yg_STE_ALLCROPS = yg_df_lite.groupby(['STE_CODE11', 'IRRIGATION'], observed = True, as_index = False).agg(YM_STE_ALLCROPS = ('YIELD_MULT', 'mean'))
yg_AUS_ALLCROPS = yg_df_lite.groupby(['IRRIGATION'], observed = True, as_index = False).agg(YM_AUS_ALLCROPS = ('YIELD_MULT', 'mean'))

yg_df = yg_df.merge(yg_SA4, how = 'left', on = ['SA4_CODE11', 'GAEZ_ID', 'IRRIGATION'])
yg_df = yg_df.merge(yg_STE, how = 'left', on = ['STE_CODE11', 'GAEZ_ID', 'IRRIGATION'])
yg_df = yg_df.merge(yg_AUS, how = 'left', on = ['GAEZ_ID', 'IRRIGATION'])
yg_df = yg_df.merge(yg_SA2_ALLCROPS, how = 'left', on = ['SA2_ID', 'IRRIGATION'])
yg_df = yg_df.merge(yg_SA4_ALLCROPS, how = 'left', on = ['SA4_CODE11', 'IRRIGATION'])
yg_df = yg_df.merge(yg_STE_ALLCROPS, how = 'left', on = ['STE_CODE11', 'IRRIGATION'])
yg_df = yg_df.merge(yg_AUS_ALLCROPS, how = 'left', on = ['IRRIGATION'])


############ Fill gaps in yield gap data
yg_df['YIELD_GAP_MULT'] = np.nan
yg_df['YIELD_GAP_SOURCE'] = 'No data'

# Grazing natural land set to 1 (i.e. no yield gap)
idx = yg_df.query('31 <= LU_ID <= 33').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = 1
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'No yield gap - grazing natural land'


# Get value from direct SA2 GAEZ value
idx = yg_df.query('(YIELD_GAP_MULT != YIELD_GAP_MULT) and (YIELD_MULT >= 1)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = yg_df['YIELD_MULT']
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'SA2 - direct value of same crop/irr status'

# Use average value for same crop and irrigation status at SA4 level
idx = yg_df.query('(YIELD_GAP_MULT != YIELD_GAP_MULT) and (YM_SA4 >= 1)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = yg_df['YM_SA4']
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'SA4 - mean from same crop/irr status'

# Use average value for same crop and irrigation status at state level
idx = yg_df.query('(YIELD_GAP_MULT != YIELD_GAP_MULT) and (YM_STE >= 1)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = yg_df['YM_STE']
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'STE - mean from same crop/irr status'

# Use average value for same crop and irrigation status at national level
idx = yg_df.query('(YIELD_GAP_MULT != YIELD_GAP_MULT) and (YM_AUS >= 1)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = yg_df['YM_AUS']
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'AUS - mean from same crop/irr status'


# Use average value for all crops and same irrigation status at SA2 level
idx = yg_df.query('(YIELD_GAP_MULT != YIELD_GAP_MULT) and (YM_SA2_ALLCROPS >= 1)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = yg_df['YM_SA2_ALLCROPS']
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'SA2 - mean from same irr status'

# Use average value for all crops and same irrigation status at SA4 level
idx = yg_df.query('(YIELD_GAP_MULT != YIELD_GAP_MULT) and (YM_SA4_ALLCROPS >= 1)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = yg_df['YM_SA4_ALLCROPS']
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'SA4 - mean from same irr status'

# Use average value for all crops and same irrigation status at state level
idx = yg_df.query('(YIELD_GAP_MULT != YIELD_GAP_MULT) and (YM_STE_ALLCROPS >= 1)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = yg_df['YM_STE_ALLCROPS']
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'STE - mean from same irr status'

# Use average value for all crops and same irrigation status at national level
idx = yg_df.query('(YIELD_GAP_MULT != YIELD_GAP_MULT) and (YM_AUS_ALLCROPS >= 1)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = yg_df['YM_AUS_ALLCROPS']
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'AUS - mean from same irr status'

# Put a yield gap multiplier ceiling of 3
idx = yg_df.query('(YIELD_GAP_MULT > 3)').index
yg_df.loc[idx, 'YIELD_GAP_MULT'] = 3
yg_df.loc[idx, 'YIELD_GAP_SOURCE'] = 'Yield gap ceiling'


# Check that we have data everywhere we need it
ludf = pd.read_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5')
ldf = ludf[['CELL_ID', 'SA2_ID', 'IRRIGATION', 'LU_ID', 'LU_DESC']]
ydf = yg_df[['SA2_ID', 'LU_ID', 'IRRIGATION', 'YIELD_GAP_MULT', 'YIELD_GAP_SOURCE']]
tmp = ldf.merge(ydf, how = 'left', on = ['SA2_ID', 'LU_ID', 'IRRIGATION']).query('LU_ID >= 5')
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])

# Export to HDF5
downcast(ydf)
ydf.to_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_yield_gap_mult.h5', key = 'SA2_yield_gap_mult', mode = 'w', format = 't')

