import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy import ndimage as nd
import rasterio, matplotlib
from rasterio import features
from rasterio.fill import fillnodata
from rasterio.warp import reproject
from rasterio.enums import Resampling
from shapely.geometry import box



############################################################################################################################################
# Initialisation. Create some helper data and functions
############################################################################################################################################

# Set some options
pd.set_option('display.width', 500)
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




############################################################################################################################################
# Load some data files
############################################################################################################################################

# Read cell_df file from disk to a new data frame for ag data with just the relevant columns
cell_df = pd.read_pickle('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.pkl')
cell_df = cell_df[['CELL_ID', 'X', 'Y', 'CELL_HA', 'SA2_ID', 'PRIMARY_V7', 'SECONDARY_V7', 'SPREAD_ID', 'SPREAD_DESC', 'IRRIGATION']] # 'COMMODITIES', 'COMMODITIES_DESC', 'IRRIGATION']]

# Read in the PROFIT MAP table as provided by CSIRO to dataframe
ag_df = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210621/T_pfe_per_product_21062021.csv', low_memory = False).drop(columns = 'rev_notes')

# Load livestock mapping data from CSIRO, drop some columns, downcast and save a lite version
# lmap = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210720/lmap.csv', low_memory = False)
# lmap = lmap.drop(columns = ['Unnamed: 0', 'ha_dairy', 'ha_pixel', 'no_sheep', 'Beef Cattle', 'Dairy Cattle', 'Sheep', 'heads_mapped_cum', 'SPREAD_colour'])
# downcast(lmap)
# lmap.to_pickle('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210720/lmap.pkl')
# lmap.to_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210720/lmap_lite.csv')
lmap = pd.read_pickle('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210720/lmap.pkl')

# Load column names and descriptions: N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210720/lmap_variable_names_description.docx
# lmap_cols = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/lmap_column_names.csv')




############################################################################################################################################
# Add livestock commodity mapping to ludf dataframe
############################################################################################################################################

# Create new integer X/Y columns to join, then join the livestock map table to the cell_df dataframe and drop uneccesary columns
cell_df['X-round'] = np.round(cell_df['X'] * 100).astype(np.int16)
cell_df['Y-round'] = np.round(cell_df['Y'] * 100).astype(np.int16)
lmap['X-round'] = np.round(lmap['X'] * 100).astype(np.int16)
lmap['Y-round'] = np.round(lmap['Y'] * 100).astype(np.int16)
ludf = cell_df.merge(lmap[['X-round', 'Y-round', 'SPREAD_id_mapped']], how = 'left', on = ['X-round', 'Y-round'])

# Create LU_ID field to hold IDs and LU_DESC field to hold names of crops and livestock
# Calculate LU_ID == SPREAD_ID and LU_DESC = SPREAD_DESC in all non-pasture cells
index = ludf.query('SPREAD_ID < 1 or SPREAD_ID > 3').index
ludf.loc[index, 'LU_ID'] = ludf['SPREAD_ID']
ludf.loc[index, 'LU_DESC'] = ludf['SPREAD_DESC']

# Classify livestock grazing and pasture type
index = ludf.query('1 <= SPREAD_ID <= 2 and SPREAD_id_mapped == 31').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [31, 'Dairy - native vegetation']

index = ludf.query('1 <= SPREAD_ID <= 2 and SPREAD_id_mapped == 32').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [32, 'Beef - native vegetation']

index = ludf.query('1 <= SPREAD_ID <= 2 and SPREAD_id_mapped == 33').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [33, 'Sheep - native vegetation']

index = ludf.query('SPREAD_ID == 3 and SPREAD_id_mapped == 31').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [34, 'Dairy - sown pasture']

index = ludf.query('SPREAD_ID == 3 and SPREAD_id_mapped == 32').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [35, 'Beef - sown pasture']

index = ludf.query('SPREAD_ID == 3 and SPREAD_id_mapped == 33').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [36, 'Sheep - sown pasture']

# Unallocated agricultural land
ludf.loc[ludf['LU_ID'].isnull(), 'LU_ID'] = 0       
ludf.loc[ludf['LU_ID'] == 0, 'LU_DESC'] = 'Unallocated agricultural land'

# Set plantation forestry (including hardwood, softwood, agroforestry) as agroforestry SPREAD commodity number (4)
index = ludf.query('SECONDARY_V7 in ["3.1 Plantation forestry", "4.1 Irrigated plantation forestry"]').index
ludf.loc[index, 'LU_ID'] = 4
ludf.loc[index, 'LU_DESC'] = 'Plantation forestry'

# Downcast
ludf['LU_ID'] = ludf['LU_ID'].astype(np.int8)
downcast(ludf)

# Export to HDF5 file
tmp_df = ludf[['CELL_ID', 'X', 'Y', 'CELL_HA', 'SA2_ID', 'PRIMARY_V7', 'SECONDARY_V7', 'SPREAD_ID', 'SPREAD_DESC', 'IRRIGATION', 'LU_ID', 'LU_DESC']]
tmp_df.to_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', key = 'cell_LU_mapping', mode = 'w', format = 't')

# Save the output to GeoTiff
with rasterio.open('N:/Planet-A/Data-Master/Profit_map/LU_ID.tif', 'w+', dtype = 'int16', nodata = -99, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(tmp_df['LU_ID']))
        
# Summarise LU_ID classes
# tmp_df.groupby('LU_ID')['LU_DESC'].first()
# tmp_df.groupby('SPREAD_ID')['SPREAD_DESC'].first()
# tmp_df.query('1 <= SPREAD_ID <= 3').groupby(['SPREAD_ID', 'IRRIGATION', 'LU_DESC'], observed = True)['LU_DESC'].count()




############################################################################################################################################
# Load Aussiegrass average pasture productivity raster to ludf dataframe
############################################################################################################################################

# Load raw Aussiegrass 0.05 degree resolution data and save the output to GeoTiff with corrected nodata value
with rasterio.open('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210720/avg_growth11/avg_growth11.tif') as src:
    meta5k = src.meta.copy()
    meta5k.update(compress = 'lzw', driver = 'GTiff', nodata = 0)
    with rasterio.open('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210720/avg_growth11/avg_growth11_nodata_fixed.tif', 'w+', **meta5k) as dst:        
        dst.write_band(1, src.read(1))
        
        # Create a destination array
        dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32)
        
        # Reproject/resample input raster to match NLUM mask (meta)
        reproject(rasterio.band(dst, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.bilinear)
        
        # Fill nodata in raster using value of nearest cell to match NLUM mask
        fill_mask = np.where(dst_array > 0, 1, 0)
        dst_array_filled = np.where(NLUM_mask == 1, fillnodata(dst_array, fill_mask, max_search_distance = 200.0), -9999)
        
        # Save the output to GeoTiff
        with rasterio.open('N:/Planet-A/Data-Master/Profit_map/PASTURE_KG_DM_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
            dst.write_band(1, dst_array_filled)
        
        # Flatten 2D array to 1D array of valid values only and add data to cell_df dataframe
        ludf['PASTURE_KG_DM_HA'] = dst_array_filled[NLUM_mask == 1]


# # Load 0.01 degree resolution resampled data that Javi used and save the output to GeoTiff for comparison
# with rasterio.open('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210720/AussieGrass/avggrowth1k.tif') as src:
        
#         # Create a destination array
#         dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32)
        
#         # Reproject/resample input raster to match NLUM mask (meta)
#         reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.bilinear)
        
#         # Fill nodata in raster using value of nearest cell to match NLUM mask
#         fill_mask = np.where(dst_array > 0, 1, 0)
#         dst_array_filled = np.where(NLUM_mask == 1, fillnodata(dst_array, fill_mask, max_search_distance = 200.0), -9999)
        
#         # Save the output to GeoTiff
#         with rasterio.open('N:/Planet-A/Data-Master/Profit_map/PASTURE_KG_DM_HA_JAVI.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
#             dst.write_band(1, dst_array_filled)
        
#         # Flatten 2D array to 1D array of valid values only and add data to cell_df dataframe
#         ludf['PASTURE_KG_DM_HA_JAVI'] = dst_array_filled[NLUM_mask == 1]


# # Put pasture data side-by-side for visual comparison
# ludf.loc[ludf['kgDMhayr'].notnull(), ['kgDMhayr', 'PASTURE_KG_DM_HA', 'PASTURE_KG_DM_HA_JAVI']].sample(100)

# # Calculate RMS error and mean absolute error
# ludf['kgDMhayr'].mean()
# ludf.eval('(PASTURE_KG_DM_HA - kgDMhayr) ** 2').mean() ** .5
# ludf.eval('PASTURE_KG_DM_HA - kgDMhayr').mean()
# ludf[['PASTURE_KG_DM_HA', 'kgDMhayr']].sample(n = 100000).corr() # 0.98 correlation




############################################################################################################################################
# Assemble livestock yield and cost data
############################################################################################################################################

# Reverse engineer the cost data to get consistent SA2/SA4 values
lmap['AC_beef_SA2'] = lmap.eval('AC_beef / yield_potential_beef')
lmap['FOC_beef_SA2'] = lmap.eval('FOC_beef / yield_potential_beef')
lmap['FLC_beef_SA2'] = lmap.eval('FLC_beef / yield_potential_beef')
lmap['FDC_beef_SA2'] = lmap.eval('FDC_beef / yield_potential_beef')

lmap['AC_sheep_SA2'] = lmap.eval('AC_sheep / yield_potential_sheep')
lmap['FOC_sheep_SA2'] = lmap.eval('FOC_sheep / yield_potential_sheep')
lmap['FLC_sheep_SA2'] = lmap.eval('FLC_sheep / yield_potential_sheep')
lmap['FDC_sheep_SA2'] = lmap.eval('FDC_sheep / yield_potential_sheep')

lmap['AC_dairy_SA2'] = lmap.eval('AC_dairy / yield_potential_dairy')
lmap['FOC_dairy_SA2'] = lmap.eval('FOC_dairy / yield_potential_dairy')
lmap['FLC_dairy_SA2'] = lmap.eval('FLC_dairy / yield_potential_dairy')
lmap['FDC_dairy_SA2'] = lmap.eval('FDC_dairy / yield_potential_dairy')

# Aggregate economic data by SA2 for beef, sheep, and dairy
col = 'SA2_ID'

feed_req = lmap.groupby([col], observed = True).agg(FEED_REQ = ('feed_req_factor', 'mean')).sort_values(by = col)

beef = lmap.groupby([col], observed = True).agg(
                    F1_BEEF = ('F1_beef', 'mean'), 
                    F3_BEEF = ('F3_beef', 'mean'), 
                    Q1_BEEF = ('Q1_beef', 'mean'), 
                    Q3_BEEF = ('Q3_beef', 'mean'), 
                    P1_BEEF = ('P1_beef', 'mean'), 
                    P3_BEEF = ('P3_beef', 'mean'),
                    AC_BEEF = ('AC_beef_SA2', 'mean'), 
                    QC_BEEF = ('QC_beef', 'mean'), 
                    FOC_BEEF = ('FOC_beef_SA2', 'mean'), 
                    FLC_BEEF = ('FLC_beef_SA2', 'mean'), 
                    FDC_BEEF = ('FDC_beef_SA2', 'mean'),
                    WR_DRN_BEEF = ('wr_drink_beef', 'mean'), 
                    WR_IRR_BEEF = ('wr_irrig_beef', 'mean'),
                    WP = ('wp', 'mean')
                    ).sort_values(by = col)

sheep = lmap.groupby([col], observed = True).agg(
                    F1_SHEEP = ('F1_sheep', 'mean'), 
                    F2_SHEEP = ('F2_sheep', 'mean'), 
                    F3_SHEEP = ('F3_sheep', 'mean'), 
                    Q1_SHEEP = ('Q1_sheep', 'mean'), 
                    Q2_SHEEP = ('Q2_sheep', 'mean'), 
                    Q3_SHEEP = ('Q3_sheep', 'mean'), 
                    P1_SHEEP = ('P1_sheep', 'mean'), 
                    P2_SHEEP = ('P2_sheep', 'mean'), 
                    P3_SHEEP = ('P3_sheep', 'mean'),
                    AC_SHEEP = ('AC_sheep_SA2', 'mean'), 
                    QC_SHEEP = ('QC_sheep', 'mean'), 
                    FOC_SHEEP = ('FOC_sheep_SA2', 'mean'), 
                    FLC_SHEEP = ('FLC_sheep_SA2', 'mean'), 
                    FDC_SHEEP = ('FDC_sheep_SA2', 'mean'),
                    WR_DRN_SHEEP = ('wr_drink_sheep', 'mean'), 
                    WR_IRR_SHEEP = ('wr_irrig_sheep', 'mean')                    
                    ).sort_values(by = col)

dairy = lmap.groupby([col], observed = True).agg(
                    F1_DAIRY = ('F1_dairy', 'mean'), 
                    Q1_DAIRY = ('Q1_dairy', 'mean'), 
                    P1_DAIRY = ('P1_dairy', 'mean'), 
                    AC_DAIRY = ('AC_dairy_SA2', 'mean'), 
                    QC_DAIRY = ('QC_dairy', 'mean'), 
                    FOC_DAIRY = ('FOC_dairy_SA2', 'mean'), 
                    FLC_DAIRY = ('FLC_dairy_SA2', 'mean'), 
                    FDC_DAIRY = ('FDC_dairy_SA2', 'mean'),
                    WR_DRN_DAIRY = ('wr_drink_dairy', 'mean'), 
                    WR_IRR_DAIRY = ('wr_irrig_dairy', 'mean')
                    ).sort_values(by = col)

# Make some space in the table by dropping colums
ludf.drop(columns = ['PRIMARY_V7', 'SECONDARY_V7', 'SPREAD_DESC', 'SPREAD_id_mapped'], inplace = True)

# Insert safe pasture utilisation rate and calculate for all grid cells
ludf.loc[ludf.query('SPREAD_ID <= 2').index, 'SAFE_PUR'] = 0.3
ludf.loc[ludf.query('SPREAD_ID == 3').index, 'SAFE_PUR'] = 0.4
# ludf.loc[ludf.query('STATE_ID == 7').index, 'SAFE_PUR'] = 0.15      # Not actually required
ludf.loc[ludf.query('Y >= -26').index, 'SAFE_PUR'] = 0.2

# Merge the feed requirement to ludf dataframe by SA2
ludf = ludf.merge(feed_req, how = 'left', on = 'SA2_ID') 



######################################
# Join the SA2-based economic data back the cell-based ludf dataframe to provide values for all cells whose SA2's support livestock
######################################

################ BEEF ################

# Calculate the yield potential for all cells based on the new pasture data 
ludf['YIELD_POT_BEEF'] = ludf.eval('SAFE_PUR * FEED_REQ * PASTURE_KG_DM_HA / (8 * 365 * 0.85)')

# # Check back against Javi's data, calculate RMS error,  mean absolute error and correlation
# ludf_ = ludf.merge(lmap[['X-round', 'Y-round', 'yield_potential_beef', 'AC_beef']], how = 'left', on = ['X-round', 'Y-round'])
# ludf_['yield_potential_beef'].mean()
# ludf_.eval('(YIELD_POT_BEEF - yield_potential_beef) ** 2').mean() ** .5
# ludf_.eval('YIELD_POT_BEEF - yield_potential_beef').mean()
# ludf_[['yield_potential_beef', 'YIELD_POT_BEEF']].sample(n = 100000).corr() # 0.98 correlation

# Merge the economic data to ludf dataframe by SA2
ludf = ludf.merge(beef, how = 'left', on = 'SA2_ID')

# Convert the costs (back to $/ha, ML/head, ML/ha)
ludf['AC_BEEF'] = ludf.eval('AC_BEEF * YIELD_POT_BEEF') # $/ha
ludf['FOC_BEEF'] = ludf.eval('FOC_BEEF * YIELD_POT_BEEF') # $/ha
ludf['FLC_BEEF'] = ludf.eval('FLC_BEEF * YIELD_POT_BEEF') # $/ha
ludf['FDC_BEEF'] = ludf.eval('FDC_BEEF * YIELD_POT_BEEF') # $/ha
ludf['WR_DRN_BEEF'] = ludf.eval('WR_DRN_BEEF / 1000000') # ML/head
ludf['WR_IRR_BEEF'] = 2.7 # ludf.eval('WR_IRR_BEEF * YIELD_POT_BEEF * 2 / 1000000') # ML/ha

# Calculate production, revenue, cost of production, and profit at full equity per hectare
ludf['PROD_BEEF_DRY'] = ludf.eval('YIELD_POT_BEEF * (F1_BEEF * Q1_BEEF + F3_BEEF * Q3_BEEF)') # tonnes/ha
ludf['REV_BEEF_DRY'] = ludf.eval('YIELD_POT_BEEF * (F1_BEEF * Q1_BEEF * P1_BEEF + F3_BEEF * Q3_BEEF * P3_BEEF)') # $/ha
ludf['COST_BEEF_DRY'] = ludf.eval('YIELD_POT_BEEF * (QC_BEEF + WR_DRN_BEEF * WP) + (AC_BEEF + FOC_BEEF + FLC_BEEF + FDC_BEEF)') # $ /ha
ludf['PFE_BEEF_DRY'] = ludf.eval('REV_BEEF_DRY - COST_BEEF_DRY') # $/ha
ludf['WATER_USE_BEEF_DRY'] = ludf.eval('WR_DRN_BEEF * YIELD_POT_BEEF') # ML/ha

ludf['PROD_BEEF_IRR'] = ludf.eval('YIELD_POT_BEEF * 2 * (F1_BEEF * Q1_BEEF + F3_BEEF * Q3_BEEF)') # tonnes/ha
ludf['REV_BEEF_IRR'] = ludf.eval('YIELD_POT_BEEF * 2 * (F1_BEEF * Q1_BEEF * P1_BEEF + F3_BEEF * Q3_BEEF * P3_BEEF)') # $/ha
ludf['COST_BEEF_IRR'] = ludf.eval('YIELD_POT_BEEF * 2 * (QC_BEEF + WR_DRN_BEEF * WP) + (AC_BEEF + FOC_BEEF + FLC_BEEF + FDC_BEEF + WR_IRR_BEEF * WP)') # $/ha
ludf['PFE_BEEF_IRR'] = ludf.eval('REV_BEEF_IRR - COST_BEEF_IRR') # $/ha
ludf['WATER_USE_BEEF_IRR'] = ludf.eval('WR_DRN_BEEF * YIELD_POT_BEEF + WR_IRR_BEEF') # ML/ha


################ SHEEP ################

# Calculate the yield potential for all cells based on the new pasture data 
ludf['YIELD_POT_SHEEP'] = ludf.eval('SAFE_PUR * FEED_REQ * PASTURE_KG_DM_HA / (1.5 * 365 * 0.85)')

# Merge the economic data to ludf dataframe by SA2
ludf = ludf.merge(sheep, how = 'left', on = 'SA2_ID') 

# Convert the costs (back to $/ha, ML/head, ML/ha)
ludf['AC_SHEEP'] = ludf.eval('AC_SHEEP * YIELD_POT_SHEEP') # $/ha
ludf['FOC_SHEEP'] = ludf.eval('FOC_SHEEP * YIELD_POT_SHEEP') # $/ha
ludf['FLC_SHEEP'] = ludf.eval('FLC_SHEEP * YIELD_POT_SHEEP') # $/ha
ludf['FDC_SHEEP'] = ludf.eval('FDC_SHEEP * YIELD_POT_SHEEP') # $/ha
ludf['WR_DRN_SHEEP'] = ludf.eval('WR_DRN_SHEEP / 1000000') # ML/head
ludf['WR_IRR_SHEEP'] = 3.1 # ludf.eval('WR_IRR_SHEEP * YIELD_POT_SHEEP * 2 / 1000000') # ML/ha

# Calculate production, revenue, cost of production, and profit at full equity per hectare
ludf['PROD_SHEEP_DRY'] = ludf.eval('YIELD_POT_SHEEP * (F1_SHEEP * Q1_SHEEP + F2_SHEEP * Q2_SHEEP + F3_SHEEP * Q3_SHEEP)') # tonnes/ha
ludf['REV_SHEEP_DRY'] = ludf.eval('YIELD_POT_SHEEP * (F1_SHEEP * Q1_SHEEP * P1_SHEEP + F2_SHEEP * Q2_SHEEP * P2_SHEEP + F3_SHEEP * Q3_SHEEP * P3_SHEEP)') # $/ha
ludf['COST_SHEEP_DRY'] = ludf.eval('YIELD_POT_SHEEP * (QC_SHEEP + WR_DRN_SHEEP * WP) + (AC_SHEEP + FOC_SHEEP + FLC_SHEEP + FDC_SHEEP)') # $ /ha
ludf['PFE_SHEEP_DRY'] = ludf.eval('REV_SHEEP_DRY - COST_SHEEP_DRY') # $/ha
ludf['WATER_USE_SHEEP_DRY'] = ludf.eval('WR_DRN_SHEEP * YIELD_POT_SHEEP') # ML/ha

ludf['PROD_SHEEP_IRR'] = ludf.eval('YIELD_POT_SHEEP * 2 * (F1_SHEEP * Q1_SHEEP + F2_SHEEP * Q2_SHEEP + F3_SHEEP * Q3_SHEEP)') # tonnes/ha
ludf['REV_SHEEP_IRR'] = ludf.eval('YIELD_POT_SHEEP * 2 * (F1_SHEEP * Q1_SHEEP * P1_SHEEP + F2_SHEEP * Q2_SHEEP * P2_SHEEP + F3_SHEEP * Q3_SHEEP * P3_SHEEP)') # $/ha
ludf['COST_SHEEP_IRR'] = ludf.eval('YIELD_POT_SHEEP * 2 * (QC_SHEEP + WR_DRN_SHEEP * WP) + (AC_SHEEP + FOC_SHEEP + FLC_SHEEP + FDC_SHEEP + WR_IRR_SHEEP * WP)') # $/ha
ludf['PFE_SHEEP_IRR'] = ludf.eval('REV_SHEEP_IRR - COST_SHEEP_IRR') # $/ha
ludf['WATER_USE_SHEEP_IRR'] = ludf.eval('WR_DRN_SHEEP * YIELD_POT_SHEEP + WR_IRR_SHEEP') # ML/ha


################ DAIRY ################

# Calculate the yield potential for all cells based on the new pasture data 
ludf['YIELD_POT_DAIRY'] = ludf.eval('SAFE_PUR * FEED_REQ * PASTURE_KG_DM_HA / (17 * 365 * 0.65)')

# Merge the economic data to ludf dataframe by SA2
ludf = ludf.merge(dairy, how = 'left', on = 'SA2_ID')

# Convert the costs (back to $/ha, ML/head, ML/ha)
ludf['AC_DAIRY'] = ludf.eval('AC_DAIRY * YIELD_POT_DAIRY') # $/ha
ludf['FOC_DAIRY'] = ludf.eval('FOC_DAIRY * YIELD_POT_DAIRY') # $/ha
ludf['FLC_DAIRY'] = ludf.eval('FLC_DAIRY * YIELD_POT_DAIRY') # $/ha
ludf['FDC_DAIRY'] = ludf.eval('FDC_DAIRY * YIELD_POT_DAIRY') # $/ha
ludf['WR_DRN_DAIRY'] = ludf.eval('WR_DRN_DAIRY / 1000000') # ML/head
ludf['WR_IRR_DAIRY'] = 3.6 # ludf.eval('WR_IRR_DAIRY * YIELD_POT_DAIRY * 2 / 1000000') # ML/ha

# Calculate production, revenue, cost of production, and profit at full equity per hectare
ludf['PROD_DAIRY_DRY'] = ludf.eval('YIELD_POT_DAIRY * (F1_DAIRY * Q1_DAIRY)') # tonnes/ha
ludf['REV_DAIRY_DRY'] = ludf.eval('YIELD_POT_DAIRY * (F1_DAIRY * Q1_DAIRY * P1_DAIRY)') # $/ha
ludf['COST_DAIRY_DRY'] = ludf.eval('YIELD_POT_DAIRY * (QC_DAIRY + WR_DRN_DAIRY * WP) + (AC_DAIRY + FOC_DAIRY + FLC_DAIRY + FDC_DAIRY)') # $ /ha
ludf['PFE_DAIRY_DRY'] = ludf.eval('REV_DAIRY_DRY - COST_DAIRY_DRY') # $/ha
ludf['WATER_USE_DAIRY_DRY'] = ludf.eval('WR_DRN_DAIRY * YIELD_POT_DAIRY') # ML/ha

ludf['PROD_DAIRY_IRR'] = ludf.eval('YIELD_POT_DAIRY * 2 * (F1_DAIRY * Q1_DAIRY)') # tonnes/ha
ludf['REV_DAIRY_IRR'] = ludf.eval('YIELD_POT_DAIRY * 2 * (F1_DAIRY * Q1_DAIRY * P1_DAIRY)') # $/ha
ludf['COST_DAIRY_IRR'] = ludf.eval('YIELD_POT_DAIRY * 2 * (QC_DAIRY + WR_DRN_DAIRY * WP) + (AC_DAIRY + FOC_DAIRY + FLC_DAIRY + FDC_DAIRY + WR_IRR_DAIRY * WP)') # $/ha
ludf['PFE_DAIRY_IRR'] = ludf.eval('REV_DAIRY_IRR - COST_DAIRY_IRR') # $/ha
ludf['WATER_USE_DAIRY_IRR'] = ludf.eval('WR_DRN_DAIRY * YIELD_POT_DAIRY + WR_IRR_DAIRY') # ML/ha


# Check against Javi's data
# ludf = ludf.merge(lmap[['X-round', 'Y-round', 'cost_pct_beef']], how = 'left', on = ['X-round', 'Y-round'])
# ludf['COST_PCENT_BEEF'] = ludf.eval('COST_DRY / REV_BEEF_DRY')
# ludf.iloc[:,-22:].sample(20)

"""
heads_mapped = safe_pur * feed_req_factor * kgDMhayr * ha_pixel * irrig_factor / (dse_per_head * 365 * grassfed_factor)
 
Parameters:
safe_pur = 0.4 for sown pasture, 0.3 for native pastures, 0.2 for pastures where y >= -26 and 0.15 in the Northern Territory
feed_req_factor = SUM(prod_ABS * DSE_per_head) / SUM(kgDMhayr * irrig_factor * ha_pixel * safe_pur / 365). This is the sum of all pixels in an SA4 and heads of all livestock types (beef, sheep and dairy). If there is less pasture available than DSEs to map, feed_req_factor will be > 1.
Otherwise we force feed_req_factor to equal 1 (so that safe_pur * feed_req_factor = safe_pur * 1 and therefore pur = safe_pur).
kgDMhayr = The AussieGrass-derived estimate of average pasture growth (kgDM) per ha and year, for all years between 2005 and 2015
ha_pixel = the real ha provided by Brett
irrig_factor = the irrigation (or productivity) factor which is dependent on pasture type and irrigation status
dse_per_head = 17 for dairy, 8 for beef, 1.5 for sheep
grassfed_factor = the proportion of feed that comes from grass (we use this to modify the number of days spent grazing). This is 0.65 for dairy and 0.85 for the rest.
Water requirements from Marinoni et al. (2012): Beef Cattle:2.7 ML/ha, Dairy cattle: 3.6 ML/ha, Sheep: 3.1 ML/ha

"""

# Downcast to make some space
downcast(ludf)

# Export to HDF5 file
tmp_df = ludf.drop(columns = ['SPREAD_ID', 'X-round', 'Y-round'])
tmp_df.to_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_livestock_data.h5', key = 'cell_livestock_data', mode = 'w', format = 't')

# x = pd.read_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_livestock_data.h5')


################### Water

ag_df['cost'] = ag_df.eval('rev * cost_pct')
ag_df['water_cost'] = ag_df.eval('WP * WR')
ag_df['cost_per_tonne'] = ag_df.eval('cost + water_cost') / ag_df['yield']
ag_df['rev_per_tonne'] = ag_df.eval('rev') / ag_df['yield']

ag_df.groupby(['SPREAD_Commodity', 'irrigation'], observed = True)['rev', 'pfe', 'rev_per_tonne', 'cost_per_tonne'].mean()

ag_df.query('irrigation == 1').groupby(['SPREAD_Commodity', 'irrigation'], observed = True)['cost', 'water_cost'].mean()
ag_df.query('irrigation == 1').groupby(['SPREAD_Commodity', 'STATE_ID'], observed = True)['WR'].mean()


ludf.query('LU_ID > 30').groupby(['LU_DESC', 'IRRIGATION'], observed = True)['WR_IRR_BEEF', 'WR_IRR_SHEEP', 'WR_IRR_DAIRY', ].mean()
ludf.groupby(['LU_DESC', 'IRRIGATION'], observed = True)['WR_IRR_BEEF', 'WR_IRR_SHEEP', 'WR_IRR_DAIRY', ].mean()

ludf['cost_per_tonne_beef_dry'] = ludf.eval('COST_BEEF_DRY / PROD_BEEF_DRY')
ludf['cost_per_tonne_beef_irr'] = ludf.eval('COST_BEEF_IRR / PROD_BEEF_IRR')
ludf.query('LU_ID > 30').groupby(['LU_DESC', 'SA2_ID', 'IRRIGATION'], observed = True)['COST_BEEF_DRY', 'COST_BEEF_IRR', 'PFE_BEEF_DRY', 'PFE_BEEF_IRR', 'cost_per_tonne_beef_dry', 'cost_per_tonne_beef_irr'].mean()






############################################################################################################################################
# Helper code
############################################################################################################################################
"""
cell_df.info()
ludf.info()

# ludf[ludf['LU_ID'] > 0].sample(200)

# cell_df.groupby(['PRIMARY_V7', 'SECONDARY_V7', 'IRRIGATION'], observed = True)[['X']].count().sort_values(by = ['PRIMARY_V7', 'SECONDARY_V7'], ascending = True)
# cell_df[cell_df['SPREAD_DESC'] == 'Agroforestry'].groupby(['SPREAD_DESC', 'SECONDARY_V7', 'IRRIGATION'], observed = True)[['X']].count().sort_values(by = ['SECONDARY_V7'], ascending = True)


# ludf.groupby(['PRIMARY_V7', 'SPREAD_DESC'], observed = True)[['SPREAD_DESC']].count().sort_values(by = 'PRIMARY_V7', ascending = True)
# ludf.groupby(['PRIMARY_V7', 'SPREAD_mapped'], observed = True)[['SPREAD_mapped']].count().sort_values(by = 'PRIMARY_V7', ascending = True)
# ludf.groupby(['PRIMARY_V7', 'SPREAD_mapped'], observed = True)[['SPREAD_mapped']].count().sort_values(by = 'PRIMARY_V7', ascending = True)

# ludf.groupby(['LU_ID', 'SPREAD_DESC'], observed = True)[['LU_ID']].count()
# ludf.groupby(['LU_ID', 'LU_DESC'], observed = True)[['LU_ID']].count()
# ludf.groupby(['PRIMARY_V7', 'LU_DESC'], observed = True)[['LU_DESC']].count().sort_values(by = 'PRIMARY_V7', ascending = True)

# ludf.groupby(['SPREAD_ID', 'SPREAD_DESC'], observed = True)[['SPREAD_DESC']].count().head(100)

# ludf.query('(SPREAD_ID >= 1) and (SPREAD_ID <= 3)').head(100)
# ludf.query('SPREAD_ID >= 1').head(100)
# ludf.info()
# ludf['LU_ID'].isnull().sum() == lmap['SPREAD_id_mapped'].isnull().sum()
# cell_df.groupby(['SPREAD_ID', 'SPREAD_DESC'], observed = True)[['SPREAD_DESC']].count().head(100)

# # Use dingo fence data to set SHEEP_MASK column - Javi: "But they shouldn't map heads of Sheep where the field "Sheep" is 0."
# ludf['SHEEP_MASK'] = 1
# ns_filter = ludf['Sheep'] == 0
# ludf.loc[ns_filter, 'SHEEP_MASK'] = 0
# ludf = ludf.drop(columns = ['Sheep'])
                     
# # Cross-check the mapped national herd size with ABS numbers
# ludf.groupby(['SPREAD_mapped'])[['heads_mapped']].sum()
# ag_df.query("SPREAD_ID >= 31 and SPREAD_ID <= 33").groupby(['SPREAD_Commodity', 'SPREAD_ID_original', 'irrigation'])[['prod_ABS']].sum()





# lmap.iloc[:, np.r_[3,12,20:45]].sort_values(by = 'SA2_ID').head(100)
# lmap.loc[(lmap['SA2_ID'] == 101011002), lmap.columns[np.r_[3:20]]].sort_values(by = 'SA2_ID')

# # Pull out beef columns
# lmap.loc[:, lmap.columns[np.r_[:16, 20, 24:33, 51, 55:61, 77, 81, 88]]].sort_values(by = 'SA2_ID')
lmap_full = lmap.merge(ludf[['X-round', 'Y-round', 'PASTURE_KG_DM_HA', 'PASTURE_KG_DM_HA_JAVI']], how = 'left', on = ['X-round', 'Y-round'])

[print(i) for i in lmap.columns]

lmap = lmap_full[['X', 'Y', 'STATE_ID', 'SA2_ID', 'SPREAD_mapped', 'kgDMhayr', 'PASTURE_KG_DM_HA', 'PASTURE_KG_DM_HA_JAVI', 'safe_pur', 'feed_req_factor', 'pur', 'head_adjustment_factor', 'yield_potential_beef', 
            'F1_beef', 'F2_beef', 'F3_beef', 'Q1_beef', 'Q2_beef', 'Q3_beef', 'P1_beef', 'P2_beef', 'P3_beef', 'rev_beef', 'AC_beef', 'QC_beef', 'FDC_beef', 'FLC_beef', 'FOC_beef', 'cost_pct_beef', 
            'wr_drink_beef', 'wr_irrig_beef', 'wp']]

# lmap.loc[(lmap['SPREAD_mapped'] == 'Beef Cattle'), lmap.columns[np.r_[3,12,16,20,23,24:33,51,55:61,79,80,87,88,92,97,99]]].sort_values(by = 'SA2_ID')

# lmap['yield_pot_beef'] = lmap.eval('pur * kgDMhayr / (8 * 365 * 0.85)')
# lmap[['yield_pot_beef', 'yield_potential_beef']]
# lmap['rev_beef_BB'] = lmap.eval('yield_potential_beef * (F1_beef * Q1_beef * P1_beef + F3_beef * Q3_beef * P3_beef)')
# lmap['costs_beef_BB1'] = lmap.eval('rev_beef * cost_pct_beef')
# lmap['costs_beef_BB2'] = lmap.eval('AC_beef + QC_beef * yield_potential_beef + FOC_beef + FLC_beef + FDC_beef')

lmap['AC_beef_SA2'] = lmap.eval('AC_beef / yield_potential_beef')
# lmap['FOC_beef_SA2'] = lmap.eval('FOC_beef / yield_potential_beef')
# lmap['FLC_beef_SA2'] = lmap.eval('FLC_beef / yield_potential_beef')
# lmap['FDC_beef_SA2'] = lmap.eval('FDC_beef / yield_potential_beef')

# lmap['AC_sheep_SA2'] = lmap.eval('AC_sheep / yield_potential_sheep')
# lmap['FOC_sheep_SA2'] = lmap.eval('FOC_sheep / yield_potential_sheep')
# lmap['FLC_sheep_SA2'] = lmap.eval('FLC_sheep / yield_potential_sheep')
# lmap['FDC_sheep_SA2'] = lmap.eval('FDC_sheep / yield_potential_sheep')

# lmap['AC_dairy_SA2'] = lmap.eval('AC_dairy / yield_potential_dairy')
# lmap['FOC_dairy_SA2'] = lmap.eval('FOC_dairy / yield_potential_dairy')
# lmap['FLC_dairy_SA2'] = lmap.eval('FLC_dairy / yield_potential_dairy')
# lmap['FDC_dairy_SA2'] = lmap.eval('FDC_dairy / yield_potential_dairy')

# lmap.loc[lmap['sa4_id'] == 102, lmap.columns[np.r_[3,12,20:45]]].sort_values(by = 'SA2_ID').head(1000)
# lmap.loc[lmap['sa4_id'] == 102, lmap.columns[np.r_[3,12,55:78]]].sort_values(by = 'SA2_ID')
# lmap.loc[lmap['sa4_id'] == 102, lmap.columns[np.r_[3,12,79:92]]]#.sort_values(by = 'SA2_ID').head(1000)

col = 'SA2_ID'

feed_req = lmap.groupby([col], observed = True).agg(FEED_REQ_FACTOR = ('feed_req_factor', 'mean')).sort_values(by = col)

beef = lmap.groupby([col], observed = True).agg(
                    # F1_BEEF = ('F1_beef', 'mean'), 
                    # F3_BEEF = ('F3_beef', 'mean'), 
                    # Q1_BEEF = ('Q1_beef', 'mean'), 
                    # Q3_BEEF = ('Q3_beef', 'mean'), 
                    # P1_BEEF = ('P1_beef', 'mean'), 
                    # P3_BEEF = ('P3_beef', 'mean'),
                    AC_BEEF = ('AC_beef_SA2', 'mean'), 
                    # QC_BEEF = ('QC_beef', 'mean'), 
                    # FOC_BEEF = ('FOC_beef_SA2', 'mean'), 
                    # FLC_BEEF = ('FLC_beef_SA2', 'mean'), 
                    # FDC_BEEF = ('FDC_beef_SA2', 'mean'),
                    # WR_DRINK_BEEF = ('wr_drink_beef', 'mean'), 
                    # WR_IRRIG_BEEF = ('wr_irrig_beef', 'mean')
                    ).sort_values(by = col)

# sheep = lmap.groupby([col], observed = True).agg(
#                     F1_SHEEP = ('F1_sheep', 'mean'), 
#                     F2_SHEEP = ('F2_sheep', 'mean'), 
#                     F3_SHEEP = ('F3_sheep', 'mean'), 
#                     Q1_SHEEP = ('Q1_sheep', 'mean'), 
#                     Q2_SHEEP = ('Q2_sheep', 'mean'), 
#                     Q3_SHEEP = ('Q3_sheep', 'mean'), 
#                     P1_SHEEP = ('P1_sheep', 'mean'), 
#                     P2_SHEEP = ('P2_sheep', 'mean'), 
#                     P3_SHEEP = ('P3_sheep', 'mean'),
#                     AC_SHEEP = ('AC_sheep_BB', 'mean'), 
#                     QC_SHEEP = ('QC_sheep', 'mean'), 
#                     FOC_SHEEP = ('FOC_sheep_BB', 'mean'), 
#                     FLC_SHEEP = ('FLC_sheep_BB', 'mean'), 
#                     FDC_SHEEP = ('FDC_sheep_BB', 'mean'),
#                     WR_DRINK_SHEEP = ('wr_drink_sheep', 'mean'), 
#                     WR_IRRIG_SHEEP = ('wr_irrig_sheep', 'mean')                    
#                     ).sort_values(by = col)

# dairy = lmap.groupby([col], observed = True).agg(
#                     F1_DAIRY = ('F1_dairy', 'mean'), 
#                     Q1_DAIRY = ('Q1_dairy', 'mean'), 
#                     P1_DAIRY = ('P1_dairy', 'mean'), 
#                     AC_DAIRY = ('AC_dairy_BB', 'mean'), 
#                     QC_DAIRY = ('QC_dairy', 'mean'), 
#                     FOC_DAIRY = ('FOC_dairy_BB', 'mean'), 
#                     FLC_DAIRY = ('FLC_dairy_BB', 'mean'), 
#                     FDC_DAIRY = ('FDC_dairy_BB', 'mean'),
#                     WR_DRINK_DAIRY = ('wr_drink_dairy', 'mean'), 
#                     WR_IRRIG_DAIRY = ('wr_irrig_dairy', 'mean'),
#                     WP = ('wp', 'mean')).sort_values(by = col)

lmap = lmap.merge(feed_req, how = 'left', on = 'SA2_ID') 
lmap = lmap.merge(beef, how = 'left', on = 'SA2_ID') 
lmap['YIELD_POT_BEEF'] = lmap.eval('safe_pur * kgDMhayr / (8 * 365 * 0.85)')
lmap['AC_BEEF_JN'] = lmap.eval('AC_BEEF * yield_potential_beef')
lmap['AC_BEEF_BB'] = lmap.eval('AC_BEEF * YIELD_POT_BEEF')

lmap['head_adjustment_factor'].describe()
lmap.query('head_adjustment_factor > 1.2').count()
lmap.query('Y >= -26')[['safe_pur']]
lmap.query('Y >= -26')[['safe_pur']].std()
lmap.query('STATE_ID == 7')[['safe_pur']].mean()


# This shows that where feed_req_factor == 1 & head_adjustment_factor == 1 then this method produces the same results as Javi
lmap[['SPREAD_mapped', 'kgDMhayr', 'PASTURE_KG_DM_HA', 'PASTURE_KG_DM_HA_JAVI', 'YIELD_POT_BEEF', 'yield_potential_beef', 'safe_pur', 'feed_req_factor', 'head_adjustment_factor', 'AC_beef', 'AC_BEEF_BB', 'AC_BEEF_JN']]
x = lmap.query('(feed_req_factor == 1) & (head_adjustment_factor == 1)')
lmap.loc[x.index, ['SPREAD_mapped', 'kgDMhayr', 'PASTURE_KG_DM_HA', 'PASTURE_KG_DM_HA_JAVI', 'YIELD_POT_BEEF', 'yield_potential_beef', 'safe_pur', 'feed_req_factor', 'head_adjustment_factor', 'AC_beef', 'AC_BEEF_BB', 'AC_BEEF_JN']]

# Calculate the safe_pur for all cells
ludf['SAFE_PUR_BB'] = 0.0
ludf[ludf['SPREAD_ID'] <= 2]['SAFE_PUR_BB'] = 0.3
idx = ludf.query('SPREAD_ID <= 2').index
ludf.loc[idx, 'SAFE_PUR_BB'] == 0.3

ludf.loc[ludf.query('SPREAD_ID <= 2').index, 'SAFE_PUR_BB'] = 0.3
ludf.loc[ludf.query('SPREAD_ID == 3').index, 'SAFE_PUR_BB'] = 0.4
ludf.loc[ludf.query('STATE_ID == 7').index, 'SAFE_PUR_BB'] = 0.15
ludf.loc[ludf.query('Y >= -26').index, 'SAFE_PUR_BB'] = 0.2



# Join the tables to the cell_df dataframe

ludf = ludf.merge(beef, how = 'left', on = 'SA2_ID') 
ludf = ludf.merge(sheep, how = 'left', on = 'SA2_ID') 
ludf = ludf.merge(dairy, how = 'left', on = 'SA2_ID')

ludf['YIELD_POT_BEEF'] = ludf.eval('SAFE_PUR * PASTURE_KG_DM_HA / (8 * 365 * 0.85)')
ludf['YIELD_POT_BEEF_IRR'] = ludf.eval('SAFE_PUR * PASTURE_KG_DM_HA * 2 / (8 * 365 * 0.85)')
ludf['YIELD_POT_SHEEP_DRY'] = ludf.eval('SAFE_PUR * PASTURE_KG_DM_HA / (1.5 * 365 * 0.85)')
ludf['YIELD_POT_SHEEP_IRR'] = ludf.eval('SAFE_PUR * PASTURE_KG_DM_HA * 2 / (1.5 * 365 * 0.85)')
ludf['YIELD_POT_DAIRY_DRY'] = ludf.eval('SAFE_PUR * PASTURE_KG_DM_HA / (17 * 365 * 0.65)')
ludf['YIELD_POT_DAIRY_IRR'] = ludf.eval('SAFE_PUR * PASTURE_KG_DM_HA * 2 / (17 * 365 * 0.65)')



# Convert back to cell-based $/ha costs
ludf['AC_BEEF_BB'] = ludf.eval('AC_BEEF * YIELD_POT_BEEF')
ludf['FOC_BEEF_BB'] = ludf.eval('FOC_BEEF * YIELD_POT_BEEF')
ludf['FLC_BEEF_BB'] = ludf.eval('FLC_BEEF * YIELD_POT_BEEF')
ludf['FDC_BEEF_BB'] = ludf.eval('FDC_BEEF * YIELD_POT_BEEF')

ludf['AC_SHEEP_BB'] = ludf.eval('AC_SHEEP * YIELD_POT_SHEEP_DRY')
ludf['FOC_SHEEP_BB'] = ludf.eval('FOC_SHEEP * YIELD_POT_SHEEP_DRY')
ludf['FLC_SHEEP_BB'] = ludf.eval('FLC_SHEEP * YIELD_POT_SHEEP_DRY')
ludf['FDC_SHEEP_BB'] = ludf.eval('FDC_SHEEP * YIELD_POT_SHEEP_DRY')

ludf['AC_DAIRY_BB'] = ludf.eval('AC_DAIRY * YIELD_POT_DAIRY_DRY')
ludf['FOC_DAIRY_BB'] = ludf.eval('FOC_DAIRY * YIELD_POT_DAIRY_DRY')
ludf['FLC_DAIRY_BB'] = ludf.eval('FLC_DAIRY * YIELD_POT_DAIRY_DRY')
ludf['FDC_DAIRY_BB'] = ludf.eval('FDC_DAIRY * YIELD_POT_DAIRY_DRY')


ludf[ludf['AC_BEEF'].notnull()]

lmap2 = lmap[['X-round', 'Y-round', 'yield_potential_beef', 'yield_potential_sheep', 'yield_potential_dairy']].merge(ludf[['X-round', 'Y-round', 'YIELD_POT_BEEF', 'YIELD_POT_SHEEP_DRY', 'YIELD_POT_DAIRY_DRY']], how = 'left', on = ['X-round', 'Y-round'])
lmap2[['yield_potential_beef', 'YIELD_POT_BEEF', 'yield_potential_sheep', 'YIELD_POT_SHEEP_DRY', 'yield_potential_dairy', 'YIELD_POT_DAIRY_DRY']]
lmap2[['yield_potential_beef', 'YIELD_POT_BEEF']].sample(n = 100000).corr() 
lmap2.eval('(yield_potential_beef - YIELD_POT_BEEF) ** 2').mean() ** .5
lmap2['yield_potential_beef'].describe()
lmap2.eval('(yield_potential_sheep - YIELD_POT_SHEEP_DRY) ** 2').mean() ** .5
lmap2['yield_potential_sheep'].describe()

lmap2.eval('(yield_potential_beef - YIELD_POT_BEEF)').mean()

lmap[['X-round', 'Y-round', 'yield_potential_beef', 'yield_potential_sheep', 'yield_potential_dairy']]
ludf

cell_df[cell_df['F1_DAIRY'] != cell_df['F1_DAIRY']].count()

ag_df[(ag_df['SPREAD_Commodity'] == 'Beef Cattle') & (ag_df['irrigation'] == 0)].groupby(['SPREAD_ID_original']).first().iloc[:, :10]

ag_df[ag_df['SPREAD_Commodity'] == 'Beef Cattle'].groupby(['SA4_ID', 'irrigation'], observed = True)[['SPREAD_ID_original']].first().iloc[:300, :10]

lmap.loc[lmap['SA2_ID'] == 101011001, 'SA2_ID': 'yield_mapped'].head(100)
lmap.info()

lmap['HC_BEEF'] = lmap.eval('safe_pur * kgDMhayr * irrig_factor / (8 * 365 * 0.85)')

lmap['HC_BEEF'] = lmap.eval('pur * kgDMhayr * irrig_factor * area_mapped / (8 * 365 * 0.85)')
lmap[['HC_BEEF', 'head_capacity_beef']].head(200)
lmap.eval('HC_BEEF == head_capacity_beef').sum()
x = (np.round(lmap['HC_BEEF'] * 100, 0) == np.round(lmap['head_capacity_beef'] * 100, 0))
x.sum()
lmap.loc[~x, ['HC_BEEF', 'head_capacity_beef']]
lmap.loc[~x, ['HC_BEEF', 'head_capacity_beef']]
lmap['feed_req_factor'].describe()
lmap['head_adjustment_factor'].describe()


# Check national herd sizes of mapped livestock heads against numbers in profit map table
lmap.groupby(['SPREAD_mapped'], observed = True)[['heads_mapped']].sum().sort_values(by = 'SPREAD_mapped')
x = ag_df[ag_df['SPREAD_ID'].between(31, 33)].groupby(['SPREAD_Commodity', 'SA2_ID'], observed = True)[['prod_ABS']].first().sort_values(by = 'SPREAD_Commodity')
x.groupby(['SPREAD_Commodity'])[['prod_ABS']].sum().sort_values(by = 'SPREAD_Commodity')

# # Calculate stocking rates (equivalent to 'yield_potential_beef' etc. in new lmap table)
# # From Javi... stocking rates (head/ha) = pur * kgDMhayr * ha_pixel * irrig_factor / (dse_per_head * 365 * grassfed_factor)
# # Grassfed_factor is 0.65 for Dairy, and 0.85 for Beef and Sheep. 17 DSE/head for dairy, 8 DSE/head for beef, 1.5 DSE/head for sheep
# lmap['BEEF_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (8 * 365 * 0.85)')
# lmap['SHEEP_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (1.5 * 365 * 0.85)')
# lmap['DAIRY_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (17 * 365 * 0.65)')
"""




############################################################################################################################################
# Join CROPS data from profit map table to the cell_df dataframe
############################################################################################################################################

# Select crops only
crops_df = ag_df.query('5 <= SPREAD_ID <= 25')

# Rename columns to avoid python built-in naming
crops_df.rename(columns = {'yield': 'Yield', 'irrigation': 'Irrigation', 'irrig_factor': 'Prod_factor'}, inplace = True) 

# Convert SPREAD_DESC to Sentence case
crops_df['SPREAD_Commodity'] = crops_df['SPREAD_Commodity'].str.capitalize()
# crops_df.groupby('SPREAD_ID')['SPREAD_Commodity'].first()


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
    crops_sum_df.loc[crops_sum_df['SPREAD_Name'].isin(['Stone fruit', 'Vegetables']), col] = crops_sum_df[col + '_STE']

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

cep_df.drop(columns = 'index')

# Do some test calculations of revenue and costs
# cep_df.eval('Yield_ha_TRUE = Production / Area', inplace = True)
# cep_df.eval('Rev_ha = Yield * Q1 * P1', inplace = True)
# cep_df.eval('Costs_ha = (AC + FDC + FOC + FLC) + (QC * Yield)', inplace = True)
# cep_df.eval('Costs_t = Costs_ha / Yield', inplace = True)

downcast(cep_df)

# Save file
cep_df.to_csv('N:/Planet-A/Data-Master/Profit_map/cep_yield_econ_SPREAD.csv')
cep_df.to_hdf('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cep_yield_econ_SPREAD.h5', key = 'cep_yield_econ_SPREAD', mode = 'w', format = 't')











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


"""
################## Explore livestock data

for col in lmap.columns:
    print(lmap[col].describe())
    print('------------------------------')

lmap.iloc[:, np.r_[3,12,20:45]].sort_values(by = 'SA2_ID').head(100)
lmap.loc[(lmap['SA2_ID'] == 101011002), lmap.columns[np.r_[3,12,24:33]]].sort_values(by = 'SA2_ID')

# Pull out beef columns
lmap.loc[(lmap['SA2_ID'] == 101011002), lmap.columns[np.r_[3,12,16,20,23,24:33,51,55:61,79,80,87,88,92]]].sort_values(by = 'SA2_ID')
lmap.loc[(lmap['SPREAD_mapped'] == 'Beef Cattle'), lmap.columns[np.r_[3,12,16,20,23,24:33,51,55:61,79,80,87,88,92,97,99]]].sort_values(by = 'SA2_ID')

lmap['rev_beef_BB'] = lmap.eval('yield_potential_beef * (F1_beef * Q1_beef * P1_beef + F3_beef * Q3_beef * P3_beef)')
lmap['costs_beef_BB1'] = lmap.eval('rev_beef * cost_pct_beef')
lmap['costs_beef_BB2'] = lmap.eval('AC_beef + QC_beef * yield_potential_beef + FOC_beef + FLC_beef + FDC_beef')
lmap['AC_beef_BB'] = lmap.eval('AC_beef / yield_potential_beef')
lmap['QC_beef_BB'] = lmap.eval('QC_beef')
lmap['FOC_beef_BB'] = lmap.eval('FOC_beef / yield_potential_beef')
lmap['FLC_beef_BB'] = lmap.eval('FLC_beef / yield_potential_beef')
lmap['FDC_beef_BB'] = lmap.eval('FDC_beef / yield_potential_beef')

lmap.loc[lmap['sa4_id'] == 102, lmap.columns[np.r_[3,12,20:45]]].sort_values(by = 'SA2_ID').head(1000)
lmap.loc[lmap['sa4_id'] == 102, lmap.columns[np.r_[3,12,55:78]]].sort_values(by = 'SA2_ID')
lmap.loc[lmap['sa4_id'] == 102, lmap.columns[np.r_[3,12,79:92]]]#.sort_values(by = 'SA2_ID').head(1000)

col = 'SA2_ID'
lmap.groupby([col], observed = True).agg(
                    F1_beef = ('F1_beef', 'mean'), 
                    F3_beef = ('F3_beef', 'mean'), 
                    Q1_beef = ('Q1_beef', 'mean'), 
                    Q3_beef = ('Q3_beef', 'mean'), 
                    P1_beef = ('P1_beef', 'mean'), 
                    P3_beef = ('P3_beef', 'mean'),
                    AC_beef_BB = ('AC_beef_BB', 'mean'), 
                    QC_beef_BB = ('QC_beef_BB', 'mean'), 
                    FOC_beef_BB = ('FOC_beef_BB', 'mean'), 
                    FLC_beef_BB = ('FLC_beef_BB', 'mean'), 
                    FDC_beef_BB = ('FDC_beef_BB', 'mean'),
                    wr_drink_beef = ('wr_drink_beef', 'mean'), 
                    wr_irrig_beef = ('wr_irrig_beef', 'mean'), 
                    wp = ('wp', 'mean')
                    ).sort_values(by = col)

lmap.groupby([col], observed = True).agg(
                    F1_Sheep_AVG = ('F1_sheep', 'mean'), 
                    F2_Sheep_AVG = ('F2_sheep', 'mean'), 
                    F3_Sheep_AVG = ('F3_sheep', 'mean'), 
                    Q1_Sheep_AVG = ('Q1_sheep', 'mean'), 
                    Q2_Sheep_AVG = ('Q2_sheep', 'mean'), 
                    Q3_Sheep_AVG = ('Q3_sheep', 'mean'), 
                    P1_Sheep_AVG = ('P1_sheep', 'mean'), 
                    P2_Sheep_AVG = ('P2_sheep', 'mean'), 
                    P3_Sheep_AVG = ('P3_sheep', 'mean')
                    ).sort_values(by = col)

lmap.groupby([col], observed = True).agg(
                    F1_Dairy_AVG = ('F1_dairy', 'mean'), 
                    Q1_Dairy_AVG = ('Q1_dairy', 'mean'), 
                    P1_Dairy_AVG = ('P1_dairy', 'mean'), 
                    ).sort_values(by = col)




ag_df[(ag_df['SPREAD_Commodity'] == 'Beef Cattle') & (ag_df['irrigation'] == 0)].groupby(['SPREAD_ID_original']).first().iloc[:, :10]

ag_df[ag_df['SPREAD_Commodity'] == 'Beef Cattle'].groupby(['SA4_ID', 'irrigation'], observed = True)[['SPREAD_ID_original']].first().iloc[:300, :10]

lmap.loc[lmap['SA2_ID'] == 101011001, 'SA2_ID': 'yield_mapped'].head(100)
lmap.info()

lmap['HC_BEEF'] = lmap.eval('safe_pur * kgDMhayr * irrig_factor / (8 * 365 * 0.85)')

lmap['HC_BEEF'] = lmap.eval('pur * kgDMhayr * irrig_factor * area_mapped / (8 * 365 * 0.85)')
lmap[['HC_BEEF', 'head_capacity_beef']].head(200)
lmap.eval('HC_BEEF == head_capacity_beef').sum()
x = (np.round(lmap['HC_BEEF'] * 100, 0) == np.round(lmap['head_capacity_beef'] * 100, 0))
x.sum()
lmap.loc[~x, ['HC_BEEF', 'head_capacity_beef']]
lmap.loc[~x, ['HC_BEEF', 'head_capacity_beef']]
lmap['feed_req_factor'].describe()
lmap['head_adjustment_factor'].describe()


# Check national herd sizes of mapped livestock heads against numbers in profit map table
lmap.groupby(['SPREAD_mapped'], observed = True)[['heads_mapped']].sum().sort_values(by = 'SPREAD_mapped')
x = ag_df[ag_df['SPREAD_ID'].between(31, 33)].groupby(['SPREAD_Commodity', 'SA2_ID'], observed = True)[['prod_ABS']].first().sort_values(by = 'SPREAD_Commodity')
x.groupby(['SPREAD_Commodity'])[['prod_ABS']].sum().sort_values(by = 'SPREAD_Commodity')

# # Calculate stocking rates (equivalent to 'yield_potential_beef' etc. in new lmap table)
# # From Javi... stocking rates (head/ha) = pur * kgDMhayr * ha_pixel * irrig_factor / (dse_per_head * 365 * grassfed_factor)
# # Grassfed_factor is 0.65 for Dairy, and 0.85 for Beef and Sheep. 17 DSE/head for dairy, 8 DSE/head for beef, 1.5 DSE/head for sheep
# lmap['BEEF_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (8 * 365 * 0.85)')
# lmap['SHEEP_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (1.5 * 365 * 0.85)')
# lmap['DAIRY_HEAD_PER_HA'] = lmap.eval('pur * kgDMhayr * irrig_factor / (17 * 365 * 0.65)')

"""
