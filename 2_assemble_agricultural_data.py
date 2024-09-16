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



############################################################################################################################################
# Initialisation. Create some helper data and functions
############################################################################################################################################

# Set some options
pd.set_option('display.width', 600)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.float_format', '{:,.4f}'.format)

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
    array_2D = np.zeros(NLUM_mask.shape)
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
# Load some base data files
############################################################################################################################################

# Read cell_df file from disk to a new data frame for ag data with just the relevant columns
cell_df = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5')
cell_df = cell_df[['CELL_ID', 'X', 'Y', 'CELL_HA', 'STE_NAME11', 'SA2_ID', 'STE_CODE11', 'SA4_CODE11', 'PRIMARY_V7', 'SECONDARY_V7', 'TERTIARY_V7', 'CLASSES_18', 'C18_DESCRIPTION', 'SPREAD_ID', 'SPREAD_DESC', 'IRRIGATION']] # 'COMMODITIES', 'COMMODITIES_DESC', 'IRRIGATION']]
cell_df.rename(columns = {'STE_CODE11': 'STE_ID', 'SA4_CODE11': 'SA4_ID'}, inplace = True)

# Read in the PROFIT MAP table as provided by CSIRO to dataframe
ag_df = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20210817/pfe_table_13082021.csv', low_memory = False).drop(columns = 'rev_notes')

# # Load livestock mapping data from CSIRO, drop some columns, downcast and save a lite version. Only need to do this once then load the lite version
# lmap = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20210910/lmap.csv', low_memory = False)
# lmap = lmap.drop(columns = ['Unnamed: 0', 'ha_dairy', 'ha_pixel', 'no_sheep', 'Beef Cattle', 'Dairy Cattle', 'Sheep', 'heads_mapped_cum', 'SPREAD_colour'])
# downcast(lmap)
# lmap.to_hdf('N:/Data-Master/Profit_map/From_CSIRO/20210910/lmap.h5', key = 'lmap', mode = 'w', format = 't')

# Load livestock map lite version, column names and descriptions: N:/Data-Master/Profit_map/From_CSIRO/20210910/lmap_variable_names_description.docx
lmap = pd.read_hdf('N:/Data-Master/Profit_map/From_CSIRO/20210910/lmap.h5')




############################################################################################################################################
# Add livestock commodity mapping to ludf dataframe
############################################################################################################################################

# Create new integer X/Y columns to join, then join the livestock map table to the cell_df dataframe and drop uneccesary columns
cell_df['X-round'] = np.round(cell_df['X'] * 100).astype(np.int16)
cell_df['Y-round'] = np.round(cell_df['Y'] * 100).astype(np.int16)
lmap['X-round'] = np.round(lmap['X'] * 100).astype(np.int16)
lmap['Y-round'] = np.round(lmap['Y'] * 100).astype(np.int16)
# ludf = cell_df.merge(lmap[['X-round', 'Y-round', 'SPREAD_id_mapped', 'kgDMhayr', 'safe_pur', 'pur', 'head_adjustment_factor', 'heads_mapped']], how = 'left', on = ['X-round', 'Y-round'])
ludf = cell_df.merge(lmap[['X-round', 'Y-round', 'SPREAD_id_mapped']], how = 'left', on = ['X-round', 'Y-round'])


# Create LU_ID field to hold IDs and LU_DESC field to hold names of crops and livestock
# Calculate LU_ID == SPREAD_ID and LU_DESC = SPREAD_DESC in all cropping cells
index = ludf.query('SPREAD_ID >= 5').index
ludf.loc[index, 'LU_ID'] = ludf['SPREAD_ID']
ludf.loc[index, 'LU_DESC'] = ludf['SPREAD_DESC']

# Add categories to LU_DESC
ludf['LU_DESC'] = ludf['LU_DESC'].cat.add_categories(['Dairy - natural land', 'Beef - natural land', 'Sheep - natural land', 'Dairy - modified land', 'Beef - modified land', 'Sheep - modified land', 'Plantation forestry', 'Unallocated - natural land', 'Unallocated - modified land'])

# Classify livestock grazing and pasture type

index = ludf.query('PRIMARY_V7 in ["1 Conservation and natural environments", "2 Production from relatively natural environments"] and SPREAD_id_mapped == 31').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [31, 'Dairy - natural land']

index = ludf.query('PRIMARY_V7 in ["1 Conservation and natural environments", "2 Production from relatively natural environments"] and SPREAD_id_mapped == 32').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [32, 'Beef - natural land']

index = ludf.query('PRIMARY_V7 in ["1 Conservation and natural environments", "2 Production from relatively natural environments"] and SPREAD_id_mapped == 33').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [33, 'Sheep - natural land']

index = ludf.query('PRIMARY_V7 in ["3 Production from dryland agriculture and plantations", "4 Production from irrigated agriculture and plantations"] and SPREAD_id_mapped == 31').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [34, 'Dairy - modified land']

index = ludf.query('PRIMARY_V7 in ["3 Production from dryland agriculture and plantations", "4 Production from irrigated agriculture and plantations"] and SPREAD_id_mapped == 32').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [35, 'Beef - modified land']

index = ludf.query('PRIMARY_V7 in ["3 Production from dryland agriculture and plantations", "4 Production from irrigated agriculture and plantations"] and SPREAD_id_mapped == 33').index
ludf.loc[index, ['LU_ID', 'LU_DESC']] = [36, 'Sheep - modified land']


# Non-agricultural land - split according to natural or modified
idx = ludf.query('SPREAD_ID == -1').index
ludf.loc[idx, 'LU_ID'] = 0
ludf.loc[idx, 'LU_DESC'] = 'Non-agricultural land'

# Unallocated agricultural land - split according to natural or modified
idx = ludf.query('(SPREAD_ID == 0 or LU_ID != LU_ID) and PRIMARY_V7 in ["1 Conservation and natural environments", "2 Production from relatively natural environments"]').index
ludf.loc[idx, 'LU_ID'] = 1
ludf.loc[idx, 'LU_DESC'] = 'Unallocated - natural land'

idx = ludf.query('(SPREAD_ID == 0 or LU_ID != LU_ID) and PRIMARY_V7 in ["3 Production from dryland agriculture and plantations", "4 Production from irrigated agriculture and plantations"]').index
ludf.loc[idx, 'LU_ID'] = 2
ludf.loc[idx, 'LU_DESC'] = 'Unallocated - modified land'

# Set plantation forestry (including hardwood, softwood, agroforestry) as agroforestry SPREAD commodity number (4)
index = ludf.query('SPREAD_ID == 4').index
ludf.loc[index, 'LU_ID'] = 0
ludf.loc[index, 'LU_DESC'] = 'Non-agricultural land'

# Fix a rogue irrigated pixel
index = ludf.query('LU_ID == 2 and PRIMARY_V7 == "4 Production from irrigated agriculture and plantations"').index 
ludf.loc[index, 'IRRIGATION'] = 0
ludf.loc[index, 'SECONDARY_V7'] = '3.2 Grazing modified pastures'
ludf.loc[index, 'PRIMARY_V7'] = '3 Production from dryland agriculture and plantations'
ludf.loc[index, 'C18_DESCRIPTION'] = 'Grazing modified pastures (3.2)'

# Fix other non-cereal crops
index = ludf.query('LU_ID == 15 and IRRIGATION == 0').index 
ludf.loc[index, ['SECONDARY_V7', 'C18_DESCRIPTION']] = ['3.3 Cropping', 'Dryland cropping (3.3)']
index = ludf.query('LU_ID == 15 and IRRIGATION == 1').index 
ludf.loc[index, ['SECONDARY_V7', 'C18_DESCRIPTION']] = ['4.4 Irrigated perennial horticulture', 'Irrigated horticulture (4.4, 4.5)']


# Downcast
ludf['LU_ID'] = ludf['LU_ID'].astype(np.int8)
downcast(ludf)

# Check 
ludf.groupby(['PRIMARY_V7', 'SECONDARY_V7', 'C18_DESCRIPTION', 'LU_ID', 'LU_DESC', 'IRRIGATION'], observed = True, as_index = False)['X'].count().sort_values(by = ['PRIMARY_V7', 'SECONDARY_V7'])

# # Check NLUM
# nlum = pd.read_csv(r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.tif.csv').drop(columns = ['Rowid'])
# nlum.groupby(['PRIMARY_V7', 'COMMODITIES', 'COMMODITIES_DESC'], observed = True)['TENURE'].count()
# nlum.groupby(['PRIMARY_V7', 'SECONDARY_V7', 'COMMODITIES_DESC'], observed = True)['TENURE'].count()
# nlum.groupby(['PRIMARY_V7', 'SECONDARY_V7', 'C18_DESCRIPTION', 'COMMODITIES_DESC'], observed = True)['TENURE'].count()
# nlum.groupby(['PRIMARY_V7', 'C18_DESCRIPTION', 'COMMODITIES', 'COMMODITIES_DESC'], observed = True)['TENURE'].count()
# nlum.groupby(['PRIMARY_V7', 'TENURE_DESC'], observed = True)['COMMODITIES'].count()

# Export to HDF5 file
tmp_df = ludf[['CELL_ID', 'X', 'Y', 'CELL_HA', 'SA2_ID', 'PRIMARY_V7', 'SECONDARY_V7', 'SPREAD_ID', 'SPREAD_DESC', 'IRRIGATION', 'LU_ID', 'LU_DESC']]
tmp_df.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', key = 'cell_LU_mapping', mode = 'w', format = 't')

# Save the output to GeoTiff
with rasterio.open('N:/Data-Master/Profit_map/LU_ID.tif', 'w+', dtype = 'int16', nodata = -99, **meta) as dst:        
    dst.write_band(1, conv_1D_to_2D(tmp_df['LU_ID']))

# Create a template from NLUM and livestock mapping to crosscheck that we have all records needed
def_df = ludf.query('LU_ID >= 5').groupby(['SA2_ID', 'LU_ID', 'IRRIGATION'], as_index = False, observed = True)['LU_DESC'].first().sort_values(by = ['SA2_ID', 'LU_ID', 'IRRIGATION'])
def_df['SA2_ID'] = pd.to_numeric(def_df['SA2_ID'], downcast = 'integer')
def_df['LU_ID'] = pd.to_numeric(def_df['LU_ID'], downcast = 'integer')
def_df = def_df.query('LU_ID >= 5')
downcast(def_df)
def_df.to_hdf('N:/Data-Master/Profit_map/NLUM_SPREAD_LU_ID_Mapped_Concordance.h5', key = 'NLUM_SPREAD_LU_ID_Mapped_Concordance', mode = 'w', format = 't')


# # Summarise LU_ID classes
# ludf.groupby(['LU_ID', 'LU_DESC', 'PRIMARY_V7'], observed = True)['LU_DESC'].count()
# # ludf.groupby('SPREAD_ID')['SPREAD_DESC'].first()
# ludf.groupby(['PRIMARY_V7', 'IRRIGATION', 'LU_DESC'], observed = True)['LU_DESC'].count()



############################################################################################################################################
# Convert LU_ID to vector GeoDataFrame, join table of LU_DESC, and save to Geopackage
############################################################################################################################################

# # Collect raster zones as rasterio shape features
# results = ({'properties': {'LU_ID': v}, 'geometry': s}
# for i, (s, v) in enumerate(features.shapes(conv_1D_to_2D(tmp_df['LU_ID']).astype(np.uint8), mask = NLUM_mask, transform = NLUM_transform)))

# # Convert rasterio shape features to GeoDataFrame
# gdfp = gpd.GeoDataFrame.from_features(list(results), crs = NLUM_crs)

# # Hack to fix error 'ValueError: No Shapely geometry can be created from null value'
# gdfp['geometry'] = gdfp.geometry.buffer(0)

# # Dissolve boundaries and convert to multipart shapes
# gdfp = gdfp.dissolve(by = 'LU_ID')

# # Load in NLUM tabular data
# table = ludf.groupby('LU_ID', observed = True, as_index = False)['LU_DESC'].first()
# table['LU_DESC'] = table['LU_DESC'].astype('object')

# # Join the table to the GeoDataFrame
# gdfp = gdfp.merge(table, on = 'LU_ID', how = 'left')

# # Save NLUM data as GeoPackage
# gdfp.to_file('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_clip.gpkg', layer = 'LU_ID', driver = 'GPKG')


# # Some rows will be NaNs because there are internal polygons
# print('Number of NULL cells =', gdfp[gdfp.isna().any(axis = 1)].shape[0])



############################################################################################################################################
# Load Aussiegrass average pasture productivity raster to ludf dataframe
############################################################################################################################################

# Load raw Aussiegrass 0.05 degree resolution data and save the output to GeoTiff with corrected nodata value
with rasterio.open('N:/Data-Master/Profit_map/From_CSIRO/20210720/avg_growth11/avg_growth11.tif') as src:
    meta5k = src.meta.copy()
    meta5k.update(compress = 'lzw', driver = 'GTiff', nodata = 0)
    with rasterio.open('N:/Data-Master/Profit_map/From_CSIRO/20210720/avg_growth11/avg_growth11_nodata_fixed.tif', 'w+', **meta5k) as dst:        
        dst.write_band(1, src.read(1))
        
        # Create a destination array
        dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32)
        
        # Reproject/resample input raster to match NLUM mask (meta)
        reproject(rasterio.band(dst, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.bilinear)
        
        # Fill nodata in raster using value of nearest cell to match NLUM mask
        fill_mask = np.where(dst_array > 0, 1, 0)
        dst_array_filled = np.where(NLUM_mask == 1, fillnodata(dst_array, fill_mask, max_search_distance = 200.0), -9999)
        
        # Save the output to GeoTiff
        with rasterio.open('N:/Data-Master/Profit_map/PASTURE_KG_DM_HA.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
            dst.write_band(1, dst_array_filled)
        
        # Flatten 2D array to 1D array of valid values only and add data to ludf dataframe
        ludf['PASTURE_KG_DM_HA'] = dst_array_filled[NLUM_mask == 1]


# # Load 0.01 degree resolution resampled data that Javi used and save the output to GeoTiff for comparison - USING THE BILINEAR INTERPOLATION ABOVE BECAUSE THIS ONE WHICH JAVI USED IS SPATIALLY OFFSET FOR SOME REASON
# with rasterio.open('N:/Data-Master/Profit_map/From_CSIRO/20210720/AussieGrass/avggrowth1k.tif') as src:
        
#         # Create a destination array
#         dst_array = np.zeros((meta.get('height'), meta.get('width')), np.float32)
        
#         # Reproject/resample input raster to match NLUM mask (meta)
#         reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.bilinear)
        
#         # Fill nodata in raster using value of nearest cell to match NLUM mask
#         fill_mask = np.where(dst_array > 0, 1, 0)
#         dst_array_filled = np.where(NLUM_mask == 1, fillnodata(dst_array, fill_mask, max_search_distance = 200.0), -9999)
        
#         # Save the output to GeoTiff
#         with rasterio.open('N:/Data-Master/Profit_map/PASTURE_KG_DM_HA_JAVI.tif', 'w+', dtype = 'float32', nodata = -9999, **meta) as dst:        
#             dst.write_band(1, dst_array_filled)
        
#         # Flatten 2D array to 1D array of valid values only and add data to ludf dataframe
#         ludf['PASTURE_KG_DM_HA'] = dst_array_filled[NLUM_mask == 1]

# # Put pasture data side-by-side for visual comparison
# ludf.loc[ludf['kgDMhayr'].notnull(), ['kgDMhayr', 'PASTURE_KG_DM_HA', 'PASTURE_KG_DM_HA_JAVI']].sample(100)

# # Calculate RMS error and mean absolute error
# ludf['kgDMhayr'].mean()
# ludf.eval('(PASTURE_KG_DM_HA - kgDMhayr) ** 2').mean() ** .5
# ludf.eval('PASTURE_KG_DM_HA - kgDMhayr').mean()
# ludf[['PASTURE_KG_DM_HA', 'kgDMhayr']].sample(n = 100000).corr() # 0.98 correlation




############################################################################################################################################
# Assemble essential livestock yield and cost data, reverse engineer to create space-filling data for switching
############################################################################################################################################

# Note: results of this are for dryland. For irrigation multiply YIELD_POT, AC, FOC, FLC, FDC by irrig_factor (i.e. 2)
 
# Aggregate feed requirement by SA2 for beef, sheep, and dairy
feed_req = lmap.groupby(['SA2_ID'], observed = True, as_index = False).agg(FEED_REQ = ('feed_req_factor', 'mean')).sort_values(by = 'SA2_ID')

############## DAIRY ##############

# Reverse engineer the cost data to get consistent SA2/SA4 values
lmap['AC_dairy_SA2'] = lmap.eval('AC_dairy / yield_potential_dairy')
lmap['FOC_dairy_SA2'] = lmap.eval('FOC_dairy / yield_potential_dairy')
lmap['FLC_dairy_SA2'] = lmap.eval('FLC_dairy / yield_potential_dairy')
lmap['FDC_dairy_SA2'] = lmap.eval('FDC_dairy / yield_potential_dairy')

dairy = lmap.groupby('SA2_ID', observed = True, as_index = False).agg(
                    SA4_ID = ('sa4_id', 'first'),
                    STATE_ID = ('STATE_ID', 'first'),
                    F1_DAIRY = ('F1_dairy', 'mean'), 
                    Q1_DAIRY = ('Q1_dairy', 'mean'), 
                    P1_DAIRY = ('P1_dairy', 'mean'), 
                    AC_DAIRY = ('AC_dairy_SA2', 'mean'), 
                    QC_DAIRY = ('QC_dairy', 'mean'), 
                    FOC_DAIRY = ('FOC_dairy_SA2', 'mean'), 
                    FLC_DAIRY = ('FLC_dairy_SA2', 'mean'), 
                    FDC_DAIRY = ('FDC_dairy_SA2', 'mean'),
                    WR_DRN_DAIRY = ('wr_drink_dairy', 'mean'), 
                    WR_IRR_DAIRY = ('wr_irrig_dairy', 'max'),
                    WP = ('wp', 'mean')
                    ).sort_values(by = 'SA2_ID')

# Calculate and join summary tables by SA4 and STE to fill in data gaps
tmp_SA4 = dairy.query('WR_IRR_DAIRY > 0').groupby('SA4_ID', observed = True, as_index = False).agg(WR_IRR_SA4 = ('WR_IRR_DAIRY', 'mean'))
tmp_STATE = dairy.query('WR_IRR_DAIRY > 0').groupby('STATE_ID', observed = True, as_index = False).agg(WR_IRR_STATE = ('WR_IRR_DAIRY', 'mean'))

dairy = dairy.merge(tmp_SA4, how = 'left', on = 'SA4_ID')
dairy = dairy.merge(tmp_STATE, how = 'left', on = 'STATE_ID')

dairy.loc[dairy.query('WR_IRR_DAIRY == 0 or WR_IRR_DAIRY != WR_IRR_DAIRY').index, 'WR_IRR_DAIRY'] = dairy['WR_IRR_SA4']
dairy.loc[dairy.query('WR_IRR_DAIRY == 0 or WR_IRR_DAIRY != WR_IRR_DAIRY').index, 'WR_IRR_DAIRY'] = dairy['WR_IRR_STATE']
dairy.loc[dairy.query('WR_IRR_DAIRY == 0 or WR_IRR_DAIRY != WR_IRR_DAIRY').index, 'WR_IRR_DAIRY'] = dairy['WR_IRR_DAIRY'].mean()

dairy['WR_DRN_DAIRY'] = dairy['WR_DRN_DAIRY'].mean()

dairy.drop(columns = ['SA4_ID', 'STATE_ID', 'WR_IRR_SA4', 'WR_IRR_STATE'], inplace = True)


############## BEEF ##############

# Reverse engineer the cost data to get consistent SA2/SA4 values
lmap['AC_beef_SA2'] = lmap.eval('AC_beef / yield_potential_beef')
lmap['FOC_beef_SA2'] = lmap.eval('FOC_beef / yield_potential_beef')
lmap['FLC_beef_SA2'] = lmap.eval('FLC_beef / yield_potential_beef')
lmap['FDC_beef_SA2'] = lmap.eval('FDC_beef / yield_potential_beef')

beef = lmap.groupby(['SA2_ID'], observed = True, as_index = False).agg(
                    STATE_ID = ('STATE_ID', 'first'),
                    SA4_ID = ('sa4_id', 'first'),
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
                    WR_IRR_BEEF = ('wr_irrig_beef', 'max')
                    ).sort_values(by = 'SA2_ID')

# Calculate and join summary tables by SA4 and STE to fill in data gaps
tmp_SA4 = beef.query('WR_IRR_BEEF > 0').groupby('SA4_ID', observed = True, as_index = False).agg(WR_IRR_SA4 = ('WR_IRR_BEEF', 'mean'))
tmp_STATE = beef.query('WR_IRR_BEEF > 0').groupby('STATE_ID', observed = True, as_index = False).agg(WR_IRR_STATE = ('WR_IRR_BEEF', 'mean'))

beef = beef.merge(tmp_SA4, how = 'left', on = 'SA4_ID')
beef = beef.merge(tmp_STATE, how = 'left', on = 'STATE_ID')

beef.loc[beef.query('WR_IRR_BEEF == 0 or WR_IRR_BEEF != WR_IRR_BEEF').index, 'WR_IRR_BEEF'] = beef['WR_IRR_SA4']
beef.loc[beef.query('WR_IRR_BEEF == 0 or WR_IRR_BEEF != WR_IRR_BEEF').index, 'WR_IRR_BEEF'] = beef['WR_IRR_STATE']
beef.loc[beef.query('WR_IRR_BEEF == 0 or WR_IRR_BEEF != WR_IRR_BEEF').index, 'WR_IRR_BEEF'] = beef['WR_IRR_BEEF'].mean()

# Fill in gaps in drinking water
beef['WR_DRN_BEEF'] = beef['WR_DRN_BEEF'].mean()

beef.drop(columns = ['SA4_ID', 'STATE_ID', 'WR_IRR_SA4', 'WR_IRR_STATE'], inplace = True)


############## SHEEP ##############

# Reverse engineer the cost data to get consistent SA2/SA4 values
lmap['AC_sheep_SA2'] = lmap.eval('AC_sheep / yield_potential_sheep')
lmap['FOC_sheep_SA2'] = lmap.eval('FOC_sheep / yield_potential_sheep')
lmap['FLC_sheep_SA2'] = lmap.eval('FLC_sheep / yield_potential_sheep')
lmap['FDC_sheep_SA2'] = lmap.eval('FDC_sheep / yield_potential_sheep')

sheep = lmap.groupby(['SA2_ID'], observed = True, as_index = False).agg(
                    STATE_ID = ('STATE_ID', 'first'),
                    SA4_ID = ('sa4_id', 'first'),
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
                    WR_IRR_SHEEP = ('wr_irrig_sheep', 'max')                    
                    ).sort_values(by = 'SA2_ID')

# Calculate and join summary tables by SA4 and STE to fill in data gaps
tmp_SA4 = sheep.query('WR_IRR_SHEEP > 0').groupby('SA4_ID', observed = True, as_index = False).agg(WR_IRR_SA4 = ('WR_IRR_SHEEP', 'mean'))
tmp_STATE = sheep.query('WR_IRR_SHEEP > 0').groupby('STATE_ID', observed = True, as_index = False).agg(WR_IRR_STATE = ('WR_IRR_SHEEP', 'mean'))

sheep = sheep.merge(tmp_SA4, how = 'left', on = 'SA4_ID')
sheep = sheep.merge(tmp_STATE, how = 'left', on = 'STATE_ID')

sheep.loc[sheep.query('WR_IRR_SHEEP == 0 or WR_IRR_SHEEP != WR_IRR_SHEEP').index, 'WR_IRR_SHEEP'] = sheep['WR_IRR_SA4']
sheep.loc[sheep.query('WR_IRR_SHEEP == 0 or WR_IRR_SHEEP != WR_IRR_SHEEP').index, 'WR_IRR_SHEEP'] = sheep['WR_IRR_STATE']
sheep.loc[sheep.query('WR_IRR_SHEEP == 0 or WR_IRR_SHEEP != WR_IRR_SHEEP').index, 'WR_IRR_SHEEP'] = sheep['WR_IRR_SHEEP'].mean()

# Fill in gaps in drinking water
sheep['WR_DRN_SHEEP'] = sheep['WR_DRN_SHEEP'].mean()

sheep.drop(columns = ['SA4_ID', 'STATE_ID', 'WR_IRR_SA4', 'WR_IRR_STATE'], inplace = True)




# Merge the feed requirement to ludf dataframe by SA2
ludf = ludf.merge(feed_req, how = 'left', on = 'SA2_ID')

# Insert current safe pasture utilisation rate and calculate for all grid cells
ludf['SAFE_PUR'] = 0.4
index = ludf.query('PRIMARY_V7 in ["1 Conservation and natural environments", "2 Production from relatively natural environments"]').index
ludf.loc[index, 'SAFE_PUR'] = 0.3
ludf.loc[ludf.query('Y >= -26').index, 'SAFE_PUR'] = 0.2
ludf.loc[ludf.query('STE_NAME11 == "Northern Territory"').index, 'SAFE_PUR'] = 0.15



############################################################################################################################################
# Join the SA2-based economic data back the cell-based ludf dataframe to provide values for all cells whose SA2's support livestock
############################################################################################################################################

# Calculate the safe pasture utilisation rate for natural land and modified land
"""safe_pur = 0.4 for sown pasture, 0.3 for native pastures, 0.2 for pastures where y >= -26 and 0.15 in the Northern Territory"""

ludf.loc[ludf.query('Y >= -26').index, 'SAFE_PUR_NATL'] = 0.2
ludf.loc[ludf.query('Y < -26').index, 'SAFE_PUR_NATL'] = 0.3
ludf.loc[ludf.query('Y >= -26').index, 'SAFE_PUR_MODL'] = 0.2
ludf.loc[ludf.query('Y < -26').index, 'SAFE_PUR_MODL'] = 0.4
ludf.loc[ludf.query('STE_NAME11 == "Northern Territory"').index, 'SAFE_PUR_MODL'] = 0.15


# Calculate the yield potential for all cells based on the new pasture data PASTURE_KG_DM_HA and new SAFE_PUR - does not include irrig_factor
ludf['YIELD_POT_DAIRY'] = ludf.eval('SAFE_PUR * FEED_REQ * PASTURE_KG_DM_HA / (17 * 365 * 0.65)')
ludf['YIELD_POT_BEEF'] = ludf.eval('SAFE_PUR * FEED_REQ * PASTURE_KG_DM_HA / (8 * 365 * 0.85)')
ludf['YIELD_POT_SHEEP'] = ludf.eval('SAFE_PUR * FEED_REQ * PASTURE_KG_DM_HA / (1.5 * 365 * 0.85)')


################ DAIRY ################

# Merge the economic data to ludf dataframe by SA2
ludf = ludf.merge(dairy, how = 'left', on = 'SA2_ID')

# Convert the costs (back to $/ha, ML/head, ML/ha)
ludf['AC_DAIRY'] = ludf.eval('AC_DAIRY * YIELD_POT_DAIRY') # $/ha
ludf['FOC_DAIRY'] = ludf.eval('FOC_DAIRY * YIELD_POT_DAIRY') # $/ha
ludf['FLC_DAIRY'] = ludf.eval('FLC_DAIRY * YIELD_POT_DAIRY') # $/ha
ludf['FDC_DAIRY'] = ludf.eval('FDC_DAIRY * YIELD_POT_DAIRY') # $/ha
ludf['WR_DRN_DAIRY'] = ludf.eval('WR_DRN_DAIRY / 1000000') # ML/head
ludf['WR_IRR_DAIRY'] = ludf.eval('WR_IRR_DAIRY / 1000000') #  ML/head

# Join some columns from lmap
ludf_ = ludf.merge(lmap[['X-round', 'Y-round', 'safe_pur', 'heads_mapped', 'yield_potential_dairy', 'AC_dairy']], how = 'left', on = ['X-round', 'Y-round'])

# Select dairy cells only 
ludf_ = ludf_.query('SPREAD_id_mapped == 31')

# Implementing this query will illustrate perfect alignment - misalignment comes because of differences in the YIELD_POT and SAFE_PUR
# ludf_ = ludf_.query('SPREAD_id_mapped == 31 and SAFE_PUR == safe_pur and YIELD_POT_DAIRY == yield_potential_dairy')

downcast(ludf_)

# Check back against Javi's data, calculate RMS error,  mean absolute error and correlation
ludf_['YIELD_POT_DAIRY'] = ludf_.eval('YIELD_POT_DAIRY * (1 + IRRIGATION)')
print('Dairy - Mean yield_potential_dairy = {:.4f} head/ha'.format(ludf_['yield_potential_dairy'].mean())) 
print('Dairy - RMSE YIELD_POT_DAIRY - yield_potential_dairy = {:.4f} head/ha'.format(ludf_.eval('(YIELD_POT_DAIRY - yield_potential_dairy) ** 2').mean() ** .5))
print('Dairy - Mean YIELD_POT_DAIRY - yield_potential_dairy = {:.4f} head/ha'.format(ludf_.eval('YIELD_POT_DAIRY - yield_potential_dairy').mean()))
print('Dairy - Corrcoeff YIELD_POT_DAIRY yield_potential_dairy = {:.4f}'.format(ludf_[['yield_potential_dairy', 'YIELD_POT_DAIRY']].corr().iloc[0,1])) # 0.98 correlation
print('Dairy - Herd size YIELD_POT_DAIRY = {:,.0f} head'.format(ludf_.eval('YIELD_POT_DAIRY * CELL_HA').sum())) # Number of dairy cattle
print('Dairy - Herd size yield_potential_dairy = {:,.0f} head'.format(ludf_.eval('yield_potential_dairy * CELL_HA').sum())) # Number of dairy cattle
print('Dairy - Herd size heads_mapped = {:,.0f} head'.format(ludf_.eval('heads_mapped').sum())) # Number of dairy cattle 

ludf_['AC_DAIRY'] = ludf_.eval('AC_DAIRY * (1 + IRRIGATION)')
print('Dairy - Mean AC_dairy = {:.4f} $/ha'.format(ludf_['AC_dairy'].mean())) 
print('Dairy - RMSE AC_DAIRY - AC_dairy = {:.4f} $/ha'.format(ludf_.eval('(AC_DAIRY - AC_dairy) ** 2').mean() ** .5))
print('Dairy - Mean AC_DAIRY - AC_dairy = {:.4f} $/ha'.format(ludf_.eval('AC_DAIRY - AC_dairy').mean()))
print('Dairy - Corrcoeff AC_DAIRY AC_dairy = {:.4f}'.format(ludf_[['AC_DAIRY', 'AC_dairy']].corr().iloc[0,1])) # 0.98 correlation
                                
# # Calculate feed requirement in tonnes of dry matter required per tonne of milk, with a floor at 0
# ludf['FEED_REQ_DAIRY_TDM_T'] = ludf.eval('(SAFE_PUR * (FEED_REQ - 1) * PASTURE_KG_DM_HA + 17 * 365 * 0.65 * YIELD_POT_DAIRY) / (YIELD_POT_DAIRY * F1_DAIRY * Q1_DAIRY)')
# ludf.loc[ludf['FEED_REQ_DAIRY_TDM_T'] < 0, 'FEED_REQ_DAIRY_TDM_T'] = 0


################ BEEF ################

# Merge the economic data to ludf dataframe by SA2
ludf = ludf.merge(beef, how = 'left', on = 'SA2_ID')

# Convert the costs (back to $/ha, ML/head, ML/ha)
ludf['AC_BEEF'] = ludf.eval('AC_BEEF * YIELD_POT_BEEF') # $/ha
ludf['FOC_BEEF'] = ludf.eval('FOC_BEEF * YIELD_POT_BEEF') # $/ha
ludf['FLC_BEEF'] = ludf.eval('FLC_BEEF * YIELD_POT_BEEF') # $/ha
ludf['FDC_BEEF'] = ludf.eval('FDC_BEEF * YIELD_POT_BEEF') # $/ha
ludf['WR_DRN_BEEF'] = ludf.eval('WR_DRN_BEEF / 1000000') # ML/head
ludf['WR_IRR_BEEF'] = ludf.eval('WR_IRR_BEEF / 1000000') #  ML/head

# Join some columns from lmap
ludf_ = ludf.merge(lmap[['X-round', 'Y-round', 'safe_pur', 'heads_mapped', 'yield_potential_beef', 'AC_beef']], how = 'left', on = ['X-round', 'Y-round'])

# Select beef cells only 
ludf_ = ludf_.query('SPREAD_id_mapped == 32')

downcast(ludf_)

# Check back against Javi's data, calculate RMS error,  mean absolute error and correlation
ludf_['YIELD_POT_BEEF'] = ludf_.eval('YIELD_POT_BEEF * (1 + IRRIGATION)')
print('Beef - Mean yield_potential_beef = {:.4f} head/ha'.format(ludf_['yield_potential_beef'].mean())) 
print('Beef - RMSE YIELD_POT_BEEF - yield_potential_beef = {:.4f} head/ha'.format(ludf_.eval('(YIELD_POT_BEEF - yield_potential_beef) ** 2').mean() ** .5))
print('Beef - Mean YIELD_POT_BEEF - yield_potential_beef = {:.4f} head/ha'.format(ludf_.eval('YIELD_POT_BEEF - yield_potential_beef').mean()))
print('Beef - Corrcoeff YIELD_POT_BEEF yield_potential_beef = {:.4f}'.format(ludf_[['yield_potential_beef', 'YIELD_POT_BEEF']].corr().iloc[0,1])) # 0.98 correlation
print('Beef - Herd size YIELD_POT_BEEF = {:,.0f} head'.format(ludf_.eval('YIELD_POT_BEEF * CELL_HA').sum())) # Number of beef cattle
print('Beef - Herd size yield_potential_beef = {:,.0f} head'.format(ludf_.eval('yield_potential_beef * CELL_HA').sum())) # Number of beef cattle
print('Beef - Herd size heads_mapped = {:,.0f} head'.format(ludf_.eval('heads_mapped').sum())) # Number of beef cattle 

ludf_['AC_BEEF'] = ludf_.eval('AC_BEEF * (1 + IRRIGATION)')
print('Beef - Mean AC_beef = {:.4f} $/ha'.format(ludf_['AC_beef'].mean())) 
print('Beef - RMSE AC_BEEF - AC_beef = {:.4f} $/ha'.format(ludf_.eval('(AC_BEEF - AC_beef) ** 2').mean() ** .5))
print('Beef - Mean AC_BEEF - AC_beef = {:.4f} $/ha'.format(ludf_.eval('AC_BEEF - AC_beef').mean()))
print('Beef - Corrcoeff AC_BEEF AC_beef = {:.4f}'.format(ludf_[['AC_BEEF', 'AC_beef']].corr().iloc[0,1])) # 0.98 correlation
                                

# # Calculate feed requirement in tonnes of dry matter required per tonne of meat, with a floor at 0
# ludf['FEED_REQ_BEEF_TDM_T'] = ludf.eval('(SAFE_PUR * (FEED_REQ - 1) * PASTURE_KG_DM_HA + 8 * 365 * 0.15 * YIELD_POT_BEEF) * 0.001 / (YIELD_POT_BEEF * (F1_BEEF * Q1_BEEF + F3_BEEF * Q3_BEEF))')
# ludf.loc[ludf['FEED_REQ_BEEF_TDM_T'] < 0, 'FEED_REQ_BEEF_TDM_T'] = 0


################ SHEEP ################

# Merge the economic data to ludf dataframe by SA2
ludf = ludf.merge(sheep, how = 'left', on = 'SA2_ID')

# Convert the costs (back to $/ha, ML/head, ML/ha)
ludf['AC_SHEEP'] = ludf.eval('AC_SHEEP * YIELD_POT_SHEEP') # $/ha
ludf['FOC_SHEEP'] = ludf.eval('FOC_SHEEP * YIELD_POT_SHEEP') # $/ha
ludf['FLC_SHEEP'] = ludf.eval('FLC_SHEEP * YIELD_POT_SHEEP') # $/ha
ludf['FDC_SHEEP'] = ludf.eval('FDC_SHEEP * YIELD_POT_SHEEP') # $/ha
ludf['WR_DRN_SHEEP'] = ludf.eval('WR_DRN_SHEEP / 1000000') # ML/head
ludf['WR_IRR_SHEEP'] = ludf.eval('WR_IRR_SHEEP / 1000000') #  ML/head

# Join some columns from lmap
ludf_ = ludf.merge(lmap[['X-round', 'Y-round', 'safe_pur', 'heads_mapped', 'yield_potential_sheep', 'AC_sheep']], how = 'left', on = ['X-round', 'Y-round'])

# Select sheep cells only 
ludf_ = ludf_.query('SPREAD_id_mapped == 33')

downcast(ludf_)

# Check back against Javi's data, calculate RMS error,  mean absolute error and correlation
ludf_['YIELD_POT_SHEEP'] = ludf_.eval('YIELD_POT_SHEEP * (1 + IRRIGATION)')
print('Sheep - Mean yield_potential_sheep = {:.4f} head/ha'.format(ludf_['yield_potential_sheep'].mean())) 
print('Sheep - RMSE YIELD_POT_SHEEP - yield_potential_sheep = {:.4f} head/ha'.format(ludf_.eval('(YIELD_POT_SHEEP - yield_potential_sheep) ** 2').mean() ** .5))
print('Sheep - Mean YIELD_POT_SHEEP - yield_potential_sheep = {:.4f} head/ha'.format(ludf_.eval('YIELD_POT_SHEEP - yield_potential_sheep').mean()))
print('Sheep - Corrcoeff YIELD_POT_SHEEP yield_potential_sheep = {:.4f}'.format(ludf_[['yield_potential_sheep', 'YIELD_POT_SHEEP']].corr().iloc[0,1])) # 0.98 correlation
print('Sheep - Herd size YIELD_POT_SHEEP = {:,.0f} head'.format(ludf_.eval('YIELD_POT_SHEEP * CELL_HA').sum())) # Number of sheep cattle
print('Sheep - Herd size yield_potential_sheep = {:,.0f} head'.format(ludf_.eval('yield_potential_sheep * CELL_HA').sum())) # Number of sheep cattle
print('Sheep - Herd size heads_mapped = {:,.0f} head'.format(ludf_.eval('heads_mapped').sum())) # Number of sheep cattle 

ludf_['AC_SHEEP'] = ludf_.eval('AC_SHEEP * (1 + IRRIGATION)')
print('Sheep - Mean AC_sheep = {:.4f} $/ha'.format(ludf_['AC_sheep'].mean())) 
print('Sheep - RMSE AC_SHEEP - AC_sheep = {:.4f} $/ha'.format(ludf_.eval('(AC_SHEEP - AC_sheep) ** 2').mean() ** .5))
print('Sheep - Mean AC_SHEEP - AC_sheep = {:.4f} $/ha'.format(ludf_.eval('AC_SHEEP - AC_sheep').mean()))
print('Sheep - Corrcoeff AC_SHEEP AC_sheep = {:.4f}'.format(ludf_[['AC_SHEEP', 'AC_sheep']].corr().iloc[0,1])) # 0.98 correlation
                                


# # Check that we have data everywhere it's needed
# ls = def_df.query('LU_ID == 31 or LU_ID == 34').groupby('SA2_ID', observed = True, as_index = False).agg(L_DAIRY = ('LU_DESC', 'count'))
# ls['L_DAIRY'] = 1
# ludf = ludf.merge(ls, how = 'left', on = 'SA2_ID')
# ludf.query('L_DAIRY == 1 and AC_DAIRY != AC_DAIRY')

# ls = def_df.query('LU_ID == 32 or LU_ID == 35').groupby('SA2_ID', observed = True, as_index = False).agg(L_BEEF = ('LU_DESC', 'count'))
# ls['L_BEEF'] = 1
# ludf = ludf.merge(ls, how = 'left', on = 'SA2_ID')
# ludf.query('L_BEEF == 1 and AC_BEEF != AC_BEEF')

# ls = def_df.query('LU_ID == 33 or LU_ID == 36').groupby('SA2_ID', observed = True, as_index = False).agg(L_SHEEP = ('LU_DESC', 'count'))
# ls['L_SHEEP'] = 1
# ludf = ludf.merge(ls, how = 'left', on = 'SA2_ID')
# ludf.query('L_SHEEP == 1 and AC_SHEEP != AC_SHEEP')


# Downcast to save space
downcast(ludf)

# Export to HDF5 file
tmp_df = ludf.drop(columns = ['X', 'Y', 'X-round', 'Y-round', 'SPREAD_id_mapped', 'STE_NAME11', 'SECONDARY_V7', 'TERTIARY_V7', 'C18_DESCRIPTION', 'CLASSES_18', 'SPREAD_ID', 'SPREAD_DESC', 'SAFE_PUR', 'YIELD_POT_BEEF', 'YIELD_POT_SHEEP', 'YIELD_POT_DAIRY'])
tmp_df.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_livestock_data.h5', key = 'cell_livestock_data', mode = 'w', format = 't')




############################################################################################################################################
# Example calculations of livestock production, costs, and water requirements
############################################################################################################################################

"""
From Javi...
    heads_mapped = safe_pur * feed_req_factor * kgDMhayr * ha_pixel * irrig_factor / (dse_per_head * 365 * grassfed_factor)
Parameters:
    safe_pur = 0.4 for sown pasture, 0.3 for native pastures, 0.2 for pastures where y >= -26 and 0.15 in the Northern Territory
    feed_req_factor = SUM(prod_ABS * DSE_per_head) / SUM(kgDMhayr * irrig_factor * ha_pixel * safe_pur / 365). This is the sum of all 
    pixels in an SA4 and heads of all livestock types (beef, sheep and dairy). 
    If there is less pasture available than DSEs to map, feed_req_factor will be > 1. Otherwise we force feed_req_factor to equal 1 
    (so that safe_pur * feed_req_factor = safe_pur * 1 and therefore pur = safe_pur).
    kgDMhayr = The AussieGrass-derived estimate of average pasture growth (kgDM) per ha and year, for all years between 2005 and 2015
    ha_pixel = the real ha provided by Brett
    irrig_factor = the irrigation (or productivity) factor which is dependent on pasture type and irrigation status
    dse_per_head = 17 for dairy, 8 for beef, 1.5 for sheep
    grassfed_factor = the proportion of feed that comes from grass (we use this to modify the number of days spent grazing). This is 0.65 for dairy and 0.85 for the rest.
    Water requirements from Marinoni et al. (2012): Beef Cattle:2.7 ML/ha, Dairy cattle: 3.6 ML/ha, Sheep: 3.1 ML/ha   ****NOT USED****
"""

################## Livestock productivity and economics cross-checks
ludf_ = ludf.copy()


################ DAIRY ################

# Calculate the yield potential (head per hectare) for natural land and modified land 
ludf_['YIELD_POT_DAIRY_DRY_NATL'] = ludf_.eval('SAFE_PUR_NATL * FEED_REQ * PASTURE_KG_DM_HA / (17 * 365 * 0.65)')
ludf_['YIELD_POT_DAIRY_DRY_MODL'] = ludf_.eval('SAFE_PUR_MODL * FEED_REQ * PASTURE_KG_DM_HA / (17 * 365 * 0.65)')
ludf_['YIELD_POT_DAIRY_IRR_NATL'] = ludf_.eval('SAFE_PUR_NATL * FEED_REQ * PASTURE_KG_DM_HA * 2 / (17 * 365 * 0.65)')
ludf_['YIELD_POT_DAIRY_IRR_MODL'] = ludf_.eval('SAFE_PUR_MODL * FEED_REQ * PASTURE_KG_DM_HA * 2 / (17 * 365 * 0.65)')

# Calculate production, revenue, cost of production, and profit at full equity per hectare
ludf_['PROD_DAIRY_DRY_NATL'] = ludf_.eval('YIELD_POT_DAIRY_DRY_NATL * (F1_DAIRY * Q1_DAIRY)') # tonnes/ha
ludf_['REV_DAIRY_DRY_NATL'] = ludf_.eval('YIELD_POT_DAIRY_DRY_NATL * (F1_DAIRY * Q1_DAIRY * P1_DAIRY)') # $/ha
ludf_['COST_DAIRY_DRY_NATL'] = ludf_.eval('YIELD_POT_DAIRY_DRY_NATL * (QC_DAIRY + WR_DRN_DAIRY * WP) + (AC_DAIRY + FOC_DAIRY + FLC_DAIRY + FDC_DAIRY)') # $ /ha
ludf_['PFE_DAIRY_DRY_NATL'] = ludf_.eval('REV_DAIRY_DRY_NATL - COST_DAIRY_DRY_NATL') # $/ha
ludf_['WATER_USE_DAIRY_DRY_NATL'] = ludf_.eval('YIELD_POT_DAIRY_DRY_NATL * WR_DRN_DAIRY') # ML/ha

ludf_['PROD_DAIRY_IRR_NATL'] = ludf_.eval('YIELD_POT_DAIRY_IRR_NATL * (F1_DAIRY * Q1_DAIRY)') # tonnes/ha
ludf_['REV_DAIRY_IRR_NATL'] = ludf_.eval('YIELD_POT_DAIRY_IRR_NATL * (F1_DAIRY * Q1_DAIRY * P1_DAIRY)') # $/ha
ludf_['COST_DAIRY_IRR_NATL'] = ludf_.eval('YIELD_POT_DAIRY_IRR_NATL * (QC_DAIRY + (WR_DRN_DAIRY + WR_IRR_DAIRY) * WP) + (AC_DAIRY + FOC_DAIRY + FLC_DAIRY + FDC_DAIRY)') # $/ha ***
ludf_['PFE_DAIRY_IRR_NATL'] = ludf_.eval('REV_DAIRY_IRR_NATL - COST_DAIRY_IRR_NATL') # $/ha
ludf_['WATER_USE_DAIRY_IRR_NATL'] = ludf_.eval('YIELD_POT_DAIRY_IRR_NATL * (WR_DRN_DAIRY + WR_IRR_DAIRY)') # ML/ha ***

ludf_['PROD_DAIRY_DRY_MODL'] = ludf_.eval('YIELD_POT_DAIRY_DRY_MODL * (F1_DAIRY * Q1_DAIRY)') # tonnes/ha
ludf_['REV_DAIRY_DRY_MODL'] = ludf_.eval('YIELD_POT_DAIRY_DRY_MODL * (F1_DAIRY * Q1_DAIRY * P1_DAIRY)') # $/ha
ludf_['COST_DAIRY_DRY_MODL'] = ludf_.eval('YIELD_POT_DAIRY_DRY_MODL * (QC_DAIRY + WR_DRN_DAIRY * WP) + (AC_DAIRY + FOC_DAIRY + FLC_DAIRY + FDC_DAIRY)') # $ /ha
ludf_['PFE_DAIRY_DRY_MODL'] = ludf_.eval('REV_DAIRY_DRY_MODL - COST_DAIRY_DRY_MODL') # $/ha
ludf_['WATER_USE_DAIRY_DRY_MODL'] = ludf_.eval('YIELD_POT_DAIRY_DRY_MODL * WR_DRN_DAIRY') # ML/ha

ludf_['PROD_DAIRY_IRR_MODL'] = ludf_.eval('YIELD_POT_DAIRY_IRR_MODL * (F1_DAIRY * Q1_DAIRY)') # tonnes/ha
ludf_['REV_DAIRY_IRR_MODL'] = ludf_.eval('YIELD_POT_DAIRY_IRR_MODL * (F1_DAIRY * Q1_DAIRY * P1_DAIRY)') # $/ha
ludf_['COST_DAIRY_IRR_MODL'] = ludf_.eval('YIELD_POT_DAIRY_IRR_MODL * (QC_DAIRY + (WR_DRN_DAIRY + WR_IRR_DAIRY) * WP) + (AC_DAIRY + FOC_DAIRY + FLC_DAIRY + FDC_DAIRY)') # $/ha ***
ludf_['PFE_DAIRY_IRR_MODL'] = ludf_.eval('REV_DAIRY_IRR_MODL - COST_DAIRY_IRR_MODL') # $/ha
ludf_['WATER_USE_DAIRY_IRR_MODL'] = ludf_.eval('YIELD_POT_DAIRY_IRR_MODL * (WR_DRN_DAIRY + WR_IRR_DAIRY)') # ML/ha ***


############### BEEF ################

# Calculate the yield potential (head per hectare) for natural land and modified land 
ludf_['YIELD_POT_BEEF_DRY_NATL'] = ludf_.eval('SAFE_PUR_NATL * FEED_REQ * PASTURE_KG_DM_HA / (8 * 365 * 0.85)')
ludf_['YIELD_POT_BEEF_DRY_MODL'] = ludf_.eval('SAFE_PUR_MODL * FEED_REQ * PASTURE_KG_DM_HA / (8 * 365 * 0.85)')
ludf_['YIELD_POT_BEEF_IRR_NATL'] = ludf_.eval('SAFE_PUR_NATL * FEED_REQ * PASTURE_KG_DM_HA * 2 / (8 * 365 * 0.85)')
ludf_['YIELD_POT_BEEF_IRR_MODL'] = ludf_.eval('SAFE_PUR_MODL * FEED_REQ * PASTURE_KG_DM_HA * 2 / (8 * 365 * 0.85)')

# Calculate production, revenue, cost of production, and profit at full equity per hectare
ludf_['PROD_BEEF_MEAT_DRY_NATL'] = ludf_.eval('YIELD_POT_BEEF_DRY_NATL * F1_BEEF * Q1_BEEF') # tonnes/ha
ludf_['PROD_BEEF_LEXP_DRY_NATL'] = ludf_.eval('YIELD_POT_BEEF_DRY_NATL * F3_BEEF * Q3_BEEF') # tonnes/ha
ludf_['REV_BEEF_DRY_NATL'] = ludf_.eval('YIELD_POT_BEEF_DRY_NATL * (F1_BEEF * Q1_BEEF * P1_BEEF + F3_BEEF * Q3_BEEF * P3_BEEF)') # $/ha
ludf_['COST_BEEF_DRY_NATL'] = ludf_.eval('YIELD_POT_BEEF_DRY_NATL * (QC_BEEF + WR_DRN_BEEF * WP) + (AC_BEEF + FOC_BEEF + FLC_BEEF + FDC_BEEF)') # $ /ha
ludf_['PFE_BEEF_DRY_NATL'] = ludf_.eval('REV_BEEF_DRY_NATL - COST_BEEF_DRY_NATL') # $/ha
ludf_['WATER_USE_BEEF_DRY_NATL'] = ludf_.eval('YIELD_POT_BEEF_DRY_NATL * WR_DRN_BEEF') # ML/ha

ludf_['PROD_BEEF_MEAT_IRR_NATL'] = ludf_.eval('YIELD_POT_BEEF_IRR_NATL * F1_BEEF * Q1_BEEF') # tonnes/ha
ludf_['PROD_BEEF_LEXP_IRR_NATL'] = ludf_.eval('YIELD_POT_BEEF_IRR_NATL * F3_BEEF * Q3_BEEF') # tonnes/ha
ludf_['REV_BEEF_IRR_NATL'] = ludf_.eval('YIELD_POT_BEEF_IRR_NATL * (F1_BEEF * Q1_BEEF * P1_BEEF + F3_BEEF * Q3_BEEF * P3_BEEF)') # $/ha
ludf_['COST_BEEF_IRR_NATL'] = ludf_.eval('YIELD_POT_BEEF_IRR_NATL * (QC_BEEF + (WR_DRN_BEEF + WR_IRR_BEEF) * WP) + (AC_BEEF + FOC_BEEF + FLC_BEEF + FDC_BEEF)') # $/ha ***
ludf_['PFE_BEEF_IRR_NATL'] = ludf_.eval('REV_BEEF_IRR_NATL - COST_BEEF_IRR_NATL') # $/ha
ludf_['WATER_USE_BEEF_IRR_NATL'] = ludf_.eval('YIELD_POT_BEEF_IRR_NATL * (WR_DRN_BEEF + WR_IRR_BEEF)') # ML/ha ***

ludf_['PROD_BEEF_MEAT_DRY_MODL'] = ludf_.eval('YIELD_POT_BEEF_DRY_MODL * F1_BEEF * Q1_BEEF') # tonnes/ha
ludf_['PROD_BEEF_LEXP_DRY_MODL'] = ludf_.eval('YIELD_POT_BEEF_DRY_MODL * F3_BEEF * Q3_BEEF') # tonnes/ha
ludf_['REV_BEEF_DRY_MODL'] = ludf_.eval('YIELD_POT_BEEF_DRY_MODL * (F1_BEEF * Q1_BEEF * P1_BEEF + F3_BEEF * Q3_BEEF * P3_BEEF)') # $/ha
ludf_['COST_BEEF_DRY_MODL'] = ludf_.eval('YIELD_POT_BEEF_DRY_MODL * (QC_BEEF + WR_DRN_BEEF * WP) + (AC_BEEF + FOC_BEEF + FLC_BEEF + FDC_BEEF)') # $ /ha
ludf_['PFE_BEEF_DRY_MODL'] = ludf_.eval('REV_BEEF_DRY_MODL - COST_BEEF_DRY_MODL') # $/ha
ludf_['WATER_USE_BEEF_DRY_MODL'] = ludf_.eval('YIELD_POT_BEEF_DRY_MODL * WR_DRN_BEEF') # ML/ha

ludf_['PROD_BEEF_MEAT_IRR_MODL'] = ludf_.eval('YIELD_POT_BEEF_IRR_MODL * F1_BEEF * Q1_BEEF') # tonnes/ha
ludf_['PROD_BEEF_LEXP_IRR_MODL'] = ludf_.eval('YIELD_POT_BEEF_IRR_MODL * F3_BEEF * Q3_BEEF') # tonnes/ha
ludf_['REV_BEEF_IRR_MODL'] = ludf_.eval('YIELD_POT_BEEF_IRR_MODL * (F1_BEEF * Q1_BEEF * P1_BEEF + F3_BEEF * Q3_BEEF * P3_BEEF)') # $/ha
ludf_['COST_BEEF_IRR_MODL'] = ludf_.eval('YIELD_POT_BEEF_IRR_MODL * (QC_BEEF + (WR_DRN_BEEF + WR_IRR_BEEF) * WP) + (AC_BEEF + FOC_BEEF + FLC_BEEF + FDC_BEEF)') # $/ha ***
ludf_['PFE_BEEF_IRR_MODL'] = ludf_.eval('REV_BEEF_IRR_MODL - COST_BEEF_IRR_MODL') # $/ha
ludf_['WATER_USE_BEEF_IRR_MODL'] = ludf_.eval('YIELD_POT_BEEF_IRR_MODL * (WR_DRN_BEEF + WR_IRR_BEEF)') # ML/ha ***


############### SHEEP ################

# Calculate the yield potential (head per hectare) for natural land and modified land 
ludf_['YIELD_POT_SHEEP_DRY_NATL'] = ludf_.eval('SAFE_PUR_NATL * FEED_REQ * PASTURE_KG_DM_HA / (1.5 * 365 * 0.85)')
ludf_['YIELD_POT_SHEEP_DRY_MODL'] = ludf_.eval('SAFE_PUR_MODL * FEED_REQ * PASTURE_KG_DM_HA / (1.5 * 365 * 0.85)')
ludf_['YIELD_POT_SHEEP_IRR_NATL'] = ludf_.eval('SAFE_PUR_NATL * FEED_REQ * PASTURE_KG_DM_HA * 2 / (1.5 * 365 * 0.85)')
ludf_['YIELD_POT_SHEEP_IRR_MODL'] = ludf_.eval('SAFE_PUR_MODL * FEED_REQ * PASTURE_KG_DM_HA * 2 / (1.5 * 365 * 0.85)')

# Calculate production, revenue, cost of production, and profit at full equity per hectare
ludf_['PROD_SHEEP_MEAT_DRY_NATL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_NATL * F1_SHEEP * Q1_SHEEP') # tonnes/ha
ludf_['PROD_SHEEP_LEXP_DRY_NATL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_NATL * F3_SHEEP * Q3_SHEEP') # tonnes/ha
ludf_['PROD_SHEEP_WOOL_DRY_NATL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_NATL * (F2_SHEEP * Q2_SHEEP)') # tonnes/ha sheep wool
ludf_['REV_SHEEP_DRY_NATL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_NATL * (F1_SHEEP * Q1_SHEEP * P1_SHEEP + F2_SHEEP * Q2_SHEEP * P2_SHEEP + F3_SHEEP * Q3_SHEEP * P3_SHEEP)') # $/ha
ludf_['COST_SHEEP_DRY_NATL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_NATL * (QC_SHEEP + WR_DRN_SHEEP * WP) + (AC_SHEEP + FOC_SHEEP + FLC_SHEEP + FDC_SHEEP)') # $ /ha
ludf_['PFE_SHEEP_DRY_NATL'] = ludf_.eval('REV_SHEEP_DRY_NATL - COST_SHEEP_DRY_NATL') # $/ha
ludf_['WATER_USE_SHEEP_DRY_NATL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_NATL * WR_DRN_SHEEP') # ML/ha

ludf_['PROD_SHEEP_MEAT_IRR_NATL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_NATL * F1_SHEEP * Q1_SHEEP') # tonnes/ha
ludf_['PROD_SHEEP_LEXP_IRR_NATL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_NATL * F3_SHEEP * Q3_SHEEP') # tonnes/ha
ludf_['PROD_SHEEP_WOOL_IRR_NATL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_NATL * (F2_SHEEP * Q2_SHEEP)') # tonnes/ha sheep wool
ludf_['REV_SHEEP_IRR_NATL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_NATL * (F1_SHEEP * Q1_SHEEP * P1_SHEEP + F2_SHEEP * Q2_SHEEP * P2_SHEEP + F3_SHEEP * Q3_SHEEP * P3_SHEEP)') # $/ha
ludf_['COST_SHEEP_IRR_NATL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_NATL * (QC_SHEEP + (WR_DRN_SHEEP + WR_IRR_SHEEP) * WP) + (AC_SHEEP + FOC_SHEEP + FLC_SHEEP + FDC_SHEEP)') # $/ha ***
ludf_['PFE_SHEEP_IRR_NATL'] = ludf_.eval('REV_SHEEP_IRR_NATL - COST_SHEEP_IRR_NATL') # $/ha
ludf_['WATER_USE_SHEEP_IRR_NATL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_NATL * (WR_DRN_SHEEP + WR_IRR_SHEEP)') # ML/ha ***

ludf_['PROD_SHEEP_MEAT_DRY_MODL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_MODL * F1_SHEEP * Q1_SHEEP') # tonnes/ha
ludf_['PROD_SHEEP_LEXP_DRY_MODL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_MODL * F3_SHEEP * Q3_SHEEP') # tonnes/ha
ludf_['PROD_SHEEP_WOOL_DRY_MODL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_MODL * (F2_SHEEP * Q2_SHEEP)') # tonnes/ha sheep wool
ludf_['REV_SHEEP_DRY_MODL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_MODL * (F1_SHEEP * Q1_SHEEP * P1_SHEEP + F2_SHEEP * Q2_SHEEP * P2_SHEEP + F3_SHEEP * Q3_SHEEP * P3_SHEEP)') # $/ha
ludf_['COST_SHEEP_DRY_MODL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_MODL * (QC_SHEEP + WR_DRN_SHEEP * WP) + (AC_SHEEP + FOC_SHEEP + FLC_SHEEP + FDC_SHEEP)') # $ /ha
ludf_['PFE_SHEEP_DRY_MODL'] = ludf_.eval('REV_SHEEP_DRY_MODL - COST_SHEEP_DRY_MODL') # $/ha
ludf_['WATER_USE_SHEEP_DRY_MODL'] = ludf_.eval('YIELD_POT_SHEEP_DRY_MODL * WR_DRN_SHEEP') # ML/ha

ludf_['PROD_SHEEP_MEAT_IRR_MODL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_MODL * F1_SHEEP * Q1_SHEEP') # tonnes/ha
ludf_['PROD_SHEEP_LEXP_IRR_MODL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_MODL * F3_SHEEP * Q3_SHEEP') # tonnes/ha
ludf_['PROD_SHEEP_WOOL_IRR_MODL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_MODL * (F2_SHEEP * Q2_SHEEP)') # tonnes/ha sheep wool
ludf_['REV_SHEEP_IRR_MODL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_MODL * (F1_SHEEP * Q1_SHEEP * P1_SHEEP + F2_SHEEP * Q2_SHEEP * P2_SHEEP + F3_SHEEP * Q3_SHEEP * P3_SHEEP)') # $/ha
ludf_['COST_SHEEP_IRR_MODL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_MODL * (QC_SHEEP + (WR_DRN_SHEEP + WR_IRR_SHEEP) * WP) + (AC_SHEEP + FOC_SHEEP + FLC_SHEEP + FDC_SHEEP)') # $/ha ***
ludf_['PFE_SHEEP_IRR_MODL'] = ludf_.eval('REV_SHEEP_IRR_MODL - COST_SHEEP_IRR_MODL') # $/ha
ludf_['WATER_USE_SHEEP_IRR_MODL'] = ludf_.eval('YIELD_POT_SHEEP_IRR_MODL * (WR_DRN_SHEEP + WR_IRR_SHEEP)') # ML/ha ***




# Check back against Javi's data, calculate RMS error,  mean absolute error and correlation
ludf_ = ludf_.merge(lmap[['X-round', 'Y-round', 'rev_dairy', 'cost_pct_dairy', 'yield_potential_sheep']], how = 'left', on = ['X-round', 'Y-round'])
print('Mean revenue dairy = {:.2f} $/ha'.format(ludf_.query('rev_dairy > 0')['rev_dairy'].mean()))
print('Mean revenue dairy DRY_NATL = {:.2f} $/ha'.format(ludf_.query('REV_DAIRY_DRY_NATL > 0')['REV_DAIRY_DRY_NATL'].mean()))
print('Mean revenue dairy IRR_NATL = {:.2f} $/ha'.format(ludf_.query('REV_DAIRY_IRR_NATL > 0')['REV_DAIRY_IRR_NATL'].mean()))
print('Mean revenue dairy DRY_MODL = {:.2f} $/ha'.format(ludf_.query('REV_DAIRY_DRY_NATL > 0')['REV_DAIRY_DRY_MODL'].mean()))
print('Mean revenue dairy IRR_MODL = {:.2f} $/ha'.format(ludf_.query('REV_DAIRY_IRR_NATL > 0')['REV_DAIRY_IRR_MODL'].mean()))

print('Mean cost dairy DRY_NATL = {:.2f} $/ha'.format(ludf_.query('COST_DAIRY_DRY_NATL > 0')['COST_DAIRY_DRY_NATL'].mean()))
print('Mean cost dairy IRR_NATL = {:.2f} $/ha'.format(ludf_.query('COST_DAIRY_IRR_NATL > 0')['COST_DAIRY_IRR_NATL'].mean()))
print('Mean cost dairy DRY_MODL = {:.2f} $/ha'.format(ludf_.query('COST_DAIRY_DRY_NATL > 0')['COST_DAIRY_DRY_MODL'].mean()))
print('Mean cost dairy IRR_MODL = {:.2f} $/ha'.format(ludf_.query('COST_DAIRY_IRR_NATL > 0')['COST_DAIRY_IRR_MODL'].mean()))





################## Livestock water cross-checks

# Calculate total water use by livestock. ABS = 1,231,627 N:/Data-Master/Profit_map/46180do001_201011.xls
print('Total water use by livestock (BB) = {:,.0f} ML'.format(
        ludf.query('SPREAD_id_mapped == 31 and IRRIGATION == 1').eval('(WR_IRR_DAIRY + WR_DRN_DAIRY) * YIELD_POT_DAIRY * CELL_HA').sum() + \
        ludf.query('SPREAD_id_mapped == 32 and IRRIGATION == 1').eval('(WR_IRR_BEEF + WR_DRN_BEEF) * YIELD_POT_BEEF * CELL_HA').sum() + \
        ludf.query('SPREAD_id_mapped == 33 and IRRIGATION == 1').eval('(WR_IRR_SHEEP + WR_DRN_SHEEP) * YIELD_POT_SHEEP * CELL_HA').sum()))

# Check against Javi's data
print('Total water use by livestock (JN) = {:,.0f} ML'.format(
        lmap.query('SPREAD_id_mapped == 31 and irrigation == 1').eval('heads_mapped * (wr_irrig_dairy + wr_drink_dairy) / 1000000').sum() + \
        lmap.query('SPREAD_id_mapped == 32 and irrigation == 1').eval('heads_mapped * (wr_irrig_beef + wr_drink_beef) / 1000000').sum() + \
        lmap.query('SPREAD_id_mapped == 33 and irrigation == 1').eval('heads_mapped * (wr_irrig_sheep + wr_drink_sheep) / 1000000').sum()))

# Area of irrigated land . ABS = 1,962,569 ha N:/Data-Master/Profit_map/46180do001_201011.xls
print('Area of irrigated land = {:,.0f} ha'.format(ludf.query('IRRIGATION == 1').eval('CELL_HA').sum()))





############################################################################################################################################
# Assemble CROP yield, economic, and water use data from profit map table and join to the ludf dataframe
############################################################################################################################################

# Select crops only
af_df_crops = ag_df.query('5 <= SPREAD_ID <= 25')

# Rename columns to avoid python built-in naming
crops_df = af_df_crops.rename(columns = {'yield': 'Yield', 
                                         'area': 'Area', 
                                         'prod': 'Prod', 
                                         'irrigation': 'Irrigation', 
                                         'SPREAD_ID': 'LU_ID', 
                                         'SPREAD_Commodity': 'LU_DESC'})
    
# Convert SPREAD_DESC to Sentence case
crops_df['LU_DESC'] = crops_df['LU_DESC'].str.capitalize()
# crops_df.groupby('LU_ID')['LU_DESC'].first()


######## Aggregate the ABS Commodity-level data to SPREAD class

# Set up area and production weighting variables for averaging such that they are > 0 to avoid 'ZeroDivisionError: Weights sum to zero, can't be normalized'
crops_df.eval('area_weight = Area + 0.01', inplace = True)
crops_df.eval('prod_weight = Prod + 0.01', inplace = True)

# Define a lambda function to compute the weighted mean
area_weighted_mean = lambda x: np.average(x, weights = crops_df.loc[x.index, 'area_weight'])
prod_weighted_mean = lambda x: np.average(x, weights = crops_df.loc[x.index, 'prod_weight'])  

# Aggregate commodity-level to SPREAD-level using weighted mean based on area of Commodity within SA2
crops_SA2_df = crops_df.groupby(['SA2_ID', 'LU_ID', 'Irrigation'], as_index = False).agg(
                    STATE_ID = ('STATE_ID', 'first'),
                    LU_DESC = ('LU_DESC', 'first'),
                    Area = ('Area', 'sum'),
                    Prod = ('Prod', 'sum'),
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

crops_STE_df = crops_df.groupby(['STATE_ID', 'LU_ID', 'Irrigation'], as_index = False).agg(
                    LU_DESC_STE = ('LU_DESC', 'first'),
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
crops_sum_df = crops_SA2_df.merge(crops_STE_df, how = 'left', on = ['STATE_ID', 'LU_ID', 'Irrigation']) 

# Sort in place
crops_sum_df.sort_values(by = ['SA2_ID', 'LU_ID', 'Irrigation'], ascending = True, inplace = True)

# Allocate state-level values for Stone Fruit and Vegetables to smooth out the variation
for col in ['Yield', 'P1', 'AC', 'QC', 'FDC', 'FLC', 'FOC', 'WR', 'WP']:
    crops_sum_df.loc[crops_sum_df['LU_DESC'].isin(['Stone fruit', 'Vegetables']), col] = crops_sum_df[col + '_STE']

# Clean up dataframe
crops_sum_df.drop(columns = ['STATE_ID', 'LU_DESC_STE', 'Yield_STE', 'P1_STE', 'AC_STE', 'QC_STE', 'FDC_STE', 'FLC_STE', 'FOC_STE', 'WR_STE', 'WP_STE'], inplace = True)

# Code to calculate revenue and costs
# crops_sum_df.eval('Yield_ha_TRUE = Production / Area', inplace = True)
# crops_sum_df.eval('Rev_ha = Yield * P1', inplace = True)
# crops_sum_df.eval('Rev_tot = Production * P1', inplace = True)
# crops_sum_df.eval('Costs_ha = (AC + FDC + FOC + FLC) + (QC * Yield) + (WR * WP)', inplace = True)
# crops_sum_df.eval('Costs_t = Costs_ha / Yield', inplace = True)

downcast(crops_sum_df)

########### Check and check again
print('Number of NaNs =', crops_sum_df[crops_sum_df.isna().any(axis=1)].shape[0])

# Join the table to the cell_df dataframe and drop uneccesary columns
adf = cell_df.merge(crops_sum_df, how = 'left', left_on = ['SA2_ID', 'SPREAD_ID', 'IRRIGATION'], right_on = ['SA2_ID', 'LU_ID', 'Irrigation']) 

# Limit cell dataframe to crops only
adf = adf.query("SPREAD_ID >= 5 and SPREAD_ID <= 25")

# Check for NaNs
print('Number of NaNs =', adf[adf.isna().any(axis = 1)].shape[0])

# Check the NLUM vs ABS commodity area
adf2 = adf[['SA2_ID', 'LU_DESC', 'IRRIGATION', 'CELL_HA', 'Area', 'Prod', 'Yield', 'WR']]

# Calculate production per cell based on NLUM area and ABS-derived yields
adf2.eval('Prod_NLUM = CELL_HA * Yield', inplace = True)
adf2.eval('WR_NLUM = CELL_HA * WR', inplace = True)
adf2.eval('WR_ABS = Area * WR', inplace = True)

# Aggregate to the level of SA2
tmp = adf2.groupby(['SA2_ID', 'LU_DESC', 'IRRIGATION']).agg(Area_NLUM = ('CELL_HA', 'sum'), 
                                                            Area_ABS = ('Area', 'first'),
                                                            Prod_NLUM = ('Prod_NLUM', 'sum'), 
                                                            Prod_ABS = ('Prod', 'first'),
                                                            WR_NLUM = ('WR_NLUM', 'sum'),
                                                            WR_ABS = ('WR_ABS', 'first')
                                                            )

# Aggregate SA2s to calculate sum over commodities and irrigation status
tmp2 = tmp.groupby(['LU_DESC', 'IRRIGATION']).agg(Area_NLUM = ('Area_NLUM', 'sum'), 
                                                  Area_ABS = ('Area_ABS', 'sum'),
                                                  Prod_NLUM = ('Prod_NLUM', 'sum'), 
                                                  Prod_ABS = ('Prod_ABS', 'sum'),
                                                  WR_NLUM = ('WR_NLUM', 'sum'), 
                                                  WR_ABS = ('WR_ABS', 'sum')
                                                  )
# Compare total NLUM vs ABS area, prductivity, and irrigation water requirement
print(tmp2)
print(tmp2.sum())


# # Further checks of total crop water use in profit map data
# af_df_crops_irr = adf.query('IRRIGATION == 1')
# af_df_crops_irr.groupby(['SA2_ID', 'SPREAD_ID'])[['WR', 'area_irrig_NLUM']].first().eval('WR * area_irrig_NLUM').sum()
# af_df_crops_irr.eval('WR * area_ABS * area_irrig_NLUM / (area_irrig_NLUM + area_dry_NLUM )').sum()
# af_df_crops_irr.eval('WR * area_ABS * pct_irrig').sum()

# # Check crop water use in the NLUM
# af_df_crops_irr.eval('WR * CELL_HA').sum()
# af_df_crops_irr['WUSE'] = af_df_crops_irr.eval('WR * CELL_HA')
# af_df_crops_irr.groupby('STE_NAME11')['WUSE'].sum()

# Join the table to the template dataframe to reduce file to only those rows necessary (i.e. those SPREAD classes mapped in the NLUM)
cdf = def_df.query('5 <= LU_ID <= 25').merge(crops_sum_df.drop(columns = 'LU_DESC'), how = 'left', left_on = ['SA2_ID', 'LU_ID', 'IRRIGATION'], right_on = ['SA2_ID', 'LU_ID', 'Irrigation']) 

# Save file
cdf.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_crop_data.h5', key = 'SA2_crop_data', mode = 'w', format = 't')




############################################################################################################################################
# Assemble off-land commodity (CHICKENS, EGGS, and PIGS) data
############################################################################################################################################

# Chickens, eggs, and pigs are 'off-land'. Demand is automatically met. There is an implication for feed pre-calculated in the demand model.
# Then the area, costs, and impacts (if any) of off-land commodities is calculated post-hoc in the reporting. These commodities do have a water requirement
# but it is minimal and safely ignored.

# Select off-land commodities (chickens, eggs, pigs) only
cep_df = ag_df.query("SPREAD_ID >= 34 and SPREAD_ID <= 36")
cep_df = cep_df[['STATE_ID', 'SA4_ID', 'SA2_ID', 'SA2_Name', 'SPREAD_Commodity', 'SPREAD_ID', 'prod_ABS', 'area_ABS', 'yield_ABS', 'Q1', 'P1', 'AC_rel', 'QC_rel', 'FDC_rel', 'FLC_rel', 'FOC_rel']].reset_index()
cep_df.drop(columns = 'index', inplace = True)

# Rename columns to avoid python built-in naming
cep_df.rename(columns = {'SPREAD_ID': 'LU_ID',
                         'SPREAD_Commodity': 'LU_DESC', 
                         'prod_ABS': 'Production', 
                         'area_ABS': 'Area', 
                         'yield_ABS': 'Yield',
                         'AC_rel': 'AC', 
                         'QC_rel': 'QC', 
                         'FDC_rel': 'FDC', 
                         'FLC_rel': 'FLC', 
                         'FOC_rel': 'FOC'}, inplace = True) 

# Print the LU_ID and LU_DESC
cep_df.groupby('LU_ID')['LU_DESC'].first()

cep_df.loc[cep_df['LU_ID'] == 34, 'LU_ID'] = 40
cep_df.loc[cep_df['LU_ID'] == 35, 'LU_ID'] = 42
cep_df.loc[cep_df['LU_ID'] == 36, 'LU_ID'] = 41

# Do some test calculations of revenue and costs
# cep_df.eval('Yield_ha_TRUE = Production / Area', inplace = True)
# cep_df.eval('Rev_ha = Yield * Q1 * P1', inplace = True)
# cep_df.eval('Costs_ha = (AC + FDC + FOC + FLC) + (QC * Yield)', inplace = True)
# cep_df.eval('Costs_t = Costs_ha / Yield', inplace = True)

downcast(cep_df)

# Save file
# cep_df.to_csv('N:/Data-Master/Profit_map/cep_yield_econ_SPREAD.csv')
cep_df.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_off_land_commodity_data.h5', key = 'SA2_off_land_commodity_data', mode = 'w', format = 't')




############################################################################################################################################
# Assemble CROP and LIVESTOCK nutrient (NPK) application and surplus data
############################################################################################################################################

################## CROPS NPK application and surplus

# Read in the crop NPK table and drop unwanted columns
npk_df = pd.read_excel('N:/Data-Master/Profit_map/From_CSIRO/20210921/T_NPK_surplus_by_SPREAD_SA2_2010_NLUM_crops_ceiling2p5_plus_n2o_kgco2e_fert.xlsx')
npk_df.drop(columns = ['kgco2e_soil_N_applied', 'kgco2e_soil_N_surplus', 'kgco2e_fert_production', 'ha_weight_SPREAD'], inplace = True)

# Read in the NLUM SA2 template to check for NaNs
def_df = pd.read_hdf('N:/Data-Master/Profit_map/NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')
              
# Join to template and check for nodata
tdf = def_df.query('5 <= LU_ID <= 25').merge(npk_df, how = 'left', left_on = ['SA2_ID', 'LU_ID', 'IRRIGATION'], right_on = ['SA2_ID', 'SPREAD_ID', 'irrigation'])
tdf[tdf.isna().any(axis = 1)].shape[0]

# Rename columns
tdf.rename(columns = {'N_applied_adjusted': 'N_APPL_KG_HA',
                      'P_applied_adjusted': 'P_APPL_KG_HA',
                      'K_applied_adjusted': 'K_APPL_KG_HA',
                      'N_surplus': 'N_SURP_KG_HA',
                      'P_surplus': 'P_SURP_KG_HA',
                      'K_surplus': 'K_SURP_KG_HA'
                      }, inplace = True)

# Select and order columns
tdf = tdf[['SA2_ID', 'LU_ID', 'LU_DESC', 'IRRIGATION', 'N_APPL_KG_HA', 'P_APPL_KG_HA', 'K_APPL_KG_HA', 'N_SURP_KG_HA', 'P_SURP_KG_HA', 'K_SURP_KG_HA']]

# Convert LU_DESC to Sentence case
tdf['LU_DESC'] = tdf['LU_DESC'].str.capitalize()

# Downcast and save file
tdf.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_crop_NPK_applic_surplus.h5', key = 'SA2_crop_NPK_applic_surplus', mode = 'w', format = 't')


################## LIVESTOCK NPK application  *** Not included in dataframe, to be calculated on the fly based on cell-based pasture growth field KG_DM_HA_YR ***

"""
Javi says...
In the documentation sent with the NPK surplus data there is a table which summarises our desktop research into NPK uptake, with references. 
There are three rows that correspond to pastures, which have different NPK uptake values. 
Pick one of your preference (we don’t information to suggest which would be best) and use it in the following formula on the livestock map (in place of NPK uptake %):

    Commodity	url	                                                                                            source	                                    N_uptake%	P_uptake%	K_uptake%	notes
Pasture	    https://www.yara.com.au/crop-nutrition/improved-pasture/improved-pasture-nutritional-summary/   Yara: Improved Pasture Nutritional Summary	0.269	    0.0265	    0.103	    Improved pasture: Average uptake for different values of t DM/ha
Pasture	    https://www.summitfertz.com.au/field-research-agronomy/pastures                                 Field Research and Agronomy Pastures	    2.75	    0.35	    1.8	        Clover Pasture
Pasture	    https://impactfertilisers.com.au/wp-content/uploads/impact-calc-nutrient-removal-chart.pdf 	                                                3	        0.3	        2.5	        Hay (Clover/ryegrass) tDM
    
NPK uptake (kg/ha) = kgDMhayr * pur * NPK uptake (%)
Then set NPK applied = NPK uptake, and NPK surplus = 0
"""

# Calculate NPK application to DRYLAND pasture on MODIFIED LAND

# Uptake taken from Impact Fertilisers data
n_uptk, p_uptk, k_uptk = 3, 0.3, 2.5 
# N ceilings taken from https://www.agric.wa.gov.au/pasture-management/boosting-winter-pasture-growth-nitrogen-fertiliser-%E2%80%93-western-australia
# P and K ceilings pro-rata'd based on N limit
n_lim = 60
p_lim = n_lim * (p_uptk / n_uptk)
k_lim = n_lim * (k_uptk / n_uptk)

# ludf['N_APPL_KG_HA_DRY_MODL'] = ludf.eval('PASTURE_KG_DM_HA * @n_uptk / 100')
# ludf.loc[ludf['N_APPL_KG_HA_DRY_MODL'] > n_lim, 'N_APPL_KG_HA_DRY_MODL'] = n_lim
# ludf['P_APPL_KG_HA_DRY_MODL'] = ludf.eval('PASTURE_KG_DM_HA * @p_uptk / 100')
# ludf.loc[ludf['P_APPL_KG_HA_DRY_MODL'] > p_lim, 'P_APPL_KG_HA_DRY_MODL'] = p_lim
# ludf['K_APPL_KG_HA_DRY_MODL'] = ludf.eval('PASTURE_KG_DM_HA * @k_uptk / 100')
# ludf.loc[ludf['K_APPL_KG_HA_DRY_MODL'] > k_lim, 'K_APPL_KG_HA_DRY_MODL'] = k_lim

# NPK application on IRRIGATED MODIFIED LAND = X_APPL_KG_HA_DRY_MODL * 2   
# NPK application on NATURAL LAND = 0                                      
# NPK surplus for all livestock = 0                                        




############################################################################################################################################
# Assemble crop and livestock GHG EMISSIONS data
############################################################################################################################################

##################  Read in the CROPS EMISSIONS table, join to cell_df, and drop unwanted columns

# Read crops emissions data
ghg_df = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20231124/T_GHG_by_SPREAD_SA2_2010_NLUM_Navarroetal_fix_pears_nuts_othernoncereal.csv')
# ghg_df.drop(columns = ['track', 'AER_ID'], inplace = True)

ghg_df = ghg_df[['SA2_ID', 'SPREAD_ID', 'irrigation', 
                 'kgco2_chem_app',
                 'kgco2_crop_management',
                 'kgco2_cult',
                 'kgco2_fert',
                 'kgco2_harvest',
                 'kgco2_irrig',
                 'kgco2_pest',
                 'kgco2_soil',
                 'kgco2_sowing',
                 'kgco2_transport']]

ghg_df = ghg_df.rename(columns = {'SPREAD_ID': 'LU_ID', 
                                'SPREAD_Commodity': 'LU_DESC',
                                'irrigation': 'IRRIGATION', 
                                'kgco2_fert': 'CO2E_KG_HA_FERT_PROD',
                                'kgco2_pest': 'CO2E_KG_HA_PEST_PROD',
                                'kgco2_irrig': 'CO2E_KG_HA_IRRIG',
                                'kgco2_chem_app': 'CO2E_KG_HA_CHEM_APPL', 
                                'kgco2_crop_management': 'CO2E_KG_HA_CROP_MGT', 
                                'kgco2_cult': 'CO2E_KG_HA_CULTIV', 
                                'kgco2_harvest': 'CO2E_KG_HA_HARVEST', 
                                'kgco2_sowing': 'CO2E_KG_HA_SOWING', 
                                'kgco2_soil': 'CO2E_KG_HA_SOIL', 
                                'kgco2_transport': 'CO2E_KG_HA_TRANSPORT'})

# Read in the NLUM SA2 template to check for NaNs (this table has every combination of LU, irr/dry, and SA2)
def_df = pd.read_hdf('N:/Data-Master/Profit_map/NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')

# Read cell_df file from disk to a new data frame for ag data with just the relevant columns
# cell_df = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5')
# cell_df = cell_df[['CELL_ID', 'CELL_HA', 'SA2_ID', 'SA4_CODE11', 'STE_CODE11', 'STE_NAME11']]
# cell_df.rename(columns = {'STE_CODE11': 'STE_ID', 'SA4_CODE11': 'SA4_ID'}, inplace = True)
lut = cell_df.groupby(['SA2_ID'], observed = True, as_index = False).agg(
                    SA4_ID = ('SA4_ID', 'first'),
                    STATE_ID = ('STE_ID', 'first')
                    ).sort_values(by = 'SA2_ID')

# Merge SA4 and STATE ID columns for gap filling
def_df = def_df.query('5 <= LU_ID <= 25').merge(lut, how = 'left', on = 'SA2_ID')

# Join GHG table to concordance template and check for nodata. 
c_ghg = def_df.merge(ghg_df, how = 'left', on = ['SA2_ID', 'LU_ID', 'IRRIGATION'])
print('Number of NaNs =', c_ghg[c_ghg.isna().any(axis=1)].shape[0]) # Number of NaNs = 402

# Calculate and join summary tables by SA4 and STATE to fill in data gaps
SA4 = c_ghg.groupby(['SA4_ID', 'LU_ID', 'IRRIGATION'], observed = True, as_index = False).agg(
                    CO2E_KG_HA_CHEM_APPL_SA4 = ('CO2E_KG_HA_CHEM_APPL', 'mean'), 
                    CO2E_KG_HA_CROP_MGT_SA4 = ('CO2E_KG_HA_CROP_MGT', 'mean'), 
                    CO2E_KG_HA_CULTIV_SA4 = ('CO2E_KG_HA_CULTIV', 'mean'), 
                    CO2E_KG_HA_FERT_PROD_SA4 = ('CO2E_KG_HA_FERT_PROD', 'mean'), 
                    CO2E_KG_HA_HARVEST_SA4 = ('CO2E_KG_HA_HARVEST', 'mean'), 
                    CO2E_KG_HA_IRRIG_SA4 = ('CO2E_KG_HA_IRRIG', 'mean'), 
                    CO2E_KG_HA_PEST_PROD_SA4 = ('CO2E_KG_HA_PEST_PROD', 'mean'), 
                    CO2E_KG_HA_SOIL_SA4 = ('CO2E_KG_HA_SOIL', 'mean'),
                    CO2E_KG_HA_SOWING_SA4 = ('CO2E_KG_HA_SOWING', 'mean'), 
                    CO2E_KG_HA_TRANSPORT_SA4 = ('CO2E_KG_HA_TRANSPORT', 'mean')
                    ).sort_values(by = ['SA4_ID', 'LU_ID', 'IRRIGATION'])
print('Number of NaNs =', SA4[SA4.isna().any(axis=1)].shape[0]) # Number of NaNs = 80

STATE = c_ghg.groupby(['STATE_ID', 'LU_ID', 'IRRIGATION'], observed = True, as_index = False).agg(
                    CO2E_KG_HA_CHEM_APPL_STATE = ('CO2E_KG_HA_CHEM_APPL', 'mean'), 
                    CO2E_KG_HA_CROP_MGT_STATE = ('CO2E_KG_HA_CROP_MGT', 'mean'), 
                    CO2E_KG_HA_CULTIV_STATE = ('CO2E_KG_HA_CULTIV', 'mean'), 
                    CO2E_KG_HA_FERT_PROD_STATE = ('CO2E_KG_HA_FERT_PROD', 'mean'), 
                    CO2E_KG_HA_HARVEST_STATE = ('CO2E_KG_HA_HARVEST', 'mean'), 
                    CO2E_KG_HA_IRRIG_STATE = ('CO2E_KG_HA_IRRIG', 'mean'), 
                    CO2E_KG_HA_PEST_PROD_STATE = ('CO2E_KG_HA_PEST_PROD', 'mean'), 
                    CO2E_KG_HA_SOIL_STATE = ('CO2E_KG_HA_SOIL', 'mean'),
                    CO2E_KG_HA_SOWING_STATE = ('CO2E_KG_HA_SOWING', 'mean'), 
                    CO2E_KG_HA_TRANSPORT_STATE = ('CO2E_KG_HA_TRANSPORT', 'mean')
                    ).sort_values(by = ['STATE_ID', 'LU_ID', 'IRRIGATION'])
print('Number of NaNs =', STATE[STATE.isna().any(axis=1)].shape[0]) # Number of NaNs = 10

AUS = c_ghg.groupby(['LU_ID', 'IRRIGATION'], observed = True, as_index = False).agg(
                    CO2E_KG_HA_CHEM_APPL_AUS = ('CO2E_KG_HA_CHEM_APPL', 'mean'), 
                    CO2E_KG_HA_CROP_MGT_AUS = ('CO2E_KG_HA_CROP_MGT', 'mean'), 
                    CO2E_KG_HA_CULTIV_AUS = ('CO2E_KG_HA_CULTIV', 'mean'), 
                    CO2E_KG_HA_FERT_PROD_AUS = ('CO2E_KG_HA_FERT_PROD', 'mean'), 
                    CO2E_KG_HA_HARVEST_AUS = ('CO2E_KG_HA_HARVEST', 'mean'), 
                    CO2E_KG_HA_IRRIG_AUS = ('CO2E_KG_HA_IRRIG', 'mean'), 
                    CO2E_KG_HA_PEST_PROD_AUS = ('CO2E_KG_HA_PEST_PROD', 'mean'), 
                    CO2E_KG_HA_SOIL_AUS = ('CO2E_KG_HA_SOIL', 'mean'),
                    CO2E_KG_HA_SOWING_AUS = ('CO2E_KG_HA_SOWING', 'mean'), 
                    CO2E_KG_HA_TRANSPORT_AUS = ('CO2E_KG_HA_TRANSPORT', 'mean')
                    ).sort_values(by = ['LU_ID', 'IRRIGATION'])
print('Number of NaNs =', AUS[AUS.isna().any(axis=1)].shape[0]) # Number of NaNs = 0

# Join GHG table to concordance template and check for nodata. 
c_ghg = c_ghg.merge(SA4, how = 'left', on = ['SA4_ID', 'LU_ID', 'IRRIGATION'])
c_ghg = c_ghg.merge(STATE, how = 'left', on = ['STATE_ID', 'LU_ID', 'IRRIGATION'])
c_ghg = c_ghg.merge(AUS, how = 'left', on = ['LU_ID', 'IRRIGATION'])

# Set up lists of GHG sources and columns suffixes
cols = ['CO2E_KG_HA_CHEM_APPL', 'CO2E_KG_HA_CROP_MGT', 'CO2E_KG_HA_CULTIV', 'CO2E_KG_HA_FERT_PROD', 'CO2E_KG_HA_HARVEST', 'CO2E_KG_HA_IRRIG', 'CO2E_KG_HA_PEST_PROD', 'CO2E_KG_HA_SOIL', 'CO2E_KG_HA_SOWING', 'CO2E_KG_HA_TRANSPORT']
suffixes = ['_SA4', '_STATE', '_AUS']

# Fill in data gaps, if SA2 is NaN use SA4 average, if SA4 value is NaN use State average, if State is NaN use national average.
for col in cols:
    for suf in suffixes: 
        tf = c_ghg[col] != c_ghg[col]
        c_ghg.loc[tf, col] = c_ghg.loc[tf, col + suf]

# Drop unwanted columns
c_ghg = c_ghg.drop(columns = [col for col in c_ghg.columns for suf in suffixes if suf in col])

# Check that all gaps are filled
print('Number of NaNs =', c_ghg[c_ghg.isna().any(axis=1)].shape[0]) # Should be zero






##################  Read in the CROPS EMISSIONS table, join to cell_df, and drop unwanted columns

# Read crops emissions data
ghg_df = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20210921/T_GHG_by_SPREAD_SA2_crops_2010_NLUM.csv')
ghg_df.drop(columns = ['SA4_ID', 'STATE_ID', 'track'], inplace = True)

# Join to template and check for nodata. Note: they only occur in kgCO2e_crop_management for valid reasons so we set them to zero.
c_ghg = def_df.query('5 <= LU_ID <= 25').merge(ghg_df, how = 'left', left_on = ['SA2_ID', 'LU_ID', 'IRRIGATION'], right_on = ['SA2_ID', 'SPREAD_ID', 'irrigation'])
print('Number of NaNs =', c_ghg[c_ghg.isna().any(axis=1)].shape[0])

# Fill in gaps in kgCO2e_pest (8 NaNs) with state averages
idx = c_ghg.query('LU_ID == 6 and kgCO2e_pest != kgCO2e_pest').index
val = c_ghg.query('LU_ID == 6 and STATE_ID == 1')['kgCO2e_pest'].mean()
c_ghg.loc[idx, 'kgCO2e_pest'] = val
idx = c_ghg.query('LU_ID == 16 and kgCO2e_pest != kgCO2e_pest').index
val = c_ghg.query('LU_ID == 16 and STATE_ID == 1')['kgCO2e_pest'].mean()
c_ghg.loc[idx, 'kgCO2e_pest'] = val
print('Number of NaNs =', c_ghg[c_ghg.isna().any(axis=1)].shape[0])


def calc_crop_GHG_with_OLD_N2O(c_ghg): 
    """Calculate fertiliser production and soil direct N emissions using ***OLD*** CSIRO N2O data"""
    
    # Merge N2O data to the rest of the crop GHG data
    # Re-order and rename columns
    cols_sorted = ['kgCO2e_fert', 'kgCO2e_pest', 'kgCO2e_irrig', 'kgCO2e_chem_app', 'kgCO2e_crop_management', 'kgCO2e_cult', 'kgCO2e_harvest', 'kgCO2e_sowing', 'kgCO2e_soil']
    cols_sorted.sort()
    c_ghg = c_ghg[['SA2_ID', 'SA4_ID', 'STATE_ID', 'LU_ID', 'SPREAD_Commodity', 'irrigation'] + cols_sorted]
    c_ghg = c_ghg.rename(columns = {'SPREAD_ID': 'LU_ID', 
                                    'SPREAD_Commodity': 'LU_DESC',
                                    'irrigation': 'IRRIGATION', 
                                    'kgCO2e_fert': 'CO2E_KG_HA_FERT_PROD',
                                    'kgCO2e_pest': 'CO2E_KG_HA_PEST_PROD',
                                    'kgCO2e_irrig': 'CO2E_KG_HA_IRRIG',
                                    'kgCO2e_chem_app': 'CO2E_KG_HA_CHEM_APPL', 
                                    'kgCO2e_crop_management': 'CO2E_KG_HA_CROP_MGT', 
                                    'kgCO2e_cult': 'CO2E_KG_HA_CULTIV', 
                                    'kgCO2e_harvest': 'CO2E_KG_HA_HARVEST', 
                                    'kgCO2e_sowing': 'CO2E_KG_HA_SOWING', 
                                    'kgCO2e_soil': 'CO2E_KG_HA_SOIL'})
    return c_ghg


def calc_crop_GHG_with_NEW_N2O(c_ghg): 
    """Calculate fertiliser production and soil direct N emissions using ***NEW*** CSIRO N2O data"""
    
    # Load the crop N2O data from the N surplus data file
    npk_df = pd.read_excel('N:/Data-Master/Profit_map/From_CSIRO/20210921/T_NPK_surplus_by_SPREAD_SA2_2010_NLUM_crops_ceiling2p5_plus_n2o_kgco2e_fert.xlsx')
    
    # Merge N2O data to the rest of the crop GHG data
    c_ghg = c_ghg.merge(npk_df[['SA2_ID', 'SPREAD_ID', 'irrigation', 'kgco2e_soil_N_applied', 'kgco2e_fert_production']], how = 'left', on = ['SA2_ID', 'SPREAD_ID', 'irrigation'])
    print('Number of NaNs =', c_ghg[c_ghg.isna().any(axis=1)].shape[0])
    
    # Re-order and rename columns
    cols_sorted = ['kgco2e_fert_production', 'kgCO2e_pest', 'kgCO2e_irrig', 'kgCO2e_chem_app', 'kgCO2e_crop_management', 'kgCO2e_cult', 'kgCO2e_harvest', 'kgCO2e_sowing', 'kgco2e_soil_N_applied']
    cols_sorted.sort()
    c_ghg = c_ghg[['SA2_ID', 'SA4_ID', 'STATE_ID', 'LU_ID', 'SPREAD_Commodity', 'irrigation'] + cols_sorted]
    c_ghg.rename(columns = {'SPREAD_ID': 'LU_ID', 
                              'SPREAD_Commodity': 'LU_DESC',
                              'irrigation': 'IRRIGATION', 
                              'kgco2e_fert_production': 'CO2E_KG_HA_FERT_PROD',
                              'kgCO2e_pest': 'CO2E_KG_HA_PEST_PROD',
                              'kgCO2e_irrig': 'CO2E_KG_HA_IRRIG',
                              'kgCO2e_chem_app': 'CO2E_KG_HA_CHEM_APPL', 
                              'kgCO2e_crop_management': 'CO2E_KG_HA_CROP_MGT', 
                              'kgCO2e_cult': 'CO2E_KG_HA_CULTIV', 
                              'kgCO2e_harvest': 'CO2E_KG_HA_HARVEST', 
                              'kgCO2e_sowing': 'CO2E_KG_HA_SOWING', 
                              'kgco2e_soil_N_applied': 'CO2E_KG_HA_SOIL'}, inplace = True)
    return c_ghg

# Calculate GHG emissions using old data or new - ***note new N2O emissions data is missing many rows***
c_ghg_with_STATE_ID = calc_crop_GHG_with_OLD_N2O(c_ghg)

# Convert LU_DESC to Sentence case
c_ghg_with_STATE_ID['LU_DESC'] = c_ghg_with_STATE_ID['LU_DESC'].str.capitalize()

# Downcast
downcast(c_ghg_with_STATE_ID)

# Drop STATE_ID
c_ghg = c_ghg_with_STATE_ID.drop(columns = 'STATE_ID')

# Save output to file
c_ghg.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_crop_GHG_data.h5', key = 'SA2_crop_GHG_data', mode = 'w', format = 't')



##################  Calculate LIVESTOCK EMISSIONS

# Read in the livestock emissions table and drop unwanted columns
ghg_ls = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20231113/T_GHG_by_SPREAD_SA2_livestock_2010_NLUM_per_head.csv')
ghg_ls.drop(columns = ['track', 'irrigation'], inplace = True)
ghg_ls.drop_duplicates(inplace = True, ignore_index = True)
ghg_ls.loc[ghg_ls.query('SPREAD_Commodity == "Dairy Cattle"').index, 'SPREAD_Commodity'] = 'DAIRY'
ghg_ls.loc[ghg_ls.query('SPREAD_Commodity == "Beef Cattle"').index, 'SPREAD_Commodity'] = 'BEEF'
ghg_ls.loc[ghg_ls.query('SPREAD_Commodity == "Sheep"').index, 'SPREAD_Commodity'] = 'SHEEP'

# Re-order and rename columns
ls_ghg = ghg_ls[['SA2_ID', 'SPREAD_Commodity', 'GHG_enteric_perHead', 'GHG_manure management_perHead', 'GHG_indirect leaching and runoff_perHead', 'GHG_dung and urine_perHead', 'GHG_Seed emissions_perHead', 'GHG_Fodder emissions_perHead', 'GHG_Fuel_perHead', 'GHG_Electricity_perHead']]
ls_ghg = ls_ghg.rename(columns = {'irrigation': 'IRRIGATION', 
                                  'GHG_enteric_perHead': 'CO2E_KG_HEAD_ENTERIC',
                                  'GHG_manure management_perHead': 'CO2E_KG_HEAD_MANURE_MGT',
                                  'GHG_indirect leaching and runoff_perHead': 'CO2E_KG_HEAD_IND_LEACH_RUNOFF',
                                  'GHG_dung and urine_perHead': 'CO2E_KG_HEAD_DUNG_URINE', 
                                  'GHG_Seed emissions_perHead': 'CO2E_KG_HEAD_SEED', 
                                  'GHG_Fodder emissions_perHead': 'CO2E_KG_HEAD_FODDER', 
                                  'GHG_Fuel_perHead': 'CO2E_KG_HEAD_FUEL', 
                                  'GHG_Electricity_perHead': 'CO2E_KG_HEAD_ELEC'
                                  })

# Calculate pivot table
lvstk_ghg_sources = ['CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_MANURE_MGT', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_SEED', 'CO2E_KG_HEAD_FODDER', 'CO2E_KG_HEAD_FUEL', 'CO2E_KG_HEAD_ELEC'].sort()
p_ls = pd.pivot_table(ls_ghg, 
                      values = lvstk_ghg_sources, 
                      index = 'SA2_ID', 
                      columns = 'SPREAD_Commodity', 
                      aggfunc = 'first'
                     ).sort_values(by = 'SA2_ID')

# Rearrange pivot table
p_ls.columns = p_ls.columns.rename('Component', level = 0)
p_ls = p_ls.reorder_levels(['SPREAD_Commodity', 'Component'], axis = 1)

# Flatten multiindex dataframe
p_ls.columns = p_ls.columns.to_flat_index()

"""Uncomment the following if you want to add GHG from livestock drinking water and pasture irrigation water supply.
   We have elected to omit these emissions sources and assume that livestock predominantly drink from natural waterbodies 
   or from gravity fed sources, leading to zero emissions. For irrigated pasture we have included emissions from winter cereals 
   production as an analogue.
"""
   
# # Calculate livestock emissions for drinking and irrigation water 
# CO2_KG_ML = 52.7145 # Value for Hay from Javi email 20/09/2021
# wr = ludf.query('LU_ID >= 31').groupby('SA2_ID', as_index = False)[['WR_IRR_DAIRY', 'WR_DRN_DAIRY', 'WR_IRR_BEEF', 'WR_DRN_BEEF', 'WR_IRR_SHEEP', 'WR_DRN_SHEEP']].first()
# wr.loc[:, 'CO2E_KG_HEAD_WATER_IRR_DAIRY'] = wr.eval('WR_IRR_DAIRY * @CO2_KG_ML') # ML/head * kg/ML = kg/head
# wr.loc[:, 'CO2E_KG_HEAD_WATER_DRN_DAIRY'] = wr.eval('WR_DRN_DAIRY * @CO2_KG_ML') # kg/head
# wr.loc[:, 'CO2E_KG_HEAD_WATER_IRR_BEEF'] = wr.eval('WR_IRR_BEEF * @CO2_KG_ML')
# wr.loc[:, 'CO2E_KG_HEAD_WATER_DRN_BEEF'] = wr.eval('WR_DRN_BEEF * @CO2_KG_ML')
# wr.loc[:, 'CO2E_KG_HEAD_WATER_IRR_SHEEP'] = wr.eval('WR_IRR_SHEEP * @CO2_KG_ML')
# wr.loc[:, 'CO2E_KG_HEAD_WATER_DRN_SHEEP'] = wr.eval('WR_DRN_SHEEP * @CO2_KG_ML')

# # Join water GHG emissions data to multi-index GHG dataframe
# p_ls = p_ls.merge(wr[['SA2_ID', 'CO2E_KG_HEAD_WATER_IRR_DAIRY', 'CO2E_KG_HEAD_WATER_DRN_DAIRY', 
#                                 'CO2E_KG_HEAD_WATER_IRR_BEEF', 'CO2E_KG_HEAD_WATER_DRN_BEEF', 
#                                 'CO2E_KG_HEAD_WATER_IRR_SHEEP', 'CO2E_KG_HEAD_WATER_DRN_SHEEP']], how = 'left', on = 'SA2_ID')

# # Remove SA2_ID from index
# p_ls.set_index('SA2_ID', drop = True, inplace = True)

# # Rename columns into multiindex format
# p_ls.rename(columns = {'CO2E_KG_HEAD_WATER_IRR_DAIRY': ('DAIRY', 'CO2E_KG_HEAD_IRR_WATER'), 'CO2E_KG_HEAD_WATER_DRN_DAIRY': ('DAIRY', 'CO2E_KG_HEAD_DRN_WATER'),
#                        'CO2E_KG_HEAD_WATER_IRR_BEEF': ('BEEF', 'CO2E_KG_HEAD_IRR_WATER'), 'CO2E_KG_HEAD_WATER_DRN_BEEF': ('BEEF', 'CO2E_KG_HEAD_DRN_WATER'),
#                        'CO2E_KG_HEAD_WATER_IRR_SHEEP': ('SHEEP', 'CO2E_KG_HEAD_IRR_WATER'), 'CO2E_KG_HEAD_WATER_DRN_SHEEP': ('SHEEP', 'CO2E_KG_HEAD_DRN_WATER')}, inplace = True)

# Merge livestock GHG to ludf dataframe by SA2 to check nodata
tmp = ludf.query('LU_ID >= 31').merge(p_ls, how = 'left', on = 'SA2_ID')
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])

# Recreate multiindex dataframe
p_ls.columns = pd.MultiIndex.from_tuples(p_ls.columns, names=['Livestock type','GHG Source'])

# Sort columns
p_ls.sort_index(axis = 1, level = 0, inplace = True)

# Downcast and save file
downcast(p_ls)
p_ls.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_livestock_GHG_data.h5', key = 'SA2_livestock_GHG_data', mode = 'w', format = 't')


# p_ls can then be joined to NLUM based on SA2_ID only. Values are per head so can be applied across livestock types on different MODL/NATL land and irrigation status.


# Calculate emissions from irrigated sown pastures as the same as from Winter cereals production (Hay has dodgy zero values for soil N20)
crop_ghg_sources = ['CO2E_KG_HA_CHEM_APPL', 
                    'CO2E_KG_HA_CROP_MGT', 
                    'CO2E_KG_HA_CULTIV', 
                    'CO2E_KG_HA_FERT_PROD', 
                    'CO2E_KG_HA_HARVEST', 
                    'CO2E_KG_HA_IRRIG', 
                    'CO2E_KG_HA_PEST_PROD', 
                    'CO2E_KG_HA_SOIL', 
                    'CO2E_KG_HA_SOWING']

# Rearrange the table structure
pivot_ghg = pd.pivot_table(c_ghg_with_STATE_ID, 
                      values = crop_ghg_sources, 
                      index = ['SA2_ID', 'SA4_ID', 'STATE_ID'],
                      columns = ['LU_DESC', 'IRRIGATION'],
                      aggfunc = 'first'
                     ).sort_values(by = 'SA2_ID')

# Select irrigated hay to represent emissions from irrigated sown pasture
irr_pasture_ghg = pivot_ghg.loc[:, (slice(None), 'Winter cereals', 1)]

# Flatten the MultiIndex to dataframe
irr_pasture_ghg = irr_pasture_ghg.droplevel(['IRRIGATION', 'LU_DESC'], axis = 1).reset_index(level = [0, 1, 2])

# Calculate and join summary tables by SA4 and STATE to fill in data gaps
tmp_SA4 = irr_pasture_ghg.groupby('SA4_ID', observed = True, as_index = False).agg(
                                    CO2E_KG_HA_CHEM_APPL_SA4 = ('CO2E_KG_HA_CHEM_APPL', 'mean'),
                                    CO2E_KG_HA_CROP_MGT_SA4  = ('CO2E_KG_HA_CROP_MGT', 'mean'),
                                    CO2E_KG_HA_CULTIV_SA4    = ('CO2E_KG_HA_CULTIV', 'mean'),
                                    CO2E_KG_HA_FERT_PROD_SA4 = ('CO2E_KG_HA_FERT_PROD', 'mean'),
                                    CO2E_KG_HA_HARVEST_SA4   = ('CO2E_KG_HA_HARVEST', 'mean'),
                                    CO2E_KG_HA_IRRIG_SA4     = ('CO2E_KG_HA_IRRIG', 'mean'),
                                    CO2E_KG_HA_PEST_PROD_SA4 = ('CO2E_KG_HA_PEST_PROD', 'mean'),
                                    CO2E_KG_HA_SOIL_SA4      = ('CO2E_KG_HA_SOIL', 'mean'),
                                    CO2E_KG_HA_SOWING_SA4    = ('CO2E_KG_HA_SOWING', 'mean'),
                                    )

tmp_STATE = irr_pasture_ghg.groupby('STATE_ID', observed = True, as_index = False).agg(
                                    CO2E_KG_HA_CHEM_APPL_STE = ('CO2E_KG_HA_CHEM_APPL', 'mean'),
                                    CO2E_KG_HA_CROP_MGT_STE  = ('CO2E_KG_HA_CROP_MGT', 'mean'),
                                    CO2E_KG_HA_CULTIV_STE    = ('CO2E_KG_HA_CULTIV', 'mean'),
                                    CO2E_KG_HA_FERT_PROD_STE = ('CO2E_KG_HA_FERT_PROD', 'mean'),
                                    CO2E_KG_HA_HARVEST_STE   = ('CO2E_KG_HA_HARVEST', 'mean'),
                                    CO2E_KG_HA_IRRIG_STE     = ('CO2E_KG_HA_IRRIG', 'mean'),
                                    CO2E_KG_HA_PEST_PROD_STE = ('CO2E_KG_HA_PEST_PROD', 'mean'),
                                    CO2E_KG_HA_SOIL_STE      = ('CO2E_KG_HA_SOIL', 'mean'),
                                    CO2E_KG_HA_SOWING_STE    = ('CO2E_KG_HA_SOWING', 'mean'),
                                    )

# Create dataframe with all SA2_ID, SA4_ID, and STE_ID
sa2_sa4_ste = ludf.groupby(['SA2_ID'], observed = True, as_index = False)[['SA4_ID', 'STE_ID']].first().sort_values(by = 'SA2_ID')

irr_pasture_ghg = sa2_sa4_ste.merge(irr_pasture_ghg.drop(columns = ['SA4_ID', 'STATE_ID']), how = 'left', on = 'SA2_ID')
irr_pasture_ghg = irr_pasture_ghg.merge(tmp_SA4, how = 'left', on = 'SA4_ID')
irr_pasture_ghg = irr_pasture_ghg.merge(tmp_STATE, how = 'left', left_on = 'STE_ID', right_on = 'STATE_ID')

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CHEM_APPL != CO2E_KG_HA_CHEM_APPL').index, 'CO2E_KG_HA_CHEM_APPL'] = irr_pasture_ghg['CO2E_KG_HA_CHEM_APPL_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CHEM_APPL != CO2E_KG_HA_CHEM_APPL').index, 'CO2E_KG_HA_CHEM_APPL'] = irr_pasture_ghg['CO2E_KG_HA_CHEM_APPL_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CHEM_APPL != CO2E_KG_HA_CHEM_APPL').index, 'CO2E_KG_HA_CHEM_APPL'] = irr_pasture_ghg['CO2E_KG_HA_CHEM_APPL'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CROP_MGT != CO2E_KG_HA_CROP_MGT').index, 'CO2E_KG_HA_CROP_MGT'] = irr_pasture_ghg['CO2E_KG_HA_CROP_MGT_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CROP_MGT != CO2E_KG_HA_CROP_MGT').index, 'CO2E_KG_HA_CROP_MGT'] = irr_pasture_ghg['CO2E_KG_HA_CROP_MGT_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CROP_MGT != CO2E_KG_HA_CROP_MGT').index, 'CO2E_KG_HA_CROP_MGT'] = irr_pasture_ghg['CO2E_KG_HA_CROP_MGT'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CULTIV != CO2E_KG_HA_CULTIV').index, 'CO2E_KG_HA_CULTIV'] = irr_pasture_ghg['CO2E_KG_HA_CULTIV_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CULTIV != CO2E_KG_HA_CULTIV').index, 'CO2E_KG_HA_CULTIV'] = irr_pasture_ghg['CO2E_KG_HA_CULTIV_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CULTIV != CO2E_KG_HA_CULTIV').index, 'CO2E_KG_HA_CULTIV'] = irr_pasture_ghg['CO2E_KG_HA_CULTIV'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_FERT_PROD != CO2E_KG_HA_FERT_PROD').index, 'CO2E_KG_HA_FERT_PROD'] = irr_pasture_ghg['CO2E_KG_HA_FERT_PROD_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_FERT_PROD != CO2E_KG_HA_FERT_PROD').index, 'CO2E_KG_HA_FERT_PROD'] = irr_pasture_ghg['CO2E_KG_HA_FERT_PROD_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_FERT_PROD != CO2E_KG_HA_FERT_PROD').index, 'CO2E_KG_HA_FERT_PROD'] = irr_pasture_ghg['CO2E_KG_HA_FERT_PROD'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_IRRIG != CO2E_KG_HA_IRRIG').index, 'CO2E_KG_HA_IRRIG'] = irr_pasture_ghg['CO2E_KG_HA_IRRIG_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_IRRIG != CO2E_KG_HA_IRRIG').index, 'CO2E_KG_HA_IRRIG'] = irr_pasture_ghg['CO2E_KG_HA_IRRIG_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_IRRIG != CO2E_KG_HA_IRRIG').index, 'CO2E_KG_HA_IRRIG'] = irr_pasture_ghg['CO2E_KG_HA_IRRIG'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_PEST_PROD != CO2E_KG_HA_PEST_PROD').index, 'CO2E_KG_HA_PEST_PROD'] = irr_pasture_ghg['CO2E_KG_HA_PEST_PROD_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_PEST_PROD != CO2E_KG_HA_PEST_PROD').index, 'CO2E_KG_HA_PEST_PROD'] = irr_pasture_ghg['CO2E_KG_HA_PEST_PROD_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_PEST_PROD != CO2E_KG_HA_PEST_PROD').index, 'CO2E_KG_HA_PEST_PROD'] = irr_pasture_ghg['CO2E_KG_HA_PEST_PROD'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOIL != CO2E_KG_HA_SOIL').index, 'CO2E_KG_HA_SOIL'] = irr_pasture_ghg['CO2E_KG_HA_SOIL_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOIL != CO2E_KG_HA_SOIL').index, 'CO2E_KG_HA_SOIL'] = irr_pasture_ghg['CO2E_KG_HA_SOIL_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOIL != CO2E_KG_HA_SOIL').index, 'CO2E_KG_HA_SOIL'] = irr_pasture_ghg['CO2E_KG_HA_SOIL'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOWING != CO2E_KG_HA_SOWING').index, 'CO2E_KG_HA_SOWING'] = irr_pasture_ghg['CO2E_KG_HA_SOWING_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOWING != CO2E_KG_HA_SOWING').index, 'CO2E_KG_HA_SOWING'] = irr_pasture_ghg['CO2E_KG_HA_SOWING_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOWING != CO2E_KG_HA_SOWING').index, 'CO2E_KG_HA_SOWING'] = irr_pasture_ghg['CO2E_KG_HA_SOWING'].mean()

# Set HARVEST emissions to zero becasue pasture is not harvested
irr_pasture_ghg['CO2E_KG_HA_HARVEST'] = 0

# Whittle down the columns 
irr_pasture_ghg = irr_pasture_ghg[['SA2_ID'] + crop_ghg_sources]


print('Number of NaNs =', irr_pasture_ghg[irr_pasture_ghg.isna().any(axis=1)].shape[0])

# Downcast and save to file
downcast(irr_pasture_ghg)
irr_pasture_ghg.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_irrigated_pasture_GHG_data.h5', key = 'SA2_irrigated_pasture_GHG_data', mode = 'w', format = 't')



################ Test 2010 livestock GHG data ################              NOTE - something is wrong with this code as it returns 83 MT CO2e

ludf_skinny = ludf_[['CELL_ID', 'CELL_HA', 'SA2_ID', 'LU_ID', 'IRRIGATION', 'SPREAD_id_mapped', 'YIELD_POT_DAIRY', 'YIELD_POT_BEEF', 'YIELD_POT_SHEEP']]

p_ls.columns = p_ls.columns.to_flat_index()
ls_t1 = ludf_skinny.merge(p_ls, how = 'left', on = 'SA2_ID')
ls_t2 = ls_t1.merge(irr_pasture_ghg, how = 'left', on = 'SA2_ID')

ls_t2['LVSTK_TCO2E'] = 0
ls_t2['IRRPAS_TCO2E'] = 0
ls_t2['TOTAL_TCO2E'] = 0

idx1 = ls_t2.query('SPREAD_id_mapped == 31').index 
ls_t2.loc[idx1, 'LVSTK_TCO2E'] = ls_t2['YIELD_POT_DAIRY'] * (ls_t2[('DAIRY', 'CO2E_KG_HEAD_ENTERIC')] + ls_t2[('DAIRY', 'CO2E_KG_HEAD_MANURE_MGT')] + ls_t2[('DAIRY', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF')] + ls_t2[('DAIRY', 'CO2E_KG_HEAD_DUNG_URINE')] + ls_t2[('DAIRY', 'CO2E_KG_HEAD_SEED')] + ls_t2[('DAIRY', 'CO2E_KG_HEAD_FODDER')] + ls_t2[('DAIRY', 'CO2E_KG_HEAD_FUEL')] + ls_t2[('DAIRY', 'CO2E_KG_HEAD_ELEC')]) / 1000
idx2 = ls_t2.query('SPREAD_id_mapped == 32').index 
ls_t2.loc[idx2, 'LVSTK_TCO2E'] = ls_t2['YIELD_POT_BEEF'] * (ls_t2[('BEEF', 'CO2E_KG_HEAD_ENTERIC')] + ls_t2[('BEEF', 'CO2E_KG_HEAD_MANURE_MGT')] + ls_t2[('BEEF', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF')] + ls_t2[('BEEF', 'CO2E_KG_HEAD_DUNG_URINE')] + ls_t2[('BEEF', 'CO2E_KG_HEAD_SEED')] + ls_t2[('BEEF', 'CO2E_KG_HEAD_FODDER')] + ls_t2[('BEEF', 'CO2E_KG_HEAD_FUEL')] + ls_t2[('BEEF', 'CO2E_KG_HEAD_ELEC')]) / 1000
idx3 = ls_t2.query('SPREAD_id_mapped == 33').index 
ls_t2.loc[idx3, 'LVSTK_TCO2E'] = ls_t2['YIELD_POT_SHEEP'] * (ls_t2[('SHEEP', 'CO2E_KG_HEAD_ENTERIC')] + ls_t2[('SHEEP', 'CO2E_KG_HEAD_MANURE_MGT')] + ls_t2[('SHEEP', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF')] + ls_t2[('SHEEP', 'CO2E_KG_HEAD_DUNG_URINE')] + ls_t2[('SHEEP', 'CO2E_KG_HEAD_SEED')] + ls_t2[('SHEEP', 'CO2E_KG_HEAD_FODDER')] + ls_t2[('SHEEP', 'CO2E_KG_HEAD_FUEL')] + ls_t2[('SHEEP', 'CO2E_KG_HEAD_ELEC')]) / 1000

ls_t2['LVSTK_TCO2E'] = ls_t2['LVSTK_TCO2E'] * cell_df['CELL_HA']

idx4 = ls_t2.query('IRRIGATION == 1').index 
ls_t2.loc[idx4, 'LVSTK_TCO2E'] = ls_t2.eval('LVSTK_TCO2E * 2')
ls_t2.loc[idx4, 'IRRPAS_TCO2E'] = ls_t2.eval('CO2E_KG_HA_CHEM_APPL + CO2E_KG_HA_CROP_MGT + CO2E_KG_HA_CULTIV + CO2E_KG_HA_FERT_PROD + CO2E_KG_HA_HARVEST + CO2E_KG_HA_IRRIG + CO2E_KG_HA_PEST_PROD + CO2E_KG_HA_SOIL + CO2E_KG_HA_SOWING') / 1000

ls_t2['IRRPAS_TCO2E'] = ls_t2.eval('IRRPAS_TCO2E * CELL_HA')


ls_t2['TOTAL_TCO2E'] = ls_t2.eval('LVSTK_TCO2E + IRRPAS_TCO2E')

print(ls_t2['LVSTK_TCO2E'].sum())
print(ls_t2['IRRPAS_TCO2E'].sum())
print(ls_t2['TOTAL_TCO2E'].sum())




############################################################################################################################################
# Assemble pesticide toxixity data CROPS and LIVESTOCK (modified land only i.e. cleared, sown pastures)
############################################################################################################################################

# Read in the tox table
tox_df = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20210630/T_toxicity_by_SPREAD_SA2_2010_NLUM.csv')
tox_df = tox_df.fillna(0)

# Increase the human toxicity numbers so that they are visible (otherwise they are very small numbers)
tox_df = tox_df.eval('CTUh_noncancer = CTUh_noncancer * 10**9')
tox_df = tox_df.eval('CTUh_cancer = CTUh_cancer * 10**9')

# Change column names
tox_df.rename(columns = {'CTUe': 'CTU_E',
                         'CTUh_noncancer': 'CTU_H_NONCANC_BN',
                         'CTUh_cancer': 'CTU_H_CANC_BN'
                        }, inplace = True)

# Create new ultimate truth table with SA4 and STE id fields so we can fill gaps
tmp_df = ludf.query('LU_ID >= 5').groupby(['SA2_ID', 'LU_ID', 'IRRIGATION'], as_index = False, observed = True)[['LU_DESC', 'SA4_ID', 'STE_ID']].first().sort_values(by = ['SA2_ID', 'LU_ID', 'IRRIGATION'])

# Add new field LU_JOIN for joining the tox data to the ultimate truth table
tmp_df.loc[tmp_df.query('5 <= LU_ID <= 25').index, 'LU_JOIN'] = tmp_df['LU_ID']
tmp_df.loc[tmp_df.query('LU_DESC.str.contains("modified land")', engine = 'python').index, 'LU_JOIN'] = 3

# Join to UT template and check for nodata
tmp_df = tmp_df.merge(tox_df.drop(columns = 'SA4_ID'), how = 'left', left_on = ['SA2_ID', 'LU_JOIN', 'IRRIGATION'], right_on = ['SA2_id', 'SPREAD_ID_original', 'irrigation'])

# Remove unwanted columns
tmp_df.drop(columns = ['SA2_id', 'SA2_name', 'STATE_ID', 'SPREAD_TXT', 'SPREAD_ID_original', 'irrigation', 'track'], inplace = True)

# Set grazing on natural land to zero (i.e. no pesticide aplication)
tmp_df.loc[tmp_df.query('LU_DESC.str.contains("natural land")', engine = 'python').index, ['LU_JOIN', 'CTU_E', 'CTU_H_NONCANC_BN', 'CTU_H_CANC_BN']] = 0

# Calculate how many NaNa
print('Number of NaNs =', tmp_df['CTU_E'].isna().sum())


# Calculate summary tables by SA4, STE and national to fill in gaps
tmp_df_lite = tmp_df.query('CTU_E > 0')

tmp_SA4 = tmp_df_lite.groupby(['SA4_ID', 'LU_JOIN', 'IRRIGATION'], observed = True, as_index = False).agg(CTU_E_MEAN_SA4 = ('CTU_E', 'mean'), 
                                                                                                          CTU_H_NONCANC_BN_MEAN_SA4 = ('CTU_H_NONCANC_BN', 'mean'),
                                                                                                          CTU_H_CANC_BN_MEAN_SA4 = ('CTU_H_CANC_BN', 'mean'))

tmp_STE = tmp_df_lite.groupby(['STE_ID', 'LU_JOIN', 'IRRIGATION'], observed = True, as_index = False).agg(CTU_E_MEAN_STE = ('CTU_E', 'mean'), 
                                                                                                          CTU_H_NONCANC_BN_MEAN_STE = ('CTU_H_NONCANC_BN', 'mean'),
                                                                                                          CTU_H_CANC_BN_MEAN_STE = ('CTU_H_CANC_BN', 'mean'))

tmp_AUS = tmp_df_lite.groupby(['LU_JOIN', 'IRRIGATION'], observed = True, as_index = False).agg(CTU_E_MEAN_AUS = ('CTU_E', 'mean'), 
                                                                                                          CTU_H_NONCANC_BN_MEAN_AUS = ('CTU_H_NONCANC_BN', 'mean'),
                                                                                                          CTU_H_CANC_BN_MEAN_AUS = ('CTU_H_CANC_BN', 'mean'))
# Merge the summary tables to the UT
tmp_df = tmp_df.merge(tmp_SA4, how = 'left', on = ['SA4_ID', 'LU_JOIN', 'IRRIGATION'])
tmp_df = tmp_df.merge(tmp_STE, how = 'left', on = ['STE_ID', 'LU_JOIN', 'IRRIGATION'])
tmp_df = tmp_df.merge(tmp_AUS, how = 'left', on = ['LU_JOIN', 'IRRIGATION'])

# Set grazing on natural land to zero (no pesticide aplication)
tmp_df.loc[tmp_df.query('LU_DESC.str.contains("natural land")', engine = 'python').index, ['CTU_E_MEAN_SA4', 'CTU_H_NONCANC_BN_MEAN_SA4', 'CTU_H_CANC_BN_MEAN_SA4', 'CTU_E_MEAN_STE', 'CTU_H_NONCANC_BN_MEAN_STE', 'CTU_H_CANC_BN_MEAN_STE']] = 0

# Fill NaNs from SA4 average
idx = tmp_df.query('CTU_E != CTU_E').index 
tmp_df.loc[idx, 'CTU_E'] = tmp_df['CTU_E_MEAN_SA4']
tmp_df.loc[idx, 'CTU_H_NONCANC_BN'] = tmp_df['CTU_H_NONCANC_BN_MEAN_SA4']
tmp_df.loc[idx, 'CTU_H_CANC_BN'] = tmp_df['CTU_H_CANC_BN_MEAN_SA4']

# Fill NaNs from state average
idx = tmp_df.query('CTU_E != CTU_E').index 
tmp_df.loc[idx, 'CTU_E'] = tmp_df['CTU_E_MEAN_STE']
tmp_df.loc[idx, 'CTU_H_NONCANC_BN'] = tmp_df['CTU_H_NONCANC_BN_MEAN_STE']
tmp_df.loc[idx, 'CTU_H_CANC_BN'] = tmp_df['CTU_H_CANC_BN_MEAN_STE']

# Fill NaNs from national average
idx = tmp_df.query('CTU_E != CTU_E').index 
tmp_df.loc[idx, 'CTU_E'] = tmp_df['CTU_E_MEAN_AUS']
tmp_df.loc[idx, 'CTU_H_NONCANC_BN'] = tmp_df['CTU_H_NONCANC_BN_MEAN_AUS']
tmp_df.loc[idx, 'CTU_H_CANC_BN'] = tmp_df['CTU_H_CANC_BN_MEAN_AUS']

# Calculate how many NaNa
print('Number of NaNs =', tmp_df['CTU_E'].isna().sum())

# Remove unnecessary columns
tmp_df = tmp_df[['SA2_ID', 'LU_ID', 'LU_DESC', 'IRRIGATION', 'CTU_E', 'CTU_H_NONCANC_BN', 'CTU_H_CANC_BN']]

# Downcast and save file
downcast(tmp_df)
tmp_df.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_crop_livestock_toxicity_data.h5', key = 'SA2_crop_livestock_toxicity_data', mode = 'w', format = 't')




