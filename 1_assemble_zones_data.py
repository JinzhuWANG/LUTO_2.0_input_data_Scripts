import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy import ndimage as nd
import numpy.ma as ma
import rasterio, matplotlib
from rasterio import features
from rasterio.warp import reproject
import lidario as lio

# Set some options
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', 100)
pd.set_option('display.min_rows', 100)
pd.set_option('display.float_format', '{:,.2f}'.format)

infile = 'N:/Data-Master/National_Landuse_Map/NLUM_2010-11_clip.tif'
outgpkg = 'N:/Data-Master/LUTO_2.0_input_data/spatial_data.gpkg'

cell_df_path = 'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/'
cell_df_fn = 'cell_zones_df'

# Read cell_df from disk
# cell_df = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5')


################################ Create some helper data and functions

# Open NLUM_ID as mask raster and get metadata
with rasterio.open(infile) as rst:
    
    # Load a 2D masked array with nodata masked out
    NLUM_ID_raster = rst.read(1, masked=True) 
    
    # Create a 0/1 binary mask
    NLUM_mask = NLUM_ID_raster.mask == False
    
    # Get metadata and update parameters
    NLUM_transform = rst.transform
    NLUM_crs = rst.crs
    meta = rst.meta.copy()
    meta.update(compress='lzw', driver='GTiff') # , dtype='int32', nodata='0')
    [meta.pop(key) for key in ['dtype', 'nodata']] # Need to add dtype and nodata manually when exporting GeoTiffs
    
    # Set up some data structures to enable conversion on 1D arrays to 2D
    array_2D = np.zeros(NLUM_ID_raster.shape)
    xy = np.nonzero(NLUM_mask)
    
    
# Convert 1D column to 2D spatial array
def conv_1D_to_2D(in_1D_array):
    array_2D[xy] = np.array(in_1D_array)
    array_2D[array_2D == 0] = np.nan
    return array_2D


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
    int64_cols = dframe.select_dtypes(include = ['int64']).columns
    dframe[int64_cols] = dframe[int64_cols].apply(pd.to_numeric, downcast = 'integer')





################################ Read NLUM raster, convert to vector GeoDataFrame, join NLUM table, and save to Geopackage

# Collect raster zones as rasterio shape features
results = ({'properties': {'NLUM_ID': v}, 'geometry': s}
for i, (s, v) in enumerate(features.shapes(NLUM_ID_raster, mask = None, transform = NLUM_transform)))

# Convert rasterio shape features to GeoDataFrame
gdfp = gpd.GeoDataFrame.from_features(list(results), crs = NLUM_crs)

# Hack to fix error 'ValueError: No Shapely geometry can be created from null value'
gdfp['geometry'] = gdfp.geometry.buffer(0)

# Dissolve boundaries and convert to multipart shapes
gdfp = gdfp.dissolve(by = 'NLUM_ID')

# Load in NLUM tabular data
NLUM_table = pd.read_csv('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_clip.tif.csv').drop(columns = ['Rowid'])

# Calculate new fields
NLUM_table['SPREAD_ID'] = NLUM_table['COMMODITIES'] 
NLUM_table['SPREAD_DESC'] = NLUM_table['COMMODITIES_DESC'] 

# Change Other non-cereal crops to SPREAD class #15
NLUM_table.loc[NLUM_table['COMMODITIES_DESC'].str.contains('non-cereal'), 'SPREAD_ID'] = 15

SPREAD_dict = {-1: 'Non-agricultural land',
                0: 'Unallocated agricultural land',
                1: 'Grazing - native pasture',
                2: 'Grazing - woodland, open forest',
                3: 'Grazing - sown pastures',
                4: 'Agroforestry',
                5: 'Winter cereals',
                6: 'Summer cereals',
                7: 'Rice',
                8: 'Winter legumes',
                9: 'Summer legumes',
                10: 'Winter oilseeds',
                11: 'Summer oilseeds',
                12: 'Sugar',
                13: 'Hay',
                14: 'Cotton',
                15: 'Other non-cereal crops',
                16: 'Vegetables',
                17: 'Citrus',
                18: 'Apples',
                19: 'Pears',
                20: 'Stone fruit',
                21: 'Tropical stone fruit',
                22: 'Nuts',
                24: 'Plantation fruit',
                25: 'Grapes'}

# Populate new column SPREAD_DESC with new SPREAD names
for key, value in SPREAD_dict.items():
    NLUM_table.loc[NLUM_table['SPREAD_ID'] == key, 'SPREAD_DESC'] = value

# Join the table to the GeoDataFrame
gdfp = gdfp.merge(NLUM_table, left_on = 'NLUM_ID', right_on = 'VALUE', how = 'left')

# Save NLUM data as GeoPackage
gdfp.to_file('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_clip.gpkg', layer = 'NLUM_2010-11_clip', driver = 'GPKG')

# Save to file
NLUM_table.to_csv('N:/Data-Master/National_Landuse_Map/NLUM_table.csv')

# Some rows will be NaNs because there are internal polygons
print('Number of NULL cells =', gdfp[gdfp.isna().any(axis = 1)].shape[0])






################################ Spit out cell_df dataframe based on raster NLUM (values of 1 - 5125) of X, Y, Z of length nCells

# Instantiate a Lidario Translator object which takes a geotiff file and returns an X, Y, Z dataframe
cell_df = lio.Translator('geotiff', 'dataframe').translate(infile)
cell_df.columns = ['X', 'Y', 'NLUM_ID']
cell_df['CELL_ID'] = cell_df.index + 1

# Downcast to save memory
cell_df['X'] = np.round(cell_df['X'], 2).astype(np.float32)
cell_df['Y'] = np.round(cell_df['Y'], 2).astype(np.float32)
cell_df['NLUM_ID'] = pd.to_numeric(np.round(cell_df['NLUM_ID']), downcast = 'integer')

# Create REALAREA grid, convert to vector, project to Australian Albers Equal Area, join to cell_df

# Convert CELL_ID to 2D array
m2D = conv_1D_to_2D(cell_df['CELL_ID']).astype(np.int32)

# Collect raster zones as rasterio shape features
results = ({'properties': {'CELL_ID': v}, 'geometry': s} for i, (s, v) in enumerate(features.shapes(m2D, mask = None, transform = NLUM_transform)))
 
# Convert rasterio shape features to GeoDataFrame
rnd_gdf = gpd.GeoDataFrame.from_features(list(results), crs = NLUM_crs)

# Project to Australian Albers
rnd_gdf = rnd_gdf.to_crs('EPSG:3577')

# Calculate the area of each cell in hectares
rnd_gdf['CELL_HA'] = rnd_gdf['geometry'].area / 10000

# Save as GeoPackage
rnd_gdf.to_file(outgpkg, layer = 'CELL_ID', driver = 'GPKG')

# Join Cell_Ha to the cell_df data frame
cell_df = cell_df.merge(rnd_gdf[['CELL_ID', 'CELL_HA']], how = 'left', on = 'CELL_ID')

# Downcast to Float32
cell_df['CELL_HA'] = cell_df['CELL_HA'].astype(np.float32)

# Rearrange columns
cell_df = cell_df[['CELL_ID', 'X', 'Y', 'CELL_HA', 'NLUM_ID']]

# Spit out GoeTiff of CELL_HA
# Convert to 2D array
cell_ha = array_2D.astype(np.float32)
cell_ha[xy] = np.array(cell_df['CELL_HA'])

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_cell_ha.tif', 'w+', dtype = 'float32', nodata = 0, **meta) as out:
    out.write_band(1, cell_ha)

# Spit out GoeTiff of CELL_ID
# Convert to 2D array
cell_id = array_2D.astype(np.float32)
cell_id[xy] = np.array(cell_df['CELL_ID'])

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_cell_id.tif', 'w+', dtype = 'float32', nodata = 0, **meta) as out:
    out.write_band(1, cell_id)
    
# Print dataframe info, check for NAs
cell_df.info()
print('Number of NULL cells =', cell_df[cell_df.isna().any(axis = 1)].shape[0])





################################ Read SA2 shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df

# Import shapefile to GeoPandas DataFrame
SA2_gdf = gpd.read_file('N:/Data-Master/Profit_map/SA2_boundaries_2011/SA2_2011_AUST.shp')

# Remove rows without geometry and reset index
SA2_gdf = SA2_gdf[SA2_gdf['geometry'].notna()].reset_index()

# Convert string column to int32 for conversion to raster
SA2_gdf['SA2_MAIN11'] = SA2_gdf['SA2_MAIN11'].astype(np.int32)

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(SA2_gdf.geometry, SA2_gdf.SA2_MAIN11)) 

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/Profit_map/SA2_boundaries_2011/SA2_raster_filled.tif', 'w+', dtype = 'int32', nodata = 0, **meta) as out:
    
    # Rasterise SA2 shapefile. Note 6 SA2s disappear during rasterisation.
    newrast = features.rasterize(shapes = shapes, fill = 0, out = out.read(1), transform = out.transform)
    
    # Find cells to fill
    msk = (newrast == 0)

    # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(msk, return_distances = False, return_indices = True)
    SA2_raster_filled = newrast[tuple(ind)]
    SA2_raster_clipped = SA2_raster_filled * NLUM_mask
    
    # Save output to GeoTiff
    out.write_band(1, SA2_raster_clipped)

# Flatten the SA2 2D array to 1D array of valid values and add to cell_df dataframe
cell_df['SA2_ID'] = SA2_raster_clipped[NLUM_mask]






################################ Create unique ID as combination of NLUM_ID and SA2_ID and convert to NLUM_SA2_gdf polygon, dissolve and save
   
diss_df = cell_df.groupby(['NLUM_ID','SA2_ID'], as_index = False).size().drop(columns = ['size'])
diss_df['NLUM_SA2_ID'] = diss_df.index + 1
cell_df = cell_df.merge(diss_df, how='left', left_on=['NLUM_ID','SA2_ID'], right_on=['NLUM_ID','SA2_ID'])
    
# Rearrange columns
cell_df = cell_df[['CELL_ID', 'X', 'Y', 'CELL_HA', 'NLUM_ID', 'NLUM_SA2_ID', 'SA2_ID']]

# Convert 1D column to 2D spatial array
arr_2D = conv_1D_to_2D(cell_df['NLUM_SA2_ID'])

# Collect raster zones as rasterio shape features
NLUM_SA2_shapes = ({'properties': {'NLUM_SA2_ID': v}, 'geometry': s} for i, (s, v) in enumerate(features.shapes(arr_2D.astype(np.int32), mask = None, transform = NLUM_transform)))
 
# Convert rasterio shape features to GeoDataFrame
NLUM_SA2_gdf = gpd.GeoDataFrame.from_features(list(NLUM_SA2_shapes))

# Hack to fix error 'ValueError: No Shapely geometry can be created from null value'
NLUM_SA2_gdf['geometry'] = NLUM_SA2_gdf.geometry.buffer(0)

# Dissolve boundaries and convert to multipart shapes
NLUM_SA2_gdf = NLUM_SA2_gdf.dissolve(by='NLUM_SA2_ID')

# Join NLUM_ID and SA2_ID to the GeoDataFrame
NLUM_SA2_gdf = NLUM_SA2_gdf.merge(diss_df, how = 'left', left_on = ['NLUM_SA2_ID'], right_on = ['NLUM_SA2_ID'])

# Load in NLUM tabular data
NLUM_table_short = NLUM_table[['VALUE', 'SPREAD_ID', 'SPREAD_DESC', 'IRRIGATION']]

# Join the COMMODITY and IRRIGATION information to the GeoDataFrame
NLUM_SA2_gdf = NLUM_SA2_gdf.merge(NLUM_table_short, left_on = 'NLUM_ID', right_on = 'VALUE', how = 'left')
NLUM_SA2_gdf = NLUM_SA2_gdf.drop(columns = ['VALUE'])
 
# Save NLUM data as GeoPackage
NLUM_SA2_gdf.to_file(outgpkg, layer = 'NLUM_SA2_gdf', driver = 'GPKG')

# Check that there are the same number of polygons (+1 outer polygon) as there are in the groupby operation
# diss_df = cell_df.groupby(['NLUM_ID','SA2_ID']).size().reset_index().rename(columns = {0: 'count'})






################################ Join SA2 table to the cell_df dataframe and optimise datatypes
    
# Join the SA2 GeoDataFrame table SA2_gdf to the cell_df dataframe
cell_df = cell_df.merge(SA2_gdf, left_on = 'SA2_ID', right_on = 'SA2_MAIN11', how = 'left')
cell_df = cell_df.drop(columns=['geometry', 'ALBERS_SQM', 'index', 'SA2_MAIN11', 'SA2_5DIG11'])

# Convert string (object) columns to integer and downcast
obj_cols = ['SA3_CODE11', 'SA4_CODE11', 'STE_CODE11']
cell_df[obj_cols] = cell_df[obj_cols].apply(pd.to_numeric, downcast = 'integer')

# Downcast int64 columns and convert object to category to save memory and space
downcast(cell_df)

# Check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])






################################ Read ABARES regions shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe

# Import shapefile to GeoPandas DataFrame
ABARES_gdf = gpd.read_file('N:/Data-Master/Australian_administrative_boundaries/abare_2018_aus/ABARE_boundaries.shp')

# Downcast CODE for conversion to raster
ABARES_gdf['AAGIS'] = pd.to_numeric(ABARES_gdf['AAGIS'], downcast = 'integer')

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(ABARES_gdf.geometry, ABARES_gdf.AAGIS)) 

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/Australian_administrative_boundaries/abare_2018_aus/ABARES_raster_filled.tif', 'w+', dtype = 'int32', nodata = 0, **meta) as out:
    
    # Rasterise NRM shapefile
    newrast = features.rasterize(shapes = shapes, fill = 0, out = out.read(1), transform = out.transform)
    
    # Find cells to fill
    msk = (newrast == 0)

    # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(msk, return_distances = False, return_indices = True)
    raster_filled = newrast[tuple(ind)]
    ABARES_raster_clipped = raster_filled * NLUM_mask
    
    # Save output to GeoTiff
    out.write_band(1, ABARES_raster_clipped)

# Flatten the NRM 2D array to 1D array of valid values only, and add SA2_ID to cell_df dataframe
cell_df['ABARES_AAGIS'] = ABARES_raster_clipped[NLUM_mask]
cell_df['ABARES_AAGIS'] = pd.to_numeric(cell_df['ABARES_AAGIS'], downcast='integer')

# Plot and print out data, check that there are no NaNs
map_in_2D(cell_df['ABARES_AAGIS'], data = 'categorical')
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])






################################ Read LGA shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe

# Import shapefile to GeoPandas DataFrame
LGA_gdf = gpd.read_file('N:/Data-Master/Australian_administrative_boundaries/lga_2010_aus/LGA10aAust.shp')

# Downcast LGA_CODE10 for conversion to raster
LGA_gdf['LGA_CODE10'] = pd.to_numeric(LGA_gdf['LGA_CODE10'], downcast = 'integer')

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(LGA_gdf.geometry, LGA_gdf.LGA_CODE10)) 

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/Australian_administrative_boundaries/lga_2010_aus/LGA_raster_filled.tif', 'w+', dtype = 'int32', nodata = 0, **meta) as out:
    
    # Rasterise shapefile
    newrast = features.rasterize(shapes = shapes, fill = 0, out = out.read(1), transform = out.transform)
    
    # Find cells to fill
    msk = (newrast == 0)

    # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(msk, return_distances = False, return_indices = True)
    raster_filled = newrast[tuple(ind)]
    LGA_raster_clipped = raster_filled * NLUM_mask
    
    # Save output to GeoTiff
    out.write_band(1, LGA_raster_clipped)

# Flatten the NRM 2D array to 1D array of valid values only, add SA2_ID to cell_df dataframe
cell_df['LGA_CODE'] = LGA_raster_clipped[NLUM_mask]
cell_df['LGA_CODE'] = pd.to_numeric(cell_df['LGA_CODE'], downcast = 'integer')

# Join LGA name to the cell_df data frame
cell_df = cell_df.merge(LGA_gdf, how = 'left', left_on = 'LGA_CODE', right_on = 'LGA_CODE10')
cell_df = cell_df.drop(columns = ['STATE_CODE', 'LGA_CODE10', 'geometry'])

# Plot and print out data, check that there are no NaNs
map_in_2D(cell_df['LGA_CODE'], data = 'categorical')
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])






################################ Read NRM regions shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
   
# Import shapefile to GeoPandas DataFrame
NRM_gdf = gpd.read_file('N:/Data-Master/Australian_administrative_boundaries/nrm_2016_aus/nrm_gda94.shp')

# Fix up an attribute error in the data
rw = NRM_gdf['CODE'] == 304310
NRM_gdf.loc[rw, 'NHT2NAME'] = 'Northern Gulf'
NRM_gdf.loc[rw, 'CODE'] = 310

# Downcast CODE for conversion to raster
NRM_gdf['CODE'] = pd.to_numeric(NRM_gdf['CODE'], downcast='integer')

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(NRM_gdf.geometry, NRM_gdf.CODE)) 

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/Australian_administrative_boundaries/nrm_2016_aus/NRM_raster_filled.tif', 'w+', dtype='int32', nodata='0', **meta) as out:
    
    # Rasterise NRM shapefile
    newrast = features.rasterize(shapes = shapes, fill = 0, out = out.read(1), transform = out.transform)
    
    # Find cells to fill
    msk = (newrast == 0)

    # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(msk, return_distances = False, return_indices = True)
    raster_filled = newrast[tuple(ind)]
    NRM_raster_clipped = raster_filled * NLUM_mask
    
    # Save output to GeoTiff
    out.write_band(1, NRM_raster_clipped)

print('Number of NRM regions in rasterized layer =', np.unique(NRM_raster_clipped).shape[0])

# Flatten the NRM 2D array to 1D array of valid values only, add NRM_CODE to cell_df dataframe
cell_df['NRM_CODE'] = NRM_raster_clipped[NLUM_mask]
cell_df['NRM_CODE'] = pd.to_numeric(cell_df['NRM_CODE'], downcast='integer')

# Simplify the table for merging
tmp = NRM_gdf.groupby(['CODE'], as_index = False)[['NHT2NAME']].first().sort_values(by = ['CODE'])

# Join NRM name to the cell_df data frame
cell_df = cell_df.merge(tmp, how = 'left', left_on = 'NRM_CODE', right_on = 'CODE')
cell_df = cell_df.drop(columns = ['CODE'])
cell_df.rename(columns = {'NHT2NAME':'NRM_NAME'}, inplace = True)

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])





################################ Read IBRA shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
   
# Import shapefile to GeoPandas DataFrame
IBRA_gdf = gpd.read_file('N:/Data-Master/Australian_administrative_boundaries/ibra7_2019_aus/ibra7_subregions.shp')

# Downcast for conversion to raster
IBRA_gdf['REC_ID'] = pd.to_numeric(IBRA_gdf['REC_ID'], downcast = 'integer')

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(IBRA_gdf.geometry, IBRA_gdf.REC_ID)) 

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/Australian_administrative_boundaries/ibra7_2019_aus/IBRA_raster_filled.tif', 'w+', dtype='int32', nodata='0', **meta) as out:
    
    # Rasterise NRM shapefile
    newrast = features.rasterize(shapes = shapes, fill = 0, out = out.read(1), transform = out.transform)
    
    # Find cells to fill
    msk = (newrast == 0)

    # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(msk, return_distances = False, return_indices = True)
    raster_filled = newrast[tuple(ind)]
    IBRA_raster_clipped = raster_filled * NLUM_mask
    
    # Save output to GeoTiff
    out.write_band(1, IBRA_raster_clipped)

# Flatten the NRM 2D array to 1D array of valid values only, add SA2_ID to cell_df dataframe
cell_df['IBRA_ID'] = IBRA_raster_clipped[NLUM_mask]
cell_df['IBRA_ID'] = pd.to_numeric(cell_df['IBRA_ID'], downcast = 'integer')

# Simplify the table for merging
tmp = IBRA_gdf.groupby(['REC_ID'], as_index = False)[['SUB_CODE_7', 'SUB_NAME_7', 'REG_CODE_7', 'REG_NAME_7']].first().sort_values(by = ['REC_ID'])

# Join LGA name to the cell_df data frame
cell_df = cell_df.merge(tmp, how = 'left', left_on = 'IBRA_ID', right_on = 'REC_ID')
cell_df = cell_df.drop(columns = ['REC_ID'])
cell_df.rename(columns={'SUB_CODE_7': 'IBRA_SUB_CODE_7', 
                        'SUB_NAME_7': 'IBRA_SUB_NAME_7', 
                        'REG_CODE_7': 'IBRA_REG_CODE_7', 
                        'REG_NAME_7': 'IBRA_REG_NAME_7'}, inplace = True)

# Downcast int64 columns and convert object to category to save memory and space
downcast(cell_df)

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])






################################ Read BOM GeoFabric HR Regions River Regions Geodatabase file in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
    
# Read file from File Geodatabase
HR_RR_gdf = gpd.read_file('N:/Data-Master/Water/GeoFabric_V3.2/HR_Regions_GDB_V3_2/HR_Regions_GDB/HR_Regions.gdb', driver = 'FileGDB', layer = 'RiverRegion')

# Simplify the table for merging, create new index
tmp = HR_RR_gdf.groupby(['RivRegName'], as_index = False)[['RivRegName']].first().sort_values(by = ['RivRegName'])
tmp['HR_RIVREG_ID'] = tmp.index + 1
HR_RR_gdf = HR_RR_gdf.merge(tmp, how = 'left', on = 'RivRegName')

# Downcast CODE for conversion to raster
HR_RR_gdf['HR_RIVREG_ID'] = pd.to_numeric(HR_RR_gdf['HR_RIVREG_ID'], downcast = 'integer')

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(HR_RR_gdf.geometry, HR_RR_gdf.HR_RIVREG_ID)) 

# Open a new GeoTiFF file
outfile = 'N:/Data-Master/Water/GeoFabric_V3.2/HR_Regions_GDB_V3_2/HR_Regions_GDB/HR_RivReg_raster_filled.tif'
with rasterio.open(outfile, 'w+', dtype = 'int32', nodata = '0', **meta) as out:
    
    # Rasterise NRM shapefile
    newrast = features.rasterize(shapes = shapes, fill = 0, out = out.read(1), transform = out.transform)
    
    # Find cells to fill
    msk = (newrast == 0)

    # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(msk, return_distances = False, return_indices = True)
    raster_filled = newrast[tuple(ind)]
    RR_raster_clipped = raster_filled * NLUM_mask
    
    # Save output to GeoTiff
    out.write_band(1, RR_raster_clipped)

# Flatten the NRM 2D array to 1D array of valid values only, add NRM_CODE to cell_df dataframe
cell_df['HR_RIVREG_ID'] = RR_raster_clipped[NLUM_mask]
cell_df['HR_RIVREG_ID'] = pd.to_numeric(cell_df['HR_RIVREG_ID'], downcast = 'integer')

# Join NRM name to the cell_df data frame
cell_df = cell_df.merge(tmp, how = 'left', on = 'HR_RIVREG_ID')
cell_df.rename(columns = {'RivRegName':'HR_RIVREG_NAME'}, inplace = True)

# Downcast int64 columns and convert object to category to save memory and space
downcast(cell_df)

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])






################################ Read BOM GeoFabric HR Regions AWRA Drainage Divisions Geodatabase file in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
    
# Read file from File Geodatabase
ESRI_gdb = 'N:/Data-Master/Water/GeoFabric_V3.2/HR_Regions_GDB_V3_2/HR_Regions_GDB/HR_Regions.gdb'
HR_DD_gdf = gpd.read_file(ESRI_gdb, driver='FileGDB', layer='AWRADrainageDivision')

# Simplify the table for merging, create new index
HR_DD_gdf['HR_DRAINDIV_ID'] = HR_DD_gdf.index + 1

# Downcast CODE for conversion to raster
HR_DD_gdf['HR_DRAINDIV_ID'] = pd.to_numeric(HR_DD_gdf['HR_DRAINDIV_ID'], downcast='integer')

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(HR_DD_gdf.geometry, HR_DD_gdf.HR_DRAINDIV_ID)) 

# Open a new GeoTiFF file
outfile = 'N:/Data-Master/Water/GeoFabric_V3.2/HR_Regions_GDB_V3_2/HR_Regions_GDB/HR_DrainDiv_raster_filled.tif'
with rasterio.open(outfile, 'w+', dtype='int32', nodata='0', **meta) as out:
    
    # Rasterise NRM shapefile
    newrast = features.rasterize(shapes = shapes, fill = 0, out = out.read(1), transform = out.transform)
    
    # Find cells to fill
    msk = (newrast == 0)

    # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(msk, return_distances = False, return_indices = True)
    raster_filled = newrast[tuple(ind)]
    DD_raster_clipped = raster_filled * NLUM_mask
    
    # Save output to GeoTiff
    out.write_band(1, DD_raster_clipped)

# Flatten the NRM 2D array to 1D array of valid values only, add NRM_CODE to cell_df dataframe
cell_df['HR_DRAINDIV_ID'] = DD_raster_clipped[NLUM_mask]
cell_df['HR_DRAINDIV_ID'] = pd.to_numeric(cell_df['HR_DRAINDIV_ID'], downcast='integer')

# Join DD name to the cell_df data frame
cell_df = cell_df.merge(HR_DD_gdf[['HR_DRAINDIV_ID', 'Division']], how='left', on='HR_DRAINDIV_ID')
cell_df.rename(columns = {'Division':'HR_DRAINDIV_NAME'}, inplace = True)

# Select rows for modification and calculate values for selected rows
index = cell_df.query('HR_DRAINDIV_NAME == "South East Coast (VICTORIA)"').index
cell_df.loc[index, 'HR_DRAINDIV_NAME'] = 'South East Coast (Victoria)'
index = cell_df.query('HR_DRAINDIV_NAME == "Tanami -Timor Sea Coast"').index
cell_df.loc[index, 'HR_DRAINDIV_NAME'] = 'Tanami-Timor Sea Coast'

# Downcast int64 columns and convert object to category to save memory and space
downcast(cell_df)

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])






################################ Read Aqueduct 3.0 Baseline Water Stress file in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
   
# Read file from GeoPackage
AD_gdf = gpd.read_file('N:/Data-Master/Water/Aqueduct_3.0_V01/baseline/annual/y2019m07d11_aqueduct30_annual_v01.gpkg', layer = 'y2019m07d11_aqueduct30_annual_v01')

AD_gdf = AD_gdf[['geometry', 'bws_cat', 'bws_label']]
AD_gdf['bws_cat'] = AD_gdf['bws_cat'] + 2
AD_gdf.loc[AD_gdf['bws_cat'].isna(), 'bws_cat'] = 0
AD_gdf['bws_cat'] = pd.to_numeric(np.round(AD_gdf['bws_cat']), downcast = 'integer')
AD_gdf.info()

# # Simplify the table for merging
AD_df = AD_gdf[['bws_cat', 'bws_label']]
AD_df = AD_df.groupby(['bws_cat'], as_index = False)[['bws_cat', 'bws_label']].first().sort_values(by = ['bws_cat'])
                            
# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(AD_gdf.geometry, AD_gdf.bws_cat)) 

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/Water/Aqueduct_3.0_V01/baseline/annual/AD_BWS_CAT_raster_filled.tif', 'w+', dtype = 'int32', nodata = 0, **meta) as out:
    
    # Rasterise NRM shapefile
    newrast = features.rasterize(shapes=shapes, fill = -99, out = out.read(1), transform = out.transform)
    
    # Find cells to fill
    msk = (newrast == 0)

    # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(msk, return_distances = False, return_indices = True)
    raster_filled = newrast[tuple(ind)]
    AD_raster_clipped = raster_filled * NLUM_mask
    
    # Save output to GeoTiff
    out.write_band(1, AD_raster_clipped)

# Flatten the 2D array to 1D array of valid values only, add NRM_CODE to cell_df dataframe
cell_df['AD_BWS_CAT'] = AD_raster_clipped[NLUM_mask]
cell_df['AD_BWS_CAT'] = pd.to_numeric(cell_df['AD_BWS_CAT'], downcast = 'integer')

# Join  to the cell_df data frame
cell_df = cell_df.merge(AD_df, how = 'left', left_on = 'AD_BWS_CAT', right_on = 'bws_cat')
cell_df = cell_df.drop(columns = ['bws_cat'])
cell_df.rename(columns = {'bws_label':'AD_BWS_LABEL'}, inplace = True)

# Downcast int64 columns and convert object to category to save memory and space
downcast(cell_df)

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])






############## Water stressed catchments (C & D) from National Water Commission (2012) Assessing water stress in Australian catchments and aquifers.
    
# Import shapefile to GeoPandas DataFrame
gdf = gpd.read_file('N:/Data-Master/Water/Water_stressed_catchments/basinStressCode.shp')

# Convert column data type for conversion to raster
gdf['STRESS_COD'] = gdf['STRESS_COD'].astype(np.uint8)

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.STRESS_COD))

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/Water/Water_stressed_catchments/WATER_STRESS_NWC.tif', 'w+', dtype = 'uint8', nodata = 255, **meta) as dst:
    
    # Rasterise shapefile
    newrast = features.rasterize(shapes=shapes, fill = 255, out = dst.read(1), transform = dst.transform)
    
    # Mask out nodata cells
    dst_array = ma.masked_where(newrast == 255, newrast)
    
    # Fill nodata in raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(dst_array.mask, return_distances = False, return_indices = True)
    raster_filled = dst_array[tuple(ind)]
    raster_clipped = np.where(NLUM_mask == 0, 255, raster_filled)
    
    # Save output to GeoTiff
    dst.write_band(1, raster_clipped)

# Flatten the 2D array to 1D array of valid values only, add data to cell_df dataframe
cell_df['WATER_STRESS_COD'] = raster_clipped[NLUM_mask == 1]

# Create a mini table to join water stress code
mt = gdf.groupby(['STRESS_COD'])[['STRESS_CLA']].first()

# Join the table to the dataframe and drop uneccesary columns
cell_df = cell_df.merge(mt, left_on='WATER_STRESS_COD', right_on='STRESS_COD', how='left')

cell_df.rename(columns = {'WATER_STRESS_COD':'WATER_STRESS_CODE', 'STRESS_CLA':'WATER_STRESS_CLASS'}, inplace = True)

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])





################################ Join NLUM land use table to the cell_df dataframe and optimise datatypes

# Join the table to the dataframe and drop uneccesary columns
cell_df = cell_df.merge(NLUM_table, left_on = 'NLUM_ID', right_on = 'VALUE', how = 'left')
cell_df = cell_df.drop(columns = ['VALUE', 'COUNT'])

# Convert string (object) columns to integer and downcast
cell_df['LU_CODEV7N'] = cell_df['LU_CODEV7N'].apply(pd.to_numeric, downcast = 'integer')

# Fix NLUM irrigation status coding error in cell_df
cell_df.loc[cell_df['SECONDARY_V7'] == '4.1 Irrigated plantation forestry', 'IRRIGATION'] = 1

# Downcast int64 columns and convert object to category to save memory and space
downcast(cell_df)

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])






################################ Identify potentially irrigable cells as:
#                                   - irrigated areas as identified by National Land and Water Resources Audit
#                                   - all cells immediately neighbouring (3 x 3) irrigated cells as mapped in NLUM
    
# Convert IRRIGATED to 2D array
irrigated = array_2D
irrigated[xy] = np.array(cell_df['IRRIGATION'])

# Identify all cells neighbouring cells mapped as irrigated in NLUM - **note that these could include non-agricultural land**
pot_irrigable = nd.maximum_filter(irrigated, size = 3).astype(np.uint8)

# Import shapefile to GeoPandas DataFrame
gdf = gpd.read_file('N:/Data-Master/Water/Irrigation_areas/irrv1ac.shp')

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['IRRV1AC_ID'] > 0))

# Open a new GeoTiFF file
with rasterio.open('N:/Data-Master/Water/Irrigation_areas/POTENTIAL_IRRIGATION_AREAS.tif', 'w+', dtype = 'uint8', nodata = 255, **meta) as out:
    
    # Rasterise shapefile
    newrast = features.rasterize(shapes = shapes, fill = 0, out = out.read(1), transform = out.transform)
    
    # Combine the neighbourhood with irrigable areas rasters
    raster_clipped = np.where(np.logical_and(NLUM_mask == 1, np.logical_or(newrast == 1, pot_irrigable == 1)), 1, 0).astype(np.uint8)
    
    # Clip raster to NLUM
    raster_clipped = np.where(NLUM_mask == 0, 255, raster_clipped)
    
    # Save output to GeoTiff
    out.write_band(1, raster_clipped)

# Flatten the SA2 2D array to 1D array of valid values only, add to cell_df dataframe
cell_df['POTENTIAL_IRRIGATION_AREAS'] = raster_clipped[NLUM_mask]

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])




################################ Reproject NVIS Extant + Pre-European Major Vegetation Groups and Subgroups rasters, fill holes to match NLUM, save GeoTiff, join to cell_df dataframe

# Reproject NVIS AIGRID to match NLUM using the 'meta' metadata and save to GeoTiff
# with rasterio.open('N:/Data-Master/NVIS\GRID_NVIS6_0_AUST_EXT_MVG\aus6_0e_mvg\w001000.adf') as src:
#     with rasterio.open('N:/Data-Master/NVIS\GRID_NVIS6_0_AUST_EXT_MVG\aus6_0e_mvg.tif', 'w+', dtype='int32', nodata='0', **meta) as dst:
#         reproject(rasterio.band(src, 1), rasterio.band(dst, 1))
#         dst_array = dst.read(1, masked = True)


############## NVIS Extant Major Vegetation Groups

with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_EXT_MVG/aus6_0e_mvg/w001000.adf') as src:
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))

# Mask out nodata cells
dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)

# Fill nodata in raster using value of nearest cell to match NLUM mask
ind = nd.distance_transform_edt(dst_array.mask, return_distances = False, return_indices = True)
NVIS_raster_filled = dst_array[tuple(ind)]
NVIS_raster_clipped = NVIS_raster_filled * NLUM_mask
    
# Save as geoTiff
with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_EXT_MVG/aus6_0e_mvg.tif', 'w+', nodata = 0, dtype = 'uint8', **meta) as dst:
    dst.write_band(1, NVIS_raster_clipped)

# Flatten 2D array to 1D array of valid values only, add NVIS to cell_df dataframe
cell_df['NVIS_EXTANT_MVG_ID'] = NVIS_raster_clipped[NLUM_mask]

# Load in look-up tables of MVG and MVS names and join to cell_df
NVIS_MVG_LUT = pd.read_csv('N:/Data-Master/NVIS/MVG_LUT.csv')

# Join the lookup table to the cell_df DataFrame
cell_df = cell_df.merge(NVIS_MVG_LUT, left_on = 'NVIS_EXTANT_MVG_ID', right_on = 'MVG_ID', how = 'left')
cell_df.rename(columns = {'Major Vegetation Group':'NVIS_EXTANT_MVG_NAME'}, inplace = True)
cell_df = cell_df.drop(columns = ['MVG_ID'])


############## NVIS Extant Major Vegetation Subgroups

with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_EXT_MVS/aus6_0e_mvs/w001000.adf') as src:
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))

# Mask out nodata cells
dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)

# Fill nodata in raster using value of nearest cell to match NLUM mask
ind = nd.distance_transform_edt(dst_array.mask, return_distances = False, return_indices = True)
NVIS_raster_filled = dst_array[tuple(ind)]
NVIS_raster_clipped = NVIS_raster_filled * NLUM_mask
    
# Save as geoTiff
with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_EXT_MVS/aus6_0e_mvs.tif', 'w+', nodata = 0, dtype = 'uint8', **meta) as dst:
    dst.write_band(1, NVIS_raster_clipped)

# Flatten 2D array to 1D array of valid values only, add NVIS to cell_df dataframe
cell_df['NVIS_EXTANT_MVS_ID'] = NVIS_raster_clipped[NLUM_mask]

# Load in look-up tables of MVG and MVS names and join to cell_df
NVIS_MVS_LUT = pd.read_csv('N:/Data-Master/NVIS/MVS_LUT.csv')

# Join the lookup table to the cell_df DataFrame
cell_df = cell_df.merge(NVIS_MVS_LUT, left_on = 'NVIS_EXTANT_MVS_ID', right_on = 'MVS_ID', how = 'left')
cell_df.rename(columns = {'Major Vegetation Subgroup':'NVIS_EXTANT_MVS_NAME'}, inplace = True)
cell_df = cell_df.drop(columns = ['MVS_ID'])


############## NVIS Pre-European Major Vegetation Groups

with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_PRE_MVG/aus6_0p_mvg/w001000.adf') as src:
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))

# Mask out nodata cells
dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)

# Fill nodata in raster using value of nearest cell to match NLUM mask
ind = nd.distance_transform_edt(dst_array.mask, return_distances = False, return_indices = True)
NVIS_raster_filled = dst_array[tuple(ind)]
NVIS_raster_clipped = NVIS_raster_filled * NLUM_mask
    
# Save as geoTiff
with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_PRE_MVG/aus6_0p_mvg.tif', 'w+', nodata = 0, dtype = 'uint8', **meta) as dst:
    dst.write_band(1, NVIS_raster_clipped)

# Flatten 2D array to 1D array of valid values only, add NVIS to cell_df dataframe
cell_df['NVIS_PRE_EURO_MVG_ID'] = NVIS_raster_clipped[NLUM_mask]

# Join the lookup table to the cell_df DataFrame
cell_df = cell_df.merge(NVIS_MVG_LUT, left_on = 'NVIS_PRE_EURO_MVG_ID', right_on = 'MVG_ID', how = 'left')
cell_df.rename(columns = {'Major Vegetation Group':'NVIS_PRE_EURO_MVG_NAME'}, inplace = True)
cell_df = cell_df.drop(columns = ['MVG_ID'])


############## NVIS Pre-European Major Vegetation Subgroups

with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_PRE_MVS/aus6_0p_mvs/w001000.adf') as src:
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))

# Mask out nodata cells
dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)

# Fill nodata in raster using value of nearest cell to match NLUM mask
ind = nd.distance_transform_edt(dst_array.mask, return_distances = False, return_indices = True)
NVIS_raster_filled = dst_array[tuple(ind)]
NVIS_raster_clipped = NVIS_raster_filled * NLUM_mask
    
# Save as geoTiff
with rasterio.open('N:/Data-Master/NVIS/GRID_NVIS6_0_AUST_PRE_MVS/aus6_0p_mvs.tif', 'w+', nodata = 0, dtype = 'uint8', **meta) as dst:
    dst.write_band(1, NVIS_raster_clipped)

# Flatten 2D array to 1D array of valid values only, add NVIS to cell_df dataframe
cell_df['NVIS_PRE_EURO_MVS_ID'] = NVIS_raster_clipped[NLUM_mask]

# Join the lookup table to the cell_df DataFrame
cell_df = cell_df.merge(NVIS_MVS_LUT, left_on = 'NVIS_PRE_EURO_MVS_ID', right_on = 'MVS_ID', how = 'left')
cell_df.rename(columns = {'Major Vegetation Subgroup':'NVIS_PRE_EURO_MVS_NAME'}, inplace = True)
cell_df = cell_df.drop(columns = ['MVS_ID'])

# Downcast int64 columns and convert object to category to save memory and space
downcast(cell_df)

# Plot and print out data, check that there are no NaNs
cell_df.info()
print('Number of grid cells =', cell_df.shape[0])
print('Number of NaNs =', cell_df[cell_df.isna().any(axis = 1)].shape[0])




# Export to HDF5 file
cell_df.to_hdf(cell_df_path + cell_df_fn + '.h5', key = cell_df_fn, mode = 'w', format = 't')





# Best not to use as we cannot release PSMA data publically due to licensing

# =============================================================================
# ################################ Add PSMA Property ID to cell_df
# def join_PSMA():
#     
#     global cell_df
#     
#     with rasterio.open('N:/Data-Master/property_boundaries\properties.tif') as rst:
#         PROP_ID_raster = rst.read(1, masked=True) # Loads a 2D masked array with nodata masked out
#     
#     # Flatten the 2D array to 1D array of valid values only
#     dataFlat = PROP_ID_raster[NLUM_mask]
#     
#     # Add NRM_CODE to cell_df dataframe
#     cell_df['PROPERTY_ID'] = dataFlat
# 
# =============================================================================
