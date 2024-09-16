import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy import ndimage as nd
import numpy.ma as ma
import rasterio, matplotlib
from rasterio import features
from rasterio.fill import fillnodata
from rasterio.warp import reproject
from rasterio.enums import Resampling

# Set some options
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

infile = r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_mask.tif'
in_cell_df_path = r'N:\Planet-A\Data-Master\LUTO_2.0_input_data\cell_zones_df.pkl'
out_cell_df_path = r'N:\Planet-A\Data-Master\LUTO_2.0_input_data\cell_biophysical_df.pkl'

# Read cell_df from disk, just grab the CELL_ID column
# cell_df = pd.read_feather(cell_df_path, columns=None, use_threads=True);
cell_df = pd.read_pickle(in_cell_df_path)[['CELL_ID', 'HR_DRAINDIV_NAME']]



################################ Create some helper functions

# Open NLUM mask raster and get metadata
with rasterio.open(infile) as rst:
    
    # Read geotiff to numpy array, loads a 2D masked array with nodata masked out
    NLUM_mask = rst.read(1) 
    
    # Get metadata and update parameters
    meta_uint8 = rst.meta.copy()
    meta_uint8.update(compress='lzw', driver = 'GTiff', dtype = 'uint8', nodata = 0)
    meta_int16 = rst.meta.copy()
    meta_int16.update(compress='lzw', driver = 'GTiff', dtype = 'int16', nodata = -9999)
    meta_int32 = rst.meta.copy()
    meta_int32.update(compress='lzw', driver = 'GTiff', dtype = 'int32', nodata = 0)
    meta_float32 = rst.meta.copy()
    meta_float32.update(compress='lzw', driver = 'GTiff', dtype = 'float32', nodata = 0)
    
    # Get the transform and bounds
    NLUM_transform = rst.transform
    bounds = rst.bounds
    
    # Set up some data structures to enable conversion on 1D arrays to 2D
    array_2D = np.zeros(NLUM_mask.shape)
    xy = np.nonzero(NLUM_mask)


# Print array stats
def desc(inarray):
    print('Shape =', inarray.shape, ' Mean =', inarray.mean(), ' Max =', inarray.max(), ' Min =', inarray.min(), ' NaNs =', np.sum(np.isnan(inarray)))

    
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
    obj_cols = dframe.select_dtypes(include = ["object"]).columns
    dframe[obj_cols] = dframe[obj_cols].astype('category')
    int64_cols = dframe.select_dtypes(include = ["int64"]).columns
    dframe[int64_cols] = dframe[int64_cols].apply(pd.to_numeric, downcast = 'integer')




############## Vegetation layer (veg - no veg) based on NVIS extant Major Vegetation Groups layer and create spatial connectivity layer to prioritise ecological restoration

with rasterio.open(r'N:\Planet-A\Data-Master\ANUCLIM_climate_data\AUS_9sec_climate_data_2021\dem-9s_p12.tif') as src:
    
    # Create an empty destination array 
    dst_array = np.zeros((meta_float32.get('height'), meta_float32.get('width')), np.float32)
    
    # Reproject/resample input raster to match NLUM mask (meta_float32)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta_float32.get('transform'), dst_crs = meta_float32.get('crs'), resampling = Resampling.bilinear)
    
    # Create mask for filling cells
    fill_mask = np.where(dst_array > 0, 1, 0)
    
    # Fill nodata using inverse distance weighted averaging and mask to NLUM
    dst_array_filled = fillnodata(dst_array, fill_mask, max_search_distance = 100.0) * NLUM_mask
    
    # Save the output to GeoTiff
    with rasterio.open(r'N:\Planet-A\Data-Master\ANUCLIM_climate_data\AUS_9sec_climate_data_2021\dem-9s_p12_filled.tif', 'w+', **meta_float32) as dst:        
        dst.write_band(1, dst_array_filled)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array_filled[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_df['AVG_AN_PREC_MM_YR'] = np.round(dataFlat).astype(np.uint16)






############## Mean annual rainfall (1975 - 2005) from ANUCLIM modelled using Australian 9 second DEM

with rasterio.open(r'N:\Planet-A\Data-Master\ANUCLIM_climate_data\AUS_9sec_climate_data_2021\dem-9s_p12.tif') as src:
    
    # Create an empty destination array 
    dst_array = np.zeros((meta_float32.get('height'), meta_float32.get('width')), np.float32)
    
    # Reproject/resample input raster to match NLUM mask (meta_float32)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta_float32.get('transform'), dst_crs = meta_float32.get('crs'), resampling = Resampling.bilinear)
    
    # Create mask for filling cells
    fill_mask = np.where(dst_array > 0, 1, 0)
    
    # Fill nodata using inverse distance weighted averaging and mask to NLUM
    dst_array_filled = fillnodata(dst_array, fill_mask, max_search_distance = 100.0) * NLUM_mask
    
    # Save the output to GeoTiff
    with rasterio.open(r'N:\Planet-A\Data-Master\ANUCLIM_climate_data\AUS_9sec_climate_data_2021\dem-9s_p12_filled.tif', 'w+', **meta_float32) as dst:        
        dst.write_band(1, dst_array_filled)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array_filled[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_df['AVG_AN_PREC_MM_YR'] = np.round(dataFlat).astype(np.uint16)




############## Water use by trees from AWRA-L

with rasterio.open(r'N:\Planet-A\Data-Master\Water\water_use_by_trees\wrimpact.tif') as src:
    
    # Create an empty destination array 
    dst_array = np.zeros((meta_float32.get('height'), meta_float32.get('width')), np.float32)
    
    # Reproject/resample input raster to match NLUM mask (meta_float32)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta_float32.get('transform'), dst_crs = meta_float32.get('crs'), resampling = Resampling.bilinear)
    
    # Create mask for filling cells
    fill_mask = np.where(dst_array > 0, 1, 0)
    
    # Fill nodata using inverse distance weighted averaging and mask to NLUM
    dst_array_filled = fillnodata(dst_array, fill_mask, max_search_distance = 100.0) * NLUM_mask
    
    # Save the output to GeoTiff
    with rasterio.open(r'N:\Planet-A\Data-Master\Water\water_use_by_trees\water_use_by_trees.tif', 'w+', **meta_float32) as dst:        
        dst.write_band(1, dst_array_filled)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array_filled[NLUM_mask == 1]
        
    # Add data to cell_df dataframe
    cell_df['WATER_USE_TREES_KL_HA'] = np.round(dataFlat * 1000).astype(np.uint16)
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])
    



############## Water license cost from BOM water trade data

wt_df = pd.read_csv('N:\Planet-A\Data-Master\Water\water_license_cost\Entitlements_Trades_downloaded_20210408.csv') 

# Remove rows with zero in price_per_ML column
wt_df = wt_df[wt_df['price_per_ML'] != 0]

# Calculate some stats on price_per_ML. Data is highly variable as a result of thin markets in many areas. Median is best metric to use.

def perc_25(g):
    return np.percentile(g, 5)

def perc_75(g):
    return np.percentile(g, 95)

pvt = pd.pivot_table(wt_df, values = 'price_per_ML', index = 'drainage_division', aggfunc = [perc_25, np.median, np.mean, perc_75, len])

# Merge pivot table to cell_df dataframe
cell_df = cell_df.merge(pvt[[('median', 'price_per_ML')]], how = 'left', left_on = 'HR_DRAINDIV_NAME', right_on = 'drainage_division')

# Drop column
cell_df = cell_df.drop(columns = 'HR_DRAINDIV_NAME')

# Rename column
cell_df.rename(columns = {('median', 'price_per_ML'):'WATER_PRICE_ML_BOM'}, inplace = True)

# Replace NaNs with zeros and change datatype
cell_df['WATER_PRICE_ML_BOM'] = cell_df['WATER_PRICE_ML_BOM'].fillna(0).astype(np.int16)




############## Water license cost from ABARES report: Burns, K, Hug, B, Lawson, K, Ahammad, H and Zhang, K 2011, Abatement potential from reforestation under selected carbon price scenarios, ABARES Special Report, Canberra, July. p35-36

# Import shapefile to GeoPandas DataFrame
gdf = gpd.read_file(r'N:\Planet-A\Data-Master\Water\water_license_cost\waterPrice\waterCostBasin.shp')

# Convert column data type for conversion to raster
gdf['waterCost'] = gdf['waterCost'].astype(np.int16)

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.waterCost))

# Open a new GeoTiFF file
outfile = r'N:\Planet-A\Data-Master\Water\water_license_cost\waterPrice\WATER_PRICE_ML_ABARES.tif'
with rasterio.open(outfile, 'w+', **meta_int16) as out:
    
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






cell_df.groupby(['HR_DRAINDIV_NAME'])[[('median', 'price_per_ML')]].mean()

cell_df.to_pickle(out_cell_df_path)


"""
wct = np.load(r'N:\Planet-A\Data-Master\LUTO_ANO1\input\input_rp_rf1\scenarioANO_rf1\wct.npy')
wp = np.load(r'N:\Planet-A\Data-Master\LUTO_ANO1\input\input_rp_rf1\scenarioANO_rf1\waterPrice.npy')


cell_df.groupby(['HR_DRAINDIV_NAME'], as_index=False)[['HR_DRAINDIV_NAME']].first().sort_values(by=['HR_DRAINDIV_NAME'])
wt_df.groupby(['water_access_entitlement_type'], as_index=False)[['water_access_entitlement_type']].first()
wt_df.groupby(['drainage_division'], as_index=False)[['price_per_ML']].median()

# Downcast int64 columns and convert object to category to save memory and space
# downcast(cell_df)

cell_df.info()

# Save dataframe to feather format
# cell_df.to_feather(cell_df_path)
cell_df.to_pickle(out_cell_df_path)
"""
