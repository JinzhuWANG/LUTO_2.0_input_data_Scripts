import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from dbfread import DBF
import rasterio, matplotlib
from rasterio.warp import reproject
from rasterio.enums import Resampling

# Set some options
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

infile = r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_mask.tif'
in_cell_df_path = r'N:\Planet-A\Data-Master\LUTO_2.0_input_data\Input_data\2D_Spatial_Snapshot\cell_zones_df.h5'
out_cell_df_path = r'N:\Planet-A\Data-Master\LUTO_2.0_input_data\Input_data\2D_Spatial_Snapshot\cell_lu_sieve_df.pkl'

# Read cell_df from disk, just grab the CELL_ID column
# cell_df = pd.read_feather(cell_df_path, columns=None, use_threads=True);
cell_lu_sieve_df = pd.read_hdf(in_cell_df_path)[['CELL_ID']]

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

    obj_cols = dframe.select_dtypes(include = ["object"]).columns
    dframe[obj_cols] = dframe[obj_cols].astype('category')
    int64_cols = dframe.select_dtypes(include = ["int64"]).columns
    dframe[int64_cols] = dframe[int64_cols].apply(pd.to_numeric, downcast = 'integer')



############## All NNTT areas
#----- Determinations from 1992 - 2020
with rasterio.open(r'N:\Planet-A\Data-Master\Indigenous_lands\Native_title_determinations\NNTT2020_Exclusive-native-title_1km.tif') as src:
      
   # Create an empty destination array 
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.int8)
    
    # Reproject/resample input raster to match NLUM mask (meta)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.nearest)
  
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['NNTT_EXCLUSIVE_TITLE_2020'] = np.round(dataFlat).astype(np.int8)
    
    plt.imshow(dst_array, vmin=0, vmax=2)
 
############## CAPAD Indigenous Protected Areas

#----- Indigenous Protected Areas gazetted between 1900 - 2020
with rasterio.open(r'N:\Planet-A\Data-Master\Protected_areas\CAPAD2020\Protected-areas_Annual_Raster-Masks_1km\Indigenous_CAPAD_Protected_Areas\CAPAD2020_Indigenous-protected-area_Mask_1km.tif') as src:
    
   # Create an empty destination array 
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.int8)
    
    # Reproject/resample input raster to match NLUM mask (meta)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.nearest)
  
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['CAPAD_IPAs_2020'] = np.round(dataFlat).astype(np.int8)
    
    plt.imshow(dst_array, vmin=0, vmax=2)     
    

#----- NLUM Nature conservation areas and All CAPAD Protected Areas classified/gazetted between 1900 - 2020
with rasterio.open(r'N:\Planet-A\Data-Master\Protected_areas\CAPAD2020\Protected-areas_Annual_Raster-Masks_1km\All_NLUM-CAPAD_Protected_Areas\NLUM1.1-2010-CAPAD2020_Protected-area_Mask_1km.tif') as src:
    
   # Create an empty destination array 
    dst_array = np.zeros((meta.get('height'), meta.get('width')), np.int8)
    
    # Reproject/resample input raster to match NLUM mask (meta)
    reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'), resampling = Resampling.nearest)
  
    # Flatten 2D array to 1D array of valid values only
    dataFlat = dst_array[NLUM_mask == 1]
        
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PROTECTED_AREAS_2020'] = np.round(dataFlat).astype(np.int8)
    
    plt.imshow(dst_array, vmin=0, vmax=2)
    
    
#----- biodiversity data
year_file = "N://Planet-A//LUF-Modelling//LUTO2.0_Reporting//Data//year_concordance.csv"
year_dataFrame = pd.read_csv(year_file)
yearlist = sorted(year_dataFrame["YEAR"].to_list())

all_biodiversity_ssp126_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/all/GeoTiffs/Biodiversity-all_ssp126_1970-2100_AUS_1km_ConditionYearly.tif"

baseYear = yearlist.index(1990)+1
with rasterio.open(all_biodiversity_ssp126_2100_file) as src:
    all_biodiversity_base = src.read(baseYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_base = all_biodiversity_base[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_HIST_1990'] = all_biodiversity_base.astype(np.float64)
    cell_lu_sieve_df['BIODIV_HIST_1990'] = cell_lu_sieve_df['BIODIV_HIST_1990'].replace({-9999.0: 0})    


all_biodiversity_ssp126_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/all/GeoTiffs/Biodiversity-all_ssp126_1970-2100_AUS_1km_ConditionYearly.tif"

futureYear = yearlist.index(2030)+1
with rasterio.open(all_biodiversity_ssp126_2100_file) as src:
    all_biodiversity_future = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_future_flat = all_biodiversity_future[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_SSP126_2030'] = all_biodiversity_future_flat.astype(np.float64)
    cell_lu_sieve_df['BIODIV_SSP126_2030'] = cell_lu_sieve_df['BIODIV_SSP126_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp126_2100_file) as src:
    all_biodiversity_future = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_future_flat = all_biodiversity_future[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_SSP126_2050'] = all_biodiversity_future_flat.astype(np.float64)
    cell_lu_sieve_df['BIODIV_SSP126_2050'] = cell_lu_sieve_df['BIODIV_SSP126_2050'].replace({-9999.0: 0})    
    

all_biodiversity_ssp245_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/all/GeoTiffs/Biodiversity-all_ssp245_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(all_biodiversity_ssp245_2100_file) as src:
    all_biodiversity_future = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_future_flat = all_biodiversity_future[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_SSP245_2030'] = all_biodiversity_future_flat.astype(np.float64)
    cell_lu_sieve_df['BIODIV_SSP245_2030'] = cell_lu_sieve_df['BIODIV_SSP245_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp245_2100_file) as src:
    all_biodiversity_future = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_future_flat = all_biodiversity_future[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_SSP245_2050'] = all_biodiversity_future_flat.astype(np.float64)
    cell_lu_sieve_df['BIODIV_SSP245_2050'] = cell_lu_sieve_df['BIODIV_SSP245_2050'].replace({-9999.0: 0})    


all_biodiversity_ssp370_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/all/GeoTiffs/Biodiversity-all_ssp370_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(all_biodiversity_ssp370_2100_file) as src:
    all_biodiversity_future = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_future_flat = all_biodiversity_future[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_SSP370_2030'] = all_biodiversity_future_flat.astype(np.float64)
    cell_lu_sieve_df['BIODIV_SSP370_2030'] = cell_lu_sieve_df['BIODIV_SSP370_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp370_2100_file) as src:
    all_biodiversity_future = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_future_flat = all_biodiversity_future[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_SSP370_2050'] = all_biodiversity_future_flat.astype(np.float64)
    cell_lu_sieve_df['BIODIV_SSP370_2050'] = cell_lu_sieve_df['BIODIV_SSP370_2050'].replace({-9999.0: 0})    


all_biodiversity_ssp585_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/all/GeoTiffs/Biodiversity-all_ssp585_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(all_biodiversity_ssp585_2100_file) as src:
    all_biodiversity_future = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_future_flat = all_biodiversity_future[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_SSP585_2030'] = all_biodiversity_future_flat.astype(np.float64)
    cell_lu_sieve_df['BIODIV_SSP585_2030'] = cell_lu_sieve_df['BIODIV_SSP585_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp585_2100_file) as src:
    all_biodiversity_future = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    all_biodiversity_future_flat = all_biodiversity_future[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIODIV_SSP585_2050'] = all_biodiversity_future_flat.astype(np.float64)
    cell_lu_sieve_df['BIODIV_SSP585_2050'] = cell_lu_sieve_df['BIODIV_SSP585_2050'].replace({-9999.0: 0})        

# Mammals groups

ssp126_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/mammals/GeoTiffs/Biodiversity-mammals_ssp126_1970-2100_AUS_1km_ConditionYearly.tif"

baseYear = yearlist.index(1990)+1
with rasterio.open(ssp126_2100_file) as src:
    base = src.read(baseYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    base_flat = base[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_HIST_1990'] = base_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_HIST_1990'] = cell_lu_sieve_df['MAMMALS_HIST_1990'].replace({-9999.0: 0})    

futureYear = yearlist.index(2030)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_SSP126_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_SSP126_2030'] = cell_lu_sieve_df['MAMMALS_SSP126_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_SSP126_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_SSP126_2050'] = cell_lu_sieve_df['MAMMALS_SSP126_2050'].replace({-9999.0: 0})    
    

ssp245_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/mammals/GeoTiffs/Biodiversity-mammals_ssp245_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_SSP245_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_SSP245_2030'] = cell_lu_sieve_df['MAMMALS_SSP245_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_SSP245_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_SSP245_2050'] = cell_lu_sieve_df['MAMMALS_SSP245_2050'].replace({-9999.0: 0})    


ssp370_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/mammals/GeoTiffs/Biodiversity-mammals_ssp370_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp370_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_SSP370_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_SSP370_2030'] = cell_lu_sieve_df['MAMMALS_SSP370_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp370_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_SSP370_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_SSP370_2050'] = cell_lu_sieve_df['MAMMALS_SSP370_2050'].replace({-9999.0: 0})    


ssp585_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/mammals/GeoTiffs/Biodiversity-mammals_ssp585_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp585_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_SSP585_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_SSP585_2030'] = cell_lu_sieve_df['MAMMALS_SSP585_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp585_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['MAMMALS_SSP585_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['MAMMALS_SSP585_2050'] = cell_lu_sieve_df['MAMMALS_SSP585_2050'].replace({-9999.0: 0})        

# Birds groups

ssp126_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/birds/GeoTiffs/Biodiversity-birds_ssp126_1970-2100_AUS_1km_ConditionYearly.tif"

baseYear = yearlist.index(1990)+1
with rasterio.open(ssp126_2100_file) as src:
    base = src.read(baseYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    base_flat = base[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_HIST_1990'] = base_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_HIST_1990'] = cell_lu_sieve_df['BIRDS_HIST_1990'].replace({-9999.0: 0})    

futureYear = yearlist.index(2030)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_SSP126_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_SSP126_2030'] = cell_lu_sieve_df['BIRDS_SSP126_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_SSP126_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_SSP126_2050'] = cell_lu_sieve_df['BIRDS_SSP126_2050'].replace({-9999.0: 0})    
    

ssp245_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/birds/GeoTiffs/Biodiversity-birds_ssp245_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_SSP245_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_SSP245_2030'] = cell_lu_sieve_df['BIRDS_SSP245_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_SSP245_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_SSP245_2050'] = cell_lu_sieve_df['BIRDS_SSP245_2050'].replace({-9999.0: 0})    


ssp370_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/birds/GeoTiffs/Biodiversity-birds_ssp370_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp370_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_SSP370_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_SSP370_2030'] = cell_lu_sieve_df['BIRDS_SSP370_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp370_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_SSP370_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_SSP370_2050'] = cell_lu_sieve_df['BIRDS_SSP370_2050'].replace({-9999.0: 0})    


ssp585_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/birds/GeoTiffs/Biodiversity-birds_ssp585_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp585_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_SSP585_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_SSP585_2030'] = cell_lu_sieve_df['BIRDS_SSP585_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp585_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['BIRDS_SSP585_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['BIRDS_SSP585_2050'] = cell_lu_sieve_df['BIRDS_SSP585_2050'].replace({-9999.0: 0})        


# Reptiles groups

ssp126_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/reptiles/GeoTiffs/Biodiversity-reptiles_ssp126_1970-2100_AUS_1km_ConditionYearly.tif"

baseYear = yearlist.index(1990)+1
with rasterio.open(ssp126_2100_file) as src:
    base = src.read(baseYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    base_flat = base[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_HIST_1990'] = base_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_HIST_1990'] = cell_lu_sieve_df['REPTILES_HIST_1990'].replace({-9999.0: 0})    

futureYear = yearlist.index(2030)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_SSP126_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_SSP126_2030'] = cell_lu_sieve_df['REPTILES_SSP126_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_SSP126_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_SSP126_2050'] = cell_lu_sieve_df['REPTILES_SSP126_2050'].replace({-9999.0: 0})    
    

ssp245_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/reptiles/GeoTiffs/Biodiversity-reptiles_ssp245_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_SSP245_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_SSP245_2030'] = cell_lu_sieve_df['REPTILES_SSP245_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_SSP245_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_SSP245_2050'] = cell_lu_sieve_df['REPTILES_SSP245_2050'].replace({-9999.0: 0})    


ssp370_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/reptiles/GeoTiffs/Biodiversity-reptiles_ssp370_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp370_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_SSP370_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_SSP370_2030'] = cell_lu_sieve_df['REPTILES_SSP370_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp370_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_SSP370_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_SSP370_2050'] = cell_lu_sieve_df['REPTILES_SSP370_2050'].replace({-9999.0: 0})    


ssp585_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/reptiles/GeoTiffs/Biodiversity-reptiles_ssp585_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp585_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_SSP585_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_SSP585_2030'] = cell_lu_sieve_df['REPTILES_SSP585_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp585_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['REPTILES_SSP585_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['REPTILES_SSP585_2050'] = cell_lu_sieve_df['REPTILES_SSP585_2050'].replace({-9999.0: 0})        


# Amphibians groups

ssp126_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/amphibians/GeoTiffs/Biodiversity-amphibians_ssp126_1970-2100_AUS_1km_ConditionYearly.tif"

baseYear = yearlist.index(1990)+1
with rasterio.open(ssp126_2100_file) as src:
    base = src.read(baseYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    base_flat = base[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_HIST_1990'] = base_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_HIST_1990'] = cell_lu_sieve_df['FROGS_HIST_1990'].replace({-9999.0: 0})    

futureYear = yearlist.index(2030)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_SSP126_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_SSP126_2030'] = cell_lu_sieve_df['FROGS_SSP126_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_SSP126_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_SSP126_2050'] = cell_lu_sieve_df['FROGS_SSP126_2050'].replace({-9999.0: 0})    
    

ssp245_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/amphibians/GeoTiffs/Biodiversity-amphibians_ssp245_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_SSP245_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_SSP245_2030'] = cell_lu_sieve_df['FROGS_SSP245_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_SSP245_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_SSP245_2050'] = cell_lu_sieve_df['FROGS_SSP245_2050'].replace({-9999.0: 0})    


ssp370_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/amphibians/GeoTiffs/Biodiversity-amphibians_ssp370_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp370_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_SSP370_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_SSP370_2030'] = cell_lu_sieve_df['FROGS_SSP370_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp370_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_SSP370_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_SSP370_2050'] = cell_lu_sieve_df['FROGS_SSP370_2050'].replace({-9999.0: 0})    


ssp585_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/amphibians/GeoTiffs/Biodiversity-amphibians_ssp585_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp585_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_SSP585_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_SSP585_2030'] = cell_lu_sieve_df['FROGS_SSP585_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp585_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['FROGS_SSP585_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['FROGS_SSP585_2050'] = cell_lu_sieve_df['FROGS_SSP585_2050'].replace({-9999.0: 0})        


# Plants groups

ssp126_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/plants/GeoTiffs/Biodiversity-plants_ssp126_1970-2100_AUS_1km_ConditionYearly.tif"

baseYear = yearlist.index(1990)+1
with rasterio.open(ssp126_2100_file) as src:
    base = src.read(baseYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    base_flat = base[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_HIST_1990'] = base_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_HIST_1990'] = cell_lu_sieve_df['PLANTS_HIST_1990'].replace({-9999.0: 0})    

futureYear = yearlist.index(2030)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_SSP126_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_SSP126_2030'] = cell_lu_sieve_df['PLANTS_SSP126_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp126_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_SSP126_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_SSP126_2050'] = cell_lu_sieve_df['PLANTS_SSP126_2050'].replace({-9999.0: 0})    
    

ssp245_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/plants/GeoTiffs/Biodiversity-plants_ssp245_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_SSP245_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_SSP245_2030'] = cell_lu_sieve_df['PLANTS_SSP245_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(ssp245_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_SSP245_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_SSP245_2050'] = cell_lu_sieve_df['PLANTS_SSP245_2050'].replace({-9999.0: 0})    


ssp370_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/plants/GeoTiffs/Biodiversity-plants_ssp370_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp370_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_SSP370_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_SSP370_2030'] = cell_lu_sieve_df['PLANTS_SSP370_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp370_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_SSP370_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_SSP370_2050'] = cell_lu_sieve_df['PLANTS_SSP370_2050'].replace({-9999.0: 0})    


ssp585_2100_file = "N:/Planet-A/Data-Master/Biodiversity_priority_areas/Biodiversity/Annual-taxa-condition_yearly_interpolated_1970-2100_1km/plants/GeoTiffs/Biodiversity-plants_ssp585_1970-2100_AUS_1km_ConditionYearly.tif"
      
futureYear = yearlist.index(2030)+1
with rasterio.open(ssp585_2100_file) as src:
    future_2030 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2030_flat = future_2030[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_SSP585_2030'] = future_2030_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_SSP585_2030'] = cell_lu_sieve_df['PLANTS_SSP585_2030'].replace({-9999.0: 0})    

futureYear = yearlist.index(2050)+1
with rasterio.open(all_biodiversity_ssp585_2100_file) as src:
    future_2050 = src.read(futureYear) # loads a 2D masked array of the year 2100         
    # Flatten 2D array to 1D array of valid values only
    future_2050_flat = future_2050[NLUM_mask == 1]    
    # Round and add data to cell_df dataframe
    cell_lu_sieve_df['PLANTS_SSP585_2050'] = future_2050_flat.astype(np.float64)
    cell_lu_sieve_df['PLANTS_SSP585_2050'] = cell_lu_sieve_df['PLANTS_SSP585_2050'].replace({-9999.0: 0})        


#----- Check file
    
for col in cell_lu_sieve_df.columns:
    print(col)
    
cell_lu_sieve_df.to_pickle(out_cell_df_path)
