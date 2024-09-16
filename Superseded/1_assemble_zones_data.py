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
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

infile = r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.tif'
outgpkg = r'N:\Planet-A\Data-Master\Profit_map\vector_spatial_data.gpkg'

cell_df_path = r'N:\Planet-A\Data-Master\LUTO_2.0_input_data\cell_zones_df.pkl'

# Read cell_df from disk
# cell_df = pd.read_feather(cell_df_path, columns=None, use_threads=True);
# cell_df = pd.read_pickle(cell_df_path)



################################ Create some helper functions

# Open NLUM_ID as mask raster and get metadata
with rasterio.open(infile) as rst:

    NLUM_ID_raster = rst.read(1, masked=True) # Loads a 2D masked array with nodata masked out
    
    # Get metadata and update parameters
    NLUM_transform = rst.transform
    meta = rst.meta.copy()
    meta.update(compress='lzw', driver='GTiff', dtype='int32', nodata='0')
    meta_uint8 = meta.copy()
    meta_uint8.update(dtype='uint8')
    bounds = rst.bounds
    
    # Set up some data structures to enable conversion on 1D arrays to 2D
    array_2D = np.zeros(NLUM_ID_raster.shape)
    xy = np.nonzero(NLUM_ID_raster.mask == False)

    
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




################################ Read NLUM raster, convert to vector GeoDataFrame, join NLUM table, and save to Geopackage
def save_NLUM_to_GeoDataFrame():

    out_fn = r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.gpkg'
    
    # Open NLUM raster
    with rasterio.open(infile) as src:
        image = src.read(1)
        
        # Collect raster zones as rasterio shape features
        results = ({'properties': {'NLUM_ID': v}, 'geometry': s}
        for i, (s, v) in enumerate(features.shapes(image, mask=None, transform=src.transform)))
    
    # Convert rasterio shape features to GeoDataFrame
    gdfp = gpd.GeoDataFrame.from_features(list(results))
    
    # Dissolve boundaries and convert to multipart shapes
    gdfp = gdfp.dissolve(by='NLUM_ID')
    
    # Load in NLUM tabular data
    NLUM_table = pd.read_csv(r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.tif.csv')
    
    # Join the table to the GeoDataFrame
    gdfp = gdfp.merge(NLUM_table, left_on='NLUM_ID', right_on='VALUE', how='left')
    gdfp = gdfp.drop(columns=['Rowid'])
    
    # Save NLUM data as GeoPackage
    gdfp.to_file(out_fn, layer='NLUM_2010-11_clip', driver="GPKG")
    
    # Some rows will be NaNs because there are internal polygons
    print('Number of NULL cells =', gdfp[gdfp.isna().any(axis=1)].shape[0])




################################ Spit out cell_df dataframe based on raster NLUM (values of 1 - 5125) of X, Y, Z of length nCells
def create_cell_df():
    
    global cell_df

    # Instantiate a LIO Translator object which takes a geotiff file and returns an X, Y, Z dataframe.
    cell_df = lio.Translator("geotiff", "dataframe").translate(infile)
    cell_df.columns = ['X', 'Y', 'NLUM_ID']
    cell_df['CELL_ID'] = cell_df.index + 1
    
    # Test to prove that the order of cells in XYZ conversion exactly the same as array flattening method
    dataFlat = NLUM_ID_raster[NLUM_ID_raster.mask == False].data
    print('Number of cells different =', sum(cell_df['NLUM_ID'] - dataFlat))
    
    # Downcast to save memory
    cell_df['X'] = pd.to_numeric(np.round(cell_df['X'] * 100), downcast='integer') # X and Y values are in hundredths of decimal degrees to cast to integer
    cell_df['Y'] = pd.to_numeric(np.round(cell_df['Y'] * 100), downcast='integer')
    cell_df['NLUM_ID'] = pd.to_numeric(np.round(cell_df['NLUM_ID']), downcast='integer')
    downcast(cell_df)

    # Create REALAREA grid, convert to vector, project to Australian Albers Equal Area, join to cell_df
    
    # Convert CELL_ID to 2D array
    m2D = conv_1D_to_2D(cell_df['CELL_ID']).astype(np.int32)
    
    # Collect raster zones as rasterio shape features
    results = ({'properties': {'CELL_ID': v}, 'geometry': s} for i, (s, v) in enumerate(features.shapes(m2D, mask=None, transform=NLUM_transform)))
     
    # Convert rasterio shape features to GeoDataFrame
    rnd_gdf = gpd.GeoDataFrame.from_features(list(results))
    
    # Assign CRS from NLUM to geodataframe
    rnd_gdf.crs = meta['crs']
    
    # Project to Australian Albers
    rnd_gdf = rnd_gdf.to_crs("EPSG:3577")
    
    # Calculate the area of each cell in hectares
    rnd_gdf["CELL_HA"] = rnd_gdf['geometry'].area / 10000
    
    # Save as GeoPackage
    rnd_gdf.to_file(outgpkg, layer='CELL_ID', driver="GPKG")
    
    # Join Cell_Ha to the cell_df data frame
    cell_df = cell_df.merge(rnd_gdf[['CELL_ID', 'CELL_HA']], how='left', on='CELL_ID')
    
    # Downcast to Float32
    cell_df['CELL_HA'] = cell_df['CELL_HA'].astype(np.float32)
    
    # Rearrange columns
    cell_df = cell_df[['CELL_ID', 'X', 'Y', 'CELL_HA', 'NLUM_ID']]
    
    # Print dataframe info
    cell_df.info()
    



################################ Read SA2 shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df
def join_SA2_ID():
    
    global cell_df

    # Import shapefile to GeoPandas DataFrame
    SA2_gdf = gpd.read_file(r'N:\Planet-A\Data-Master\Profit_map\SA2_boundaries_2011\SA2_2011_AUST.shp')
    print('Number of SA2s in original shapefile =', SA2_gdf['SA2_MAIN11'].nunique())
    
    # Remove rows without geometry and reset index
    SA2_gdf = SA2_gdf[SA2_gdf['geometry'].notna()].reset_index()
    
    # Convert string column to int32 for conversion to raster
    SA2_gdf['SA2_MAIN11'] = SA2_gdf['SA2_MAIN11'].astype(np.int32)
    
    # Save SA2 data as GeoPackage
    SA2_gdf.to_file(outgpkg, layer='SA2_gdf', driver="GPKG")
    
    print('Number of SA2s in GeoDataFrame with NULL geometries removed =', SA2_gdf['SA2_MAIN11'].nunique())
    
    # Convert SA2 shapefile to raster and fillnodata to match NLUM mask
    
    # Access geometry and field to rasterise
    shapes = ((geom, value) for geom, value in zip(SA2_gdf.geometry, SA2_gdf.SA2_MAIN11)) 
    
    # Open a new GeoTiFF file
    outfile = r'N:\Planet-A\Data-Master\Profit_map\SA2_boundaries_2011\SA2_raster_filled.tif'
    with rasterio.open(outfile, 'w+', **meta) as out:
        
        # Rasterise SA2 shapefile. Note 6 SA2s disappear during rasterisation.
        newrast = features.rasterize(shapes=shapes, fill=0, out=out.read(1), transform=out.transform)
        
        # Find cells to fill
        msk = (newrast == 0)
    
        # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
        ind = nd.distance_transform_edt(msk, return_distances=False, return_indices=True)
        SA2_raster_filled = newrast[tuple(ind)]
        SA2_raster_clipped = SA2_raster_filled * (NLUM_ID_raster.mask == False)
        
        # Save output to GeoTiff
        out.write_band(1, SA2_raster_clipped)
    
    print('Number of SA2s in rasterized layer =', np.unique(SA2_raster_clipped).shape[0])
    
    # Flatten the SA2 2D array to 1D array of valid values only
    dataFlat = SA2_raster_clipped[NLUM_ID_raster.mask == False]
    
    # Add SA2_ID to cell_df dataframe
    cell_df['SA2_ID'] = dataFlat
    
    # Plot and print out SA2 data
    map_in_2D(cell_df['SA2_ID'], data='categorical')

    return SA2_gdf


################################ Create unique ID as combination of NLUM_ID and SA2_ID and convert to NLUM_SA2_gdf polygon, dissolve and save
def add_NLUM_SA2_ID():
    
    global cell_df
    
    diss_df = cell_df.groupby(['NLUM_ID','SA2_ID'], as_index=False).size().drop(columns=['size'])
    diss_df['NLUM_SA2_ID'] = diss_df.index + 1
    cell_df = cell_df.merge(diss_df, how='left', left_on=['NLUM_ID','SA2_ID'], right_on=['NLUM_ID','SA2_ID'])
        
    # Rearrange columns
    cell_df = cell_df[['CELL_ID', 'X', 'Y', 'CELL_HA', 'NLUM_ID', 'NLUM_SA2_ID', 'SA2_ID']]
    
    # Convert 1D column to 2D spatial array
    arr_2D = conv_1D_to_2D(cell_df['NLUM_SA2_ID'])
    
    # Collect raster zones as rasterio shape features
    NLUM_SA2_shapes = ({'properties': {'NLUM_SA2_ID': v}, 'geometry': s} for i, (s, v) in enumerate(features.shapes(arr_2D.astype(np.int32), mask=None, transform=NLUM_transform)))
     
    # Convert rasterio shape features to GeoDataFrame
    NLUM_SA2_gdf = gpd.GeoDataFrame.from_features(list(NLUM_SA2_shapes))
    
    # Dissolve boundaries and convert to multipart shapes
    NLUM_SA2_gdf = NLUM_SA2_gdf.dissolve(by='NLUM_SA2_ID')
    
    # Join NLUM_ID and SA2_ID to the GeoDataFrame
    NLUM_SA2_gdf = NLUM_SA2_gdf.merge(diss_df, how='left', left_on=['NLUM_SA2_ID'], right_on=['NLUM_SA2_ID'])
    
    # Load in NLUM tabular data
    NLUM_table = pd.read_csv(r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.tif.csv')
    NLUM_table = NLUM_table[['VALUE', 'COMMODITIES', 'COMMODITIES_DESC', 'IRRIGATION']]
     
    # Join the COMMODITY and IRRIGATION information to the GeoDataFrame
    NLUM_SA2_gdf = NLUM_SA2_gdf.merge(NLUM_table, left_on='NLUM_ID', right_on='VALUE', how='left')
    NLUM_SA2_gdf = NLUM_SA2_gdf.drop(columns=['VALUE'])
     
    # Save NLUM data as GeoPackage
    NLUM_SA2_gdf.to_file(outgpkg, layer='NLUM_SA2_gdf', driver="GPKG")
    
    # Check that there are the same number of polygons (+1 outer polygon) as there are in the groupby operation
    diss_df = cell_df.groupby(['NLUM_ID','SA2_ID']).size().reset_index().rename(columns={0:'count'})
    
    # Downcast int64 columns and convert object to category to save memory and space
    downcast(cell_df)




################################ Join SA2 table to the cell_df dataframe and optimise datatypes
def join_SA2_table():
    
    global cell_df
    
    # Join the SA2 GeoDataFrame table SA2_gdf to the cell_df dataframe
    cell_df = cell_df.merge(SA2_gdf, left_on='SA2_ID', right_on='SA2_MAIN11', how='left')
    cell_df = cell_df.drop(columns=['geometry', 'ALBERS_SQM', 'index', 'SA2_MAIN11', 'SA2_5DIG11'])
    
    # Convert string (object) columns to integer and downcast
    obj_cols = ['SA3_CODE11', 'SA4_CODE11', 'STE_CODE11']
    cell_df[obj_cols] = cell_df[obj_cols].apply(pd.to_numeric, downcast = 'integer')
    
    # Downcast int64 columns and convert object to category to save memory and space
    downcast(cell_df)
    
    cell_df.info()
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])




################################ Read ABARES regions shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
def join_ABARES():
    
    global cell_df

    # Import shapefile to GeoPandas DataFrame
    ABARES_gdf = gpd.read_file(r'N:\Planet-A\Data-Master\Australian_administrative_boundaries\abare_2018_aus\ABARE_boundaries.shp')
    print('Number of ABARES regions in original shapefile =', ABARES_gdf['AAGIS'].nunique())
    
    # Downcast CODE for conversion to raster
    ABARES_gdf['AAGIS'] = pd.to_numeric(ABARES_gdf['AAGIS'], downcast='integer')
    
    # Access geometry and field to rasterise
    shapes = ((geom, value) for geom, value in zip(ABARES_gdf.geometry, ABARES_gdf.AAGIS)) 
    
    # Open a new GeoTiFF file
    outfile = r'N:\Planet-A\Data-Master\Australian_administrative_boundaries\abare_2018_aus\ABARES_raster_filled.tif'
    with rasterio.open(outfile, 'w+', **meta) as out:
        
        # Rasterise NRM shapefile
        newrast = features.rasterize(shapes=shapes, fill=0, out=out.read(1), transform=out.transform)
        
        # Find cells to fill
        msk = (newrast == 0)
    
        # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
        ind = nd.distance_transform_edt(msk, return_distances=False, return_indices=True)
        raster_filled = newrast[tuple(ind)]
        ABARES_raster_clipped = raster_filled * (NLUM_ID_raster.mask == False)
        
        # Save output to GeoTiff
        out.write_band(1, ABARES_raster_clipped)
    
    print('Number of ABARES regions in rasterized layer =', np.unique(ABARES_raster_clipped).shape[0])
    
    # Flatten the NRM 2D array to 1D array of valid values only
    dataFlat = ABARES_raster_clipped[NLUM_ID_raster.mask == False]
    
    # Add SA2_ID to cell_df dataframe
    cell_df['ABARES_AAGIS'] = dataFlat
    cell_df['ABARES_AAGIS'] = pd.to_numeric(cell_df['ABARES_AAGIS'], downcast='integer')
    
    # Plot and print out data
    map_in_2D(cell_df['ABARES_AAGIS'], data='categorical')
    cell_df.info()
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])




################################ Read LGA shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
def join_LGAs():
    
    global cell_df

    # Import shapefile to GeoPandas DataFrame
    LGA_gdf = gpd.read_file(r'N:\Planet-A\Data-Master\Australian_administrative_boundaries\lga_2010_aus\LGA10aAust.shp')
    print('Number of LGAs in original shapefile =', LGA_gdf['LGA_CODE10'].nunique())
    
    # Downcast LGA_CODE10 for conversion to raster
    LGA_gdf['LGA_CODE10'] = pd.to_numeric(LGA_gdf['LGA_CODE10'], downcast='integer')
    
    # Access geometry and field to rasterise
    shapes = ((geom, value) for geom, value in zip(LGA_gdf.geometry, LGA_gdf.LGA_CODE10)) 
    
    # Open a new GeoTiFF file
    outfile = r'N:\Planet-A\Data-Master\Australian_administrative_boundaries\lga_2010_aus\LGA_raster_filled.tif'
    with rasterio.open(outfile, 'w+', **meta) as out:
        
        # Rasterise shapefile
        newrast = features.rasterize(shapes=shapes, fill=0, out=out.read(1), transform=out.transform)
        
        # Find cells to fill
        msk = (newrast == 0)
    
        # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
        ind = nd.distance_transform_edt(msk, return_distances=False, return_indices=True)
        raster_filled = newrast[tuple(ind)]
        LGA_raster_clipped = raster_filled * (NLUM_ID_raster.mask == False)
        
        # Save output to GeoTiff
        out.write_band(1, LGA_raster_clipped)
    
    print('Number of NRM regions in rasterized layer =', np.unique(LGA_raster_clipped).shape[0])
    
    # Flatten the NRM 2D array to 1D array of valid values only
    dataFlat = LGA_raster_clipped[NLUM_ID_raster.mask == False]
    
    # Add SA2_ID to cell_df dataframe
    cell_df['LGA_CODE'] = dataFlat
    cell_df['LGA_CODE'] = pd.to_numeric(cell_df['LGA_CODE'], downcast='integer')
    
    # Join LGA name to the cell_df data frame
    cell_df = cell_df.merge(LGA_gdf, how='left', left_on='LGA_CODE', right_on='LGA_CODE10')
    cell_df = cell_df.drop(columns=['STATE_CODE', 'LGA_CODE10', 'geometry'])
    
    # Plot and print out data
    map_in_2D(cell_df['LGA_CODE'], data='categorical')
    cell_df.info()
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])




################################ Read NRM regions shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
def join_NRM():
    
    global cell_df
    
    # Import shapefile to GeoPandas DataFrame
    NRM_gdf = gpd.read_file(r'N:\Planet-A\Data-Master\Australian_administrative_boundaries\nrm_2016_aus\nrm_gda94.shp')
    print('Number of NRM regions in original shapefile =', NRM_gdf['CODE'].nunique())
    
    # Fix up an attribute error in the data
    rw = NRM_gdf['CODE'] == 304310
    NRM_gdf.loc[rw, 'NHT2NAME'] = 'Northern Gulf'
    NRM_gdf.loc[rw, 'CODE'] = 310
    
    # Downcast CODE for conversion to raster
    NRM_gdf['CODE'] = pd.to_numeric(NRM_gdf['CODE'], downcast='integer')
    
    # Access geometry and field to rasterise
    shapes = ((geom, value) for geom, value in zip(NRM_gdf.geometry, NRM_gdf.CODE)) 
    
    # Open a new GeoTiFF file
    outfile = r'N:\Planet-A\Data-Master\Australian_administrative_boundaries\nrm_2016_aus\NRM_raster_filled.tif'
    with rasterio.open(outfile, 'w+', **meta) as out:
        
        # Rasterise NRM shapefile
        newrast = features.rasterize(shapes=shapes, fill=0, out=out.read(1), transform=out.transform)
        
        # Find cells to fill
        msk = (newrast == 0)
    
        # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
        ind = nd.distance_transform_edt(msk, return_distances=False, return_indices=True)
        raster_filled = newrast[tuple(ind)]
        NRM_raster_clipped = raster_filled * (NLUM_ID_raster.mask == False)
        
        # Save output to GeoTiff
        out.write_band(1, NRM_raster_clipped)
    
    print('Number of NRM regions in rasterized layer =', np.unique(NRM_raster_clipped).shape[0])
    
    # Flatten the NRM 2D array to 1D array of valid values only
    dataFlat = NRM_raster_clipped[NLUM_ID_raster.mask == False]
    
    # Add NRM_CODE to cell_df dataframe
    cell_df['NRM_CODE'] = dataFlat
    cell_df['NRM_CODE'] = pd.to_numeric(cell_df['NRM_CODE'], downcast='integer')
    
    # Simplify the table for merging
    tmp = NRM_gdf.groupby(['CODE'], as_index=False)[['NHT2NAME']].first().sort_values(by=['CODE'])
    
    # Join NRM name to the cell_df data frame
    cell_df = cell_df.merge(tmp, how='left', left_on='NRM_CODE', right_on='CODE')
    cell_df = cell_df.drop(columns=['CODE'])
    cell_df.rename(columns = {'NHT2NAME':'NRM_NAME'}, inplace = True)
    
    # Plot and print out data
    map_in_2D(cell_df['NRM_CODE'], data='categorical')
    cell_df.info()
        
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])




################################ Read IBRA shapefile in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
def join_IBRA():
    
    global cell_df
       
    # Import shapefile to GeoPandas DataFrame
    IBRA_gdf = gpd.read_file(r'N:\Planet-A\Data-Master\Australian_administrative_boundaries\ibra7_2019_aus\ibra7_subregions.shp')
    print('Number of IBRA subregions in original shapefile =', IBRA_gdf['SUB_CODE_7'].nunique())
    print('Number of IBRA regions in original shapefile =', IBRA_gdf['REG_CODE_7'].nunique())
    
    # Downcast for conversion to raster
    IBRA_gdf['REC_ID'] = pd.to_numeric(IBRA_gdf['REC_ID'], downcast='integer')
    
    # Convert shapefile to raster and fillnodata to match NLUM mask
    
    # Access geometry and field to rasterise
    shapes = ((geom, value) for geom, value in zip(IBRA_gdf.geometry, IBRA_gdf.REC_ID)) 
    
    # Open a new GeoTiFF file
    outfile = r'N:\Planet-A\Data-Master\Australian_administrative_boundaries\ibra7_2019_aus\IBRA_raster_filled.tif'
    with rasterio.open(outfile, 'w+', **meta) as out:
        
        # Rasterise NRM shapefile
        newrast = features.rasterize(shapes=shapes, fill=0, out=out.read(1), transform=out.transform)
        
        # Find cells to fill
        msk = (newrast == 0)
    
        # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
        ind = nd.distance_transform_edt(msk, return_distances=False, return_indices=True)
        raster_filled = newrast[tuple(ind)]
        IBRA_raster_clipped = raster_filled * (NLUM_ID_raster.mask == False)
        
        # Save output to GeoTiff
        out.write_band(1, IBRA_raster_clipped)
    
    print('Number of IBRA subregions in rasterized layer =', np.unique(IBRA_raster_clipped).shape[0])
    
    # Flatten the NRM 2D array to 1D array of valid values only
    dataFlat = IBRA_raster_clipped[NLUM_ID_raster.mask == False]
    
    # Add SA2_ID to cell_df dataframe
    cell_df['IBRA_ID'] = dataFlat
    cell_df['IBRA_ID'] = pd.to_numeric(cell_df['IBRA_ID'], downcast='integer')
    
    # Simplify the table for merging
    tmp = IBRA_gdf.groupby(['REC_ID'], as_index=False)[['SUB_CODE_7', 'SUB_NAME_7', 'REG_CODE_7', 'REG_NAME_7']].first().sort_values(by=['REC_ID'])
    
    # Join LGA name to the cell_df data frame
    cell_df = cell_df.merge(tmp, how='left', left_on='IBRA_ID', right_on='REC_ID')
    cell_df = cell_df.drop(columns=['REC_ID'])
    cell_df.rename(columns={'SUB_CODE_7': 'IBRA_SUB_CODE_7', 
                            'SUB_NAME_7': 'IBRA_SUB_NAME_7', 
                            'REG_CODE_7': 'IBRA_REG_CODE_7', 
                            'REG_NAME_7': 'IBRA_REG_NAME_7'}, inplace=True)
    
    # Downcast int64 columns and convert object to category to save memory and space
    downcast(cell_df)
    
    # Plot and print out data
    map_in_2D(cell_df['IBRA_ID'], data='categorical')
    cell_df.info()
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])




################################ Read BOM GeoFabric HR Regions River Regions Geodatabase file in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
def join_BOM_RR():
    
    global cell_df
    
    # Read file from File Geodatabase
    ESRI_gdb = 'N:\Planet-A\Data-Master\Water\GeoFabric_V3.2\HR_Regions_GDB_V3_2\HR_Regions_GDB\HR_Regions.gdb'
    HR_RR_gdf = gpd.read_file(ESRI_gdb, driver="FileGDB", layer='RiverRegion')
    
    print('Number of regions in original shapefile =', HR_RR_gdf['RivRegName'].nunique())
    
    # Simplify the table for merging, create new index
    tmp = HR_RR_gdf.groupby(['RivRegName'], as_index=False)[['RivRegName']].first().sort_values(by=['RivRegName'])
    tmp['HR_RIVREG_ID'] = tmp.index + 1
    HR_RR_gdf = HR_RR_gdf.merge(tmp, how='left', on='RivRegName')
    
    # Downcast CODE for conversion to raster
    HR_RR_gdf['HR_RIVREG_ID'] = pd.to_numeric(HR_RR_gdf['HR_RIVREG_ID'], downcast='integer')
    
    # Access geometry and field to rasterise
    shapes = ((geom, value) for geom, value in zip(HR_RR_gdf.geometry, HR_RR_gdf.HR_RIVREG_ID)) 
    
    # Open a new GeoTiFF file
    outfile = r"N:\Planet-A\Data-Master\Water\GeoFabric_V3.2\HR_Regions_GDB_V3_2\HR_Regions_GDB\HR_RivReg_raster_filled.tif"
    with rasterio.open(outfile, 'w+', **meta) as out:
        
        # Rasterise NRM shapefile
        newrast = features.rasterize(shapes=shapes, fill=0, out=out.read(1), transform=out.transform)
        
        # Find cells to fill
        msk = (newrast == 0)
    
        # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
        ind = nd.distance_transform_edt(msk, return_distances=False, return_indices=True)
        raster_filled = newrast[tuple(ind)]
        RR_raster_clipped = raster_filled * (NLUM_ID_raster.mask == False)
        
        # Save output to GeoTiff
        out.write_band(1, RR_raster_clipped)
    
    print('Number of regions in rasterized layer =', np.unique(RR_raster_clipped).shape[0])
    
    # Flatten the NRM 2D array to 1D array of valid values only
    dataFlat = RR_raster_clipped[NLUM_ID_raster.mask == False]
    
    # Add NRM_CODE to cell_df dataframe
    cell_df['HR_RIVREG_ID'] = dataFlat
    cell_df['HR_RIVREG_ID'] = pd.to_numeric(cell_df['HR_RIVREG_ID'], downcast='integer')
    
    # Join NRM name to the cell_df data frame
    cell_df = cell_df.merge(tmp, how='left', on='HR_RIVREG_ID')
    cell_df.rename(columns = {'RivRegName':'HR_RIVREG_NAME'}, inplace = True)
    
    # Downcast int64 columns and convert object to category to save memory and space
    downcast(cell_df)
    
    # Plot and print out data
    map_in_2D(cell_df['HR_RIVREG_ID'], data='categorical')
    cell_df.info()
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])




################################ Read BOM GeoFabric HR Regions AWRA Drainage Divisions Geodatabase file in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
def join_BOM_DD():
    
    global cell_df
    
    # Read file from File Geodatabase
    ESRI_gdb = 'N:\Planet-A\Data-Master\Water\GeoFabric_V3.2\HR_Regions_GDB_V3_2\HR_Regions_GDB\HR_Regions.gdb'
    HR_DD_gdf = gpd.read_file(ESRI_gdb, driver="FileGDB", layer='AWRADrainageDivision')
    
    print('Number of regions in original shapefile =', HR_DD_gdf['Division'].nunique())
    
    # Simplify the table for merging, create new index
    HR_DD_gdf['HR_DRAINDIV_ID'] = HR_DD_gdf.index + 1
    
    # Downcast CODE for conversion to raster
    HR_DD_gdf['HR_DRAINDIV_ID'] = pd.to_numeric(HR_DD_gdf['HR_DRAINDIV_ID'], downcast='integer')
    
    # Access geometry and field to rasterise
    shapes = ((geom, value) for geom, value in zip(HR_DD_gdf.geometry, HR_DD_gdf.HR_DRAINDIV_ID)) 
    
    # Open a new GeoTiFF file
    outfile = r"N:\Planet-A\Data-Master\Water\GeoFabric_V3.2\HR_Regions_GDB_V3_2\HR_Regions_GDB\HR_DrainDiv_raster_filled.tif"
    with rasterio.open(outfile, 'w+', **meta) as out:
        
        # Rasterise NRM shapefile
        newrast = features.rasterize(shapes=shapes, fill=0, out=out.read(1), transform=out.transform)
        
        # Find cells to fill
        msk = (newrast == 0)
    
        # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
        ind = nd.distance_transform_edt(msk, return_distances=False, return_indices=True)
        raster_filled = newrast[tuple(ind)]
        DD_raster_clipped = raster_filled * (NLUM_ID_raster.mask == False)
        
        # Save output to GeoTiff
        out.write_band(1, DD_raster_clipped)
    
    print('Number of regions in rasterized layer =', np.unique(DD_raster_clipped).shape[0])
    
    # Flatten the NRM 2D array to 1D array of valid values only
    dataFlat = DD_raster_clipped[NLUM_ID_raster.mask == False]
    
    # Add NRM_CODE to cell_df dataframe
    cell_df['HR_DRAINDIV_ID'] = dataFlat
    cell_df['HR_DRAINDIV_ID'] = pd.to_numeric(cell_df['HR_DRAINDIV_ID'], downcast='integer')
    
    # Join DD name to the cell_df data frame
    cell_df = cell_df.merge(HR_DD_gdf[['HR_DRAINDIV_ID', 'Division']], how='left', on='HR_DRAINDIV_ID')
    cell_df.rename(columns = {'Division':'HR_DRAINDIV_NAME'}, inplace = True)
    
   # Select rows for modification and calculate values for selected rows
    index = cell_df.query("HR_DRAINDIV_NAME == 'South East Coast (VICTORIA)'").index
    cell_df.loc[index, 'HR_DRAINDIV_NAME'] = 'South East Coast (Victoria)'
    index = cell_df.query("HR_DRAINDIV_NAME == 'Tanami -Timor Sea Coast'").index
    cell_df.loc[index, 'HR_DRAINDIV_NAME'] = 'Tanami-Timor Sea Coast'
    
    # Downcast int64 columns and convert object to category to save memory and space
    downcast(cell_df)
    
    # Plot and print out data
    map_in_2D(cell_df['HR_DRAINDIV_ID'], data='categorical')
    cell_df.info()
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])




################################ Read Aqueduct 3.0 Baseline Water Stress file in to GeoPandas GeoDataFrame, convert to raster to match NLUM, save, join to cell_df dataframe
def join_Aqueduct():
    
    global cell_df
    
    # Read file from GeoPackage
    AD_file = 'N:/Planet-A/Data-Master/Water/Aqueduct_3.0_V01/baseline/annual/y2019m07d11_aqueduct30_annual_v01.gpkg'
    AD_gdf = gpd.read_file(AD_file, layer='y2019m07d11_aqueduct30_annual_v01')
    
    AD_gdf = AD_gdf[['geometry', 'bws_cat', 'bws_label']]
    AD_gdf['bws_cat'] = AD_gdf['bws_cat'] + 2
    AD_gdf.loc[AD_gdf['bws_cat'].isna(), 'bws_cat'] = 0
    AD_gdf['bws_cat'] = pd.to_numeric(np.round(AD_gdf['bws_cat']), downcast='integer')
    AD_gdf.info()
    
    # # Simplify the table for merging
    AD_df = AD_gdf[['bws_cat', 'bws_label']]
    AD_df = AD_df.groupby(['bws_cat'], as_index=False)[['bws_cat', 'bws_label']].first().sort_values(by=['bws_cat'])
                                
    # Access geometry and field to rasterise
    shapes = ((geom, value) for geom, value in zip(AD_gdf.geometry, AD_gdf.bws_cat)) 
    
    # Open a new GeoTiFF file
    outfile = "N:/Planet-A/Data-Master/Water/Aqueduct_3.0_V01/baseline/annual/AD_BWS_CAT_raster_filled.tif"
    
    with rasterio.open(outfile, 'w+', **meta) as out:
        
        # Rasterise NRM shapefile
        newrast = features.rasterize(shapes=shapes, fill=-99, out=out.read(1), transform=out.transform)
        
        # Find cells to fill
        msk = (newrast == 0)
    
        # Fill nodata in SA2 raster using value of nearest cell to match NLUM mask
        ind = nd.distance_transform_edt(msk, return_distances=False, return_indices=True)
        raster_filled = newrast[tuple(ind)]
        AD_raster_clipped = raster_filled * (NLUM_ID_raster.mask == False)
        
        # Save output to GeoTiff
        out.write_band(1, AD_raster_clipped)
    
    print('Number of regions in rasterized layer =', np.unique(AD_raster_clipped).shape[0])
    
    # Flatten the 2D array to 1D array of valid values only
    dataFlat = AD_raster_clipped[NLUM_ID_raster.mask == False]
    
    # Add NRM_CODE to cell_df dataframe
    cell_df['AD_BWS_CAT'] = dataFlat
    cell_df['AD_BWS_CAT'] = pd.to_numeric(cell_df['AD_BWS_CAT'], downcast='integer')
    
    # Join  to the cell_df data frame
    cell_df = cell_df.merge(AD_df, how='left', left_on='AD_BWS_CAT', right_on='bws_cat')
    cell_df = cell_df.drop(columns=['bws_cat'])
    cell_df.rename(columns = {'bws_label':'AD_BWS_LABEL'}, inplace = True)
    
    # Downcast int64 columns and convert object to category to save memory and space
    downcast(cell_df)
    
    # Plot and print out data
    map_in_2D(cell_df['AD_BWS_CAT'], data='categorical')
    cell_df.info()
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])




############## Water stressed catchments (C & D) from National Water Commission (2012) Assessing water stress in Australian catchments and aquifers.

# Import shapefile to GeoPandas DataFrame
gdf = gpd.read_file(r'N:\Planet-A\Data-Master\Water\Water_stressed_catchments\basinStressCode.shp')

# Convert column data type for conversion to raster
gdf['STRESS_COD'] = gdf['STRESS_COD'].astype(np.int8)

# Access geometry and field to rasterise
shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.STRESS_COD))

# Open a new GeoTiFF file
outfile = r'N:\Planet-A\Data-Master\Water\Water_stressed_catchments\WATER_STRESS_NWC.tif'
with rasterio.open(outfile, 'w+', dtype = 'int8', nodata = -9999, **meta) as out:
    
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


gdf.groupby(['STRESS_COD'])[['STRESS_CLA']].first()



################################ Join NLUM land use table to the cell_df dataframe and optimise datatypes
def join_NLUM():
    
    global cell_df
    
    # Read in the NLUM table
    tmp = pd.read_csv(r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.tif.csv')
    
    # Join the table to the dataframe and drop uneccesary columns
    cell_df = cell_df.merge(tmp, left_on='NLUM_ID', right_on='VALUE', how='left')
    cell_df = cell_df.drop(columns=['Rowid', 'VALUE', 'COUNT'])
    
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])
    
    # Convert string (object) columns to integer and downcast
    cell_df['LU_CODEV7N'] = cell_df['LU_CODEV7N'].apply(pd.to_numeric, downcast = 'integer')
    
    # Downcast int64 columns and convert object to category to save memory and space
    downcast(cell_df)
    
    # Print info about the table
    cell_df.info()




################################ Identify potentially irrigable cells as all cells immediately neighbouring (3 x 3) currently irrigated cells
def join_POT_IRRIGABLE():
    
    # Convert IRRIGATED to 2D array
    irrigated = array_2D.astype(np.int8)
    irrigated[xy] = np.array(cell_df['IRRIGATION'])
    # irrigated[array_2D == 0] = np.nan
    
    # Identify all cells neighbouring cells mapped as irrigated in NLUM - **note that these could include non-agricultural land**
    pot_irrigable = nd.maximum_filter(irrigated, size = 3)
    
    # Flatten the SA2 2D array to 1D array of valid values only
    dataFlat = pot_irrigable[NLUM_ID_raster.mask == False]
    
    # Add to cell_df dataframe
    cell_df['POT_IRRIGABLE'] = dataFlat




################################ Reproject NVIS Extant + Pre-European Major Vegetation Groups and Subgroups rasters, fill holes to match NLUM, save GeoTiff, join to cell_df dataframe

def join_NVIS():
        
    # Reproject NVIS AIGRID to match NLUM using the "meta" metadata and save to GeoTiff
    # with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_EXT_MVG\aus6_0e_mvg\w001000.adf') as src:
    #     with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_EXT_MVG\aus6_0e_mvg.tif', 'w+', **meta) as dst:
    #         reproject(rasterio.band(src, 1), rasterio.band(dst, 1))
    #         dst_array = dst.read(1, masked = True)
    
    
    ############## NVIS Extant Major Vegetation Groups
    
    global cell_df
    
    with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_EXT_MVG\aus6_0e_mvg\w001000.adf') as src:
        dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
        reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))
    
    # Mask out nodata cells
    dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)
    
    # Fill nodata in raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(dst_array.mask, return_distances=False, return_indices=True)
    NVIS_raster_filled = dst_array[tuple(ind)]
    NVIS_raster_clipped = NVIS_raster_filled * (NLUM_ID_raster.mask == False)
        
    # Save as geoTiff
    with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_EXT_MVG\aus6_0e_mvg.tif', 'w+', **meta_uint8) as dst:
        dst.write_band(1, NVIS_raster_clipped)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = NVIS_raster_clipped[NLUM_ID_raster.mask == False]
        
    # Add NVIS to cell_df dataframe
    cell_df['NVIS_EXTANT_MVG_ID'] = dataFlat
    
    # Load in look-up tables of MVG and MVS names and join to cell_df
    NVIS_MVG_LUT = pd.read_csv(r'N:\Planet-A\Data-Master\NVIS\MVG_LUT.csv')

    # Join the lookup table to the cell_df DataFrame
    cell_df = cell_df.merge(NVIS_MVG_LUT, left_on='NVIS_EXTANT_MVG_ID', right_on='MVG_ID', how='left')
    cell_df.rename(columns = {'Major Vegetation Group':'NVIS_EXTANT_MVG_NAME'}, inplace = True)
    cell_df = cell_df.drop(columns=['MVG_ID'])
    
    
    ############## NVIS Extant Major Vegetation Subgroups
    
    with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_EXT_MVS\aus6_0e_mvs\w001000.adf') as src:
        dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
        reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))
    
    # Mask out nodata cells
    dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)
    
    # Fill nodata in raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(dst_array.mask, return_distances=False, return_indices=True)
    NVIS_raster_filled = dst_array[tuple(ind)]
    NVIS_raster_clipped = NVIS_raster_filled * (NLUM_ID_raster.mask == False)
        
    # Save as geoTiff
    with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_EXT_MVS\aus6_0e_mvs.tif', 'w+', **meta_uint8) as dst:
        dst.write_band(1, NVIS_raster_clipped)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = NVIS_raster_clipped[NLUM_ID_raster.mask == False]
        
    # Add NVIS to cell_df dataframe
    cell_df['NVIS_EXTANT_MVS_ID'] = dataFlat
    
    # Load in look-up tables of MVG and MVS names and join to cell_df
    NVIS_MVS_LUT = pd.read_csv(r'N:\Planet-A\Data-Master\NVIS\MVS_LUT.csv')

    # Join the lookup table to the cell_df DataFrame
    cell_df = cell_df.merge(NVIS_MVS_LUT, left_on='NVIS_EXTANT_MVS_ID', right_on='MVS_ID', how='left')
    cell_df.rename(columns = {'Major Vegetation Subgroup':'NVIS_EXTANT_MVS_NAME'}, inplace = True)
    cell_df = cell_df.drop(columns=['MVS_ID'])

    
    ############## NVIS Pre-European Major Vegetation Groups
    
    with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_PRE_MVG\aus6_0p_mvg\w001000.adf') as src:
        dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
        reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))
    
    # Mask out nodata cells
    dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)
    
    # Fill nodata in raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(dst_array.mask, return_distances=False, return_indices=True)
    NVIS_raster_filled = dst_array[tuple(ind)]
    NVIS_raster_clipped = NVIS_raster_filled * (NLUM_ID_raster.mask == False)
        
    # Save as geoTiff
    with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_PRE_MVG\aus6_0p_mvg.tif', 'w+', **meta_uint8) as dst:
        dst.write_band(1, NVIS_raster_clipped)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = NVIS_raster_clipped[NLUM_ID_raster.mask == False]
        
    # Add NVIS to cell_df dataframe
    cell_df['NVIS_PRE-EURO_MVG_ID'] = dataFlat
    
    # Join the lookup table to the cell_df DataFrame
    cell_df = cell_df.merge(NVIS_MVG_LUT, left_on='NVIS_PRE-EURO_MVG_ID', right_on='MVG_ID', how='left')
    cell_df.rename(columns = {'Major Vegetation Group':'NVIS_PRE-EURO_MVG_NAME'}, inplace = True)
    cell_df = cell_df.drop(columns=['MVG_ID'])

    
    ############## NVIS Pre-European Major Vegetation Subgroups
    
    with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_PRE_MVS\aus6_0p_mvs\w001000.adf') as src:
        dst_array = np.zeros((meta.get('height'), meta.get('width')), np.uint8)
        reproject(rasterio.band(src, 1), dst_array, dst_transform = meta.get('transform'), dst_crs = meta.get('crs'))
    
    # Mask out nodata cells
    dst_array = ma.masked_where((dst_array >= 99) | (dst_array == 0), dst_array)
    
    # Fill nodata in raster using value of nearest cell to match NLUM mask
    ind = nd.distance_transform_edt(dst_array.mask, return_distances=False, return_indices=True)
    NVIS_raster_filled = dst_array[tuple(ind)]
    NVIS_raster_clipped = NVIS_raster_filled * (NLUM_ID_raster.mask == False)
        
    # Save as geoTiff
    with rasterio.open(r'N:\Planet-A\Data-Master\NVIS\GRID_NVIS6_0_AUST_PRE_MVS\aus6_0p_mvs.tif', 'w+', **meta_uint8) as dst:
        dst.write_band(1, NVIS_raster_clipped)
    
    # Flatten 2D array to 1D array of valid values only
    dataFlat = NVIS_raster_clipped[NLUM_ID_raster.mask == False]
        
    # Add NVIS to cell_df dataframe
    cell_df['NVIS_PRE-EURO_MVS_ID'] = dataFlat

    # Join the lookup table to the cell_df DataFrame
    cell_df = cell_df.merge(NVIS_MVS_LUT, left_on='NVIS_PRE-EURO_MVS_ID', right_on='MVS_ID', how='left')
    cell_df.rename(columns = {'Major Vegetation Subgroup':'NVIS_PRE-EURO_MVS_NAME'}, inplace = True)
    cell_df = cell_df.drop(columns=['MVS_ID'])

    # Downcast int64 columns and convert object to category to save memory and space
    downcast(cell_df)
    
    cell_df.info()
    
    # Check that there are no NaNs
    print('Number of grid cells =', cell_df.shape[0])
    print('Number of NaNs =', cell_df[cell_df.isna().any(axis=1)].shape[0])



# Save dataframe to feather format
# cell_df.to_feather(cell_df_path)
# cell_df.to_pickle(cell_df_path)

save_NLUM_to_GeoDataFrame()
create_cell_df()
SA2_gdf = join_SA2_ID()
add_NLUM_SA2_ID()
join_SA2_table()
join_ABARES()
join_LGAs()
join_NRM()
join_IBRA()
join_BOM_RR()
join_BOM_DD()
join_Aqueduct()
join_NLUM()
join_POT_IRRIGABLE()
join_NVIS()

cell_df.to_pickle(cell_df_path)
# cell_df = pd.read_pickle(cell_df_path)







# Best not to use as we cannot release PSMA data publically due to licensing

# =============================================================================
# ################################ Add PSMA Property ID to cell_df
# def join_PSMA():
#     
#     global cell_df
#     
#     with rasterio.open(r'N:\Planet-A\Data-Master\property_boundaries\properties.tif') as rst:
#         PROP_ID_raster = rst.read(1, masked=True) # Loads a 2D masked array with nodata masked out
#     
#     # Flatten the 2D array to 1D array of valid values only
#     dataFlat = PROP_ID_raster[NLUM_ID_raster.mask == False]
#     
#     # Add NRM_CODE to cell_df dataframe
#     cell_df['PROPERTY_ID'] = dataFlat
# 
# =============================================================================
