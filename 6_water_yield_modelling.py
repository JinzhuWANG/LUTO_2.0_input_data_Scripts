# Script calculates water yield under climate change using INVEST

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio, matplotlib, h5py, os
from rasterio.fill import fillnodata
from rasterio.warp import calculate_default_transform, reproject, Resampling
import concurrent.futures as cf
from ftplib import FTP
from itertools import product
from multiprocessing import Pool

# Set some options
pd.set_option('display.width', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:,.2f}'.format)


################################ Create some helper functions

  
# Convert 1D column to 2D spatial array
def conv_1D_to_2D(in_1D_array):
    array_2D[xy] = np.array(in_1D_array)
    return array_2D.astype(in_1D_array.dtype)


# Print array stats
def desc(inarray):
    print('Shape =', inarray.shape, ' Mean =', inarray.mean(), ' Max =', inarray.max(), ' Min =', inarray.min(), ' NaNs =', np.sum(np.isnan(inarray)))


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





############## Download files from the Soil and Landscape Grid of Australia

def download_file(path):
    ftp = FTP('qld.auscover.org.au')  # connect to host, default port
    ftp.login()
    ftp.cwd('/tern-soils/Products/National_digital_soil_property_maps/' + path[0])
    with open('N:/Planet-A/Data-Master/Soil_Landscape_Grid_Australia/' + path[1], 'wb') as fp:
        ftp.retrbinary('RETR ' + path[1], fp.write)
    ftp.quit()





############## Downsample Soil Landscape Grid of Australia rasters to 1km

def downsample_to_1km(fn):
    
    with rasterio.open('N:/Planet-A/Data-Master/Soil_Landscape_Grid_Australia/' + fn) as src:
        
        # Create an empty destination array 
        dst_array = np.zeros((NLUM_height, NLUM_width), np.float32)
        
        # Reproject/resample input raster to match NLUM mask (meta)
        reproject(rasterio.band(src, 1), dst_array, dst_transform = NLUM_transform, dst_crs = NLUM_crs, resampling = Resampling.bilinear)
        
        # Create mask for filling cells
        if fn == 'DER_000_999_EV_N_P_AU_NAT_C_20150601_fixed.tif': fill_mask = np.where((dst_array > 0) & (dst_array < 250), 1, 0)
        else: fill_mask = np.where(dst_array > 0, 1, 0)
            
        # Fill nodata using inverse distance weighted averaging and mask to NLUM
        dst_array_filled = fillnodata(dst_array, fill_mask, max_search_distance = 100.0) * NLUM_mask
        
        # Save the output to GeoTiff
        with rasterio.open('N:/Planet-A/Data-Master/Water/Water_yield_modelling/' + fn[:-4] + '_1km.tif', 'w+', dtype = 'float32', nodata = 0, **meta) as dst:        
            if 'DE' in fn: mult = 1000
            else: mult = 0.01
            dst.write_band(1, dst_array_filled * mult) # multiplier to get correct units for INVEST





############## Reproject dataset to Albers to run in INVEST software (data needs to be projected)

def reproj_for_INVEST():
    
    file_list = [('N:/Planet-A/Data-Master/WorldClim_CMIP6/Australia/Australia_1km/Annual_20-year_snapshots/Historical_1970-2000/', 'wc2.1_2.5m_prec_Historical_1970-2000_AUS_1km_AnnAvgTot.tif'),
                 ('N:/Planet-A/Data-Master/WorldClim_CMIP6/Australia/Australia_1km/Annual_20-year_snapshots/Historical_1970-2000/', 'wc2.1_2.5m_evap_Historical_1970-2000_AUS_1km_AnnAvgTot.tif'),
                 ('N:/Planet-A/Data-Master/Water/Water_yield_modelling/', 'DES_000_200_EV_N_P_AU_NAT_C_20140801_1km.tif'),
                 ('N:/Planet-A/Data-Master/Water/Water_yield_modelling/', 'AWC_mean.tif'), 
                 ('N:/Planet-A/Data-Master/National_Landuse_Map/', 'NLUM_2010-11_mask.tif')]
    
    dst_crs = 'EPSG:3577'
    
    for file in file_list:
        with rasterio.open(file[0] + file[1]) as src:
            transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
        
            with rasterio.open('N:/Planet-A/Data-Master/Water/Water_yield_modelling/Projected_data_for_INVEST/' + file[1], 'w+', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(source = rasterio.band(src, i),
                              destination = rasterio.band(dst, i),
                              src_transform = src.transform,
                              src_crs = src.crs,
                              dst_transform = transform,
                              dst_crs = dst_crs,
                              resampling = Resampling.nearest)
            



############## Water yield modelling under climate change

def calculate_water_yield_GCMs(params):
    
    # DELETE
    # params = ('GCM-Ensemble', 'ssp126', 'SR')
    
    # Get GCM, SSP, plant type, soil_depth, AWC_brick
    gcm, ssp, pt, soil_depth, AWC_brick = params
    
    # Set some key parameters. Evapotranspiration coefficients from INVEST kc_calculator.xlsx, Eucapypt rooting depth from http://docs.kfri.res.in/KFRI-RR/KFRI-RR136.pdf
    Z = 10
    if pt == 'SR': Kc, root_depth = 0.832, 1000
    else: Kc, root_depth = 1.008, 3500
    
    # Set paths to input and output data folders
    in_path = 'N:/Planet-A/Data-Master/WorldClim_CMIP6/Australia/Australia_1km/Annual_yearly_interpolated_1970-2100/HDF5/'
    out_path_gtif = 'N:/Planet-A/Data-Master/Water/Water_yield_modelling/Water_yield_projections/GeoTiff/'
    out_path_h5 = 'N:/Planet-A/Data-Master/Water/Water_yield_modelling/Water_yield_projections/HDF5/'
    
    os.makedirs(out_path_gtif, exist_ok = True)
    os.makedirs(out_path_h5, exist_ok = True)
    
    # Open NLUM mask raster and get metadata
    with rasterio.open('N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as src:
        NLUM_mask = src.read(1)
        
        # Get metadata and update parameters (note: count = 91 indicates multibands rasters where each band is a year)
        meta = src.meta.copy()
        meta.update(compress = 'lzw', driver = 'GTiff', dtype = 'float32', nodata = -9999, count = 131)
        
        # Set up some data structures to enable conversion on 1D arrays to 2D
        array_2D = np.zeros(NLUM_mask.shape, dtype = np.float32) 
        xy = tuple([a.astype(np.int16) for a in np.nonzero(NLUM_mask == 1)])
                   
    # Load yearly rainfall and potential evapotranspiration data and transpose to support numpy broadcasting
    prec_fn = 'wc2.1_2.5m_prec_' + gcm + '_' + ssp + '_1970-2100_AUS_1km_AnnAvgTot_Yearly'
    ph5 = in_path + prec_fn + '.h5'
    with h5py.File(ph5, 'r') as prec_h5:
        prec = prec_h5[prec_fn][...].T
    
    # Load potential evapotranspiration, calculate phi as the evapotranspiration coefficient Kc x potential evapotranspiration / precipitation
    evap_fn = 'wc2.1_2.5m_evap_' + gcm + '_' + ssp + '_1970-2100_AUS_1km_AnnAvgTot_Yearly'
    eh5 = in_path + evap_fn + '.h5'
    with h5py.File(eh5, 'r') as evap_h5:
        phi = Kc * evap_h5[evap_fn][...].T / prec
    
    # Find the minimum of soil depth and root depth
    depth = np.minimum(root_depth, soil_depth)
    
    # Calculate plant available water content (mm) based on the weighted average AWC (values between 0 and 1) of the hydrologically active soil zone based on the data for 6 soil depths
    awc_mm = np.where(depth < 5,    AWC_brick[0, :],
              np.where(depth < 15,  (AWC_brick[0, :] * 5 + AWC_brick[1, :] * (depth - 5)) / depth, 
              np.where(depth < 30,  (AWC_brick[0, :] * 5 + AWC_brick[1, :] * 10 + AWC_brick[2, :] * (depth - 15)) / depth, 
              np.where(depth < 60,  (AWC_brick[0, :] * 5 + AWC_brick[1, :] * 10 + AWC_brick[2, :] * 15 + AWC_brick[3, :] * (depth - 30)) / depth, 
              np.where(depth < 100, (AWC_brick[0, :] * 5 + AWC_brick[1, :] * 10 + AWC_brick[2, :] * 15 + AWC_brick[3, :] * 30 + AWC_brick[4, :] * (depth - 60)) / depth, 
                                    (AWC_brick[0, :] * 5 + AWC_brick[1, :] * 10 + AWC_brick[2, :] * 15 + AWC_brick[3, :] * 30 + AWC_brick[4, :] * 40 + AWC_brick[5, :] * (depth - 100)) / depth))))) * depth
        
    # Calculate climate_w variable and cap at 5.0
    climate_w = (Z * awc_mm / prec) + 1.25
    climate_w[climate_w > 5.0] = 5.0
    
    # Compute ratio of actual evapotranspiration to precipitation using Budyko
    aet_p = 1.0 + phi - (1.0 + phi ** climate_w) ** (1.0 / climate_w)
    
    # Take the minimum of the following values (phi, aet_p) to determine the evapotranspiration partition of the water balance (??? INVEST code says see users guide???)
    aet_p = np.where(phi < aet_p, phi, aet_p)
    
    # Water yield calculation (mm), convert to ML per ha by dividing by 100, and transpose back to original shape
    water_yield_ML_HA = ((1 - aet_p) * prec / 100).T
    
    # Save water yield data to HDF5 
    fn = 'Water_yield_' + gcm + '_' + ssp + '_1970-2100_' + pt + '_ML_HA'
    with h5py.File(out_path_h5 + fn + '.h5', 'w') as h5f:
        h5f.create_dataset(fn + '.h5', data = water_yield_ML_HA, chunks = True)
            
    # Convert 1D arrays to 2D and save as multiband GeoTiff
    with rasterio.open(out_path_gtif + fn + '.tif', 'w+', **meta) as dst:
        for i in range(water_yield_ML_HA.shape[1]):
            array_2D[xy] = water_yield_ML_HA[:, i]
            array_2D[NLUM_mask == 0] = -9999
            dst.write_band(i + 1, array_2D)




def calculate_ensembles(params):
    
    # For full production run, comment out when testing
    ssp, pt = params
    # ssp, pt = 'ssp126', 'DR'
    
    # Specify the GCMs, SSPs, periods, and Layers as lists
    gcms = ['BCC-CSM2-MR', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'MRI-ESM2-0']
    
    # Set some paths to input and output data
    inpath = 'N:/Planet-A/Data-Master/Water/Water_yield_modelling/Water_yield_projections/HDF5/'
    path_gtif = 'N:/Planet-A/Data-Master/Water/Water_yield_modelling/Water_yield_projections/GeoTiff/'
    path_hdf5 = 'N:/Planet-A/Data-Master/Water/Water_yield_modelling/Water_yield_projections/HDF5/'
    path_LUTO = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/'
    
    nbands = 131
    
    os.makedirs(path_gtif, exist_ok = True)
    
    # Open NLUM mask raster and get mask raster and metadata
    with rasterio.open('N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as rst:
        NLUM_mask = rst.read(1)
        meta = rst.meta.copy()
        meta.update(compress = 'lzw', driver = 'GTiff', dtype = 'float32', nodata = -9999)
        meta.pop('count') # Need to add count manually when exporting GeoTiffs
        
        # Set some data structures to enable conversion on 1D arrays to 2D
        array_2D = np.zeros(NLUM_mask.shape, dtype = np.float32)
        xy = tuple([a.astype(np.int16) for a in np.nonzero(NLUM_mask == 1)])
    
    # Create container for data for each GCM for creating ensembles
    brick = np.zeros((len(gcms), nbands, np.sum(NLUM_mask == 1))).astype(np.float32) - 9999
    
    # Loop through GCMs to load annual layers
    for i, gcm in enumerate(gcms):
        
        # Load water yield HDF5 file into the brick data container
        fn = 'Water_yield_' + gcm + '_' + ssp + '_1970-2100_' + pt + '_ML_HA.h5'
        with h5py.File(inpath + fn, 'r') as h5:
            brick[i, ...] = h5[fn][...].T
    
    # Create ensemble averages and stdevs, mask the output
    ens_mean = np.mean(brick, axis = 0)
    ens_min = np.min(brick, axis = 0)
    ens_max = np.max(brick, axis = 0)
    
    dic = {'min': ens_min, 'mean': ens_mean, 'max': ens_max}
    
    # Save water yield data to HDF5 and GeoTiff bricks 1970 - 2100
    for key, data in dic.items():
        fn = 'Water_yield_GCM-Ensemble_' + ssp + '_1970-2100_' + pt + '_ML_HA_' + key
        with h5py.File(path_hdf5 + fn + '.h5', 'w') as h5f:
            h5f.create_dataset(fn, data = data, chunks = True)
        
        # Convert 1D arrays to 2D and save as multiband GeoTiff
        with rasterio.open(path_gtif + fn + '.tif', 'w+', **meta, count = nbands) as dst:
            for i in range(nbands):
                array_2D[xy] = data[i, :]
                array_2D[NLUM_mask == 0] = -9999
                dst.write_band(i + 1, array_2D)
                
    # Save mean water yield data to HDF5 brick for input into LUTO 2010 - 2100
    fn = 'Water_yield_GCM-Ensemble_' + ssp + '_2010-2100_' + pt + '_ML_HA_mean'
    with h5py.File(path_LUTO + fn + '.h5', 'w') as h5f:
        h5f.create_dataset(fn, data = ens_mean[40:, :], chunks = True)
        
    # Convert historical 1D arrays to 2D and save as GeoTiff
    if ssp == 'ssp126':
        with rasterio.open(path_gtif + 'Water_yield_Historical_1970-2000_' + pt + '_ML_HA.tif', 'w+', **meta, count = 1) as dst:
            array_2D[xy] = ens_mean[15, :] # Column 15 in the brick is 1985 which is the historical data
            array_2D[NLUM_mask == 0] = -9999
            dst.write_band(1, array_2D)



if __name__ == '__main__':
    
    # Open NLUM mask raster and get metadata
    with rasterio.open('N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as rst:
    
        # Read geotiff to numpy array
        NLUM_mask = rst.read(1) # Loads a 2D masked array with nodata masked out
        
        # Get transform and metadata and update parameters
        NLUM_transform = rst.transform
        NLUM_height = rst.height
        NLUM_width = rst.width
        NLUM_crs = rst.crs
        meta = rst.meta.copy()
        meta.update(compress='lzw', driver='GTiff') # dtype='int32', nodata='-99')
        [meta.pop(key) for key in ['dtype', 'nodata']] # Need to add dtype and nodata manually when exporting GeoTiffs
            
        # Set some data structures to enable conversion on 1D arrays to 2D
        array_2D = np.zeros(NLUM_mask.shape, dtype = np.float32) -99
        xy = tuple([a.astype(np.int16) for a in np.nonzero(NLUM_mask == 1)])
    
    ############ Download SLGA files
    
    # Specify a list of folder/filename tuples of required 
    paths = [('Depth_of_Soil', 'DES_000_200_EV_N_P_AU_NAT_C_20140801.tif'),
             ('Depth_of_Regolith', 'DER_000_999_EV_N_P_AU_NAT_C_20150601_fixed.tif'),
             ('AWC', 'AWC_000_005_EV_N_P_AU_NAT_C_20140801.tif'),
             ('AWC', 'AWC_005_015_EV_N_P_AU_NAT_C_20140801.tif'),
             ('AWC', 'AWC_015_030_EV_N_P_AU_NAT_C_20140801.tif'),
             ('AWC', 'AWC_030_060_EV_N_P_AU_NAT_C_20140801.tif'),
             ('AWC', 'AWC_060_100_EV_N_P_AU_NAT_C_20140801.tif'),
             ('AWC', 'AWC_100_200_EV_N_P_AU_NAT_C_20140801.tif')]
    
    # Download using 7 threads using a with statement to ensure threads are cleaned up promptly
    # with cf.ThreadPoolExecutor(max_workers = 9) as executor:
    #     executor.map(download_file, paths)
    
    # Downsample to 1km resolution in parallel
    # fns = [path[1] for path in paths]
    # with cf.ThreadPoolExecutor(max_workers = 9) as executor:
    #     out = executor.map(downsample_to_1km, fns)
    
    ############ 
    
    # # Load 2D input data arrays and flatten to 1D
    # with rasterio.open('N:/Planet-A/Data-Master/Water/Water_yield_modelling/DER_000_999_EV_N_P_AU_NAT_C_20150601_fixed_1km.tif') as rst:
    #     soil_depth = np.round(rst.read(1))[NLUM_mask == 1].astype(np.float32)
        
    # # Load depth-specific AWC layers and assemble into data brick
    # fns = ['AWC_000_005_EV_N_P_AU_NAT_C_20140801_1km.tif', 'AWC_005_015_EV_N_P_AU_NAT_C_20140801_1km.tif', 'AWC_015_030_EV_N_P_AU_NAT_C_20140801_1km.tif', 'AWC_030_060_EV_N_P_AU_NAT_C_20140801_1km.tif', 'AWC_060_100_EV_N_P_AU_NAT_C_20140801_1km.tif', 'AWC_100_200_EV_N_P_AU_NAT_C_20140801_1km.tif']
    # AWC_brick = np.zeros((len(fns), np.sum(NLUM_mask == 1)), dtype = np.float32)   
    # for i, fn in enumerate(fns):
    #     with rasterio.open('N:/Planet-A/Data-Master/Water/Water_yield_modelling/' + fn) as rst:
    #         AWC_brick[i, :] = rst.read(1)[NLUM_mask == 1]
    
    
    # List GCMs, SSPs & plant types
    gcms = ['BCC-CSM2-MR', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'MRI-ESM2-0']
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    plant_type = ['SR', 'DR'] # Short-rooted plants (crops, grass) vs deep-rooted (trees)
    
    # # Create a list of parameters for calculating water yield for each GCM, SSP, and plant type
    # params_list = list(product(plant_type))
    # params_list = [list(x) for x in params_list]
    # [x.extend([soil_depth, AWC_brick]) for x in params_list]
    
    # # Calculate water yield historical in parallel
    # with Pool(2) as pool:
    #     ret = pool.map(calculate_water_yield_historical, params_list, chunksize = 1)
            
    # # Create a list of parameters for calculating water yield for each GCM, SSP, and plant type
    # params_list = list(product(gcms, ssps, plant_type))
    # params_list = [list(x) for x in params_list]
    # [x.extend([soil_depth, AWC_brick]) for x in params_list]
    
    # # Calculate water yield in parallel
    # with Pool(8) as pool:
    #     ret = pool.map(calculate_water_yield_GCMs, params_list, chunksize = 1)
    
    # Calculate water yield ensembles in parallel
    params_list = list(product(ssps, plant_type))
    with Pool(8) as pool:
        ret = pool.map(calculate_ensembles, params_list, chunksize = 1)

    
    
    
    
    # Calculate depth-weighted mean AWC for use in the INVEST software (only takes 1 layer)
    # AWC_mean = (AWC_brick[0, :] * 5 + AWC_brick[1, :] * 10 + AWC_brick[2, :] * 15 + AWC_brick[3, :] * 30 + AWC_brick[4, :] * 40 + AWC_brick[5, :] * 100) / 200
    # with rasterio.open('N:/Planet-A/Data-Master/Water/Water_yield_modelling/AWC_mean.tif', 'w+', dtype = 'float32', nodata = -99, **meta) as out:
    #     out.write_band(1, conv_1D_to_2D(AWC_mean))   
    
    
    
    """ Helpful code for exploring results
    
    # Read cell_df from disk, just grab the CELL_ID column
    cell_df = pd.read_pickle('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.pkl')[['CELL_ID', 'CELL_HA', 'HR_DRAINDIV_NAME']]
    cell_df['WATER_USE_TREES_ML_HA'] = pd.read_pickle('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.pkl')[['WATER_USE_TREES_KL_HA']] / 1000
    
    cell_df['water_impact_of_trees_ML_HA'] = cell_df.eval('water_yield_under_crops_ML_HA - water_yield_under_forest_ML_HA')
    
    cell_df['water_yield_under_crops_GL'] = cell_df.eval('water_yield_under_crops_ML_HA * CELL_HA / 1000')
    cell_df['water_yield_under_forest_GL'] = cell_df.eval('water_yield_under_forest_ML_HA * CELL_HA / 1000')
    cell_df['water_impact_of_trees_GL'] = cell_df.eval('water_impact_of_trees_ML_HA * CELL_HA / 1000')

    cell_df['WATER_USE_TREES_GL'] = cell_df.eval('WATER_USE_TREES_ML_HA * CELL_HA / 1000')
    
    
    # Open a new GeoTiFF file
    with rasterio.open('N:/Planet-A/Data-Master/Water/Water_yield_modelling/water_impact_of_trees_ML_HA.tif', 'w+', dtype = 'float32', nodata = -99, **meta) as out:
        out.write_band(1, conv_1D_to_2D(cell_df['water_impact_of_trees_ML_HA']))
    
    cell_df.groupby('HR_DRAINDIV_NAME')[['water_yield_under_crops_GL', 'water_yield_under_forest_GL', 'water_impact_of_trees_GL', 'WATER_USE_TREES_GL']].sum()
    """