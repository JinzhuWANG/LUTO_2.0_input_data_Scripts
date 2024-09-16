# Quantifies the impacts of climate change on crops using the Challinor model
# Calculates climate damage for each GCM and for the ensemble of GCMs
# Simulates uncertainty at 95% confidence interval using the standard deviation of temp and rainfall and S.E.M. of Challinor coefficients
# Key outputs are low, med, high estimates of climate impacts
# Takes about 18 hours to run on Denethor

import rasterio, h5py, os
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from itertools import product

# Calculates CO2 trajectories
def assemble_CO2_data():

    in_CO2_path = 'N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/'

    # Load CO2 data (Meinshausen et al.) and concatenate historical data (2010 - 2014) with projected data (2015 onwards)
    CO2_hist = pd.read_csv(in_CO2_path + 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_0000-2014.csv')
    CO2_ssp126_tmp = pd.read_csv(in_CO2_path + 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-IMAGE-ssp126-1-2-1_gr1-GMNHSH_2015-2500.csv')
    CO2_ssp245_tmp = pd.read_csv(in_CO2_path + 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-2-1_gr1-GMNHSH_2015-2500.csv')
    CO2_ssp370_tmp = pd.read_csv(in_CO2_path + 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-AIM-ssp370-1-2-1_gr1-GMNHSH_2015-2500.csv')
    CO2_ssp850_tmp = pd.read_csv(in_CO2_path + 'mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-REMIND-MAGPIE-ssp585-1-2-1_gr1-GMNHSH_2015-2500.csv')
    
    CO2_df = CO2_hist.loc[CO2_hist['year'] >= 2010, ['year', 'data_mean_sh']].reset_index(drop = True)
    CO2_ssp126 = CO2_ssp126_tmp.loc[(CO2_ssp126_tmp['year'] >= 2015) & (CO2_ssp126_tmp['year'] <= 2100), ['year', 'data_mean_sh']].reset_index(drop = True)
    CO2_ssp245 = CO2_ssp245_tmp.loc[(CO2_ssp245_tmp['year'] >= 2015) & (CO2_ssp245_tmp['year'] <= 2100), ['year', 'data_mean_sh']].reset_index(drop = True)
    CO2_ssp370 = CO2_ssp370_tmp.loc[(CO2_ssp370_tmp['year'] >= 2015) & (CO2_ssp370_tmp['year'] <= 2100), ['year', 'data_mean_sh']].reset_index(drop = True)
    CO2_ssp850 = CO2_ssp850_tmp.loc[(CO2_ssp850_tmp['year'] >= 2015) & (CO2_ssp850_tmp['year'] <= 2100), ['year', 'data_mean_sh']].reset_index(drop = True)
    
    CO2_ssp126 = pd.concat([CO2_df, CO2_ssp126]).reset_index(drop = True)
    CO2_ssp245 = pd.concat([CO2_df, CO2_ssp245]).reset_index(drop = True)
    CO2_ssp370 = pd.concat([CO2_df, CO2_ssp370]).reset_index(drop = True)
    CO2_ssp850 = pd.concat([CO2_df, CO2_ssp850]).reset_index(drop = True)
    
    # Calculate CO2 deltas
    CO2_ssp126['Delta_PPM'] = (CO2_ssp126['data_mean_sh'] - CO2_ssp126.loc[CO2_ssp126['year'] == 2010, 'data_mean_sh'][0]).astype(np.float32)
    CO2_ssp245['Delta_PPM'] = (CO2_ssp245['data_mean_sh'] - CO2_ssp245.loc[CO2_ssp245['year'] == 2010, 'data_mean_sh'][0]).astype(np.float32)
    CO2_ssp370['Delta_PPM'] = (CO2_ssp370['data_mean_sh'] - CO2_ssp370.loc[CO2_ssp370['year'] == 2010, 'data_mean_sh'][0]).astype(np.float32)
    CO2_ssp850['Delta_PPM'] = (CO2_ssp850['data_mean_sh'] - CO2_ssp850.loc[CO2_ssp850['year'] == 2010, 'data_mean_sh'][0]).astype(np.float32)
    
    # Save dataframes
    CO2_ssp126.to_pickle(in_CO2_path + 'CO2_ssp126_2010-2100_dataframe.pkl')
    CO2_ssp245.to_pickle(in_CO2_path + 'CO2_ssp245_2010-2100_dataframe.pkl')
    CO2_ssp370.to_pickle(in_CO2_path + 'CO2_ssp370_2010-2100_dataframe.pkl')
    CO2_ssp850.to_pickle(in_CO2_path + 'CO2_ssp850_2010-2100_dataframe.pkl')
    
    
    
# Calculates crop yield damage for each GCM
def calc_crop_damage_for_GCMs(gcm):
    
    # List SSPs
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    # Set paths to input and output data folders
    in_CO2_path = 'N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/'
    in_path = 'N:/Planet-A/Data-Master/WorldClim_CMIP6/Australia/Australia_1km/Annual_yearly_interpolated_1970-2100/HDF5/'
    out_path_gtif = 'N:/Planet-A/Data-Master/Climate_damage/Climate_damage_crops/'
    out_path_deltas = 'N:/Planet-A/Data-Master/Climate_damage/Climate_deltas/'
    out_path = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/'
    
    os.makedirs(out_path_gtif, exist_ok = True)
    os.makedirs(out_path_deltas, exist_ok = True)
    
    # Open NLUM mask raster and get metadata
    with rasterio.open('N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as src:
        NLUM_mask = src.read(1)
        
        # Get metadata and update parameters (note: count = 91 indicates multibands rasters where each band is a year)
        meta = src.meta.copy()
        meta.update(compress = 'lzw', driver = 'GTiff', dtype = 'float32', nodata = -9999, count = 91)
        
        # Set up some data structures to enable conversion on 1D arrays to 2D
        array_2D = np.zeros(NLUM_mask.shape, dtype = np.float32) 
        xy = tuple([a.astype(np.int16) for a in np.nonzero(NLUM_mask == 1)])
    
    # Load CO2 data
    CO2_ssp126 = pd.read_pickle(in_CO2_path + 'CO2_ssp126_2010-2100_dataframe.pkl')
    CO2_ssp245 = pd.read_pickle(in_CO2_path + 'CO2_ssp245_2010-2100_dataframe.pkl')
    CO2_ssp370 = pd.read_pickle(in_CO2_path + 'CO2_ssp370_2010-2100_dataframe.pkl')
    CO2_ssp850 = pd.read_pickle(in_CO2_path + 'CO2_ssp850_2010-2100_dataframe.pkl')
    
    # Add CO2 deltas to dictionary
    CO2_dict = {'ssp126': CO2_ssp126, 'ssp245': CO2_ssp245, 'ssp370': CO2_ssp370, 'ssp585': CO2_ssp850}
    
    # Loop through SSPs to calculate climate change yield damage
    for ssp in ssps:
        
        # Load yearly rainfall and temp data
        prec_fn = 'wc2.1_2.5m_prec_' + gcm + '_' + ssp + '_1970-2100_AUS_1km_AnnAvgTot_Yearly'
        if gcm == 'GCM-Ensemble': ph5 = in_path + prec_fn + '_mean.h5'
        else: ph5 = in_path + prec_fn + '.h5'
        with h5py.File(ph5, 'r') as prec_h5:
            prec = prec_h5[prec_fn][:, 40:]
            
        tavg_fn = 'wc2.1_2.5m_tavg_' + gcm + '_' + ssp + '_1970-2100_AUS_1km_AnnAvgTot_Yearly'
        if gcm == 'GCM-Ensemble': th5 = in_path + tavg_fn + '_mean.h5'
        else: th5 = in_path + tavg_fn + '.h5'
        with h5py.File(th5, 'r') as tavg_h5:
            tavg = tavg_h5[tavg_fn][:, 40:]
        
        # Create deltaP and deltaT arrays (note: arrays are transposed to enable numpy broadcasting). 
        prec_delta = (100 * (prec.T - prec[:, 0]) / prec[:, 0]).T
        tavg_delta = (tavg.T - tavg[:, 0]).T
        
        # Convert 1D delta arrays to 2D and save as multiband GeoTiff - prec_delta
        with rasterio.open(out_path_deltas + 'Climate_delta_prec_' + gcm + '_' + ssp + '_2010-2100.tif', 'w+', **meta) as dst:
            for i in range(prec_delta.shape[1]):
                array_2D[xy] = prec_delta[:, i]
                array_2D[NLUM_mask == 0] = -9999
                dst.write_band(i + 1, array_2D)
        
        # Convert 1D delta arrays to 2D and save as multiband GeoTiff - tavg_delta
        with rasterio.open(out_path_deltas + 'Climate_delta_tavg_' + gcm + '_' + ssp + '_2010-2100.tif', 'w+', **meta) as dst:
            for i in range(tavg_delta.shape[1]):
                array_2D[xy] = tavg_delta[:, i]
                array_2D[NLUM_mask == 0] = -9999
                dst.write_band(i + 1, array_2D)
            
        # Challinor Table 1 model. Note: intercept (-5.4) not included as it induces a block shift of -5.4% even in 2010. In other words we 'calibrated' the Challinor model to 2010.
        yld_delta = (prec_delta * 0.53 + tavg_delta * -4.9) + 0.06 * np.array(CO2_dict[ssp]['Delta_PPM'])
        
        # Put a floor on yield decline at -100%
        yld_delta = np.where(yld_delta < -100, -100, yld_delta)
        
        # Save climate damage crop (yield delta) to HDF5 and GeoTiff - spit out all HDF5s or just the ensembles?
        # fn = 'Climate_damage_crops_' + gcm + '_' + ssp + '_2010-2100.h5'
        # with h5py.File(out_path + fn, 'w') as h5f:
        #     h5f.create_dataset(fn, data = np.around(yld_delta).astype(np.int8), chunks = True)
        if gcm == 'GCM-Ensemble':
            fn = 'Climate_damage_crops_' + gcm + '_' + ssp + '_2010-2100_med'
            with h5py.File(out_path + fn + '.h5', 'w') as h5f:
                h5f.create_dataset(fn, data = yld_delta, chunks = True)
                
        # Convert 1D arrays to 2D and save as multiband GeoTiff
        if gcm == 'GCM-Ensemble': fn = 'Climate_damage_crops_' + gcm + '_' + ssp + '_2010-2100_med'
        else: fn = 'Climate_damage_crops_' + gcm + '_' + ssp + '_2010-2100'
        with rasterio.open(out_path_gtif + 'Climate_damage_crops_' + gcm + '_' + ssp + '_2010-2100.tif', 'w+', **meta) as dst:
            for i in range(yld_delta.shape[1]):
                array_2D[xy] = yld_delta[:, i]
                array_2D[NLUM_mask == 0] = -9999
                dst.write_band(i + 1, array_2D)



# Calculates GCM high and low estimates using uncertainty in both climate (i.e., across GCMs) and the Challinor model
def simulate_yield_damage_uncertainty(ssp):
    
    # Set paths to input and output data folders
    in_CO2_path = 'N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/'
    in_path = 'N:/Planet-A/Data-Master/WorldClim_CMIP6/Australia/Australia_1km/Annual_yearly_interpolated_1970-2100/HDF5/'
    out_path = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/'
    sims_path = 'D:/bbrett/'
    
    # Load CO2 data
    CO2_ssp126 = pd.read_pickle(in_CO2_path + 'CO2_ssp126_2010-2100_dataframe.pkl')
    CO2_ssp245 = pd.read_pickle(in_CO2_path + 'CO2_ssp245_2010-2100_dataframe.pkl')
    CO2_ssp370 = pd.read_pickle(in_CO2_path + 'CO2_ssp370_2010-2100_dataframe.pkl')
    CO2_ssp850 = pd.read_pickle(in_CO2_path + 'CO2_ssp850_2010-2100_dataframe.pkl')
    
    # Add CO2 deltas to dictionary
    CO2_dict = {'ssp126': CO2_ssp126, 'ssp245': CO2_ssp245, 'ssp370': CO2_ssp370, 'ssp585': CO2_ssp850}
    
    # Set prec and temp filenames
    fn_prec = 'wc2.1_2.5m_prec_GCM-Ensemble_' + ssp + '_1970-2100_AUS_1km_AnnAvgTot_Yearly'
    fn_tavg = 'wc2.1_2.5m_tavg_GCM-Ensemble_' + ssp + '_1970-2100_AUS_1km_AnnAvgTot_Yearly'
    
    # Load GCM Ensemble yearly rainfall and temp mean and standard deviation for the years 2010 - 2100
    with h5py.File(in_path + fn_prec + '_mean.h5', 'r') as prec_h5:
        prec = prec_h5[fn_prec][:, 40:]
    with h5py.File(in_path + fn_tavg + '_mean.h5', 'r') as tavg_h5:
        tavg = tavg_h5[fn_tavg][:, 40:]
    with h5py.File(in_path + fn_prec +'_stdev.h5', 'r') as prec_h5:
        prec_std = prec_h5[fn_prec][:, 40:]
    with h5py.File(in_path + fn_tavg + '_stdev.h5', 'r') as tavg_h5:
        tavg_std = tavg_h5[fn_tavg][:, 40:]
    
    # Specify the number of simulations for Monte Carlo analysis of low, med, high ensemble estimates
    nsims = 100
    
    # Create a HDF5 dataset to hold the simulations
    sfn = 'Climate_damage_crops_' + ssp + '_' + str(nsims) + '_Simulations_2010-2100'
    if os.path.exists(sims_path + sfn + '.h5'): os.remove(sims_path + sfn + '.h5')
    with h5py.File(sims_path + sfn + '.h5', 'a') as sf:
        yld_delta_dset = sf.create_dataset(sfn, (nsims, prec.shape[0], prec.shape[1]), dtype = np.int8, chunks = True)
    
        # Monte Carlo simulation of yield deltas
        for i in range(nsims):
            
            # Random draw of Challinor et al coefficients
            params_rnd = np.random.normal([0.53, -4.9, 0.06], [0.18, 1.25, 0.02]).astype(np.float32)
            
            # Random draw of temp and prec layers based on ensemble mean and std layers
            prec_rnd = np.random.normal(prec, prec_std).astype(np.float32)
            tavg_rnd = np.random.normal(tavg, tavg_std).astype(np.float32)
            
            # Create deltaP and deltaT for staochastic arrays (note: arrays are transposed to enable numpy broadcasting)
            prec_delta = (100 * (prec_rnd.T - prec[:, 0]) / prec[:, 0]).T
            tavg_delta = (tavg_rnd.T - tavg[:, 0]).T
            
            # Calculate yield delta using Challinor model and add to data container
            yld_delta_dset[i, ...] = np.around((prec_delta * params_rnd[0] + tavg_delta * params_rnd[1]) + params_rnd[2] * np.array(CO2_dict[ssp]['Delta_PPM'])).astype(np.int8)


        
def calc_high_and_low_estimates():
    
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    out_path = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/'
    out_path_gtif = 'N:/Planet-A/Data-Master/Climate_damage/Climate_damage_crops/'
    sims_path = 'D:/bbrett/'
    nsims = 100

    # Open NLUM mask raster and get metadata
    with rasterio.open('N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as src:
        NLUM_mask = src.read(1)
        
        # Get metadata and update parameters (note: count = 91 indicates multibands rasters where each band is a year)
        meta = src.meta.copy()
        meta.update(compress = 'lzw', driver = 'GTiff', dtype = 'float32', nodata = -9999, count = 91)
        
        # Set up some data structures to enable conversion on 1D arrays to 2D
        array_2D = np.zeros(NLUM_mask.shape, dtype = np.float32) 
        xy = tuple([a.astype(np.int16) for a in np.nonzero(NLUM_mask == 1)])
        
    for ssp in ssps:
        
        sfn = 'Climate_damage_crops_' + ssp + '_' + str(nsims) + '_Simulations_2010-2100'
        
        # Load entire file from disk
        with h5py.File(sims_path + sfn + '.h5', 'r') as yld_h5:
            yld_delta = yld_h5[sfn][()]
            
            # Calculate the mean of simulated yield deltas
            yld_delta_mean = np.mean(yld_delta, axis = 0).astype(np.float32)
            
            # Calculate the standard deviation of yield deltas
            yld_delta_std = np.std(yld_delta, axis = 0).astype(np.float32)
        
        # Release the memory
        yld_delta = None
        del yld_delta
    
        # Calculate high and low estimates as 2.5% and 97.5% limits of the distribution from simulations (med estimates are exported in function above)
        lh_dict = {'low': -1.96, 'high': 1.96}
        
        for lh in lh_dict.items():
            
            # Calculate low/high limits
            data = yld_delta_mean + (lh[1] * yld_delta_std)
            
            # Smooth the low/high limits using Savitsky-Golay filter
            data = savgol_filter(data, 51, 3, axis = 1)
            
            # Save climate damage crop (yield delta) to HDF5 and GeoTiff
            fn = 'Climate_damage_crops_GCM-Ensemble_' + ssp + '_2010-2100_' + lh[0]
            with h5py.File(out_path + fn + '.h5', 'w') as h5f:
                h5f.create_dataset(fn, data = data, chunks = True)
                
            # Convert 1D arrays to 2D and save as multiband GeoTiff
            with rasterio.open(out_path_gtif + fn + '.tif', 'w+', **meta) as dst:
                for i in range(data.shape[1]):
                    array_2D[xy] = data[:, i]
                    array_2D[NLUM_mask == 0] = -9999
                    dst.write_band(i + 1, array_2D)



# Main execution code
if __name__ == '__main__':
    
    # assemble_CO2_data()
    
    # List GCMs
    gcms = ['GCM-Ensemble', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'MRI-ESM2-0']
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

    with Pool(9) as pool:
        ret = pool.map(calc_crop_damage_for_GCMs, gcms, chunksize = 1)

    with Pool(4) as pool:
        ret = pool.map(simulate_yield_damage_uncertainty, ssps, chunksize = 1)

    calc_high_and_low_estimates()
    

"""
# Some code for plotting the original against the filtered data
h1 = h5py.File('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/Climate_damage_crops_GCM-Ensembles_ssp585_2010-2100_low.h5', 'r')
h1a = h1['Climate_damage_crops_GCM-Ensembles_ssp585_2010-2100_low.h5'][()]

h1 = h5py.File('N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/Climate_damage_crops_GCM-Ensembles_ssp585_2010-2100_low.h5t', 'r')
h1t = h1['Climate_damage_crops_GCM-Ensembles_ssp585_2010-2100_low.h5'][()]

plt.plot(range(2010, 2101), h1a[11110, :], color='red', label = 'Filter')
plt.plot(range(2010, 2101), h1t[11110, :], color='blue', label = 'Filter2')
plt.show()


# Applies Savitsky-Golay filter on files to save recreating them
def smooth_high_and_low_estimates(ssps_lhs):
    
    ssp = ssps_lhs[0]
    lh = ssps_lhs[1]
    
    pth = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/4D_Spatial_SSP_Timeseries/'
    fn = 'Climate_damage_crops_GCM-Ensembles_' + ssp + '_2010-2100_' + lh + '.h5'
    
    with h5py.File(pth + fn, 'r') as h5:
        arr = h5[fn][()]
    
    # Smooth the low/high limits using Savitsky-Golay filter
    arr = savgol_filter(arr, 51, 3, axis = 1)

    with h5py.File(pth + fn + 't', 'w') as h5f:
        h5f.create_dataset(fn, data = arr, chunks = True)
"""