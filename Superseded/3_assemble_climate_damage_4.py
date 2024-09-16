import rasterio, h5py
import numpy as np
import pandas as pd
from multiprocessing import Pool


# Define the function
def runGCMs(gcm):
    
    # List SSPs
    ssps = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    
    # Set paths to input and output data folders
    in_CO2_path = 'N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/'
    in_path = 'N:/Planet-A/Data-Master/Climate_damage/Climate_projection_data/'
    out_GEOTIFF_path = 'N:/Planet-A/Data-Master/Climate_damage/Climate_damage_crops/'
    out_delta_path = 'N:/Planet-A/Data-Master/Climate_damage/Climate_deltas/'
    out_path = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/Climate_damage_crops/'
    
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
        # prec = np.load(in_path + 'wc2.1_2.5m_prec_' + gcm + '_' + ssp + '_2010-2100.npy')
        # tavg = np.load(in_path + 'wc2.1_2.5m_tavg_' + gcm + '_' + ssp + '_2010-2100.npy')
        with h5py.File(in_path + 'wc2.1_2.5m_prec_' + gcm + '_' + ssp + '_2010-2100.h5', 'r') as prec_h5:
            prec = prec_h5['wc2.1_2.5m_prec_' + gcm + '_' + ssp + '_2010-2100'][...]
        with h5py.File(in_path + 'wc2.1_2.5m_tavg_' + gcm + '_' + ssp + '_2010-2100.h5', 'r') as prec_h5:
            tavg = prec_h5['wc2.1_2.5m_tavg_' + gcm + '_' + ssp + '_2010-2100'][...]
        
        # Create deltaP and deltaT arrays (note: arrays are transposed to enable numpy broadcasting). 
        prec_delta = (100 * (prec.T - prec[:, 0]) / prec[:, 0]).T
        tavg_delta = (tavg.T - tavg[:, 0]).T
        
        # Convert 1D delta arrays to 2D and save as multiband GeoTiff - prec_delta
        with rasterio.open(out_delta_path + 'Climate_delta_prec_' + gcm + '_' + ssp + '_2010-2100.tif', 'w+', **meta) as dst:
            for i in range(prec_delta.shape[1]):
                array_2D[xy] = prec_delta[:, i]
                array_2D[NLUM_mask == 0] = -9999
                dst.write_band(i + 1, array_2D)
        
        # Convert 1D delta arrays to 2D and save as multiband GeoTiff - tavg_delta
        with rasterio.open(out_delta_path + 'Climate_delta_tavg_' + gcm + '_' + ssp + '_2010-2100.tif', 'w+', **meta) as dst:
            for i in range(tavg_delta.shape[1]):
                array_2D[xy] = tavg_delta[:, i]
                array_2D[NLUM_mask == 0] = -9999
                dst.write_band(i + 1, array_2D)

        
        # Challinor Table 1 model. Note: intercept (-5.4) not included as it induces a block shift of -5.4% even in 2010. In other words we 'calibrated' the Challinor model to 2010.
        yld_delta = (prec_delta * 0.53 + tavg_delta * -4.9) + 0.06 * np.array(CO2_dict[ssp]['Delta_PPM'])
        
        # Put a floor on yield decline at -100%, round, and convert to int8
        yld_delta = np.where(yld_delta < -100, -100, yld_delta)
        
        # Save climate damage crop (yield delta) to HDF5 and GeoTiff
        fn = 'Climate_damage_crops_' + gcm + '_' + ssp + '_2010-2100.npy'
        # np.save(out_path + fn, np.around(yld_delta).astype(np.int8))
        with h5py.File(out_path + fn, 'w') as h5f:
            h5f.create_dataset(fn, data = np.around(yld_delta).astype(np.int8), chunks = True)
                
        # Convert 1D arrays to 2D and save as multiband GeoTiff
        with rasterio.open(out_GEOTIFF_path + 'Climate_damage_crops_' + gcm + '_' + ssp + '_2010-2100.tif', 'w+', **meta) as dst:
            for i in range(yld_delta.shape[1]):
                array_2D[xy] = yld_delta[:, i]
                array_2D[NLUM_mask == 0] = -9999
                dst.write_band(i + 1, array_2D)


# Main execution code
if __name__ == '__main__':
    
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
    
    # List GCMs
    gcms = ['GCM-Ensembles', 'BCC-CSM2-MR', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'MRI-ESM2-0']
    
    # Run in series (s) or parallel (p)
    sp = 'p' 
    
    if sp == 's':
        
        # Run in series
        for gcm in gcms:
            runGCMs(gcm)
    
    else: 
        # Start multiprocessing pool and run in parallel
        pool = Pool(9)
        ret = pool.map(runGCMs, gcms, chunksize = 1)

            
