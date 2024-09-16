import os, rasterio, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from multiprocessing import Pool

# List GCMs and SSPs
gcms = ['GCM_ensembles']# , 'BCC-CSM2-MR', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'MRI-ESM2-0']
ssps = ['ssp126']#, 'ssp245', 'ssp370', 'ssp585']

# Set paths to input and output data folders
in_path = 'N:/Planet-A/Data-Master/Climate_damage/Climate_projection_data/'
out_path = 'N:/Planet-A/Data-Master/Climate_damage/Climate_damage_crops/'
LUTO_data_path = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/Climate_damage_crops/'

# Open NLUM mask raster and get metadata
with rasterio.open('N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as src:
    NLUM_mask = src.read(1)
    
    # Get metadata and update parameters (note: count = 91 indicates multibands rasters where each band is a year)
    NLUM_transform = src.transform
    meta = src.meta.copy()
    meta.update(compress = 'lzw', driver = 'GTiff', dtype = 'float32', nodata = -9999, count = 91)
    
    # Set up some data structures to enable conversion on 1D arrays to 2D
    array_2D = np.zeros(NLUM_mask.shape)
    xy = np.nonzero(NLUM_mask == 1)

# Load CO2 data (Meinshausen et al.) and concatenate historical data (2010 - 2014) with projected data (2015 onwards)
CO2_hist = pd.read_csv('N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_CMIP_UoM-CMIP-1-2-0_gr1-GMNHSH_0000-2014.csv')
CO2_ssp126_tmp = pd.read_csv('N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-IMAGE-ssp126-1-2-1_gr1-GMNHSH_2015-2500.csv')
CO2_ssp245_tmp = pd.read_csv('N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-MESSAGE-GLOBIOM-ssp245-1-2-1_gr1-GMNHSH_2015-2500.csv')
CO2_ssp370_tmp = pd.read_csv('N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-AIM-ssp370-1-2-1_gr1-GMNHSH_2015-2500.csv')
CO2_ssp850_tmp = pd.read_csv('N:/Planet-A/Data-Master/Climate_damage/CO2_downloads/mole-fraction-of-carbon-dioxide-in-air_input4MIPs_GHGConcentrations_ScenarioMIP_UoM-REMIND-MAGPIE-ssp585-1-2-1_gr1-GMNHSH_2015-2500.csv')

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
CO2_ssp126['Delta_PPM'] = CO2_ssp126['data_mean_sh'] - CO2_ssp126.loc[CO2_ssp126['year'] == 2010, 'data_mean_sh'][0]
CO2_ssp245['Delta_PPM'] = CO2_ssp245['data_mean_sh'] - CO2_ssp245.loc[CO2_ssp245['year'] == 2010, 'data_mean_sh'][0]
CO2_ssp370['Delta_PPM'] = CO2_ssp370['data_mean_sh'] - CO2_ssp370.loc[CO2_ssp370['year'] == 2010, 'data_mean_sh'][0]
CO2_ssp850['Delta_PPM'] = CO2_ssp850['data_mean_sh'] - CO2_ssp850.loc[CO2_ssp850['year'] == 2010, 'data_mean_sh'][0]

# Add CO2 deltas to dictionary
CO2_dict = {'ssp126': CO2_ssp126, 'ssp245': CO2_ssp245, 'ssp370': CO2_ssp370, 'ssp585': CO2_ssp850}

# Loop through GCMs and SSPs to calculate climate change yield damage
for gcm in gcms:
    for ssp in ssps:
        
        prec = np.load(in_path + 'wc2.1_2.5m_prec_' + gcm + '_' + ssp + '_2010-2100.npy')
        tavg = np.load(in_path + 'wc2.1_2.5m_tavg_' + gcm + '_' + ssp + '_2010-2100.npy')
        
        # Create deltaP and deltaT arrays (note: arrays are transposed to enable numpy broadcasting)
        prec_delta = (prec.T - prec[:, 0]).T
        tavg_delta = (tavg.T - tavg[:, 0]).T
        
        yld_delta = -5.4 + (prec_delta * 0.52 + tavg_delta * -5.33)  + 0.06 * np.array(CO2_dict[ssp]['Delta_PPM'])

        fn = 'Climate_damage_crops_' + gcm + '_' + ssp + '_2010-2100.npy'
        np.save(LUTO_data_path + fn, yld_delta)
            
        # Convert 1D arrays to 2D and save as multiband GeoTiff
        with rasterio.open(out_path + fn[:-3] + '.tif', 'w+', **meta) as dst:
            
            # Convert to 2D and specify nodata as -9999
            for i in range(yld_delta.shape[1]):
                array_2D[xy] = yld_delta[:, i]
                array_2D[NLUM_mask == 0] = -9999
                
                # Save the output to GeoTiff
                dst.write_band(i + 1, array_2D.astype(np.float32))
            print(fn)


            
