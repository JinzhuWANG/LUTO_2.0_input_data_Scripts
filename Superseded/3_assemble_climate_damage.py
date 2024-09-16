import os, rasterio
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

gcms = ['GCM_ensembles']#, 'BCC-CSM2-MR', 'CanESM5', 'CNRM-CM6-1', 'CNRM-ESM2-1', 'IPSL-CM6A-LR', 'MIROC6', 'MIROC-ES2L', 'MRI-ESM2-0']
ssps = ['ssp126'] #, 'ssp245', 'ssp370', 'ssp585']
years = ['2021-2040', '2041-2060', '2061-2080', '2081-2100']
layers = ['prec', 'tavg']

# Set some paths to input and output data
inannpth = 'N:/Planet-A/Data-Master/WorldClim_CMIP6/Australia/Australia_1km_masked/Annual_climate_layers/'
outyrlypth = 'N:/Planet-A/Data-Master/WorldClim_CMIP6/Australia/Australia_1km_masked/Yearly_climate_layers_1970-2100/'

# Open NLUM mask raster and get metadata
with rasterio.open('N:/Planet-A/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif') as src:
    NLUM_mask = src.read(1) == 1

    # Get metadata and update parameters
    NLUM_transform = src.transform
    meta = src.meta.copy()
    meta.update(compress = 'lzw', driver = 'GTiff', dtype = 'float32', nodata = -9999)
    
    # Set up some data structures to enable conversion on 1D arrays to 2D
    array_2D = np.zeros(NLUM_mask.shape)
    xy = np.nonzero(NLUM_mask)


# Load historical prec and temp layers and flatten to 1D
with rasterio.open(inannpth + 'Historical_1970-2020/wc2.1_2.5m_prec_historical_1970-2020_Annual.tif') as src:
    prec_hist = src.read(1)[NLUM_mask]
    
with rasterio.open(inannpth + 'Historical_1970-2020/wc2.1_2.5m_tavg_historical_1970-2020_Annual.tif') as src:
    tavg_hist = src.read(1)[NLUM_mask]

hist_dict = {'prec': prec_hist, 'tavg': tavg_hist}
midpoints = [1995, 2030, 2050, 2070, 2090]

for gcm in gcms:
    
    if not os.path.exists(outyrlypth + gcm): os.mkdir(outyrlypth + gcm)
                        
    for ssp in ssps:
            
        for layer in layers:
            
            dc_training = np.zeros((np.sum(NLUM_mask), 5), dtype = np.float32)
            dc_training[:, 0] = hist_dict[layer]
            
            for i, year in enumerate(years):
            
                if gcm == 'GCM_ensembles': fn1 = 'wc2.1_2.5m_' + layer + '_' + ssp + '_' + year + '_Annual_Ensemble_Mean.tif'
                else: fn1 = 'wc2.1_2.5m_' + layer + '_' + gcm + '_' + ssp + '_' + year + '_Annual.tif'
                
                fullpath = inannpth + gcm + '/' + ssp + '/' + fn1
                with rasterio.open(fullpath) as src:
                    year_data = src.read(1)[NLUM_mask]
                
                dc_training[:, i + 1] = year_data
            
            
            linfit = interp1d(midpoints, dc_training, axis = 1, fill_value = "extrapolate", kind = 'linear')
            
            dc_full_timeseries = linfit(list(range(1970, 2101, 1))).astype(np.float32)
            
            fn = 'N:/Planet-A/Data-Master/Climate_damage/Climate_projection_data/wc2.1_2.5m_' + layer + '_' + gcm + '_' + ssp + '_2010-2100.tif'
            np.save(fn, dc_full_timeseries[:, 40:])
            
            for i in range(dc_full_timeseries.shape[1]):
                array_2D[xy] = dc_full_timeseries[:, i]
                array_2D[array_2D == 0] = -9999
                
                # Save the output to GeoTiff
                outfullpath = outyrlypth + gcm + '/' + ssp + '/'
                if not os.path.exists(outfullpath): os.mkdir(outfullpath)
                
                fn = 'wc2.1_2.5m_' + layer + '_' + gcm + '_' + ssp + '_' + str(1970 + i) + '_Annual.tif'

                with rasterio.open(outfullpath + fn, 'w+', **meta) as dst:        
                    dst.write_band(1, array_2D.astype(np.float32))
                print(fn)
            
            

'''
fig = plt.figure()
ax = plt.axes()
x = list(range(1970, 2100))
ax.plot(x, dc_full_timeseries[0, :])
ax.plot(midpoints, dc_training[0, :], 'o', color='black')
'''
            
