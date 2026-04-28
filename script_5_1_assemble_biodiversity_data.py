
import os, re
import subprocess
from matplotlib.table import Cell
import netCDF4 # necessary for xarray to read/write netCDF files
import rasterio, fiona
import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import geopandas as gpd

from glob import glob
from itertools import product
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from rasterio import features
from rasterio.warp import reproject
from pyproj import CRS
from affine import Affine



###############################################################################################
#                                  Global variables                                           #
###############################################################################################

NLUM = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8') 
NLUM_zero = NLUM.copy() * 0


# Paths
bio_Carla_EnviroSuit_dir = 'N:/Data-Master/Biodiversity/Environmental-suitability'
bio_Carla_GTIFF_dir  = f'{bio_Carla_EnviroSuit_dir}/Annual-species-suitability_20-year_snapshots_5km'
bio_Carla_NetCDF_dir = f'{bio_Carla_EnviroSuit_dir}/Annual-species-suitability_20-year_snapshots_5km_to_NetCDF'

SNES_ECNES_dir = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES'
NVIS_PRE_1750_path = 'N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL'
NVIS_SAVE_path = 'N:/Data-Master/NVIS/Processed'
HCAS_condition = 'N:/Data-Master/Habitat_condition_assessment_system/Data/Processed/HABITAT_CONDITION.csv'
# Constants
Unalloc_nat_code = 23
bio_presence_weight = {'LIKELY': 0.8, 'MAYBE': 0.3}

# Upstream data
zones = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5',
    key='cell_zones_df',
    columns=['X', 'Y', 'CELL_HA']
)
bioph = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5', 
    key = 'cell_biophysical_df', 
    columns=['NATURAL_AREA_INC_WATER', 'DCCEEW_NCI', 'NATURAL_AREA_CONNECTIVITY']
)
lumap = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', 
    key = 'cell_LU_mapping', 
    columns=['LU_DESC','LU_ID_LUTO']
)


# Get real area for each cell
real_area_ha = zones['CELL_HA'].values
real_area_ha_2D = NLUM_zero.copy().astype(np.float32)
np.place(real_area_ha_2D.values, NLUM.values, real_area_ha)

# Get the index of cells that are in natural state, and inside/outside the LUTO study area
natural_cells = np.logical_not(bioph['NATURAL_AREA_INC_WATER'].values)              # flip the values to make 1 natural, 0 non-natural
idx_in_LUTO = np.logical_not(np.isin(lumap['LU_DESC'], ['Non-agricultural land']))  # shape=6956407, sum=4218733
idx_out_LUTO = np.isin(lumap['LU_DESC'], ['Non-agricultural land'])                 # shape=6956407, sum=2737674
idx_out_LUTO_natural = idx_out_LUTO & natural_cells                                 # shape=6956407, sum=2677065
idx_out_LUTO_non_natural = idx_out_LUTO & np.logical_not(natural_cells)             # shape=6956407, sum=60609


# Get land-use degradation data
biodiv_degrade_lookup = pd.read_csv(HCAS_condition).set_index(['lu'])['PERCENTILE_50'].to_dict()
biodiv_degrade_lookup = {k:v*(1/biodiv_degrade_lookup[Unalloc_nat_code]) for k,v in biodiv_degrade_lookup.items()}
biodiv_degrade_lookup[-1] = 0  
biodiv_degrade_ly = np.vectorize(biodiv_degrade_lookup.get, otypes=[np.float32])(lumap['LU_ID_LUTO'].values).astype(np.float32)
biodiv_degrade_ly[idx_out_LUTO_natural] = 1.0

biodiv_degrade_ly_2D = NLUM_zero.copy().astype(np.float32)
np.place(biodiv_degrade_ly_2D.values, NLUM.values, biodiv_degrade_ly)


# Get the 2D layers that are in natural state, inside/outside the LUTO study area
idx_in_LUTO_2D = NLUM_zero.copy()
np.place(idx_in_LUTO_2D.values, NLUM.values, idx_in_LUTO.astype('uint8'))
idx_out_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_out_LUTO_natural_2D.values, NLUM.values, idx_out_LUTO_natural.astype('uint8'))




###############################################################################################
#                  Process Speciese Suitability data (GBF8) with Xarray                       #
###############################################################################################



'''
Historical and future habitat suitability and condition projections for terrestrial vertebrate and 
vascular plant species (total ~106k species * 5km resolution).

Each species-layer represents the map of rescaled (0-100) suitability for the given species. By squashing 
all ~106k layers into a single layer with the Zonation algorithm, LUTO can determined the overall importances 
for all species.

The biodiversity data has a structure of:
- SSP:          {'ssp126', 'ssp245', 'ssp370', 'ssp585'}                            Use all SSPs in LUTO
- model: 	    {'GCM-Ensembles', 'historic'}                                       Use only 'GCM-Ensembles' in LUTO
- group: 	    {'amphibians', 'birds', 'mammals', 'plants', 'reptiles'}            Use all groups in LUTO
- species: 	    {'Abelmoschus_ficulneus' ... 'Zyzomys_woodwardi'}                   Use all species in LUTO
- year: 	    {1990, 2030, 2050, 2070, 2090}                                      Use all years in LUTO
- mode: 	    {'EnviroSuit', 'EnviroSuit_max', 'EnviroSuit_min', 'historic'}      Use 'historic' and 'EnviroSuit' in LUTO

And the data has a metadata of:
- 808 rows *  978 columns
- 5km resolution
- uint8 data type
- nodata value: 255
- CRS: EPSG:4283


To incoporate this data to LUTO, we use xarray to combine all GeoTIFF files into a single NetCDF file. Essentialy, the resuting nc file 
can be thought as a data cube of 4 dimensions: (year * species (group) * x * y), with group information attached to the species dimension.
'''



def find_str(row: pd.Series) -> list:
    """
    Extracts relevant information from the given row's path and returns it as a list.
    """
    reg_year = re.compile(r'_(\d{4})_').findall(row['path'])[0]
    
    if int(reg_year) < 2010:
        return ['historic', 'historic', reg_year, 'historic']
    
    reg_model = re.compile(rf'{row["species"]}_(.*)_ssp').findall(row['path'])[0]
    reg_ssp = re.compile(r'_(ssp\d*)_').findall(row['path'])[0]
    reg_mode = re.compile(r'km_(.*).tif').findall(row['path'])[0]
    return [reg_model, reg_ssp, int(reg_year), reg_mode]


def get_all_path(root_dir:str, save_path:str):
    """
    Retrieves the paths of all TIFF files in the specified root directory and saves them to a CSV file.
    """
    records = []
    for dirpath, _, filenames in tqdm(os.walk(root_dir)):
        for f in filenames:
            if f.endswith('.tif'):
                group, species = os.path.normpath(dirpath).split('\\')[-2:]
                records.append({'group':group, 'species':species, 'path':os.path.join(dirpath, f)})
    
    # Convert all records to df            
    df = pd.DataFrame(records)            
    df[['model', 'ssp', 'year', 'mode']] = df.apply(lambda x: pd.Series(find_str(x)), axis=1) 
    df.to_csv(save_path, index=False)


# Get all the paths of the GeoTIFF files
if not os.path.exists(f'{bio_Carla_NetCDF_dir}/bio_file_paths_raw.csv'):
    # Create a csv file recording all the paths, group, species, model, ssp, year, mode
    get_all_path(bio_Carla_GTIFF_dir, f'{bio_Carla_NetCDF_dir}/bio_file_paths_condition.csv')
else:
    # Read the existing csv file
    df = pd.read_csv(f'{bio_Carla_NetCDF_dir}/bio_file_paths_raw.csv' )



# Get the first GeoTIFF file to get the shape of the data
bio_arr = rxr.open_rasterio(df['path'][0], chunks='auto').sel(band=1).drop_vars('band')
bio_arr.values = np.arange(bio_arr.sizes['y'] * bio_arr.sizes['x']).reshape(bio_arr.sizes['y'], bio_arr.sizes['x'])
bio_coord_x = xr.DataArray(bio_arr['x'].values, dims=['x'])
bio_coord_y = xr.DataArray(bio_arr['y'].values, dims=['y'])

# Calculate the real area for each bio cell in hectares
results = ({'properties': {'cell_bio': v}, 'geometry': s} 
           for i, (s, v) in enumerate(features.shapes(bio_arr.values, mask = None, transform = bio_arr.rio.transform())))
rnd_gdf = gpd.GeoDataFrame.from_features(list(results), crs = NLUM.rio.crs)
rnd_gdf = rnd_gdf.to_crs('EPSG:3577')
rnd_gdf['CELL_HA'] = rnd_gdf['geometry'].area / 10000
bio_arr_area_ha = bio_arr.copy()
bio_arr_area_ha.values = rnd_gdf['CELL_HA'].values.reshape(bio_arr.sizes['y'], bio_arr.sizes['x'])


# Convert the index to xarray; 1D with cell as the primary dimension, and y, x as the coordinates
idx_in_LUTO_2D = NLUM_zero.copy()
np.place(idx_in_LUTO_2D.values, NLUM.values, idx_in_LUTO.astype('uint8'))
idx_out_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_out_LUTO_natural_2D.values, NLUM.values, idx_out_LUTO_natural.astype('uint8'))

# Get the coordinates of the cells that are in natural state, inside/outside the LUTO study area
idx_in_LUTO_2D_bio = idx_in_LUTO_2D.interp(x=bio_coord_x, y=bio_coord_y, method='nearest', kwargs={'fill_value': 0}).astype('bool')
idx_out_LUTO_natural_2D_bio = idx_out_LUTO_natural_2D.interp(x=bio_coord_x, y=bio_coord_y, method='nearest', kwargs={'fill_value': 0}).astype('bool')





# --------------------------------------Convert GeoTIFF to NetCDF--------------------------------------------

# Filter out the ensemble data
ensemble_df = df.query('model == "GCM-Ensembles" & mode == "EnviroSuit"').drop(columns=['model'])
valid_species = ensemble_df['species'].unique()                  
historic_df = df.query('model == "historic" and species.isin(@valid_species) and ~path.str.contains("5x5")').sort_values(['group', 'species']).reset_index(drop=True)


# Save ensemble data to nc
for ssp, mode in product(ensemble_df['ssp'].unique(), ensemble_df['mode'].unique()):
    
    # Get all the data for the given ssp and mode
    in_df = ensemble_df.query(f'ssp == "{ssp}" and mode == "{mode}"')
    in_df = pd.concat([historic_df, in_df])
    in_df = in_df.sort_values(['year', 'species']).reset_index(drop=True)
    
    # Create an empty array to store the data
    ensemble_arr = xr.DataArray(
        np.zeros((
            df['year'].nunique(),  
            len(historic_df['species']), 
            bio_arr.sizes['y'], 
            bio_arr.sizes['x']), dtype='uint8'
        ), 
        dims=['year', 'species', 'y', 'x'], 
        coords={
            'year':sorted(df['year'].unique()),
            'species':historic_df['species'], 
            'y':bio_arr['y'],
            'x':bio_arr['x'],
            'group': ('species', historic_df['group'])
        }
    )
    
    # Parallel processing to put the data into the empty array
    def get_arr(row):
        ds = rxr.open_rasterio(row['path']).sel(band=1).drop_vars('band')
        ds = xr.where(ds == ds.rio.nodata, 0, ds)
        return row['year'], row['species'], ds.values

    tasks = (delayed(get_arr)(row) for _,row in in_df.iterrows())

    for year,species,arr in tqdm(Parallel(n_jobs=-1, return_as='generator')(tasks), total=len(in_df)):
        ensemble_arr.loc[year, species] = arr


    # Save to nc, chunked by year, species, leave x, y as unlimited
    ensemble_arr.name = 'data'
    ensemble_arr.to_netcdf(
        f'{bio_Carla_NetCDF_dir}/bio_{ssp}_{mode}.nc', 
        mode='w', 
        encoding={'data': {
            "compression": "gzip", 
            "compression_opts": 5,  
            "dtype": 'uint8',
            "chunksizes": (1, 1, ensemble_arr.sizes['y'], ensemble_arr.sizes['x'])}}, 
        engine='h5netcdf'
    )

    del ensemble_arr



# -------------------- Calculate biodiversity contribution by group ------------------------------------------

# Search for biodiversity NetCDF files
bio_suitability_ncs = glob(f'{bio_Carla_NetCDF_dir}/*_EnviroSuit.nc')
  
# Save nc to disk
for nc in bio_suitability_ncs:
    
    fname = os.path.basename(nc).replace('_EnviroSuit.nc', '_EnviroSuit_group')
    bio_species_suitability = xr.open_dataset(nc, chunks={'year':1, 'species':1})['data']
    years = set(bio_species_suitability['year'].values)
    groups = set(bio_species_suitability['group'].values)
    
    # Calculate the sum of biodiversity contribution scores for each group
    group_arr_sum = bio_species_suitability.groupby('group').sum()

    # Save to nc, chunked by year, group, leave x, y as unlimited
    group_arr_sum.name = 'data'
    group_arr_sum.to_netcdf(
        f'{bio_Carla_NetCDF_dir}/{fname}.nc', 
        mode='w', 
        encoding={'data': {
            'compression': 'gzip', 
            'compression_opts': 9, 
            'dtype': 'uint32',
            'chunksizes': (1, 1, group_arr_sum.sizes['y'], group_arr_sum.sizes['x'])}}, 
        engine='h5netcdf'
    )





# ------------------- Calculate the biodiversity score for each species  ------------------------------------------


# Calculate the contribution, with real_area weighted
bio_condition_ncs = glob(f'{bio_Carla_NetCDF_dir}/*_EnviroSuit.nc')

for nc in bio_condition_ncs:
    
    fname = os.path.basename(nc).replace('_EnviroSuit.nc', '_EnviroSuit_Score')
    # Biodiversity scores for ALL Australia, inside LUTO study area, and outside LUTO study area
    score_sources = ['all', 'in', 'out']
    # Read the data
    bio_suitability = xr.open_dataarray(nc, chunks={'year': 1, 'species': 1})
    years = sorted([2010] + list(bio_suitability['year'].values))
 
    # Calculate the biodiversity score for each species
    bio_suitability_sum = xr.DataArray(
        np.zeros((len(years), bio_suitability.sizes['species'], len(score_sources)), dtype='float32'),
        dims=['year', 'species', 'source'],
        coords={'year':years, 'species':bio_suitability['species'], 'source':score_sources}
    )

    def get_val(sel_year, sel_species):

        arr = bio_suitability.sel(species=sel_species).interp(year=sel_year, method='linear').compute()
        # Reproject the data to match NLUM
        arr = arr.rio.set_crs(NLUM.rio.crs)
        arr = arr.rio.reproject_match(NLUM, resample=rasterio.enums.Resampling.bilinear) 
        # Multiply by the real area (ha) to get the biodiversity suitability score (i.e., area weighted suitability)
        arr = (arr * real_area_ha_2D).astype('float32')
        
        # Get the sum of biodiversity suitability score for all Australia, inside LUTO study area, and outside LUTO study area
        all_sum = arr.sum(['y', 'x']).values
        out_sum = arr.where(idx_out_LUTO_natural_2D).sum(['y', 'x']).values
        
        if sel_year == 2010:
            arr = arr * biodiv_degrade_ly_2D
            in_sum = arr.where(idx_in_LUTO_2D).sum(['y', 'x']).values
        else:
            in_sum = np.nan
            
        return sel_year, sel_species, all_sum, in_sum, out_sum
        
    tasks = [
        delayed(get_val)(yr, sp) 
        for sp in bio_suitability['species'].values
        for yr in years
    ]
    for yr, sp, val_sum, val_in, val_out in tqdm(Parallel(n_jobs=-1, return_as='generator')(tasks), total=len(tasks)):
        bio_suitability_sum.loc[yr, sp] = [val_sum, val_in, val_out]

    # Save to csv
    bio_suitability_sum.to_dataframe('BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA').reset_index().to_csv(f'{bio_Carla_NetCDF_dir}/{fname}.csv', index=False)




# Get the biodiversity target
'''
The habitat suitability baselines are same for all SSPs, so here use SSP245 to calculate the baseline
'''
bio_score_baseline = pd.read_csv(f'{bio_Carla_NetCDF_dir}/bio_ssp245_EnviroSuit_Score.csv').query('year == 2010')
bio_score_baseline = bio_score_baseline.pivot(index=['species'], columns='source', values='BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA').reset_index()
bio_score_baseline['HABITAT_SUITABILITY_BASELINE_PERCENT'] = bio_score_baseline.eval('(`in` + `out`) / `all`') * 100

# Create a habitat suitability target csv file
bio_target = bio_score_baseline[['species', 'HABITAT_SUITABILITY_BASELINE_PERCENT','all','in','out']].copy()
bio_target = bio_target.rename(columns={
    'all': 'HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA',
    'in': 'HABITAT_SUITABILITY_BASELINE_SCORE_INSIDE_LUTO',
    'out': 'HABITAT_SUITABILITY_BASELINE_SCORE_OUTSIDE_LUTO'
})


bio_target.insert(2, 'USER_DEFINED_TARGET_PERCENT_2100', np.nan)
bio_target.insert(2, 'USER_DEFINED_TARGET_PERCENT_2050', np.nan)
bio_target.insert(2, 'USER_DEFINED_TARGET_PERCENT_2030', np.nan)
bio_target.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF8_TARGET.csv', index=False)


# Get the biodiversity suitability area weighted scores for each SSP
bio_scores = pd.DataFrame()

for f in glob(f'{bio_Carla_NetCDF_dir}/*EnviroSuit_Score.csv'):
    ssp = re.compile(r'bio_ssp(\d*)_').findall(f)[0]
    bio_out = pd.read_csv(f).query('year != 1990').query('source == "out"').drop(columns=['source']).set_index(['species', 'year'])
    bio_out = bio_out.rename(columns={'BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA': f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{ssp}'})
    bio_scores = pd.concat([bio_scores, bio_out], axis=1)

bio_scores = bio_scores.reset_index()
bio_scores.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF8_SCORES.csv', index=False)




# ------------------- Calculate the biodiversity score for each group  ------------------------------------------

# Calculate the contribution, with real_area weighted
bio_condition_ncs = glob(f'{bio_Carla_NetCDF_dir}/*_EnviroSuit_group.nc')

for nc in bio_condition_ncs:
    fname = os.path.basename(nc).replace('_group.nc', '_group_Score')
    # Biodiversity scores for ALL Australia, inside LUTO study area, and outside LUTO study area
    score_sources = ['all', 'in', 'out']
    # Read the data
    bio_group = xr.open_dataarray(nc, chunks={'year':1,'group':1})
    
    years = sorted([2010] + list(bio_group['year'].values))

    # Calculate the biodiversity score for each group
    bio_group_sum = xr.DataArray(
        np.zeros((len(years), bio_group.sizes['group'], len(score_sources)), dtype='float32'),
        dims=['year', 'group', 'source'],
        coords={'year':years, 'group':bio_group['group'], 'source':score_sources}
    )


    def get_val(sel_year, sel_group):

        arr = bio_group.sel(group=sel_group).interp(year=sel_year, method='linear').compute()
        # Reproject the data to match NLUM
        arr = arr.rio.set_crs(NLUM.rio.crs)
        arr = arr.rio.reproject_match(NLUM, resample=rasterio.enums.Resampling.bilinear) 
        # Multiply by the real area (ha) to get the biodiversity suitability score (i.e., area weighted suitability)
        arr = (arr * real_area_ha_2D).astype('float32')
        
        # Get the sum of biodiversity suitability score for all Australia, inside LUTO study area, and outside LUTO study area
        all_sum = arr.sum(['y', 'x']).values
        out_sum = arr.where(idx_out_LUTO_natural_2D).sum(['y', 'x']).values
        
        if sel_year == 2010:
            arr = arr * biodiv_degrade_ly_2D
            in_sum = arr.where(idx_in_LUTO_2D).sum(['y', 'x']).values
        else:
            in_sum = np.nan
            
        return sel_year, sel_group, all_sum, in_sum, out_sum
        
    tasks = [
        delayed(get_val)(yr, gp) 
        for gp in bio_group['group'].values
        for yr in years
    ]
    
    for yr, gp, val_sum, val_in, val_out in tqdm(Parallel(n_jobs=5, return_as='generator')(tasks), total=len(tasks)):
        bio_group_sum.loc[yr, gp] = [val_sum, val_in, val_out]
        
        
    bio_group_sum.to_dataframe('BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA').reset_index().to_csv(f'{bio_Carla_NetCDF_dir}/{fname}.csv', index=False)
    
    
    
# Get the biodiversity target
'''
The habitat suitability baselines are same for all SSPs, so here use SSP245 to calculate the baseline
'''
bio_score_baseline = pd.read_csv(f'{bio_Carla_NetCDF_dir}/bio_ssp245_EnviroSuit_group_Score.csv').query('year == 2010')
bio_score_baseline = bio_score_baseline.pivot(index=['group'], columns='source', values='BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA').reset_index()
bio_score_baseline['HABITAT_SUITABILITY_BASELINE_PERCENT'] = bio_score_baseline.eval('(`in` + `out`) / `all`') * 100

# Create a habitat suitability target csv file
bio_target = bio_score_baseline[['group', 'HABITAT_SUITABILITY_BASELINE_PERCENT','in','all','out']].copy()
bio_target = bio_target.rename(columns={
    'all': 'HABITAT_SUITABILITY_BASELINE_SCORE_ALL_AUSTRALIA',
    'in': 'HABITAT_SUITABILITY_BASELINE_SCORE_INSIDE_LUTO',
    'out': 'HABITAT_SUITABILITY_BASELINE_SCORE_OUTSIDE_LUTO',
})

bio_target.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF8_TARGET_GROUP.csv', index=False)
    
    
# Get the biodiversity suitability area weighted scores for each SSP
bio_scores = pd.DataFrame()

for f in glob(f'{bio_Carla_NetCDF_dir}/*group_Score.csv'):
    ssp = re.compile(r'bio_ssp(\d*)_').findall(f)[0]
    bio_out = pd.read_csv(f).query('source == "out"').drop(columns=['source']).set_index(['group', 'year'])
    bio_out = bio_out.rename(columns={'BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA': f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{ssp}'})
    bio_scores = pd.concat([bio_scores, bio_out], axis=1)

bio_scores = bio_scores.reset_index()
bio_scores.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF8_SCORES_GROUP.csv', index=False)





################################################################################
#           Process Biodiversity Data (DCCEEW) (GBF4) with Xarray             #
################################################################################


# ------------------- Rasterise SNES/ECNES data to GEOTIFF ------------------------------------------

'''
Save SNES/ECNES to TIFFS.

 - 0.0: No presence
 - 0.0–1.0: Area proportion of species/community presence within the cell
 - NaN: No data (outside NLUM mask)

A csv file containing the metadata of the dissolved vector data and paths to the 
rasterised data is also saved to '{SNES_ECNES_dir}/Processed'.

Note: The 'LIKELY' and 'MAYBE' layers are not overlapped, 'MAYBE' layers are 
surrounding the 'LIKELY' layers.

'''


# Set parameters
n_workers = 20
presence_dict = {1: 'MAYBE', 2: 'LIKELY'}


# Read the reference raster data
ref_mask = NLUM.values
ref_meta = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'nodata': 255,
    'width': NLUM.rio.width,
    'height': NLUM.rio.height,
    'count': 1,
    'crs': NLUM.rio.crs,
    'transform': NLUM.rio.transform(),
    'compress': 'lzw',
}
ref_meta_float = {**ref_meta, 'dtype': 'float32', 'nodata': np.nan}
ref_transform = ref_meta['transform']




# ------------------- Rasterise SNES/ECNES data to GEOTIFF ------------------------------------------

# Read the SNES biodiversity data; dissolve the data by 'SCIENTIFIC_NAME'
if not os.path.exists(f"{SNES_ECNES_dir}/Processed/snes_dissolve.gpkg"):
    snes = gpd.read_file(f"{SNES_ECNES_dir}/SNES_version_6 March 2025/snes_public_gdb.gdb", driver="OpenFileGDB", layer="SNES_Public")
    snes_dissolve = snes.dissolve(by=['SCIENTIFIC_NAME','PRESENCE_CATEGORY']).reset_index()
    snes_dissolve.to_file(f"{SNES_ECNES_dir}/Processed/snes_dissolve.gpkg")
else:
    snes_dissolve = gpd.read_file(f"{SNES_ECNES_dir}/Processed/snes_dissolve.gpkg")
    
# Read the ECNES biodiversity data; dissolve the data by 'COMMUNITY'   
if not os.path.exists(f"{SNES_ECNES_dir}/Processed/ecnes_dissolve.gpkg"):
    ecnes = gpd.read_file(f"{SNES_ECNES_dir}/ECNES_versoin_4 September 2024/ECnes_public_gdb.gdb", driver="OpenFileGDB", layer="ECnes_public")
    ecnes_dissolve = ecnes.dissolve(by=['COMMUNITY', 'CATEGORY']).reset_index()
    ecnes_dissolve.to_file(f"{SNES_ECNES_dir}/Processed/ecnes_dissolve.gpkg")
else:
    ecnes_dissolve = gpd.read_file(f"{SNES_ECNES_dir}/Processed/ecnes_dissolve.gpkg")
    
    
    
    
# Filter the data to include only the species that are significant for the LUTO study area
snes_filter = (
    "(MARINE.isna() or MARINE == 'Listed - overfly marine area') "
    "and ("
        "THREATENED_STATUS.isin(['Critically Endangered', 'Vulnerable', 'Endangered', 'Extinct in the wild']) "
        "or MIGRATORY_STATUS == 'Migratory' "
    ")"
)

ecnes_filter = (
    "EPBC.isin(['Critically Endangered',  'Endangered']) "
    "and COMMUNITY != 'Giant Kelp Marine Forests of South East Australia'"  # This is a marine community.
)

snes_dissolve = snes_dissolve.query(snes_filter)
ecnes_dissolve = ecnes_dissolve.query(ecnes_filter)


# Function to get the presence value and save path for each row of the dissolved data
def get_presense_and_save_path(row):
    # Get value for rasterisation polygon (1 for 'maybe present', 2 for 'likely present')
    if 'PRES_RANK' in row:  # ECNES data
        val = row['PRES_RANK']
        name = re.sub(r'[^a-zA-Z0-9]', '_', row['COMMUNITY'])
        save_path = f'{SNES_ECNES_dir}/Processed/ECNES/{name}_{presence_dict[val]}.tif'
    else:                   # SNES data
        val = row['PRESENCE_RANK']
        name = re.sub(r'[^a-zA-Z0-9]', '_', row['SCIENTIFIC_NAME'])
        save_path = f'{SNES_ECNES_dir}/Processed/SNES/{row["TAXON_GROUP"]}/{name}/{name}_{presence_dict[val]}.tif'
    
    # Replace spaces with underscores
    save_path = save_path.replace(' ', '_')
    return val, save_path



# Build a fine-resolution grid (10x NLUM) for rasterise-then-average approach
_fine_scale = 10   # ECNES is in 100m resolution, so it is 10x more fine than the NLUM grid (1km resolution)
_fine_transform = Affine(
    ref_transform.a / _fine_scale, 0, ref_transform.c,
    0, ref_transform.e / _fine_scale, ref_transform.f
)
_fine_shape = (ref_mask.shape[0] * _fine_scale, ref_mask.shape[1] * _fine_scale)


# Function to rasterise polygon at fine resolution, then average-resample to NLUM grid
def rasterize(row):
    val, save_path = get_presense_and_save_path(row)

    # Rasterise at fine resolution (binary: 1 where polygon covers, 0 elsewhere)
    fine_arr = rasterio.features.rasterize(
        [(row["geometry"], 1)],
        out_shape=_fine_shape,
        transform=_fine_transform,
        all_touched=False,
        dtype='uint8',
    )

    # Average-resample from fine grid to NLUM grid to get area proportion [0.0–1.0]
    dst_arr = np.zeros(ref_mask.shape, dtype=np.float32)
    reproject(
        fine_arr,
        dst_arr,
        src_transform=_fine_transform,
        src_crs=ref_meta['crs'],
        dst_transform=ref_transform,
        dst_crs=ref_meta['crs'],
        resampling=rasterio.enums.Resampling.average,
    )

    # Apply mask (NaN outside NLUM); values are raw proportion [0.0–1.0]
    dst_arr = np.where(ref_mask, dst_arr, np.nan).astype(np.float32)

    with rasterio.open(save_path, 'w', **ref_meta_float) as dst:
        dst.write(dst_arr, 1)
        


# Create folders for SNES data
for _,row in snes_dissolve.iterrows():
    _,tif_path = get_presense_and_save_path(row)
    folder = os.path.dirname(tif_path)
    if os.path.exists(folder):
        continue
    os.makedirs(folder, exist_ok=True)
    
# Create folders for ECNES data; Only a single folder to store all the data
if not os.path.exists(f'{SNES_ECNES_dir}/Processed/ECNES'):
    os.makedirs(f'{SNES_ECNES_dir}/Processed/ECNES', exist_ok=True)
    
    

# Rasterise and save the SNES data to GEOTIFF
tasks = [delayed(rasterize)(row) for _,row in snes_dissolve.iterrows()]
for _ in tqdm(Parallel(n_jobs=-1, return_as='generator')(tasks), total=len(tasks)):
    pass

# Save SNES attributes to csv
snes_meta = snes_dissolve.copy().drop(columns='geometry')
snes_meta['TIF_PATH'] = snes_meta.apply(lambda x: get_presense_and_save_path(x)[1], axis=1)
snes_meta.to_csv(f'{SNES_ECNES_dir}/Processed/DCCEEW_SNES_meta.csv', index=False)



# Rasterise and save the ECNES data to GEOTIFF
tasks = [delayed(rasterize)(row) for _,row in ecnes_dissolve.iterrows()]

for out in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
    pass

# Save ECNES attributes to csv
ecnes_meta = ecnes_dissolve.copy().drop(columns='geometry')
ecnes_meta['TIF_PATH'] = ecnes_meta.apply(lambda x: get_presense_and_save_path(x)[1], axis=1)
ecnes_meta.to_csv(f'{SNES_ECNES_dir}/Processed/DCCEEW_ECNES_meta.csv', index=False)





# ------------------- Assemble SNES data into a single array ------------------------------------------

# Read DCCEEW SNES GeoTIFF file paths
SNES_meta = pd.read_csv(f'{SNES_ECNES_dir}/Processed/DCCEEW_SNES_meta.csv')

# Create an empty array to store the data (float32 for cell-fraction proportions)
SNES_arr = xr.DataArray(
    np.zeros((
        SNES_meta['SCIENTIFIC_NAME'].nunique(), 
        SNES_meta['PRESENCE_RANK'].nunique(), 
        NLUM.sum().item()),  dtype=np.float32
    ),
    dims=['species', 'presence', 'cell'],
    coords={
        'species':SNES_meta['SCIENTIFIC_NAME'].unique(),
        'presence': [presence_dict[r] for r in SNES_meta['PRESENCE_RANK'].unique()],
        'cell':np.arange(NLUM.sum().item())}
)


# Parallel processing to put the data into the empty array
def get_arr(row):
    ds = rxr.open_rasterio(row['TIF_PATH']).sel(band=1).drop_vars('band')
    ds = ds.values[np.nonzero(NLUM.values)].astype(np.float32)
    return row['SCIENTIFIC_NAME'], presence_dict[row['PRESENCE_RANK']], ds


tasks = (delayed(get_arr)(row) for _,row in SNES_meta.iterrows())
for species,rank,arr in tqdm(Parallel(n_jobs=20, return_as='generator')(tasks), total=len(SNES_meta)):
    SNES_arr.loc[species, rank] = arr
    
    
    
# Save raw data to nc, will be used for calculating targets
SNES_arr.name = 'data'
SNES_arr.to_netcdf(
    f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES.nc',
    mode='w',
    encoding={'data': {
        "compression": "gzip",
        "compression_opts": 5,
        "dtype": 'float32',
        "chunksizes": (1, 1, SNES_arr.sizes['cell'])}},
    engine='h5netcdf'
)


# Combine LIKELY and MAYBE into a single weighted layer:
SNES_likely = SNES_arr.sel(presence='LIKELY') * bio_presence_weight['LIKELY']
SNES_maybe = SNES_arr.sel(presence='MAYBE') * bio_presence_weight['MAYBE']
SNES_likely_and_maybe = np.maximum(SNES_likely, SNES_maybe)

SNES_arr_weighted = xr.DataArray(
    np.stack([SNES_likely.values, SNES_likely_and_maybe.values]),
    dims=['presence', 'species', 'cell'],
    coords={'presence': ['LIKELY', 'LIKELY_AND_MAYBE'], 'species': SNES_arr.species, 'cell': SNES_arr.cell}
)

SNES_arr_weighted.name = 'data'
SNES_arr_weighted.to_netcdf(
    f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES_weighted.nc',
    mode='w',
    encoding={'data': {
        "compression": "gzip",
        "compression_opts": 5,
        "dtype": 'float32',
        "chunksizes": (1, 1, SNES_arr_weighted.sizes['cell'])}},
    engine='h5netcdf'
)





# ------------------- Calculate the biodiversity score for SNES  ------------------------------------------

SNES_arr = xr.open_dataarray(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES.nc', chunks={'species': 1, 'presence': 1})

def get_area(arr):
    score_area_weighted_all_Australia = arr * zones['CELL_HA'].values
    score_area_weighted_in_LUTO = arr * idx_in_LUTO * zones['CELL_HA'].values * biodiv_degrade_ly
    score_area_weighted_out_LUTO_nat = arr * idx_out_LUTO_natural * zones['CELL_HA'].values
    score_area_weighted_out_LUTO_non_nat = arr * idx_out_LUTO_non_natural * zones['CELL_HA'].values
    return [{
        'ALL_HA':score_area_weighted_all_Australia.sum(), 
        'IN_LUTO_HA':score_area_weighted_in_LUTO.sum(),
        'NATURAL_OUT_LUTO_HA':score_area_weighted_out_LUTO_nat.sum(),
        'NON_NATURAL_OUT_LUTO_HA':score_area_weighted_out_LUTO_non_nat.sum()
    }]


# Calculate the area weighted scores for each species
tasks = []
for species, presence in product(SNES_arr['species'].values, SNES_arr['presence'].values):
    arr = SNES_arr.sel(species=species, presence=presence).values
    def set_val(species, presence, arr):
        arr_df = pd.DataFrame(get_area(arr))
        arr_df['SCIENTIFIC_NAME'] = species
        arr_df['PRESENCE_RANK'] = presence
        return arr_df
    tasks.append(delayed(set_val)(species, presence, arr))

SNES_in_out_LUTO_area = pd.DataFrame()
for arr in tqdm(Parallel(n_jobs=20, return_as='generator')(tasks), total=len(tasks)):
    SNES_in_out_LUTO_area = pd.concat([SNES_in_out_LUTO_area, arr], ignore_index=True)
 


# Get the shared atributs of the SNES data
SNES_meta_att = SNES_meta.groupby(['SCIENTIFIC_NAME']).aggregate('first')
SNES_meta_att = SNES_meta_att.drop(columns=['PRESENCE_CATEGORY', 'PRESENCE_RANK','SHAPE_Length', 'SHAPE_Area', 'TIF_PATH']).reset_index()

# Save the inside/outside LUTO data to csv
SNES_df = SNES_in_out_LUTO_area.copy()
SNES_df['BASELINE_LEVEL_ALL_AUSTRALIA']         = SNES_df['ALL_HA']
SNES_df['BASEYEAR_SCORE_INSIDE_LUTO_NATURAL']   = SNES_df['IN_LUTO_HA']
SNES_df['BASEYEAR_SCORE_OUT_LUTO_NATURAL']      = SNES_df['NATURAL_OUT_LUTO_HA']
SNES_df['BASEYEAR_SCORE_OUT_LUTO_NON_NATURAL']  = SNES_df['NON_NATURAL_OUT_LUTO_HA']
SNES_df['BASEYEAR_SCORE']                       = SNES_df['IN_LUTO_HA'] + SNES_df['NATURAL_OUT_LUTO_HA']
SNES_df['BASEYEAR_LEVEL']                       = SNES_df['BASEYEAR_SCORE'] / SNES_df['ALL_HA'] * 100
SNES_df['ATTAINABLE_LEVEL']                     = (1 - SNES_df['NON_NATURAL_OUT_LUTO_HA'] / SNES_df['ALL_HA']) * 100

# Drop unneeded columns, and split the data into three dataframes based on the PRESENCE_RANK
SNES_df = SNES_df.drop(columns=['ALL_HA', 'IN_LUTO_HA', 'NATURAL_OUT_LUTO_HA', 'NON_NATURAL_OUT_LUTO_HA', 'BASEYEAR_SCORE_OUT_LUTO_NON_NATURAL', 'BASEYEAR_SCORE'])
SNES_df_LIKELY = SNES_df.query('PRESENCE_RANK == "LIKELY"').copy().drop(columns=['PRESENCE_RANK'])
SNES_df_LIKELY_MAYBE = SNES_df.query('PRESENCE_RANK == "MAYBE"').copy().drop(columns=['PRESENCE_RANK'])

# Append suffix to the columns for the LIKELY and MAYBE dataframes
SNES_df_LIKELY.columns = [f'{col}_LIKELY' if col != 'SCIENTIFIC_NAME' else 'SCIENTIFIC_NAME' for col in SNES_df_LIKELY.columns]
SNES_df_LIKELY_MAYBE.columns = [f'{col}_LIKELY_MAYBE' if col != 'SCIENTIFIC_NAME' else 'SCIENTIFIC_NAME' for col in SNES_df_LIKELY_MAYBE.columns]

# Add user defined columns to the LIKELY and MAYBE dataframes
SNES_df_LIKELY.insert(0, 'TARGET_LEVEL_2100_LIKELY', np.nan)
SNES_df_LIKELY.insert(0, 'TARGET_LEVEL_2050_LIKELY', np.nan)
SNES_df_LIKELY.insert(0, 'TARGET_LEVEL_2030_LIKELY', np.nan)

SNES_df_LIKELY_MAYBE.insert(0, 'TARGET_LEVEL_2100_LIKELY_MAYBE', np.nan)
SNES_df_LIKELY_MAYBE.insert(0, 'TARGET_LEVEL_2050_LIKELY_MAYBE', np.nan)
SNES_df_LIKELY_MAYBE.insert(0, 'TARGET_LEVEL_2030_LIKELY_MAYBE', np.nan)

# Merge the LIKELY and MAYBE dataframes, and append the shared attributes
SNES_df = SNES_df_LIKELY.merge(SNES_df_LIKELY_MAYBE, on='SCIENTIFIC_NAME', how='outer')
SNES_df = SNES_df.merge(SNES_meta_att, on='SCIENTIFIC_NAME')


# Reorder the columns
cols = ['SCIENTIFIC_NAME','VERNACULAR_NAME',

        'ATTAINABLE_LEVEL_LIKELY',
        'BASEYEAR_LEVEL_LIKELY',
        'TARGET_LEVEL_2030_LIKELY',
        'TARGET_LEVEL_2050_LIKELY',
        'TARGET_LEVEL_2100_LIKELY',

        'ATTAINABLE_LEVEL_LIKELY_MAYBE',
        'BASEYEAR_LEVEL_LIKELY_MAYBE',
        'TARGET_LEVEL_2030_LIKELY_MAYBE',
        'TARGET_LEVEL_2050_LIKELY_MAYBE',
        'TARGET_LEVEL_2100_LIKELY_MAYBE',

        'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY',
        'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY',
        'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY',

        'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY_MAYBE',
        'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY_MAYBE',
        'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY_MAYBE',

        'LISTED_TAXON_ID','MAP_TAXON_ID', 'THREATENED_STATUS',
        'MIGRATORY_STATUS', 'MARINE', 'CETACEAN', 'EXTRACT_DATE', 'TAXON_GROUP',
        'TAXON_FAMILY', 'TAXON_ORDER', 'TAXON_CLASS', 'TAXON_PHYLUM',
        'TAXON_KINGDOM', 'OTHER_IDS', 'CELL_SIZE', 'REGIONS', 'ATTRIBUTION',
        'SPRAT_PROFILE']

SNES_df = SNES_df[cols]
SNES_df.to_csv(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES_target.csv', index=False)





# ------------------- Assemble ECNES data into a single array ------------------------------------------

# Read DCCEEW ECNES GeoTIFF file paths
ECNES_meta = pd.read_csv(f'{SNES_ECNES_dir}/Processed/DCCEEW_ECNES_meta.csv')

# Create an empty array to store the data (float32 for cell-fraction proportions)
ECNES_arr = xr.DataArray(
    np.zeros((ECNES_meta['COMMUNITY'].nunique(), ECNES_meta['PRES_RANK'].nunique(), NLUM.sum().item()), dtype=np.float32),
    dims=['species', 'presence', 'cell'],
    coords={'species':ECNES_meta['COMMUNITY'].unique(), 'presence':[presence_dict[r] for r in ECNES_meta['PRES_RANK'].unique()], 'cell':np.arange(NLUM.sum().item())}
)

def get_arr(row):
    arr = rasterio.open(row['TIF_PATH']).read(1).astype('float32')
    arr = arr[np.nonzero(NLUM.values)]
    return row['COMMUNITY'], presence_dict[row['PRES_RANK']], arr

tasks = (delayed(get_arr)(row) for _,row in ECNES_meta.iterrows())
for species,rank,arr in tqdm(Parallel(n_jobs=20, return_as='generator')(tasks), total=len(ECNES_meta)):
    ECNES_arr.loc[species,rank] = arr


# Save raw data to nc, raw data will be used to calculate targets
ECNES_arr.name = 'data'
ECNES_arr.to_netcdf(
    f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES.nc',
    mode='w',
    encoding={'data': {
        "compression": "gzip",
        "compression_opts": 5,
        "dtype": 'float32',
        "chunksizes": (1, 1, ECNES_arr.sizes['cell'])}},
    engine='h5netcdf'
)



# Combine LIKELY and MAYBE into a single weighted layer:
ECNES_likely = ECNES_arr.sel(presence='LIKELY') * bio_presence_weight['LIKELY']
ECNES_maybe = ECNES_arr.sel(presence='MAYBE') * bio_presence_weight['MAYBE']
ECNES_likely_and_maybe = np.maximum(ECNES_likely, ECNES_maybe)

ECNES_arr_weighted = xr.DataArray(
    np.stack([ECNES_likely.values, ECNES_likely_and_maybe.values]),
    dims=['presence', 'species', 'cell'],
    coords={'presence': ['LIKELY', 'LIKELY_AND_MAYBE'], 'species': ECNES_arr.species, 'cell': ECNES_arr.cell}
).transpose('species', 'presence', 'cell')

ECNES_arr_weighted.name = 'data'
ECNES_arr_weighted.to_netcdf(
    f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES_weighted.nc',
    mode='w',
    encoding={'data': {
        "compression": "gzip",
        "compression_opts": 5,
        "dtype": 'float32',
        "chunksizes": (1, 1, ECNES_arr_weighted.sizes['cell'])}},
    engine='h5netcdf'
)





# ------------------- Calculate the biodiversity score for ECNES  ------------------------------------------

ECNES_arr = xr.open_dataarray(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES.nc', chunks={'species': 1, 'presence': 1})


def get_area(arr):
    score_area_weighted_all_Australia       = arr * zones['CELL_HA'].values
    score_area_weighted_in_LUTO             = arr * idx_in_LUTO * zones['CELL_HA'].values * biodiv_degrade_ly
    score_area_weighted_out_LUTO_nat        = arr * idx_out_LUTO_natural * zones['CELL_HA'].values
    score_area_weighted_out_LUTO_non_nat    = arr * idx_out_LUTO_non_natural * zones['CELL_HA'].values
    return [{
        'ALL_HA':score_area_weighted_all_Australia.sum(),
        'IN_LUTO_HA':score_area_weighted_in_LUTO.sum(),
        'NATURAL_OUT_LUTO_HA':score_area_weighted_out_LUTO_nat.sum(),
        'NON_NATURAL_OUT_LUTO_HA':score_area_weighted_out_LUTO_non_nat.sum()
    }]
    

# Calculate the area weighted scores for each species
tasks = []
for species, presence in product(ECNES_arr['species'].values, ECNES_arr['presence'].values):
    arr = ECNES_arr.sel(species=species, presence=presence).values
    def set_val(species, presence, arr):
        arr_df = pd.DataFrame(get_area(arr))
        arr_df['COMMUNITY'] = species
        arr_df['PRES_RANK'] = presence
        return arr_df
    tasks.append(delayed(set_val)(species, presence, arr))
    

# Parallel processing to put the data into the empty array
ECNES_in_out_LUTO_area = pd.DataFrame()
for arr in tqdm(Parallel(n_jobs=20, return_as='generator')(tasks), total=len(tasks)):
    ECNES_in_out_LUTO_area = pd.concat([ECNES_in_out_LUTO_area, arr], ignore_index=True)



# Get the shared atributs of the ECNES data
ECNES_meta_att = ECNES_meta.groupby(['COMMUNITY']).aggregate('first')
ECNES_meta_att = ECNES_meta_att.drop(columns=['PRES_RANK', 'SHAPE_Length', 'SHAPE_Area', 'TIF_PATH']).reset_index()

# Save the inside/outside LUTO data to csv
ECNES_df = ECNES_in_out_LUTO_area.copy().reset_index()
ECNES_df['BASELINE_LEVEL_ALL_AUSTRALIA']            = ECNES_df['ALL_HA']
ECNES_df['BASEYEAR_SCORE_INSIDE_LUTO_NATURAL']      = ECNES_df['IN_LUTO_HA']
ECNES_df['BASEYEAR_SCORE_OUT_LUTO_NATURAL']         = ECNES_df['NATURAL_OUT_LUTO_HA']
ECNES_df['BASEYEAR_SCORE']                          = ECNES_df['IN_LUTO_HA'] + ECNES_df['NATURAL_OUT_LUTO_HA']
ECNES_df['BASEYEAR_LEVEL']                          = ECNES_df['BASEYEAR_SCORE'] / ECNES_df['ALL_HA'] * 100
ECNES_df['ATTAINABLE_LEVEL']                        = (1 - ECNES_df['NON_NATURAL_OUT_LUTO_HA'] / ECNES_df['ALL_HA']) * 100

# Fill the missing COMMUNITY and PRES_RANK with nan
re_index = pd.MultiIndex.from_product([ECNES_df['COMMUNITY'].unique(), ECNES_df['PRES_RANK'].unique()], names=['COMMUNITY', 'PRES_RANK'])
ECNES_df = ECNES_df.set_index(['COMMUNITY', 'PRES_RANK']).reindex(re_index).reset_index()

# Drop unneeded columns, and split the data into three dataframes based on the PRES_RANK
ECNES_df = ECNES_df.drop(columns=['ALL_HA', 'IN_LUTO_HA', 'NATURAL_OUT_LUTO_HA', 'NON_NATURAL_OUT_LUTO_HA', 'BASEYEAR_SCORE'])
ECNES_df_LIKELY = ECNES_df.query('PRES_RANK == "LIKELY"').copy().drop(columns=['PRES_RANK'])
ECNES_df_LIKELY_MAYBE = ECNES_df.query('PRES_RANK == "MAYBE"').copy().drop(columns=['PRES_RANK'])

# Append suffix to the columns for the LIKELY and MAYBE dataframes
ECNES_df_LIKELY.columns = [f'{col}_LIKELY' if col != 'COMMUNITY' else 'COMMUNITY' for col in ECNES_df_LIKELY.columns]
ECNES_df_LIKELY_MAYBE.columns = [f'{col}_LIKELY_MAYBE' if col != 'COMMUNITY' else 'COMMUNITY' for col in ECNES_df_LIKELY_MAYBE.columns]

# Add user defined columns to the LIKELY and MAYBE dataframes
ECNES_df_LIKELY.insert(0, 'TARGET_LEVEL_2100_LIKELY', np.nan)
ECNES_df_LIKELY.insert(0, 'TARGET_LEVEL_2050_LIKELY', np.nan)
ECNES_df_LIKELY.insert(0, 'TARGET_LEVEL_2030_LIKELY', np.nan)

ECNES_df_LIKELY_MAYBE.insert(0, 'TARGET_LEVEL_2100_LIKELY_MAYBE', np.nan)
ECNES_df_LIKELY_MAYBE.insert(0, 'TARGET_LEVEL_2050_LIKELY_MAYBE', np.nan)
ECNES_df_LIKELY_MAYBE.insert(0, 'TARGET_LEVEL_2030_LIKELY_MAYBE', np.nan)

# Merge the LIKELY and MAYBE dataframes, and append the shared attributes
ECNES_df = ECNES_df_LIKELY.merge(ECNES_df_LIKELY_MAYBE, on='COMMUNITY', how='outer')
ECNES_df = ECNES_df.merge(ECNES_meta_att, on='COMMUNITY')

# Reorder the columns
cols = ['COMMUNITY',

        'ATTAINABLE_LEVEL_LIKELY',
        'BASEYEAR_LEVEL_LIKELY',
        'TARGET_LEVEL_2030_LIKELY',
        'TARGET_LEVEL_2050_LIKELY',
        'TARGET_LEVEL_2100_LIKELY',

        'ATTAINABLE_LEVEL_LIKELY_MAYBE',
        'BASEYEAR_LEVEL_LIKELY_MAYBE',
        'TARGET_LEVEL_2030_LIKELY_MAYBE',
        'TARGET_LEVEL_2050_LIKELY_MAYBE',
        'TARGET_LEVEL_2100_LIKELY_MAYBE',

        'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY',
        'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY',
        'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY',

        'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY_MAYBE',
        'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY_MAYBE',
        'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY_MAYBE',

        'CATEGORY', 'COM_ID','EPBC', 'EXTRACTED', 'CELL_SIZE', 'REGIONS', 'CITATION', 'SPRAT']

ECNES_df = ECNES_df[cols]
ECNES_df.to_csv(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES_target.csv', index=False)







# ------------------- Apply Zonation algorithm to merged data ------------------------------------------

# Define the path to the Zonation executable
zonation_exe = 'C:/Program Files (x86)/Zonation5/z5.exe'

# Save weighted SNES and ECNES data to GeoTIFFs for Zonation input
SNES_arr_weighted = xr.open_dataarray(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES_weighted.nc', chunks={})
ECNES_arr_weighted = xr.open_dataarray(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES_weighted.nc', chunks={})

xy = np.nonzero(NLUM.values)
def _save_weighted_tif(data_array, species, presence, out_dir, subdir, meta):
    arr_2D = np.full((meta['height'], meta['width']), np.nan, dtype=np.float32)
    arr_2D[xy] = data_array.sel(species=species, presence=presence).values
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', species)
    save_path = f'{out_dir}/Processed/SNES_ECNES_WEIGHTED/{subdir}/{safe_name}_{presence}.tif'
    with rasterio.open(save_path, 'w', **meta) as dst:
        dst.write(arr_2D, 1)

_weighted_tasks = [
    delayed(_save_weighted_tif)(da, species, presence, SNES_ECNES_dir, subdir, ref_meta_float)
    for da, subdir in ((SNES_arr_weighted, 'SNES'), (ECNES_arr_weighted, 'ECNES'))
    for species, presence in product(da['species'].values, da['presence'].values)
]

for _ in tqdm(Parallel(n_jobs=-1, return_as='generator')(_weighted_tasks), total=len(_weighted_tasks)):
    pass



# Collect the TIF paths for the LIKELY and MAYBE layers for SNES and ECNES, and save to txt files for Zonation input
snes_likely_tifs =      glob(f'{SNES_ECNES_dir}/Processed/SNES_ECNES_WEIGHTED/SNES/*_LIKELY.tif')
ecnes_likely_tifs =     glob(f'{SNES_ECNES_dir}/Processed/SNES_ECNES_WEIGHTED/ECNES/*_LIKELY.tif')
snes_likely_may_tifs =  glob(f'{SNES_ECNES_dir}/Processed/SNES_ECNES_WEIGHTED/SNES/*_LIKELY_AND_MAYBE.tif')
ecnes_likely_may_tifs = glob(f'{SNES_ECNES_dir}/Processed/SNES_ECNES_WEIGHTED/ECNES/*_LIKELY_AND_MAYBE.tif')
mnes_likely_tifs = snes_likely_tifs + ecnes_likely_tifs
mnes_likely_may_tifs = snes_likely_may_tifs + ecnes_likely_may_tifs


# Save the TIF path to txt files
with open(f'{SNES_ECNES_dir}/Processed/Zonation/SNES_likely_files.txt', 'w') as f_snes_likely,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/ECNES_likely_files.txt', 'w') as f_ecnes_likely,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/SNES_likely_may_files.txt', 'w') as f_snes_likely_may,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/ECNES_likely_may_files.txt', 'w') as f_ecnes_likely_may,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/MNES_likely_files.txt', 'w') as f_mnes_likely,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/MNES_likely_may_files.txt', 'w') as f_mnes_likely_may:
         
    f_snes_likely.write('filename\n')
    f_snes_likely.write('\n'.join(f'"{p}"' for p in snes_likely_tifs))
    f_ecnes_likely.write('filename\n')
    f_ecnes_likely.write('\n'.join(f'"{p}"' for p in ecnes_likely_tifs))
    f_snes_likely_may.write('filename\n')
    f_snes_likely_may.write('\n'.join(f'"{p}"' for p in snes_likely_may_tifs))
    f_ecnes_likely_may.write('filename\n')
    f_ecnes_likely_may.write('\n'.join(f'"{p}"' for p in ecnes_likely_may_tifs))
    f_mnes_likely.write('filename\n')
    f_mnes_likely.write('\n'.join(f'"{p}"' for p in mnes_likely_tifs))
    f_mnes_likely_may.write('filename\n')
    f_mnes_likely_may.write('\n'.join(f'"{p}"' for p in mnes_likely_may_tifs))
    
    
# Create mask and hierarchy TIF
with rasterio.open(snes_likely_may_tifs[0]) as src:
    meta = src.meta.copy()
    meta.update({
        'dtype': 'uint8',
        'nodata': 0,
        'compress': 'lzw',
        'count': 1,
        'width': NLUM.rio.width,
        'height': NLUM.rio.height,
        'transform': NLUM.rio.transform(),
    })
    
    zone_mask = NLUM.values.astype('uint8')
    zone_hierarchy = ((zone_mask == 1) * (idx_in_LUTO_2D == 0)).astype('uint8')
    
    with rasterio.open(f'{SNES_ECNES_dir}/Processed/Zonation/zone_hierarchy.tif', 'w', **meta) as zone_hierarchy_dst,\
         rasterio.open(f'{SNES_ECNES_dir}/Processed/Zonation/zone_mask.tif', 'w', **meta) as zone_mask_dst:
        zone_hierarchy_dst.write(zone_hierarchy, 1)
        zone_mask_dst.write(zone_mask, 1)


# Create the zonation settings file
with open(f'{SNES_ECNES_dir}/Processed/Zonation/snes_likely_settings.txt', 'w') as snes_likely_settings,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/ecnes_likely_settings.txt', 'w') as ecnes_likely_settings,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/snes_likely_may_settings.txt', 'w') as snes_likely_may_settings,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/ecnes_likely_may_settings.txt', 'w') as ecnes_likely_may_settings,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/mnes_likely_settings.txt', 'w') as mnes_likely_settings,\
     open(f'{SNES_ECNES_dir}/Processed/Zonation/mnes_likely_may_settings.txt', 'w') as mnes_likely_may_settings:
         
    snes_likely_settings.write((
        f'feature list file = {SNES_ECNES_dir}/Processed/Zonation/SNES_likely_files.txt\n'
        f'analysis area mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_mask.tif\n'
        f'hierarchic mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_hierarchy.tif\n'
    ))
    
    ecnes_likely_settings.write((
        f'feature list file = {SNES_ECNES_dir}/Processed/Zonation/ECNES_likely_files.txt\n'
        f'analysis area mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_mask.tif\n'
        f'hierarchic mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_hierarchy.tif\n'
    ))

    snes_likely_may_settings.write((
        f'feature list file = {SNES_ECNES_dir}/Processed/Zonation/SNES_likely_may_files.txt\n'
        f'analysis area mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_mask.tif\n'
        f'hierarchic mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_hierarchy.tif\n'
    ))

    ecnes_likely_may_settings.write((
        f'feature list file = {SNES_ECNES_dir}/Processed/Zonation/ECNES_likely_may_files.txt\n'
        f'analysis area mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_mask.tif\n'
        f'hierarchic mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_hierarchy.tif\n'
    ))

    mnes_likely_settings.write((
        f'feature list file = {SNES_ECNES_dir}/Processed/Zonation/MNES_likely_files.txt\n'
        f'analysis area mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_mask.tif\n'
        f'hierarchic mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_hierarchy.tif\n'
    ))

    mnes_likely_may_settings.write((
        f'feature list file = {SNES_ECNES_dir}/Processed/Zonation/MNES_likely_may_files.txt\n'
        f'analysis area mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_mask.tif\n'
        f'hierarchic mask layer = {SNES_ECNES_dir}/Processed/Zonation/zone_hierarchy.tif\n'
    ))

  

# Execute zonation in parallel
zonation_runs = [
    ('snes_likely_settings.txt',     'SNES_likely_Priority'),
    ('ecnes_likely_settings.txt',    'ECNES_likely_Priority'),
    ('snes_likely_may_settings.txt', 'SNES_likely_may_Priority'),
    ('ecnes_likely_may_settings.txt','ECNES_likely_may_Priority'),
    ('mnes_likely_settings.txt',     'MNES_likely_Priority'),
    ('mnes_likely_may_settings.txt', 'MNES_likely_may_Priority'),
]

zonation_procs = [
    subprocess.Popen([
        zonation_exe, '--mode=CAZMAX', '-ah',
        f'{SNES_ECNES_dir}/Processed/Zonation/{settings}',
        f'{SNES_ECNES_dir}/Processed/Zonation/{output}'
    ]) for settings, output in zonation_runs
]

for proc, (settings, _) in zip(zonation_procs, zonation_runs):
    rc = proc.wait()
    if rc != 0:
        print(f'WARNING: Zonation failed for {settings} (exit code {rc})')


# Merge all zonation layers and save as NetCDF
zonation_layers = {
    'ECNES_likely_may': f'{SNES_ECNES_dir}/Processed/Zonation/ECNES_likely_may_Priority/rankmap.tif',
    'ECNES_likely': f'{SNES_ECNES_dir}/Processed/Zonation/ECNES_likely_Priority/rankmap.tif',
    'SNES_likely_may': f'{SNES_ECNES_dir}/Processed/Zonation/SNES_likely_may_Priority/rankmap.tif',
    'SNES_likely': f'{SNES_ECNES_dir}/Processed/Zonation/SNES_likely_Priority/rankmap.tif',
    'MNES_likely_may': f'{SNES_ECNES_dir}/Processed/Zonation/MNES_likely_may_Priority/rankmap.tif',
    'MNES_likely': f'{SNES_ECNES_dir}/Processed/Zonation/MNES_likely_Priority/rankmap.tif',
}

zonation_arr = xr.DataArray(
    np.zeros((len(zonation_layers), NLUM.sum().item()), dtype=np.float32),
    dims=['layer', 'cell'],
    coords={'layer':list(zonation_layers.keys()), 'cell':np.arange(NLUM.sum().item())}
)

for layer, path in zonation_layers.items():
    with rasterio.open(path) as src:
        arr = src.read(1)
        arr = arr[np.nonzero(NLUM.values)]
        zonation_arr.loc[dict(layer=layer)] = arr
    

# Save to nc
zonation_arr.name = 'data'
zonation_arr.to_netcdf(
    f'{SNES_ECNES_dir}/Processed/bio_NES_Zonation.nc',
    mode='w', 
    encoding={'data': {
        "compression": "gzip", 
        "compression_opts": 5,
        "dtype": 'float32'
        },
    },
    engine='h5netcdf'
)



###############################################################################################
#                  Process Speciese Conservation Priority data (GBF2) with Xarray             #
###############################################################################################


# ----------------- Calculate the rank2area performance curves for conservation priority data -----------------

Biodiversity_conserve_performance = pd.DataFrame()

# Get conservation priority raster/csv
for ssp in ['ssp126', 'ssp245', 'ssp370', 'ssp585']:
    
    # Read the conservation priority data, select the cells of 'inside LUTO study area'
    ly = rxr.open_rasterio(f'{bio_Carla_EnviroSuit_dir}/Zonation/{ssp}/{ssp}_zonation_rank_1km.tif'
        ).squeeze('band'
        ).drop_vars('band'
        ).sel(
            x=xr.DataArray(zones['X'].values, dims='cell', coords={'cell':zones.index}), 
            y=xr.DataArray(zones['Y'].values, dims='cell', coords={'cell':zones.index}), 
            method='nearest', 
            drop=True
        ).assign_coords(area=('cell', real_area_ha)
        ).drop_vars(['x', 'y', 'spatial_ref'])

    # Select cells inside LUTO study area, calculate the conservation priority statistics
    ly_stats = ly.sel(cell=idx_in_LUTO
        ).to_dataframe(name='PRIORITY_RANK'
        ).reset_index(
        ).sort_values('PRIORITY_RANK', ascending=False
        ).assign(
            AREA_COVERAGE_PERCENT=(
                lambda df: 
                    df['area'].astype("float").cumsum()     # Needs to convert cumsum to float to avoid numerical issues
                    / df['area'].sum() 
                    * 100
                ),              
            PRIORITY_RANK_CUMSUM_CONTRIBUTION=(
                lambda df: 
                    (df['PRIORITY_RANK'].astype("float") * df['area']).cumsum() 
                    / (df['PRIORITY_RANK'] * df['area']).sum() * 100
                ),
            source=ssp
        ).drop(columns=['area', 'cell'])
    
    # Select rows with AREA_COVERAGE_PERCENT closest to integer values from 0 to 100
    ly_stats = ly_stats.iloc[
        abs(np.arange(101).reshape(-1, 1)  - ly_stats['AREA_COVERAGE_PERCENT'].values).argmin(axis=1)].copy()
    ly_stats['AREA_COVERAGE_PERCENT'] = np.arange(101)
    
    # Save the conservation priority data to the array
    Biodiversity_conserve_performance = pd.concat([Biodiversity_conserve_performance, ly_stats])
    
    
# Get conservation priority raster/csv
for nes in ['ECNES_likely_may', 'ECNES_likely', 'SNES_likely_may', 'SNES_likely', 'MNES_likely_may', 'MNES_likely' ]:
    
    ly = rxr.open_rasterio(f'{SNES_ECNES_dir}/Processed/Zonation/{nes}_Priority/rankmap.tif'
        ).squeeze('band'
        ).drop_vars('band'
        ).sel(
            x=xr.DataArray(zones['X'].values, dims='cell', coords={'cell':zones.index}), 
            y=xr.DataArray(zones['Y'].values, dims='cell', coords={'cell':zones.index}),
            method='nearest', 
            drop=True 
        ).assign_coords(area=('cell', real_area_ha)
        ).drop_vars(['x', 'y', 'spatial_ref'])
    ly_stats = ly.sel(cell=idx_in_LUTO
        ).to_dataframe(name='PRIORITY_RANK'
        ).reset_index(
        ).sort_values('PRIORITY_RANK', ascending=False
        ).assign(
            AREA_COVERAGE_PERCENT=(
                lambda df: 
                    df['area'].astype("float").cumsum()     # Needs to convert cumsum to float to avoid numerical issues
                    / df['area'].sum() 
                    * 100
                ),              
            PRIORITY_RANK_CUMSUM_CONTRIBUTION=(
                lambda df: 
                    (df['PRIORITY_RANK'].astype("float") * df['area']).cumsum() 
                    / (df['PRIORITY_RANK'] * df['area']).sum() * 100
                ),
            source=nes
        ).drop(columns=['area', 'cell'])
    ly_stats = ly_stats.iloc[
        abs(np.arange(101).reshape(-1, 1) - ly_stats['AREA_COVERAGE_PERCENT'].values).argmin(axis=1)].copy()
    ly_stats['AREA_COVERAGE_PERCENT'] = np.arange(101)
    Biodiversity_conserve_performance = pd.concat([Biodiversity_conserve_performance, ly_stats])


# Save csv to Excel
with pd.ExcelWriter(f'{SNES_ECNES_dir}/Processed/Biodiversity_conserve_performance.xlsx') as writer:
    for source, df in Biodiversity_conserve_performance.groupby('source'):
        df = df[['AREA_COVERAGE_PERCENT', 'PRIORITY_RANK', 'PRIORITY_RANK_CUMSUM_CONTRIBUTION']]
        df.to_excel(writer, sheet_name=source, index=False)




################################################################################
#           Process Vegetation Data (NVIS)  (GBF3) with Xarray                 #
################################################################################


'''
Reproject NVIS Extant + Pre-European Major Vegetation Groups and Subgroups rasters to match NLUM, save to GeoTiff and NetCDF
'''

mask_GEOTIFF = 'N:/Data-Master/National_Landuse_Map/NLUM_2010-11_clip.tif'
area_ha_GEOTIFF = 'N:/Data-Master/National_Landuse_Map/NLUM_2010-11_cell_ha.tif'

# Get metadata from mask_GEOTIFF
with rasterio.open(mask_GEOTIFF) as rst:
    # Load a 2D masked array with nodata masked out
    NLUM_ID_raster = rst.read(1, masked=True) 
    NLUM_mask = NLUM_ID_raster.mask == False
    # Get metadata and update parameters
    NLUM_transform = rst.transform
    NLUM_crs = rst.crs
    meta = rst.meta.copy()
    meta.update(compress='lzw', driver='GTiff') # , dtype='int32', nodata='0')
    [meta.pop(key) for key in ['dtype', 'nodata', 'count', 'driver']] # Need to add dtype and nodata manually when exporting GeoTiffs

# Get real area in hectares from area_ha_GEOTIFF
with rasterio.open(area_ha_GEOTIFF) as rst:
    NLUM_area = rst.read(1)
    NLUM_area_mask = NLUM_area[NLUM_mask]
    

############## Functions

def reproject_and_average(val:int, src_arr:np.ndarray, src_trans:Affine, src_crs:CRS, target_meta:dict):
    zero_arr = np.zeros((meta.get('height'), meta.get('width')), np.float32)
    band_arr = (src_arr == val).astype(np.uint8)
    reproject(
        band_arr, 
        zero_arr, 
        resampling=rasterio.enums.Resampling.average, 
        src_transform=src_trans, 
        src_crs=src_crs, 
        dst_transform=target_meta.get('transform'), 
        dst_crs=target_meta.get('crs')
    )
    return np.round(zero_arr * 100, 0).astype(np.uint8)


############## List raster layers in NVIS geoDatabase folder

# Present Vegetation Groups and Subgroups
fiona.listlayers('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb')
     
# Pre1750 Vegetation Groups and Subgroups
fiona.listlayers('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb')


# Set paths and layer names
files = [
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb','NVIS7_0_AUST_PRE_MVG_ALB', 'VAT_NVIS7_0_AUST_PRE_MVG_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS_V7_0_AUST_PRE.gdb','NVIS7_0_AUST_PRE_MVS_ALB', 'VAT_NVIS7_0_AUST_PRE_MVS_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb','NVIS7_0_AUST_EXT_MVG_ALB', 'VAT_NVIS7_0_AUST_EXT_MVG_ALB'),
    ('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS_V7_0_AUST_EXT.gdb','NVIS7_0_AUST_EXT_MVS_ALB', 'VAT_NVIS7_0_AUST_EXT_MVS_ALB')
]

# Set number of workers for parallel processing
n_workers = 10

# Loop through each raster layer and reproject to match NLUM
for gdb_path, layer_raster, layer_attribute in files:
    
    # Read NVIS raster and reproject to match NLUM
    with rasterio.open(f'OpenFileGDB:{gdb_path}:{layer_raster}') as src: 
        src_arr = src.read(1)
    # Load in look-up tables of MVG and MVS names
    src_att = gpd.read_file(gdb_path, layer=layer_attribute)
    # Rename column that contains 'NAME' to 'NAME'
    src_att = src_att.rename(columns={src_att.filter(like='NAME').columns[0]: 'NAME'})
    src_att = src_att[['Value', 'NAME']]
    src_att.to_csv(f'{NVIS_SAVE_path}/{layer_raster}_lookup.csv', index=False)


    # Create a list of delayed jobs, so we can reproject and average rasters in parallel with `n_workers`
    jobs = [delayed(reproject_and_average)(val, src_arr, src.transform, src.crs, meta) for val in src_att['Value']]
    dst_array = np.stack(Parallel(n_jobs=n_workers)(jobs), axis=0)
    
    
    # Save reprojected raster to GeoTiff
    save_path = f'{NVIS_SAVE_path}/{layer_raster}.tif'
    with rasterio.open(save_path, 'w', **meta, PROFILE='GEOTIFF', count=dst_array.shape[0], dtype=dst_array.dtype) as dst:
        # Write each band to the raster
        for i in range(dst_array.shape[0]):
            dst.write(dst_array[i], i+1)
        # Set band descriptions
        dst.descriptions = tuple(src_att['Value'].astype(str).str.zfill(2).values)
        

    # Get the cells based on NLUM mask
    dst_array_flat = dst_array[:,NLUM_mask]
    # Create xarray DataArray with group and cell dimensions
    dst_array_xr = xr.DataArray(
        dst_array_flat, 
        dims=['group', 'cell'], 
        coords={'group':src_att['NAME'], 'cell':np.arange(dst_array_flat.shape[1])}
    )
    
    
    # Save xarray DataArray to NetCDF
    save_path = f'{NVIS_SAVE_path}/{layer_raster}.nc'
    encoding = {'data': {"compression": "gzip", "compression_opts": 5,  "dtype": 'uint8'}} 
    dst_array_xr.name = 'data'
    dst_array_xr.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')




# --------------- Remove invalid vegetation class; Apply spatial mask ---------------

# Some groups are undertmined and should be exclude from the analysis, such as 'Other ...', 'Unknown/no data', and 'Unclassified'.
rm_names = ['Unknown/no data', 'Unknown/No data']


# Calculate the sum of all groups for each raster
for gdb_path, layer_raster, layer_attribute in files:
    
    # Read NVIS raster and filter out the groups that should be removed
    dst_array_xr = xr.load_dataarray(f'{NVIS_SAVE_path}/{layer_raster}.nc')
    dst_array_xr = dst_array_xr.sel(group=~dst_array_xr.group.isin(rm_names))
    
    # Reorder the groups lexicographically
    dst_array_xr = dst_array_xr.sortby('group')
    
    # Save xarray DataArray to NetCDF
    encoding = {'data': {"compression": "gzip", "compression_opts": 5,  "dtype": 'uint8'}}
    output_layer_name = layer_raster.replace('_ALB', '')
    
    # Use each separate group layer, which is the percentage [0-100], to represent the cell
    save_path = f'{NVIS_SAVE_path}/{output_layer_name}.nc'
    dst_array_xr.name = 'data'
    dst_array_xr.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')
    



# --------------- Get the sum of areas (ha) for pre-1750 ---------------

# Read NVIS data
NVIS_pre_mvg_xr = xr.load_dataarray(f'{NVIS_SAVE_path}/NVIS7_0_AUST_PRE_MVG.nc') / 100  # Convert percentage to fraction
NVIS_pre_mvs_xr = xr.load_dataarray(f'{NVIS_SAVE_path}/NVIS7_0_AUST_PRE_MVS.nc') / 100  # Convert percentage to fraction


# Total vegataion area (ha) pre-1750
NVIS_pre_mvg_total_ha = NVIS_pre_mvg_xr * zones['CELL_HA'].values[None, :]
NVIS_pre_mvs_total_ha = NVIS_pre_mvs_xr * zones['CELL_HA'].values[None, :]
NVIS_pre_mvg_total_ha_df = NVIS_pre_mvg_total_ha.sum(dim='cell').to_dataframe('AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA').reset_index()
NVIS_pre_mvs_total_ha_df = NVIS_pre_mvs_total_ha.sum(dim='cell').to_dataframe('AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA').reset_index()


# Vegataion area outside the LUTO study area
NVIS_pre_mvg_outside_ha = NVIS_pre_mvg_xr.sel(cell=idx_out_LUTO_natural) * zones['CELL_HA'].values[None, idx_out_LUTO_natural]
NVIS_pre_mvs_outside_ha = NVIS_pre_mvs_xr.sel(cell=idx_out_LUTO_natural) * zones['CELL_HA'].values[None, idx_out_LUTO_natural]
NVIS_pre_mvg_outside_ha_df = NVIS_pre_mvg_outside_ha.sum(dim='cell').to_dataframe('AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA').reset_index()
NVIS_pre_mvs_outside_ha_df = NVIS_pre_mvs_outside_ha.sum(dim='cell').to_dataframe('AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA').reset_index()


# Vegataion area inside the LUTO study area
NVIS_pre_mvg_inside_ha = NVIS_pre_mvg_xr.sel(cell=idx_in_LUTO) * zones['CELL_HA'].values[None, idx_in_LUTO] * biodiv_degrade_ly[idx_in_LUTO]
NVIS_pre_mvs_inside_ha = NVIS_pre_mvs_xr.sel(cell=idx_in_LUTO) * zones['CELL_HA'].values[None, idx_in_LUTO] * biodiv_degrade_ly[idx_in_LUTO]
NVIS_pre_mvg_inside_ha_df = NVIS_pre_mvg_inside_ha.sum(dim='cell').to_dataframe('AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_HA').reset_index()
NVIS_pre_mvs_inside_ha_df = NVIS_pre_mvs_inside_ha.sum(dim='cell').to_dataframe('AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_HA').reset_index()


# Concatenate the dataframes
NVIS_pre_mvg = NVIS_pre_mvg_total_ha_df.merge(NVIS_pre_mvg_outside_ha_df, on='group').merge(NVIS_pre_mvg_inside_ha_df, on='group')
NVIS_pre_mvs = NVIS_pre_mvs_total_ha_df.merge(NVIS_pre_mvs_outside_ha_df, on='group').merge(NVIS_pre_mvs_inside_ha_df, on='group')


# Calculate the percentage of base-year biodiversity socre to pre-1750 level of the base year
NVIS_pre_mvg.insert(1, 'BASE_YR_PERCENT', NVIS_pre_mvg.eval(
    '(AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_HA + AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA) \
    / AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA * 100'))

NVIS_pre_mvs.insert(1, 'BASE_YR_PERCENT', NVIS_pre_mvs.eval(
    '(AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_HA + AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA) \
    / AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA * 100'))


# Append a user-defined target column
NVIS_pre_mvg.insert(2, 'USER_DEFINED_TARGET_PERCENT_2100', 50)
NVIS_pre_mvg.insert(2, 'USER_DEFINED_TARGET_PERCENT_2050', 50)
NVIS_pre_mvg.insert(2, 'USER_DEFINED_TARGET_PERCENT_2030', 30)

NVIS_pre_mvs.insert(2, 'USER_DEFINED_TARGET_PERCENT_2100', 50)
NVIS_pre_mvs.insert(2, 'USER_DEFINED_TARGET_PERCENT_2050', 50)
NVIS_pre_mvs.insert(2, 'USER_DEFINED_TARGET_PERCENT_2030', 30)

# Combine all CSVs and save them to Excel
csv_files = {
    'NVIS_MVG': NVIS_pre_mvg,
    'NVIS_MVS': NVIS_pre_mvs
}

with pd.ExcelWriter(NVIS_SAVE_path + '/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.xlsx') as writer:
    for sheet_name, df in csv_files.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)



