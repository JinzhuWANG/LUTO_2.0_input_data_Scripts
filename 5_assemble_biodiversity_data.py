# The script uses `xarray` to process data, therefore is different compared to rasterio based methods.

# Author:		Jinzhu WANG
# Email: 		wangjinzhulala@gmail.com
# Last update: 	17 Feb, 2025



import os, re
import netCDF4
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



# Global variables
NLUM = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8') 
NLUM_zero = NLUM.copy() * 0

bio_Carla_GTIFF_dir  = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km'
bio_Carla_NetCDF_dir = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km_to_NetCDF'

SNES_TIF_path = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF'
bio_DCCEEW_dir = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/To_NetCDF'

NVIS_PRE_1750_path = 'N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL'

HCAS_condition = 'N:/Data-Master/Habitat_condition_assessment_system/Data/Processed/HABITAT_CONDITION.csv'
Unalloc_nat_code = 23

# Read previouse raw data
zones = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5', key='cell_zones_df', columns=['X', 'Y', 'CELL_HA'])
bioph = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5', key = 'cell_biophysical_df', columns=['NATURAL_AREA_INC_WATER'])
lumap = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', key = 'cell_LU_mapping', columns=['LU_DESC','LU_ID_LUTO'])

# Get land-use degradation data
biodiv_degrade_lookup = pd.read_csv(HCAS_condition).set_index(['lu'])['PERCENTILE_50'].to_dict()
biodiv_degrade_lookup = {k:v*(1/biodiv_degrade_lookup[Unalloc_nat_code]) for k,v in biodiv_degrade_lookup.items()}
biodiv_degrade_lookup[-1] = 1  # -1 means outside the study area, so we set their degrade score to 1 (no degrade).
biodiv_degrade_ly = np.vectorize(biodiv_degrade_lookup.get)(lumap['LU_ID_LUTO']).astype(np.float32)
biodiv_degrade_ly_2D = NLUM_zero.copy().astype(np.float32)
np.place(biodiv_degrade_ly_2D.values, NLUM.values, biodiv_degrade_ly)

# Get real area for each cell
real_area_ha = zones['CELL_HA'].values
real_area_ha_2D = NLUM_zero.copy().astype(np.float32)
np.place(real_area_ha_2D.values, NLUM.values, real_area_ha)

# Get the index of cells that are in natural state, and inside/outside the LUTO study area
natural_cells = np.logical_not(bioph['NATURAL_AREA_INC_WATER'].values)  # 0 is natural, 1 is non-natural; so we flip the values to make 1 natural
idx_in_LUTO_natural = np.isin(lumap['LU_DESC'], ['Beef - natural land', 'Dairy - natural land', 'Sheep - natural land', 'Unallocated - natural land'])
idx_out_LUTO = np.isin(lumap['LU_DESC'], ['Non-agricultural land'])     # shape=6956407, sum=2737674
idx_out_LUTO_natural = idx_out_LUTO & natural_cells

# Get the 2D layers that are in natural state, inside/outside the LUTO study area
idx_in_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_in_LUTO_natural_2D.values, NLUM.values, idx_in_LUTO_natural.astype('uint8'))
idx_out_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_out_LUTO_natural_2D.values, NLUM.values, idx_out_LUTO_natural.astype('uint8'))




################################################################################
#                  Process Biodiversity Data (Carla) with Xarray               #
################################################################################



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
results = ({'properties': {'cell_bio': v}, 'geometry': s} for i, (s, v) in enumerate(features.shapes(bio_arr.values, mask = None, transform = bio_arr.rio.transform())))
rnd_gdf = gpd.GeoDataFrame.from_features(list(results), crs = NLUM.rio.crs)
rnd_gdf = rnd_gdf.to_crs('EPSG:3577')
rnd_gdf['CELL_HA'] = rnd_gdf['geometry'].area / 10000
bio_arr_area_ha = bio_arr.copy()
bio_arr_area_ha.values = rnd_gdf['CELL_HA'].values.reshape(bio_arr.sizes['y'], bio_arr.sizes['x'])


# Convert the index to xarray; 1D with cell as the primary dimension, and y, x as the coordinates
idx_in_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_in_LUTO_natural_2D.values, NLUM.values, idx_in_LUTO_natural.astype('uint8'))
idx_out_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_out_LUTO_natural_2D.values, NLUM.values, idx_out_LUTO_natural.astype('uint8'))

# Get the coordinates of the cells that are in natural state, inside/outside the LUTO study area
idx_in_LUTO_natural_2D_bio = idx_in_LUTO_natural_2D.interp(x=bio_coord_x, y=bio_coord_y, method='nearest', kwargs={'fill_value': 0}).astype('bool')
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
    
    # Parallel processing put the data into the empty array
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
            "compression_opts": 9,  
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

    
    fname = os.path.basename(nc).replace('_EnviroSuit.nc', '_Condition')
    bio_species_suitability = xr.open_dataset(nc, chunks={'year':1, 'species':1})['data']
    years = set(bio_species_suitability['year'].values)
    groups = set(bio_species_suitability['group'].values)
    
    # Create an empty array to store the data
    group_arr_contribution = xr.DataArray(
        np.zeros((
            len(years),  
            len(groups), 
            bio_species_suitability.sizes['y'], 
            bio_species_suitability.sizes['x']), dtype='float32'
        ), 
        dims=['year', 'group', 'y', 'x'], 
        coords={
            'year':sorted(years),
            'group':sorted(groups), 
            'y':bio_species_suitability['y'],
            'x':bio_species_suitability['x']
        }
    )
    
    # Calculate the biodiversity contribution scores for each group
    for sel_group in groups:
        
        group_arr = bio_species_suitability.groupby('group')[sel_group]
        # Divide by the number of species to avoide large number overflows in later sum calculation
        group_arr = group_arr.astype('float32') / group_arr.sizes['species']        
        # Calculate the contribution of the group to the total biodiversity
        group_arr_contr = group_arr.sum('species') / group_arr.sel(year=1990, drop=True).sum(['species', 'y', 'x'])
        # Multiply by the real area (ha) to get the area weighted contribution
        group_arr_contribution.loc[:, sel_group] = group_arr_contr.values * bio_arr_area_ha


    # Save to nc, chunked by year, group, leave x, y as unlimited
    group_arr_contribution.name = 'data'
    group_arr_contribution.to_netcdf(
        f'{bio_Carla_NetCDF_dir}/{fname}_group.nc', 
        mode='w', 
        encoding={'data': {
            'compression': 'gzip', 
            'compression_opts': 9, 
            'dtype': 'float32',
            'chunksizes': (1, 1, group_arr_contribution.sizes['y'], group_arr_contribution.sizes['x'])}}, 
        engine='h5netcdf'
    )
    
    del group_arr_contribution




# ------------------- Calculate the biodiversity score for each species  ------------------------------------------

# Calculate the contribution, with real_area weighted
bio_condition_ncs = glob(f'{bio_Carla_NetCDF_dir}/*_EnviroSuit.nc')

for nc in bio_condition_ncs:
    
    fname = os.path.basename(nc).replace('_EnviroSuit.nc', '_EnviroSuit_Score')
    # Biodiversity scores for ALL Australia, inside LUTO study area, and outside LUTO study area
    score_sources = ['all', 'in', 'out']
    # Read the data
    bio_suitability = xr.open_dataarray(nc, chunks={'year':1,'group':1})

    # Calculate the biodiversity score for each species
    bio_suitability_sum = xr.DataArray(
        np.zeros((bio_suitability.sizes['year'], bio_suitability.sizes['species'], len(score_sources)), dtype='float32'),
        dims=['year', 'species', 'source'],
        coords={'year':bio_suitability['year'], 'species':bio_suitability['species'], 'source':score_sources}
    )

    def get_val(sel_year, sel_species):

        arr = bio_suitability.sel(species=sel_species).interp(year=sel_year, method='linear').compute()
        # Reproject the data to match NLUM
        arr = arr.rio.set_crs(NLUM.rio.crs)
        arr = arr.rio.reproject_match(NLUM, resample=rasterio.enums.Resampling.bilinear) 
        # Multiply by the real area (ha) to get the biodiversity suitability score (i.e., area weighted suitability)
        arr = (arr * real_area_ha_2D).astype('float32')
        
        if sel_year == 1990:
            # Sum of biodiversity suitability score without degradation
            all_sum = arr.sum(['y', 'x']).values
            # Biodiversity suitability score with degradation
            arr = arr * biodiv_degrade_ly_2D
            in_sum = arr.where(idx_in_LUTO_natural_2D).sum(['y', 'x']).values
            out_sum = arr.where(idx_out_LUTO_natural_2D).sum(['y', 'x']).values
        else:
            all_sum = np.nan
            in_sum = np.nan
            out_sum = arr.where(idx_out_LUTO_natural_2D).sum(['y', 'x']).values
        return sel_year, sel_species, all_sum, in_sum, out_sum
        
    tasks = [
        delayed(get_val)(yr, sp) 
        for sp in bio_suitability['species'].values
        for yr in bio_suitability['year'].values
    ]
    for yr, sp, val_sum, val_in, val_out in tqdm(Parallel(n_jobs=-1, return_as='generator')(tasks), total=len(tasks)):
        bio_suitability_sum.loc[yr, sp] = [val_sum, val_in, val_out]

    # Save to csv
    bio_suitability_sum.to_dataframe('BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA').reset_index().to_csv(f'{bio_Carla_NetCDF_dir}/{fname}.csv', index=False)




# Get the biodiversity target
'''
The habitat suitability baselines are same for all SSPs, so here use SSP245 to calculate the baseline
'''
bio_score_baseline = pd.read_csv(f'{bio_Carla_NetCDF_dir}/bio_ssp245_EnviroSuit_Score.csv').query('year == 1990')
bio_score_baseline = bio_score_baseline.pivot(index=['species'], columns='source', values='BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA').reset_index()
bio_score_baseline['HABITAT_SUITABILITY_BASELINE'] = bio_score_baseline.eval('(`in` + `out`) / `all`') * 100

# Create a habitat suitability target csv file
bio_target = bio_score_baseline[['species', 'HABITAT_SUITABILITY_BASELINE']].copy()
bio_target.insert(2, 'USER_DEFINED_TARGET_PERCENT_2100', np.nan)
bio_target.insert(2, 'USER_DEFINED_TARGET_PERCENT_2050', np.nan)
bio_target.insert(2, 'USER_DEFINED_TARGET_PERCENT_2030', np.nan)
bio_target.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF4A_TARGET.csv', index=False)


# Get the biodiversity suitability area weighted scores for each SSP
bio_scores = pd.DataFrame()

for f in glob(f'{bio_Carla_NetCDF_dir}/*_Score.csv'):
    ssp = re.compile(r'bio_ssp(\d*)_').findall(f)[0]
    bio_out = pd.read_csv(f).query('year != 1990').query('source == "out"').drop(columns=['source']).set_index(['species', 'year'])
    bio_out = bio_out.rename(columns={'BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA': f'OUTSIDE_LUTO_NATURAL_AREA_WEIGHTED_HA_SSP{ssp}'})
    bio_scores = pd.concat([bio_scores, bio_out], axis=1)

bio_scores = bio_scores.reset_index()
bio_scores = bio_scores.merge(bio_score_baseline[['species', 'all']], on='species', how='left')
bio_scores = bio_scores.rename(columns={'all': 'HABITAT_SUITABILITY_BASELINE'})
bio_scores.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF4A_SCORES.csv', index=False)





################################################################################
#           Process Biodiversity Data (DCCEEW) with Xarray                     #
################################################################################


# ------------------- Rasterise SNES/ECNES data to GEOTIFF ------------------------------------------


# Set parameters
n_workers = 50

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


# Read the SNES biodiversity data
snes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/snes_public_gdb.gdb", driver="OpenFileGDB", layer="SNES_Public")
ecnes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/ECnes_public_gdb.gdb", driver="OpenFileGDB", layer="ECnes_public")

# Define the k-v pair for presence 
presence_dict = {1: 'MAYBE', 2: 'LIKELY'}

# Dissolve to merge, and save the dissolved data
snes_dissolve = snes.dissolve(by=['SCIENTIFIC_NAME','PRESENCE_CATEGORY']).reset_index()
ecnes_dissolve = ecnes.dissolve(by=['COMMUNITY', 'CATEGORY']).reset_index()

if not os.path.exists(f"{SNES_TIF_path}/snes_dissolve.geojson"):
    snes_dissolve.to_file(f"{SNES_TIF_path}/DISSOLVED_VECTOR/snes_dissolve.geojson")
if not os.path.exists(f"{SNES_TIF_path}/ecnes_dissolve.geojson"):
    ecnes_dissolve.to_file(f"{SNES_TIF_path}/DISSOLVED_VECTOR/ecnes_dissolve.geojson")


def get_presVal_savePath(row):
    # Get value for rasterisation polygon (1 for 'maybe present', 2 for 'likely present')
    if 'PRES_RANK' in row:  # ECNES data
        val = row['PRES_RANK']
        name = row['COMMUNITY'].replace('/', '_')
        save_path = f'{SNES_TIF_path}/ECNES/{name}_{presence_dict[val]}.tif'
    else:                   # SNES data
        val = row['PRESENCE_RANK']
        name = row['SCIENTIFIC_NAME'].replace('/', '_')
        save_path = f'{SNES_TIF_path}/SNES/{row["TAXON_GROUP"]}/{name}/{name}_{presence_dict[val]}.tif'
    
    # Replace spaces with underscores
    save_path = save_path.replace(' ', '_')
    return val, save_path



# Function to rasterise the data, note here converting the rasterised data to boolean
def rasterize(row):
    val, save_path = get_presVal_savePath(row)
    # Rasterise the polygon
    arr = rasterio.features.rasterize(
        [(row["geometry"], val)],
        out_shape=ref_mask.shape,
        transform=ref_meta['transform'],
        all_touched=False,
        dtype='uint8',
    )
    # Apply mask, 255 will be used for nodata
    arr = np.where(ref_mask, arr, 255)
    # Save to GEOTIFF
    with rasterio.open(save_path, 'w', **ref_meta) as dst:
        dst.write(arr, 1)
        


# Create folders for SNES data
for _,row in snes_dissolve.iterrows():
    tif_path = get_presVal_savePath(row)[1]
    folder = os.path.dirname(tif_path)
    if os.path.exists(folder):
        continue
    os.makedirs(folder, exist_ok=True)
    
# Create folders for ECNES data; Only a single folder to store all the data
if not os.path.exists(f'{SNES_TIF_path}/ECNES'):
    os.makedirs(f'{SNES_TIF_path}/ECNES', exist_ok=True)
    
    

# Rasterise and save the SNES data to GEOTIFF
tasks = [delayed(rasterize)(row) for _,row in snes_dissolve.iterrows()]
for _ in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
    pass

# Save SNES attributes to csv
snes_meta = snes_dissolve.copy().drop(columns='geometry')
snes_meta['TIF_PATH'] = snes_meta.apply(lambda x: get_presVal_savePath(x)[1], axis=1)
snes_meta.to_csv(f'{SNES_TIF_path}/DCCEEW_SNES_meta.csv', index=False)



# Rasterise and save the ECNES data to GEOTIFF
tasks = [delayed(rasterize)(row) for _,row in ecnes_dissolve.iterrows()]

raster_arr = []
for out in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
    raster_arr.append(out)

# Save ECNES attributes to csv
ecnes_meta = ecnes_dissolve.copy().drop(columns='geometry')
ecnes_meta['TIF_PATH'] = ecnes_meta.apply(lambda x: get_presVal_savePath(x)[1], axis=1)
ecnes_meta.to_csv(f'{SNES_TIF_path}/DCCEEW_ECNES_meta.csv', index=False)



# ------------------- Masking GEOTIFFs and save SNES to NetCDF ------------------------------------------

# Read DCCEEW SNES GeoTIFF file paths
SNES_meta = pd.read_csv('N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/DCCEEW_SNES_meta.csv')


# Create an empty array to store the data
SNES_arr = xr.DataArray(
    np.zeros((SNES_meta['SCIENTIFIC_NAME'].nunique(), SNES_meta['PRESENCE_RANK'].nunique(), NLUM.sum().item()), dtype=np.bool_),
    dims=['species', 'presence', 'cell'],
    coords={'species':SNES_meta['SCIENTIFIC_NAME'].unique(), 'presence':SNES_meta['PRESENCE_RANK'].unique(), 'cell':np.arange(NLUM.sum().item())}
)


# Create an empty dataframes to store the inside/outside LUTO data
SNES_in_out_LUTO_area = pd.DataFrame({
    'ALL_HA': np.zeros(len(SNES_meta)),
    'NATURAL_IN_LUTO_HA': np.zeros(len(SNES_meta)),
    'NATURAL_OUT_LUTO_HA': np.zeros(len(SNES_meta))
}, index=SNES_meta.set_index(['SCIENTIFIC_NAME', 'PRESENCE_RANK']).index)


# Parallel processing to put the data into the empty array
def get_arr(row):
    ds = rxr.open_rasterio(row['TIF_PATH']).sel(band=1).drop_vars('band')
    ds = xr.where(ds == ds.rio.nodata, 0, ds)
    ds = xr.where(ds.isin([1, 2]), 1, 0).astype(np.bool_)      # 1 is 'MAYBE', 2 is 'LIKELY'. We convert them to 1 so that we can use bool_ type
    ds = ds.values.ravel()[np.flatnonzero(NLUM.values)]
    # Multiply by the real area (ha) to get the area weighted contribution
    ds_all = ds * zones['CELL_HA'].values
    # Multiply by the by the degradation score to get degraded habitat significance score
    ds_in_LUTO = ds * idx_in_LUTO_natural * zones['CELL_HA'].values * biodiv_degrade_ly
    ds_out_LUTO = ds * idx_out_LUTO_natural * zones['CELL_HA'].values
    return row['SCIENTIFIC_NAME'], row['PRESENCE_RANK'], ds, ds_all.sum(), ds_in_LUTO.sum(), ds_out_LUTO.sum()

tasks = (delayed(get_arr)(row) for _,row in SNES_meta.iterrows())
for species,rank,arr,all_area,in_area,out_area in tqdm(Parallel(n_jobs=20, return_as='generator')(tasks), total=len(SNES_meta)):
    SNES_arr.loc[species] = arr
    SNES_in_out_LUTO_area.loc[species, rank] = [all_area, in_area, out_area]
    

# Save to nc, chunked by species
SNES_arr.name = 'data'
SNES_arr.to_netcdf(
    f'{bio_DCCEEW_dir}/bio_DCCEEW_SNES.nc', 
    mode='w', 
    encoding={'data': {
        "compression": "gzip", 
        "compression_opts": 9,  
        "dtype": 'bool',
        "chunksizes": (1, 1, SNES_arr.sizes['cell'])}}, 
    engine='h5netcdf'
)


# Get the shared atributs of the SNES data
SNES_meta_att = SNES_meta.groupby(['SCIENTIFIC_NAME']).apply(lambda df: df.iloc[0], include_groups=False)
SNES_meta_att = SNES_meta_att.drop(columns=['PRESENCE_CATEGORY', 'PRESENCE_RANK','SHAPE_Length', 'SHAPE_Area', 'TIF_PATH']).reset_index()

# Save the inside/outside LUTO data to csv
SNES_df = SNES_in_out_LUTO_area.copy().reset_index()
SNES_df['HABITAT_SIGNIFICANCE_PRESTINE_AUSTRALIA'] = SNES_df['ALL_HA']
SNES_df['HABITAT_SIGNIFICANCE_BASELINE_SCORE'] = SNES_df['NATURAL_IN_LUTO_HA'] + SNES_df['NATURAL_OUT_LUTO_HA']
SNES_df['HABITAT_SIGNIFICANCE_BASELINE_PERCENT'] = SNES_df['HABITAT_SIGNIFICANCE_BASELINE_SCORE'] / SNES_df['ALL_HA'] * 100

# Fill the missing SCIENTIFIC_NAME and PRESENCE_RANK with nan
re_index = pd.MultiIndex.from_product([SNES_df['SCIENTIFIC_NAME'].unique(), SNES_df['PRESENCE_RANK'].unique()], names=['SCIENTIFIC_NAME', 'PRESENCE_RANK'])
SNES_df = SNES_df.set_index(['SCIENTIFIC_NAME', 'PRESENCE_RANK']).reindex(re_index).reset_index()

# Drop unneeded columns, and split the data into three dataframes based on the PRESENCE_RANK
SNES_df = SNES_df.drop(columns=['ALL_HA', 'NATURAL_IN_LUTO_HA', 'NATURAL_OUT_LUTO_HA'])
SNES_df_LIKELY = SNES_df.query('PRESENCE_RANK == 2').copy().drop(columns=['PRESENCE_RANK'])
SNES_df_MAYBE = SNES_df.query('PRESENCE_RANK == 1').copy().drop(columns=['PRESENCE_RANK'])

# Append suffix to the columns for the LIKELY and MAYBE dataframes
SNES_df_LIKELY.columns = [f'{col}_LIKELY' if col != 'SCIENTIFIC_NAME' else 'SCIENTIFIC_NAME' for col in SNES_df_LIKELY.columns]
SNES_df_MAYBE.columns = [f'{col}_MAYBE' if col != 'SCIENTIFIC_NAME' else 'SCIENTIFIC_NAME' for col in SNES_df_MAYBE.columns]

# Add user defined columns to the LIKELY and MAYBE dataframes
SNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2100_LIKELY', np.nan)
SNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2050_LIKELY', np.nan)
SNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2030_LIKELY', np.nan)

SNES_df_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2100_MAYBE', np.nan)
SNES_df_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2050_MAYBE', np.nan)
SNES_df_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2030_MAYBE', np.nan)

# Merge the LIKELY and MAYBE dataframes, and append the shared attributes
SNES_df = SNES_df_LIKELY.merge(SNES_df_MAYBE, on='SCIENTIFIC_NAME', how='outer')
SNES_df = SNES_df.merge(SNES_meta_att, on='SCIENTIFIC_NAME')


# Reorder the columns
cols = ['SCIENTIFIC_NAME',
        
        'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2030_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2050_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2100_LIKELY',
         
        'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2030_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2050_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2100_MAYBE',
        
        'HABITAT_SIGNIFICANCE_PRESTINE_AUSTRALIA_LIKELY',
        'HABITAT_SIGNIFICANCE_PRESTINE_AUSTRALIA_MAYBE',
        
         'LISTED_TAXON_ID',
         'MAP_TAXON_ID', 'VERNACULAR_NAME', 'THREATENED_STATUS',
         'MIGRATORY_STATUS', 'MARINE', 'CETACEAN', 'EXTRACT_DATE', 'TAXON_GROUP',
         'TAXON_FAMILY', 'TAXON_ORDER', 'TAXON_CLASS', 'TAXON_PHYLUM',
         'TAXON_KINGDOM', 'OTHER_IDS', 'CELL_SIZE', 'REGIONS', 'ATTRIBUTION',
         'SPRAT_PROFILE']

SNES_df = SNES_df[cols]
SNES_df.to_csv(f'{bio_DCCEEW_dir}/bio_DCCEEW_SNES_AREA_HA.csv', index=False)



# ------------------- Masking GEOTIFFs and save ECNES to NetCDF ------------------------------------------

# Read DCCEEW ECNES GeoTIFF file paths
ECNES_meta = pd.read_csv('N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/DCCEEW_ECNES_meta.csv')

# Create an empty array to store the data
ECNES_arr = xr.DataArray(
    np.zeros((ECNES_meta['COMMUNITY'].nunique(), ECNES_meta['PRES_RANK'].nunique(), NLUM.sum().item()), dtype=np.bool_),
    dims=['species', 'presence', 'cell'],
    coords={'species':ECNES_meta['COMMUNITY'].unique(), 'presence':ECNES_meta['PRES_RANK'].unique(), 'cell':np.arange(NLUM.sum().item())}
)

# Create an empty dataframes to store the inside/outside LUTO data
ECNES_in_out_LUTO_area = pd.DataFrame({
    'ALL_HA': np.zeros(len(ECNES_meta)),
    'NATURAL_IN_LUTO_HA': np.zeros(len(ECNES_meta)),
    'NATURAL_OUT_LUTO_HA': np.zeros(len(ECNES_meta))
}, index=ECNES_meta.set_index(['COMMUNITY', 'PRES_RANK']).index)


# Parallel processing put the data into the empty array
def get_arr(row):
    ds = rxr.open_rasterio(row['TIF_PATH']).sel(band=1).drop_vars('band')
    ds = xr.where(ds == ds.rio.nodata, 0, ds)
    ds = xr.where(ds.isin([1, 2]), 1, 0)        # 1 is 'MAYBE', 2 is 'LIKELY'. We convert them to 1 so that we can use bool_ type
    ds = ds.values.ravel()[np.flatnonzero(NLUM.values)]
    # Multiply by the real area (ha) to get the area weighted contribution
    ds_all = ds * zones['CELL_HA'].values
    # Multiply by the by the degradation score (2010) to get land-use degraded score
    ds_in_LUTO = ds * idx_in_LUTO_natural * zones['CELL_HA'].values * biodiv_degrade_ly
    ds_out_LUTO = ds * idx_out_LUTO_natural * zones['CELL_HA'].values
    return row['COMMUNITY'], row['PRES_RANK'], ds, ds_all.sum(), ds_in_LUTO.sum(), ds_out_LUTO.sum()

tasks = (delayed(get_arr)(row) for _,row in ECNES_meta.iterrows())
for species,rank,arr,all_area,in_area,out_area in tqdm(Parallel(n_jobs=20, return_as='generator')(tasks), total=len(ECNES_meta)):
    ECNES_arr.loc[species] = arr
    ECNES_in_out_LUTO_area.loc[species, rank] = [all_area, in_area, out_area]

# Save to nc, chunked by species
ECNES_arr.name = 'data'
ECNES_arr.to_netcdf(
    f'{bio_DCCEEW_dir}/bio_DCCEEW_ECNES.nc', 
    mode='w', 
    encoding={'data': {
        "compression": "gzip", 
        "compression_opts": 9,  
        "dtype": 'bool',
        "chunksizes": (1, 1, ECNES_arr.sizes['cell'])}}, 
    engine='h5netcdf'
)

# Get the shared atributs of the ECNES data
ECNES_meta_att = ECNES_meta.groupby(['COMMUNITY']).apply(lambda df: df.iloc[0], include_groups=False)
ECNES_meta_att = ECNES_meta_att.drop(columns=['PRES_RANK', 'SHAPE_Length', 'SHAPE_Area', 'TIF_PATH']).reset_index()

# Save the inside/outside LUTO data to csv
ECNES_df = ECNES_in_out_LUTO_area.copy().reset_index()
ECNES_df['HABITAT_SIGNIFICANCE_PRESTINE_AUSTRALIA'] = ECNES_df['ALL_HA']
ECNES_df['HABITAT_SIGNIFICANCE_BASELINE_SCORE'] = ECNES_df['NATURAL_IN_LUTO_HA'] + ECNES_df['NATURAL_OUT_LUTO_HA']
ECNES_df['HABITAT_SIGNIFICANCE_BASELINE_PERCENT'] = ECNES_df['HABITAT_SIGNIFICANCE_BASELINE_SCORE'] / ECNES_df['ALL_HA'] * 100

# Fill the missing COMMUNITY and PRES_RANK with nan
re_index = pd.MultiIndex.from_product([ECNES_df['COMMUNITY'].unique(), ECNES_df['PRES_RANK'].unique()], names=['COMMUNITY', 'PRES_RANK'])
ECNES_df = ECNES_df.set_index(['COMMUNITY', 'PRES_RANK']).reindex(re_index).reset_index()
    
# Drop unneeded columns, and split the data into three dataframes based on the PRES_RANK
ECNES_df = ECNES_df.drop(columns=['ALL_HA', 'NATURAL_IN_LUTO_HA', 'NATURAL_OUT_LUTO_HA'])
ECNES_df_LIKELY = ECNES_df.query('PRES_RANK == 2').copy().drop(columns=['PRES_RANK'])
ECNES_df_MAYBE = ECNES_df.query('PRES_RANK == 1').copy().drop(columns=['PRES_RANK'])

# Append suffix to the columns for the LIKELY and MAYBE dataframes
ECNES_df_LIKELY.columns = [f'{col}_LIKELY' if col != 'COMMUNITY' else 'COMMUNITY' for col in ECNES_df_LIKELY.columns]
ECNES_df_MAYBE.columns = [f'{col}_MAYBE' if col != 'COMMUNITY' else 'COMMUNITY' for col in ECNES_df_MAYBE.columns]

# Add user defined columns to the LIKELY and MAYBE dataframes
ECNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2100_LIKELY', np.nan)
ECNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2050_LIKELY', np.nan)
ECNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2030_LIKELY', np.nan)

ECNES_df_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2100_MAYBE', np.nan)
ECNES_df_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2050_MAYBE', np.nan)
ECNES_df_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2030_MAYBE', np.nan)

# Merge the LIKELY and MAYBE dataframes, and append the shared attributes
ECNES_df = ECNES_df_LIKELY.merge(ECNES_df_MAYBE, on='COMMUNITY', how='outer')
ECNES_df = ECNES_df.merge(ECNES_meta_att, on='COMMUNITY')

# Reorder the columns
cols = ['COMMUNITY',
        
        'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2030_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2050_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2100_LIKELY',
         
        'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2030_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2050_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2100_MAYBE',
        
        'HABITAT_SIGNIFICANCE_PRESTINE_AUSTRALIA_LIKELY',
        'HABITAT_SIGNIFICANCE_PRESTINE_AUSTRALIA_MAYBE',
        
        'CATEGORY', 'COM_ID','EPBC', 'EXTRACTED', 'CELL_SIZE', 'REGIONS', 'CITATION', 'SPRAT']

ECNES_df = ECNES_df[cols]
ECNES_df.to_csv(f'{bio_DCCEEW_dir}/bio_DCCEEW_ECNES_AREA_HA.csv', index=False)





################################################################################
#           Process Biodiversity Data (NVIS) with Xarray                       #
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
    src_att.to_csv(f'{os.path.dirname(gdb_path)}/{layer_raster}_lookup.csv', index=False)


    # Create a list of delayed jobs, so we can reproject and average rasters in parallel with `n_workers`
    jobs = [delayed(reproject_and_average)(val, src_arr, src.transform, src.crs, meta) for val in src_att['Value']]
    dst_array = np.stack(Parallel(n_jobs=n_workers)(jobs), axis=0)
    
    
    # Save reprojected raster to GeoTiff
    save_path = f'{os.path.dirname(gdb_path)}/{layer_raster}.tif'
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
    save_path = f'{os.path.dirname(gdb_path)}/{layer_raster}.nc'
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}} 
    dst_array_xr.name = 'data'
    dst_array_xr.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')




# --------------- Remove invalid vegetation class; Apply spatial mask ---------------

# Get group names for both pre-European and extant vegetation
PRE_mvg_groups = pd.read_csv('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS7_0_AUST_PRE_MVG_ALB_lookup.csv')['NAME'].tolist()
PRE_mvs_groups = pd.read_csv('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL/NVIS7_0_AUST_PRE_MVS_ALB_lookup.csv')['NAME'].tolist()
EXT_mvg_groups = pd.read_csv('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS7_0_AUST_EXT_MVG_ALB_lookup.csv')['NAME'].tolist()
EXT_mvs_groups = pd.read_csv('N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_EXT_ALL/NVIS7_0_AUST_EXT_MVS_ALB_lookup.csv')['NAME'].tolist()


# Some groups are undertmined and should be exclude from the analysis, such as 'Other ...', 'Unknown/no data', and 'Unclassified'.
rm_names = ['Unknown/no data', 'Unknown/No data']


# Calculate the sum of all groups for each raster
for gdb_path, layer_raster, layer_attribute in files:
    
    # Read NVIS raster and filter out the groups that should be removed
    dst_array_xr = xr.load_dataarray(f'{os.path.dirname(gdb_path)}/{layer_raster}.nc')
    dst_array_xr = dst_array_xr.sel(group=~dst_array_xr.group.isin(rm_names))
    
    # Reorder the groups lexicographically
    dst_array_xr = dst_array_xr.sortby('group')
    
    # Optioin-1: Use the index of the largest group value to represent the cell
    dst_array_xr_argmax = dst_array_xr.argmax(dim='group')     
    dst_array_xr_argmax_area_ha = np.bincount(dst_array_xr_argmax.values, weights=NLUM_area_mask, minlength=dst_array_xr_argmax.max().values+1)
    dst_array_xr_argmax_area_ha = pd.DataFrame({'group':dst_array_xr.coords['group'], 'AREA_HA':dst_array_xr_argmax_area_ha})
    
    # Option-2: Split each group as a separate layer, which is the percentage [0-100] of the group in each cell
    dst_array_xr_area_ha = dst_array_xr * NLUM_area_mask[None,:]
    dst_array_xr_group_area_ha = dst_array_xr_area_ha.sum(dim='cell').compute().to_dataframe(name='AREA_HA').reset_index()
    
    # Save xarray DataArray to NetCDF
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}}
    output_layer_name = layer_raster.replace('_ALB', '')
    
    save_path = f'{os.path.dirname(gdb_path)}/{output_layer_name}_LOW_SPATIAL_DETAIL.nc'
    dst_array_xr_argmax.name = 'data'
    dst_array_xr_argmax.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')
    
    save_path = f'{os.path.dirname(gdb_path)}/{output_layer_name}_HIGH_SPATIAL_DETAIL.nc'
    dst_array_xr.name = 'data'
    dst_array_xr.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')
    
 




# --------------- Get the sum of areas (ha) for pre-1750 ---------------

# Read NVIS data
NVIS_pre_mvg_xr_low_spatial_detail = xr.load_dataarray(f'{NVIS_PRE_1750_path}/NVIS7_0_AUST_PRE_MVG_LOW_SPATIAL_DETAIL.nc')
NVIS_pre_mvs_xr_low_spatial_detail = xr.load_dataarray(f'{NVIS_PRE_1750_path}/NVIS7_0_AUST_PRE_MVS_LOW_SPATIAL_DETAIL.nc')
NVIS_pre_mvg_xr_high_spatial_detail = xr.load_dataarray(f'{NVIS_PRE_1750_path}/NVIS7_0_AUST_PRE_MVG_HIGH_SPATIAL_DETAIL.nc') / 100  # Convert percentage to fraction
NVIS_pre_mvs_xr_high_spatial_detail = xr.load_dataarray(f'{NVIS_PRE_1750_path}/NVIS7_0_AUST_PRE_MVS_HIGH_SPATIAL_DETAIL.nc') / 100  # Convert percentage to fraction

# Get the NVIS names
NVIS_pre_mvg_names = NVIS_pre_mvg_xr_high_spatial_detail.coords['group'].values.tolist()
NVIS_pre_mvs_names = NVIS_pre_mvs_xr_high_spatial_detail.coords['group'].values.tolist()


# --------------- Total vegataion area (ha) pre-1750 ---------------

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------
NVIS_pre_mvg_total_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvg_xr_low_spatial_detail.values,
    weights = zones['CELL_HA'].values,
    minlength = NVIS_pre_mvg_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvs_total_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvs_xr_low_spatial_detail.values,
    weights = zones['CELL_HA'].values,
    minlength = NVIS_pre_mvs_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvg_total_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvg_names, 'TOTAL_AREA_HA': NVIS_pre_mvg_total_ha_low_spatial_detail})
NVIS_pre_mvs_total_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvs_names, 'TOTAL_AREA_HA': NVIS_pre_mvs_total_ha_low_spatial_detail})




# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------
NVIS_pre_mvg_total_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail * zones['CELL_HA'].values[None, :]
NVIS_pre_mvs_total_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail * zones['CELL_HA'].values[None, :]
NVIS_pre_mvg_total_ha_df_high_spatial_detail = NVIS_pre_mvg_total_ha_high_spatial_detail.sum(dim='cell').to_dataframe('TOTAL_AREA_HA').reset_index()
NVIS_pre_mvs_total_ha_df_high_spatial_detail = NVIS_pre_mvs_total_ha_high_spatial_detail.sum(dim='cell').to_dataframe('TOTAL_AREA_HA').reset_index()





# --------------- Vegataion area outside the LUTO study area ---------------

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------
NVIS_pre_mvg_outside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvg_xr_low_spatial_detail.sel(cell=idx_out_LUTO_natural).values, 
    weights = zones['CELL_HA'].values[idx_out_LUTO_natural],
    minlength = NVIS_pre_mvg_xr_low_spatial_detail.max().values + 1
)


NVIS_pre_mvs_outside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvs_xr_low_spatial_detail.sel(cell=idx_out_LUTO_natural).values,
    weights = zones['CELL_HA'].values[idx_out_LUTO_natural],
    minlength = NVIS_pre_mvs_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvg_outside_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvg_names,'OUTSIDE_LUTO_AREA_HA': NVIS_pre_mvg_outside_ha_low_spatial_detail})
NVIS_pre_mvs_outside_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvs_names,'OUTSIDE_LUTO_AREA_HA': NVIS_pre_mvs_outside_ha_low_spatial_detail})


# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------

NVIS_pre_mvg_outside_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail.sel(cell=idx_out_LUTO_natural) * zones['CELL_HA'].values[None, idx_out_LUTO_natural]     
NVIS_pre_mvs_outside_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail.sel(cell=idx_out_LUTO_natural) * zones['CELL_HA'].values[None, idx_out_LUTO_natural]     
NVIS_pre_mvg_outside_ha_df_high_spatial_detail = NVIS_pre_mvg_outside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('OUTSIDE_LUTO_AREA_HA').reset_index()
NVIS_pre_mvs_outside_ha_df_high_spatial_detail = NVIS_pre_mvs_outside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('OUTSIDE_LUTO_AREA_HA').reset_index()





# --------------- Vegataion area inside the LUTO study area ---------------

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------

NVIS_pre_mvg_inside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvg_xr_low_spatial_detail.sel(cell=idx_in_LUTO_natural).values, 
    weights = zones['CELL_HA'].values[idx_in_LUTO_natural],
    minlength = NVIS_pre_mvg_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvs_inside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvs_xr_low_spatial_detail.sel(cell=idx_in_LUTO_natural).values,
    weights = zones['CELL_HA'].values[idx_in_LUTO_natural],
    minlength = NVIS_pre_mvs_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvg_inside_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvg_names,'INSIDE_LUTO_AREA_HA': NVIS_pre_mvg_inside_ha_low_spatial_detail})
NVIS_pre_mvs_inside_ha_df_low_spatial_detail = pd.DataFrame({'group':NVIS_pre_mvs_names,'INSIDE_LUTO_AREA_HA': NVIS_pre_mvs_inside_ha_low_spatial_detail})


# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------
NVIS_pre_mvg_inside_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail.sel(cell=idx_in_LUTO_natural) * zones['CELL_HA'].values[None, idx_in_LUTO_natural]
NVIS_pre_mvs_inside_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail.sel(cell=idx_in_LUTO_natural) * zones['CELL_HA'].values[None, idx_in_LUTO_natural]
NVIS_pre_mvg_inside_ha_df_high_spatial_detail = NVIS_pre_mvg_inside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('INSIDE_LUTO_AREA_HA').reset_index()
NVIS_pre_mvs_inside_ha_df_high_spatial_detail = NVIS_pre_mvs_inside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('INSIDE_LUTO_AREA_HA').reset_index()





# ------------- Combine 'HIGH' and 'LOW' -------------

# Concatenate the two dataframes
NVIS_pre_mvg_low_spatial_detail = NVIS_pre_mvg_total_ha_df_low_spatial_detail.merge(NVIS_pre_mvg_outside_ha_df_low_spatial_detail, on='group')
NVIS_pre_mvg_high_spatial_detail = NVIS_pre_mvg_total_ha_df_high_spatial_detail.merge(NVIS_pre_mvg_outside_ha_df_high_spatial_detail, on='group')

NVIS_pre_mvs_low_spatial_detail = NVIS_pre_mvs_total_ha_df_low_spatial_detail.merge(NVIS_pre_mvs_outside_ha_df_low_spatial_detail, on='group')
NVIS_pre_mvs_high_spatial_detail = NVIS_pre_mvs_total_ha_df_high_spatial_detail.merge(NVIS_pre_mvs_outside_ha_df_high_spatial_detail, on='group')

# Append a user-defined target column
NVIS_pre_mvg_low_spatial_detail['CONSERVATION_TARGET_PCT'] = 30
NVIS_pre_mvg_high_spatial_detail['CONSERVATION_TARGET_PCT'] = 30
NVIS_pre_mvs_low_spatial_detail['CONSERVATION_TARGET_PCT'] = 30
NVIS_pre_mvs_high_spatial_detail['CONSERVATION_TARGET_PCT'] = 30

# Save to CSV
NVIS_pre_mvg_low_spatial_detail.to_csv(NVIS_PRE_1750_path + '/NVIS_MVG_LOW_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvg_high_spatial_detail.to_csv(NVIS_PRE_1750_path + '/NVIS_MVG_HIGH_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvs_low_spatial_detail.to_csv(NVIS_PRE_1750_path + '/NVIS_MVS_LOW_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvs_high_spatial_detail.to_csv(NVIS_PRE_1750_path + '/NVIS_MVS_HIGH_SPATIAL_DETAIL.csv', index=False)




# ------------------------- TMP -------------------------

save_path = 'N:/LUF-Modelling/LUTO2_JZ/TEMP/vegetation_pre_calc_area_ha'

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------

NVIS_pre_mvg_low_spatial_detail_area_ha = pd.concat([
    NVIS_pre_mvg_total_ha_df_low_spatial_detail.set_index('group'), 
    NVIS_pre_mvg_outside_ha_df_low_spatial_detail.set_index('group'),
    NVIS_pre_mvg_inside_ha_df_low_spatial_detail.set_index('group')], axis=1).reset_index()

NVIS_pre_mvg_low_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_AREA_HA'] = NVIS_pre_mvg_low_spatial_detail_area_ha.eval('OUTSIDE_LUTO_AREA_HA	+ INSIDE_LUTO_AREA_HA')
NVIS_pre_mvg_low_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_to_TOTAL'] = NVIS_pre_mvg_low_spatial_detail_area_ha.eval('INSIDE_OUTSIDE_SUM_AREA_HA / TOTAL_AREA_HA')
NVIS_pre_mvg_low_spatial_detail_area_ha.to_csv(f'{save_path}/NVIS_pre_mvg_low_spatial_detail_area_ha.csv', index=False)

NVIS_pre_mvg_high_spatial_detail_area_ha = pd.concat([
    NVIS_pre_mvg_total_ha_df_high_spatial_detail.set_index('group'), 
    NVIS_pre_mvg_outside_ha_df_high_spatial_detail.set_index('group'),
    NVIS_pre_mvg_inside_ha_df_high_spatial_detail.set_index('group')], axis=1).reset_index()

NVIS_pre_mvg_high_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_AREA_HA'] = NVIS_pre_mvg_high_spatial_detail_area_ha.eval('OUTSIDE_LUTO_AREA_HA	+ INSIDE_LUTO_AREA_HA')
NVIS_pre_mvg_high_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_to_TOTAL'] = NVIS_pre_mvg_high_spatial_detail_area_ha.eval('INSIDE_OUTSIDE_SUM_AREA_HA / TOTAL_AREA_HA')
NVIS_pre_mvg_high_spatial_detail_area_ha.to_csv(f'{save_path}//NVIS_pre_mvg_high_spatial_detail_area_ha.csv', index=False)



# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------

NVIS_pre_mvs_low_spatial_detail_area_ha = pd.concat([
    NVIS_pre_mvs_total_ha_df_low_spatial_detail.set_index('group'), 
    NVIS_pre_mvs_outside_ha_df_low_spatial_detail.set_index('group'),
    NVIS_pre_mvs_inside_ha_df_low_spatial_detail.set_index('group')], axis=1).reset_index()

NVIS_pre_mvs_low_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_AREA_HA'] = NVIS_pre_mvs_low_spatial_detail_area_ha.eval('OUTSIDE_LUTO_AREA_HA	+ INSIDE_LUTO_AREA_HA')
NVIS_pre_mvs_low_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_to_TOTAL'] = NVIS_pre_mvs_low_spatial_detail_area_ha.eval('INSIDE_OUTSIDE_SUM_AREA_HA / TOTAL_AREA_HA')
NVIS_pre_mvs_low_spatial_detail_area_ha.to_csv(f'{save_path}//NVIS_pre_mvs_low_spatial_detail_area_ha.csv', index=False)


NVIS_pre_mvs_high_spatial_detail_area_ha = pd.concat([
    NVIS_pre_mvs_total_ha_df_high_spatial_detail.set_index('group'), 
    NVIS_pre_mvs_outside_ha_df_high_spatial_detail.set_index('group'),
    NVIS_pre_mvs_inside_ha_df_high_spatial_detail.set_index('group')], axis=1).reset_index()

NVIS_pre_mvs_high_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_AREA_HA'] = NVIS_pre_mvs_high_spatial_detail_area_ha.eval('OUTSIDE_LUTO_AREA_HA	+ INSIDE_LUTO_AREA_HA')
NVIS_pre_mvs_high_spatial_detail_area_ha['INSIDE_OUTSIDE_SUM_to_TOTAL'] = NVIS_pre_mvs_high_spatial_detail_area_ha.eval('INSIDE_OUTSIDE_SUM_AREA_HA / TOTAL_AREA_HA')
NVIS_pre_mvs_high_spatial_detail_area_ha.to_csv(f'{save_path}/NVIS_pre_mvs_high_spatial_detail_area_ha.csv', index=False)

# --------------------------  TMP END --------------------------