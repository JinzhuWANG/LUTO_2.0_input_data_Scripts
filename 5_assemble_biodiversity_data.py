# The script uses `xarray` to process data, therefore is different compared to rasterio based methods.

# Author:		Jinzhu WANG
# Email: 		wangjinzhulala@gmail.com
# Last update: 	6 Feb, 2025


'''
The benefits of using xarray is its multi-dimension labeling for NDArray, inherent parallelizing,
and raster processing (with rioxarray) capabilities.

The key technical consideration here is how to deal with the ~10k layers in a reasonable time.
We choose to use a 5km spatial resolution to reduce data size, and leverage the parallelising 
of xarray to speed up the processing.
'''


import os, re
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




################################################################################
#                  Process Biodiversity Data (Carla) with Xarray               #
################################################################################


# -----------------------------------------------------------------------------------------------------
#                   Biodiversity suitability dataset from Carla Archibald 
# -----------------------------------------------------------------------------------------------------

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



# Global variables
NLUM = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8') 

bio_GTIFF_dir  = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km'
bio_NetCDF_dir = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km_to_NetCDF'


def find_str(row: pd.Series) -> list:
    """
    Extracts relevant information from the given row's path and returns it as a list.

    Args:
        row (pandas.Series): A pandas Series object representing a row of data.
    Returns:
        list: A list containing the extracted information from the row's path.
    Raises:
        IndexError: If the regular expression fails to find a match for the year.
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

    Parameters:
    - root_dir (str): The root directory to search for TIFF files.
    - save_path (str): The path to save the CSV file.

    Returns:
    None
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
if not os.path.exists(f'{bio_NetCDF_dir}/bio_file_paths_raw.csv'):
    # Create a csv file recording all the paths, group, species, model, ssp, year, mode
    get_all_path(bio_GTIFF_dir, f'{bio_NetCDF_dir}/bio_file_paths_condition.csv')
else:
    # Read the existing csv file
    df = pd.read_csv(f'{bio_NetCDF_dir}/bio_file_paths_raw.csv' )



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


# Read previouse raw data
zones = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5', key='cell_zones_df', columns=['X', 'Y', 'CELL_HA'])
bioph = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5', key = 'cell_biophysical_df', columns=['NATURAL_AREA_INC_WATER'])
lumap = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', key = 'cell_LU_mapping', columns=['LU_DESC'])


# Get the index of cells that are in natural state, and inside/outside the LUTO study area
natural_cells = np.logical_not(bioph['NATURAL_AREA_INC_WATER'].values)  # 0 is natural, 1 is non-natural; so we flip the values to make 1 natural
idx_in_LUTO_natural = np.isin(lumap['LU_DESC'], ['Beef - natural land', 'Dairy - natural land', 'Sheep - natural land', 'Unallocated - natural land'])
idx_out_LUTO = np.isin(lumap['LU_DESC'], ['Non-agricultural land'])     # shape=6956407, sum=2737674
idx_out_LUTO_natural = idx_out_LUTO & natural_cells

# Convert the index to xarray; 1D with cell as the primary dimension, and y, x as the coordinates
NLUM_zero = NLUM.copy() * 0
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
        f'{bio_NetCDF_dir}/bio_{ssp}_{mode}.nc', 
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
bio_suitability_ncs = glob(f'{bio_NetCDF_dir}/*_EnviroSuit.nc')
  
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
        f'{bio_NetCDF_dir}/{fname}_group.nc', 
        mode='w', 
        encoding={'data': {
            'compression': 'gzip', 
            'compression_opts': 9, 
            'dtype': 'float32',
            'chunksizes': (1, 1, group_arr_contribution.sizes['y'], group_arr_contribution.sizes['x'])}}, 
        engine='h5netcdf'
    )
    
    del group_arr_contribution




# ------------------- Calculate the biodiversity score for each species ------------------------------------------

# Calculate the contribution, with real_area weighted
bio_condition_ncs = glob(f'{bio_NetCDF_dir}/*_EnviroSuit.nc')

for nc in bio_condition_ncs:
    
    fname = os.path.basename(nc).replace('_EnviroSuit.nc', '_Condition')
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
        arr = bio_suitability.sel(year=sel_year, species=sel_species).compute()
        all_sum = arr.sum(['y', 'x'])
        in_sum = arr.where(idx_in_LUTO_natural_2D_bio).sum(['y', 'x'])
        out_sum = arr.where(idx_out_LUTO_natural_2D_bio).sum(['y', 'x'])
        return sel_year, sel_species, all_sum, in_sum.values, out_sum.values
        
    tasks = [
        delayed(get_val)(yr, sp) 
        for sp in bio_suitability['species']
        for yr in bio_suitability['year']
    ]
    for yr, sp, val_sum, val_in, val_out in tqdm(Parallel(n_jobs=-1, return_as='generator')(tasks), total=len(tasks)):
        bio_suitability_sum.loc[yr, sp] = [val_sum, val_in, val_out]

    # Save to csv
    bio_suitability_sum.to_dataframe('BIO_SCORE_HA').reset_index().to_csv(f'{bio_NetCDF_dir}/{fname}.csv', index=False)




# Get the biodiversity score for the baseline year (1990), as well as the in/out LUTO scores
bio_in_and_out = pd.DataFrame()

for bio_scores in glob(f'{bio_NetCDF_dir}/*_Condition.csv'):
    
    ssp = re.compile(r'bio_ssp(\d*)_').findall(bio_scores)[0]
    bio_baseline = pd.read_csv(bio_scores).query('year == 1990').query('source == "all"')
    bio_out = pd.read_csv(bio_scores).query('year != 1990').query('source == "out"')
    
    
    # Combine baseline and in/out LUTO scores
    bio_df = pd.concat([bio_baseline, bio_out], ignore_index=True).sort_values(['species', 'source', 'year'])
    bio_df['SSP'] = ssp
    bio_in_and_out = pd.concat([bio_in_and_out, bio_df])
    
# Save to disk
bio_in_and_out.to_csv(f'{bio_NetCDF_dir}/BIODIVERSITY_GBF4A_SCORES.csv', index=False)

bio_target = pd.DataFrame({
    'species':bio_in_and_out['species'].unique(), 
    'USER_DEFINED_TARGET_PERCENT':np.nan}
)

bio_target.to_csv(f'{bio_NetCDF_dir}/BIODIVERSITY_GBF4A_TARGET.csv', index=False)





################################################################################
#           Process Biodiversity Data (DCCEEW) with Xarray                     #
################################################################################


# ------------------- Rasterise SNES/ECNES data to GEOTIFF ------------------------------------------


# Set parameters
n_workers = 50
SNES_TIF_PATH = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF'


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

if not os.path.exists(f"{SNES_TIF_PATH}/snes_dissolve.geojson"):
    snes_dissolve.to_file(f"{SNES_TIF_PATH}/DISSOLVED_VECTOR/snes_dissolve.geojson")
if not os.path.exists(f"{SNES_TIF_PATH}/ecnes_dissolve.geojson"):
    ecnes_dissolve.to_file(f"{SNES_TIF_PATH}/DISSOLVED_VECTOR/ecnes_dissolve.geojson")


def get_presVal_savePath(row):
    # Get value for rasterisation polygon (1 for 'maybe present', 2 for 'likely present')
    if 'PRES_RANK' in row:  # ECNES data
        val = row['PRES_RANK']
        name = row['COMMUNITY'].replace('/', '_')
        save_path = f'{SNES_TIF_PATH}/ECNES/{name}_{presence_dict[val]}.tif'
    else:                   # SNES data
        val = row['PRESENCE_RANK']
        name = row['SCIENTIFIC_NAME'].replace('/', '_')
        save_path = f'{SNES_TIF_PATH}/SNES/{row["TAXON_GROUP"]}/{name}/{name}_{presence_dict[val]}.tif'
    
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
if not os.path.exists(f'{SNES_TIF_PATH}/ECNES'):
    os.makedirs(f'{SNES_TIF_PATH}/ECNES', exist_ok=True)
    
    

# Rasterise and save the SNES data to GEOTIFF
tasks = [delayed(rasterize)(row) for _,row in snes_dissolve.iterrows()]
for _ in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
    pass

# Save SNES attributes to csv
snes_meta = snes_dissolve.copy().drop(columns='geometry')
snes_meta['TIF_PATH'] = snes_meta.apply(lambda x: get_presVal_savePath(x)[1], axis=1)
snes_meta.to_csv(f'{SNES_TIF_PATH}/DCCEEW_SNES_meta.csv', index=False)



# Rasterise and save the ECNES data to GEOTIFF
tasks = [delayed(rasterize)(row) for _,row in ecnes_dissolve.iterrows()]

raster_arr = []
for out in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
    raster_arr.append(out)

# Save ECNES attributes to csv
ecnes_meta = ecnes_dissolve.copy().drop(columns='geometry')
ecnes_meta['TIF_PATH'] = ecnes_meta.apply(lambda x: get_presVal_savePath(x)[1], axis=1)
ecnes_meta.to_csv(f'{SNES_TIF_PATH}/DCCEEW_ECNES_meta.csv', index=False)






# ------------------- Masking GEOTIFFs and save SNES to NetCDF ------------------------------------------

bio_DCCEEW_dir = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/To_NetCDF'

# Read DCCEEW SNES GeoTIFF file paths
SNES_meta = pd.read_csv('N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/DCCEEW_SNES_meta.csv')
ECNES_meta = pd.read_csv('N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/DCCEEW_ECNES_meta.csv')


# Create an empty array to store the data
SNES_arr = xr.DataArray(
    np.zeros((len(SNES_meta), NLUM.sum().values), dtype=np.bool_), 
    dims=['species', 'cell'], 
    coords={'species':SNES_meta['SCIENTIFIC_NAME'], 'cell':np.arange(NLUM.sum().values)}
).assign_coords({
    k: ('species', v.tolist())
    for k, v in SNES_meta.items()
    if k not in ['SCIENTIFIC_NAME', 'TIF_PATH']
})


# Parallel processing put the data into the empty array
def get_arr(row):
    ds = rxr.open_rasterio(row['TIF_PATH']).sel(band=1).drop_vars('band')
    ds = xr.where(ds == ds.rio.nodata, 0, ds)
    ds = xr.where(ds.isin([1, 2]), 1, 0)        # 1 is 'MAYBE', 2 is 'LIKELY'. We convert them to 1 so that we can use bool_ type
    ds = ds.values.ravel()[np.flatnonzero(NLUM.values)]
    return row['SCIENTIFIC_NAME'], ds

tasks = (delayed(get_arr)(row) for _,row in SNES_meta.iterrows())
for species,arr in tqdm(Parallel(n_jobs=-1, return_as='generator')(tasks), total=len(SNES_meta)):
    SNES_arr.loc[species] = arr


# Save to nc, chunked by year, species, leave x, y as unlimited
SNES_arr.name = 'data'
SNES_arr.to_netcdf(
    f'{bio_DCCEEW_dir}/bio_DCCEEW_SNES.nc', 
    mode='w', 
    encoding={'data': {
        "compression": "gzip", 
        "compression_opts": 9,  
        "dtype": 'bool',
        "chunksizes": (1, SNES_arr.sizes['cell'])}}, 
    engine='h5netcdf'
)



# ------------------- Masking GEOTIFFs and save ECNES to NetCDF ------------------------------------------

# Read DCCEEW ECNES GeoTIFF file paths
ECNES_meta = pd.read_csv('N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/DCCEEW_ECNES_meta.csv')

# Create an empty array to store the data
ECNES_arr = xr.DataArray(
    np.zeros((len(ECNES_meta), NLUM.sum().values), dtype=np.bool_), 
    dims=['species', 'cell'], 
    coords={'species':ECNES_meta['COMMUNITY'], 'cell':np.arange(NLUM.sum().values)}
).assign_coords({
    k: ('species', v.tolist())
    for k, v in ECNES_meta.items() 
    if k not in ['COMMUNITY', 'TIF_PATH']}
)


# Parallel processing put the data into the empty array
def get_arr(row):
    ds = rxr.open_rasterio(row['TIF_PATH']).sel(band=1).drop_vars('band')
    ds = xr.where(ds == ds.rio.nodata, 0, ds)
    ds = xr.where(ds.isin([1, 2]), 1, 0)        # 1 is 'MAYBE', 2 is 'LIKELY'. We convert them to 1 so that we can use bool_ type
    ds = ds.values.ravel()[np.flatnonzero(NLUM.values)]
    return row['COMMUNITY'], ds

tasks = (delayed(get_arr)(row) for _,row in ECNES_meta.iterrows())
for species,arr in tqdm(Parallel(n_jobs=10, return_as='generator')(tasks), total=len(ECNES_meta)):
    ECNES_arr.loc[species] = arr


# Save to nc, chunked by year, species, leave x, y as unlimited
ECNES_arr.name = 'data'
ECNES_arr.to_netcdf(
    f'{bio_DCCEEW_dir}/bio_DCCEEW_ECNES.nc', 
    mode='w', 
    encoding={'data': {
        "compression": "gzip", 
        "compression_opts": 9,  
        "dtype": 'bool',
        "chunksizes": (1, ECNES_arr.sizes['cell'])}}, 
    engine='h5netcdf'
)





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

# Read raw zones and lumap database
zones = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5', key='cell_zones_df', columns=['CELL_HA'])
bioph = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5', key = 'cell_biophysical_df', columns=['NATURAL_AREA_INC_WATER'])
lumap = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', key = 'cell_LU_mapping', columns=['LU_DESC'])

natural_cells = np.logical_not(bioph['NATURAL_AREA_INC_WATER'].values) # 0 is natural, 1 is non-natural; so we flip the values to make 1 natural

# Get the index of cells that are outside the LUTO study area, AND, also in natural state
idx_out_LUTO = np.isin(lumap['LU_DESC'], ['Non-agricultural land'])     # shape=6956407, sum=2737674
idx_out_LUTO_natural = idx_out_LUTO & natural_cells

# Get the index of cells that are inside the LUTO study area, AND, also in natural state
idx_in_LUTO_natural = np.isin(lumap['LU_DESC'], ['Beef - natural land', 'Dairy - natural land', 'Sheep - natural land', 'Unallocated - natural land'])



# Read NVIS data
PRE1750_path = 'N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL'

NVIS_pre_mvg_xr_low_spatial_detail = xr.load_dataarray(f'{PRE1750_path}/NVIS7_0_AUST_PRE_MVG_LOW_SPATIAL_DETAIL.nc')
NVIS_pre_mvs_xr_low_spatial_detail = xr.load_dataarray(f'{PRE1750_path}/NVIS7_0_AUST_PRE_MVS_LOW_SPATIAL_DETAIL.nc')
NVIS_pre_mvg_xr_high_spatial_detail = xr.load_dataarray(f'{PRE1750_path}/NVIS7_0_AUST_PRE_MVG_HIGH_SPATIAL_DETAIL.nc') / 100  # Convert percentage to fraction
NVIS_pre_mvs_xr_high_spatial_detail = xr.load_dataarray(f'{PRE1750_path}/NVIS7_0_AUST_PRE_MVS_HIGH_SPATIAL_DETAIL.nc') / 100  # Convert percentage to fraction

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
NVIS_pre_mvg_low_spatial_detail.to_csv(PRE1750_path + '/NVIS_MVG_LOW_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvg_high_spatial_detail.to_csv(PRE1750_path + '/NVIS_MVG_HIGH_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvs_low_spatial_detail.to_csv(PRE1750_path + '/NVIS_MVS_LOW_SPATIAL_DETAIL.csv', index=False)
NVIS_pre_mvs_high_spatial_detail.to_csv(PRE1750_path + '/NVIS_MVS_HIGH_SPATIAL_DETAIL.csv', index=False)




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