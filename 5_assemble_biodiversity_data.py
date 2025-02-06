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




########################################################################
#                  Process Biodiversity Data with Xarray               #
########################################################################


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
- mode: 	    {'EnviroSuit', 'EnviroSuit_max', 'EnviroSuit_min', 'historic'}      Use only 'EnviroSuit' in LUTO

And the data has a metadata of:
- 808 rows *  978 columns
- 5km resolution
- int8 data type
- nodata value: 255
- CRS: EPSG:4283


To incoporate this data to LUTO, we use xarray to combine all GeoTIFF files into a single NetCDF file. Essentialy, the nc file 
can be thought as a data cube of 5 dimensions: (year * species, * x * y), with group information attached to the species dimension.
'''


import os, re
import netCDF4
import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd

from glob import glob
from itertools import product
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from scipy.ndimage import distance_transform_edt
from rasterio.enums import Resampling

# Global variables
NLUM = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8')

bio_GTIFF_dir  = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km'
bio_NetCDF_dir = 'N:/Data-Master/Biodiversity/Environmental-suitability/Annual-species-suitability_20-year_snapshots_5km_to_NetCDF'



def replace_with_nearest(map_: np.ndarray, filler: int) -> np.ndarray:
    """
    Replaces invalid values in the input array with the nearest non-filler values.

    Parameters:
        map_ (np.ndarray, 2D): The input array.
        filler (int): The value to be considered as invalid.

    Returns:
        np.ndarray (2D): The array with invalid values replaced by the nearest non-invalid values.
    """
    # Create a mask for invalid values
    mask = (map_ == filler)
    # Perform distance transform on the mask
    _, nearest_indices = distance_transform_edt(mask, return_indices=True)
    # Replace the invalid values with the nearest non-invalid values
    map_[mask] = map_[tuple(nearest_indices[:, mask])]
    
    return map_


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


# Create an tempalate biodiversity MASK in netcdf format
bio_mask = rxr.open_rasterio(df.iloc[0]['path']).squeeze('band').drop_vars('band').astype('uint8')
bio_mask = xr.where(bio_mask != bio_mask.rio.nodata, 1, 0).astype('uint8')
bio_mask = bio_mask.rio.write_crs(NLUM.rio.crs)
bio_mask.name = 'data'
bio_mask.to_netcdf(
    f'{bio_NetCDF_dir}/bio_mask.nc', 
    mode='w', 
    encoding={'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}},
    engine='h5netcdf'
)


# Create an ID map for the biodiversity data
id_map = np.arange(bio_mask.size).reshape(bio_mask.shape)
id_map = xr.DataArray(
    id_map, 
    dims=['y', 'x'], 
    coords={'y': bio_mask.coords['y'], 'x': bio_mask.coords['x']})

id_map = id_map.rio.write_crs(bio_mask.rio.crs)
id_map = id_map.rio.write_transform(bio_mask.rio.transform())
id_map = id_map.rio.reproject_match(NLUM, Resampling = Resampling.nearest, nodata=bio_mask.size + 1).chunk('auto')
    
id_map.attrs = {}
id_map.name = 'data'
encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint32'}} 
id_map.to_netcdf(f'{bio_NetCDF_dir}/bio_id_map.nc', encoding=encoding, engine='h5netcdf')




# --------------------------------------Convert GeoTIFF to NetCDF--------------------------------------------

# Filter out the ensemble data
ensemble_df = df.query('model == "GCM-Ensembles" & mode == "EnviroSuit"').drop(columns=['model'])
valid_species = ensemble_df['species'].unique()                  
historic_df = df.query('model == "historic" and species.isin(@valid_species) and ~path.str.contains("5x5")')


# Define the function to convert tif to nc
def process_row(row,bio_mask=bio_mask):
    ds = rxr.open_rasterio(row['path']).sel(band=1).drop_vars('band')           # Only select the first band
    ds.values = replace_with_nearest(ds.values, ds.rio.nodata).astype('uint8')  # Replace nodata with nearest valide value
    ds = ds.expand_dims({'year':[row['year']], 'species':[row['species']]})     # Append year and species as dims
    ds = ds.assign_coords(group=('species', [row['group']]))                    # Attach group to species dim
    ds['x'] = bio_mask['x']
    ds['y'] = bio_mask['y']
    return ds  


def tif_to_nc(df, ssp, mode):
    # Multi-threading to read TIF and expand dims
    in_df = df.query(f'ssp == "{ssp}" and mode == "{mode}"')
    tasks = (delayed(process_row)(row) for _,row in in_df.iterrows())
    para_obj = Parallel(n_jobs=-1, return_as='generator')
    return [result for result in tqdm(para_obj(tasks), total=len(in_df))]

        
# Save ensemble data to nc !!!!!!!!!! This will take ~5 hours to finish !!!!!!!!!!      
historic_xr = tif_to_nc(historic_df, 'historic', 'historic')
for ssp, mode in product(ensemble_df['ssp'].unique(), ensemble_df['mode'].unique()):
    # Pass if the file already exists
    if os.path.exists(f'{bio_NetCDF_dir}/bio_{ssp}_{mode}.nc'):
        print(f'{ssp}_{mode}.nc already exists')
        continue

    # get the data
    ensemble_arrs = tif_to_nc(ensemble_df, ssp, mode)
    ensemble_arrs = xr.combine_by_coords(historic_xr + ensemble_arrs, fill_value=0, combine_attrs='drop')

    # Save to nc
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}} 
    ensemble_arrs.name = 'data'
    ensemble_arrs.to_netcdf(f'{bio_NetCDF_dir}/bio_{ssp}_{mode}.nc', mode='w', encoding=encoding, engine='h5netcdf')

    del ensemble_arrs


# -------------------- Calculate biodiversity contribution ------------------------------------------

# Search for biodiversity NetCDF files
n_chunks = 1000
n_workers = 15
bio_raw_ncs = glob(f'{bio_NetCDF_dir}/*EnviroSuit.nc')
bio_xr_mask = xr.open_dataset(f'{bio_NetCDF_dir}/bio_mask.nc')['data'].astype(np.bool_)
  
# Save nc to disk
for nc in bio_raw_ncs:
    fname = os.path.basename(nc).replace('EnviroSuit', 'Condition').replace('.nc', '')
    bio_xr_raw = xr.open_dataset(nc, chunks='auto')['data']

    # Calculate the biodiversity contribution scores for each species.
    # Contribution is the percentage of each cell's value to the sum of whole layer for 1990
    def process_chunks(data, sel_species):
        return (data.sel(species=sel_species) 
                / (data.sel(year=1990, species=sel_species).astype('uint64') * bio_xr_mask).sum(['x', 'y'])
                * 100).astype(np.float32).compute()
        
    tasks = [
        delayed(process_chunks)(bio_xr_raw, sel_species) 
        for sel_species in np.array_split(bio_xr_raw['species'].values, n_chunks)
    ]
    
    bio_species_contributions = xr.combine_by_coords(list(
        tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=n_chunks))
    )
    
    bio_species_contributions.to_netcdf(
        f'{bio_NetCDF_dir}/{fname}.nc', 
        mode='w', 
        encoding={'data': {'compression': 'gzip', 'compression_opts': 9, 'dtype': 'float32'}}, 
        engine='h5netcdf'
    )
    
    del bio_species_contributions
    
    
# -------------------- Calculate biodiversity contribution by group ------------------------------------------

# Search for biodiversity NetCDF files
n_workers = 15
bio_condition_ncs = glob(f'{bio_NetCDF_dir}/*Condition.nc')
  
# Save nc to disk
for nc in bio_condition_ncs:
    
    fname = os.path.basename(nc).replace('.nc', '')
    bio_species_contributions = xr.open_dataset(nc, chunks='auto')['data']
    
    # Calculate the biodiversity contribution scores for each group
    def process_chunks(data, sel_group):
        return data.sel(group=sel_group).mean('species').compute()
    
    tasks = [
        delayed(process_chunks)(bio_species_contributions, sel_group)
        for sel_group in set(bio_species_contributions['group'].values)
    ]

    bio_contribution_group = xr.combine_by_coords(list(
        tqdm(Parallel(n_jobs=min(len(tasks), n_workers), return_as='generator')(tasks), total=len(tasks)))
    )

    bio_contribution_group.to_netcdf(
        f'{bio_NetCDF_dir}/{fname}_group.nc', 
        mode='w', 
        encoding={'data': {'compression': 'gzip', 'compression_opts': 9, 'dtype': 'float32'}}, 
        engine='h5netcdf'
    )
    
    del bio_contribution_group




# -------------------- Reproject group biodiversity contribution to 1km ------------------------------------------

bio_suitablity_nc = glob(f'{bio_NetCDF_dir}/*Condition_group.nc')

for nc in bio_suitablity_nc:
    
    fname = os.path.basename(nc).replace('.nc', '')
    
    # Read the data
    bio_xr = xr.open_dataset(nc, chunks='auto')['data']
    bio_xr = bio_xr.rio.write_crs(NLUM.rio.crs)
    
    # Reproject the data to 1km using parallel processing
    def reproject_chunk(from_arr, to_arr, year, group):
        reproj_arr = from_arr.rio.reproject_match(to_arr, Resampling = Resampling.bilinear, nodata=0)
        return reproj_arr.expand_dims({'year':[year], 'group':[group]})
    
    tasks = [
        delayed(reproject_chunk)(bio_xr.sel(year=year, group=group), NLUM, year, group)
        for group in bio_xr['group'].values
        for year in bio_xr['year'].values
    ]
    
    bio_xr = xr.combine_by_coords(list(
        tqdm(Parallel(n_jobs=min(n_workers, len(tasks)),return_as='generator')(tasks), total=len(tasks)))
    )
    
    # Apply NLUM mask
    sel_y = xr.DataArray(np.nonzero(NLUM)[0].values, dims='cell')
    sel_x = xr.DataArray(np.nonzero(NLUM)[1].values, dims='cell')
    bio_xr = bio_xr['data'].isel(y=sel_y, x=sel_x)


    bio_xr.to_netcdf(
        f'{bio_NetCDF_dir}/{fname}_1km.nc', 
        mode='w', 
        encoding={'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'float32'}},
        engine='h5netcdf'
    )
    
    del bio_xr












# ------------------- Biodiversity Dataset 1) from Carla Archibald ---------------------------
'''
Historical and future habitat suitability and condition projections for terrestrial vertebrate and 
vascular plant species (total ~106k species * 5km resolution).


Each species-layer represents the map of rescaled (0-100) suitability for the given species.

LUTO uses this dataset to add biodiversity constraints:
(1) <Completed> By squashing all ~106k layers into a single layer with the Zonation algorithm, LUTO can 
determined the overall importances for all species.

(2) <TODO> By sperating all species into different groups (plant, mammals, ...) or endangered status, LUTO
can prioritise the conservation for a specific group or endanger level.
'''

# The code can be found below:
# N:/Data-Master/Biodiversity/Processing_as_LUTO_input/biodiversity_contribution_Species_Occurrence_Records




# ------------------- Biodiversity Dataset 2) from DCCEWW ---------------------------
'''
This dataset is originaly provided as a vector data. Unlike Calar's data that each cell has a float number 
representing a species suitability, this dataset use "may occur" and "likely to occur" to indicate the presense
of a spcies.

<TODO>
To incoporate this data to LUTO, we preprocessed it with below steps:
(1) Rasterising the vector data to GEOTIFF format.
(2) Use the Zonation algorithm to squash all layers into a single layer of overall biodiversity importance.
(3) Normalizing the layers based on species group or endanger status, calculate each species contribution to the subgroup.
'''

# The code can be found below:
# N:/Data-Master/Biodiversity/Processing_as_LUTO_input/biodiversity_contribution_DCCEWW


