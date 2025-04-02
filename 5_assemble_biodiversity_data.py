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




###############################################################################################
#                                  Global variables                                           #
###############################################################################################

NLUM = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8') 
NLUM_zero = NLUM.copy() * 0

bio_Carla_EnviroSuit_dir = 'N:/Data-Master/Biodiversity/Environmental-suitability'
bio_Carla_GTIFF_dir  = f'{bio_Carla_EnviroSuit_dir}/Annual-species-suitability_20-year_snapshots_5km'
bio_Carla_NetCDF_dir = f'{bio_Carla_EnviroSuit_dir}/Annual-species-suitability_20-year_snapshots_5km_to_NetCDF'

SNES_TIF_path = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF'
bio_DCCEEW_dir = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/To_NetCDF'

NVIS_PRE_1750_path = 'N:/Data-Master/NVIS/NVIS_V7_0_AUST_RASTERS_PRE_ALL'

HCAS_condition = 'N:/Data-Master/Habitat_condition_assessment_system/Data/Processed/HABITAT_CONDITION.csv'
Unalloc_nat_code = 23

# Read previouse raw data
zones = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5', key='cell_zones_df', columns=['X', 'Y', 'CELL_HA'])
bioph = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5', key = 'cell_biophysical_df', columns=['NATURAL_AREA_INC_WATER'])
lumap = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5', key = 'cell_LU_mapping', columns=['LU_DESC','LU_ID_LUTO'])


# Get real area for each cell
real_area_ha = zones['CELL_HA'].values
real_area_ha_2D = NLUM_zero.copy().astype(np.float32)
np.place(real_area_ha_2D.values, NLUM.values, real_area_ha)

# Get the index of cells that are in natural state, and inside/outside the LUTO study area
natural_cells = np.logical_not(bioph['NATURAL_AREA_INC_WATER'].values)              # 0 is natural, 1 is non-natural; so we flip the values to make 1 natural
idx_in_LUTO = np.logical_not(np.isin(lumap['LU_DESC'], ['Non-agricultural land']))  # shape=6956407, sum=4218733
idx_out_LUTO = np.isin(lumap['LU_DESC'], ['Non-agricultural land'])                 # shape=6956407, sum=2737674
idx_in_LUTO_natural = idx_in_LUTO & natural_cells                                   # shape=6956407, sum=3267523
idx_out_LUTO_natural = idx_out_LUTO & natural_cells                                 # shape=6956407, sum=2677065


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
#                  Process Speciese Conservation Priority data (GBF2) with Xarray             #
###############################################################################################

SSPs = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

# Empty list to store the conservation priority performance data
GBF2_conserve_performance = pd.DataFrame()

# Get conservation priority raster/csv
for ssp in SSPs:
    
    # Read the conservation priority data, select the cells of all Australia
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
            AREA_COVERAGE_PERCENT=lambda df: df['area'].astype("float").cumsum() / df['area'].sum() * 100,              # Needs to convert cumsum to float to avoid numerical issues
            PRIORITY_RANK_CUMSUM_CONTRIBUTION=lambda df: (df['PRIORITY_RANK'].astype("float") * df['area']).cumsum() / (df['PRIORITY_RANK'] * df['area']).sum() * 100,
            ssp=ssp
        ).drop(columns=['area', 'cell'])
    
    # Select rows with AREA_COVERAGE_PERCENT closest to integer values from 0 to 100
    ly_stats = ly_stats.iloc[abs((np.arange(101)).reshape(-1, 1) - ly_stats['AREA_COVERAGE_PERCENT'].values).argmin(axis=1)].copy()
    ly_stats['AREA_COVERAGE_PERCENT'] = np.arange(101)
    
    # Save the conservation priority data to the array
    GBF2_conserve_performance = pd.concat([GBF2_conserve_performance, ly_stats])


# Save csv to Excel
with pd.ExcelWriter(f'{bio_Carla_NetCDF_dir}/GBF2_conserve_performance.xlsx') as writer:
    for ssp, df in GBF2_conserve_performance.groupby('ssp'):
        df = df[['AREA_COVERAGE_PERCENT', 'PRIORITY_RANK', 'PRIORITY_RANK_CUMSUM_CONTRIBUTION']]
        df.to_excel(writer, sheet_name=ssp, index=False)






###############################################################################################
#                  Process Speciese Suitability data (GBF4A) with Xarray                      #
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
np.place(idx_in_LUTO_2D.values, NLUM.values, idx_in_LUTO_natural.astype('uint8'))
idx_out_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_out_LUTO_natural_2D.values, NLUM.values, idx_out_LUTO_natural.astype('uint8'))

# Get the coordinates of the cells that are in natural state, inside/outside the LUTO study area
idx_in_LUTO_natural_2D_bio = idx_in_LUTO_2D.interp(x=bio_coord_x, y=bio_coord_y, method='nearest', kwargs={'fill_value': 0}).astype('bool')
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

species_selected = ['Abutilon_grandifolium','Acacia_baeuerlenii','Glaphyromorphus_punctulatus','Goodenia_minutiflora']

# Calculate the contribution, with real_area weighted
bio_condition_ncs = glob(f'{bio_Carla_NetCDF_dir}/*_EnviroSuit.nc')

for nc in bio_condition_ncs:
    
    fname = os.path.basename(nc).replace('_EnviroSuit.nc', '_EnviroSuit_Score')
    # Biodiversity scores for ALL Australia, inside LUTO study area, and outside LUTO study area
    score_sources = ['all', 'in', 'out']
    # Read the data
    bio_suitability = xr.open_dataarray(nc, chunks={'year': 1, 'species': 1}).sel(species=species_selected)
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
bio_target.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF4A_TARGET.csv', index=False)


# Get the biodiversity suitability area weighted scores for each SSP
bio_scores = pd.DataFrame()

for f in glob(f'{bio_Carla_NetCDF_dir}/*EnviroSuit_Score.csv'):
    ssp = re.compile(r'bio_ssp(\d*)_').findall(f)[0]
    bio_out = pd.read_csv(f).query('year != 1990').query('source == "out"').drop(columns=['source']).set_index(['species', 'year'])
    bio_out = bio_out.rename(columns={'BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA': f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{ssp}'})
    bio_scores = pd.concat([bio_scores, bio_out], axis=1)

bio_scores = bio_scores.reset_index()
bio_scores.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF4A_SCORES.csv', index=False)




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
        for gp in bio_suitability['species'].values
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

bio_target.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF4A_TARGET_GROUP.csv', index=False)
    
    
# Get the biodiversity suitability area weighted scores for each SSP
bio_scores = pd.DataFrame()

for f in glob(f'{bio_Carla_NetCDF_dir}/*group_Score.csv'):
    ssp = re.compile(r'bio_ssp(\d*)_').findall(f)[0]
    bio_out = pd.read_csv(f).query('year != 1990').query('source == "out"').drop(columns=['source']).set_index(['group', 'year'])
    bio_out = bio_out.rename(columns={'BIO_SUITABILITY_AREA_WEIGHTED_SCORE_HA': f'OUTSIDE_LUTO_NATURAL_SUITABILITY_AREA_WEIGHTED_HA_SSP{ssp}'})
    bio_scores = pd.concat([bio_scores, bio_out], axis=1)

bio_scores = bio_scores.reset_index()
bio_scores.to_csv(f'{bio_Carla_NetCDF_dir}/BIODIVERSITY_GBF4A_SCORES_GROUP.csv', index=False)





################################################################################
#           Process Biodiversity Data (DCCEEW) (GBF4B) with Xarray             #
################################################################################


# ------------------- Rasterise SNES/ECNES data to GEOTIFF ------------------------------------------

'''
This section rasterises the SNES and ECNES data to GEOTIFF files. The data is dissolved by 'SCIENTIFIC_NAME' for SNES 
and 'COMMUNITY' for ECNES.

Each cell is assigned a value of 1 for 'maybe present' and 2 for 'likely present'. The rasterised data is then saved
to the '{SNES_TIF_path}/DISSOLVED_VECTOR/' folder. A csv file containing the metadata of the dissolved vector data and 
paths to the rasterised data is also saved to '{SNES_TIF_path}/DCCEEW_SNES_meta.csv'.

Note: The 'LIKELY' and 'MAYBE' layers are not overlapped, 'MAYBE' layers are surrounding the 'LIKELY' layers.

'''


# Set parameters
n_workers = 20

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


# Define the k-v pair for presence 
presence_dict = {1: 'MAYBE', 2: 'LIKELY'}   


# Read the SNES biodiversity data; dissolve the data by 'SCIENTIFIC_NAME'
if os.path.exists(f"{SNES_TIF_path}/DISSOLVED_VECTOR/snes_dissolve.geojson"):
    snes_dissolve = gpd.read_file(f"{SNES_TIF_path}/DISSOLVED_VECTOR/snes_dissolve.geojson")
else:
    snes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/snes_public_gdb.gdb", driver="OpenFileGDB", layer="SNES_Public")
    snes_dissolve = snes.dissolve(by=['SCIENTIFIC_NAME','PRESENCE_CATEGORY']).reset_index()
    snes_dissolve.to_file(f"{SNES_TIF_path}/DISSOLVED_VECTOR/snes_dissolve.geojson")

# Read the ECNES biodiversity data; dissolve the data by 'COMMUNITY'   
if os.path.exists(f"{SNES_TIF_path}/DISSOLVED_VECTOR/ecnes_dissolve.geojson"):
    ecnes_dissolve = gpd.read_file(f"{SNES_TIF_path}/DISSOLVED_VECTOR/ecnes_dissolve.geojson")
else:
    ecnes = gpd.read_file("N:/Data-Master/Biodiversity/DCCEEW/ECnes_public_gdb.gdb", driver="OpenFileGDB", layer="ECnes_public")
    ecnes_dissolve = ecnes.dissolve(by=['COMMUNITY', 'CATEGORY']).reset_index()
    ecnes_dissolve.to_file(f"{SNES_TIF_path}/DISSOLVED_VECTOR/ecnes_dissolve.geojson")


def get_presense_and_save_path(row):
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
    val, save_path = get_presense_and_save_path(row)
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
    tif_path = get_presense_and_save_path(row)[1]
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
snes_meta['TIF_PATH'] = snes_meta.apply(lambda x: get_presense_and_save_path(x)[1], axis=1)
snes_meta.to_csv(f'{SNES_TIF_path}/DCCEEW_SNES_meta.csv', index=False)



# Rasterise and save the ECNES data to GEOTIFF
tasks = [delayed(rasterize)(row) for _,row in ecnes_dissolve.iterrows()]

raster_arr = []
for out in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
    raster_arr.append(out)

# Save ECNES attributes to csv
ecnes_meta = ecnes_dissolve.copy().drop(columns='geometry')
ecnes_meta['TIF_PATH'] = ecnes_meta.apply(lambda x: get_presense_and_save_path(x)[1], axis=1)
ecnes_meta.to_csv(f'{SNES_TIF_path}/DCCEEW_ECNES_meta.csv', index=False)



# ------------------- Merge 'LIKELY' and 'MAYBE' layer ------------------------------------------
'''
For each SNES/ECNES species, we have rasterised them into two layers: 'LIKELY' and 'MAYBE' (some only has 'LIKELY'), and use
1 to indicate 'MAYBE' and 2 for 'LIKELY'. 

Here we merge the two layers into a single layer by assiging 0.8 for 'LIKELY' and 0.3 for 'MAYBE'. The merged layer is then saved
to the '{SNES_TIF_path}/LIKELY_MAYBE_MERGED/' folder. A csv file containing the metadata of the dissolved vector data and paths to the rasterised
data is also saved to '{SNES_TIF_path}/DCCEEW_SNES_meta_merged.csv'.

Note: The 'LIKELY' and 'MAYBE' layers are not overlapped, 'MAYBE' layers are surrounding the 'LIKELY' layers. So we just need to assign
values to cells and then sum them up to get the merged layer.

'''

# Define the cell values for 'LIKELY' and 'MAYBE'
bio_raw2val = {2: 0.8, 1: 0.3} # 2 is 'LIKELY', 1 is 'MAYBE'; this is a mapping from raw data to the values we want to assign

# Update the metadata
ref_meta.update({'dtype': 'float32', 'nodata': np.nan})

# Read the SNES metadata
snes_meta = pd.read_csv(f'{SNES_TIF_path}/DCCEEW_SNES_meta.csv')

# Function to merge the 'LIKELY' and 'MAYBE' layers
def merge_arr(in_df):
    arr_merge = []
    for _,row in in_df.iterrows():
        # Get the raw value and raw save-path
        raw_val,_ = get_presense_and_save_path(row)
        # Map the raw value to the new value
        arr = rasterio.open(row['TIF_PATH']).read(1).astype('float32')
        arr = np.where(arr == raw_val, bio_raw2val[raw_val], 0)
        arr_merge.append(arr)
    return np.stack(arr_merge).sum(axis=0)



# Merge SNES data
tasks = []
save_paths = pd.DataFrame()
for _,df in snes_meta.groupby(['SCIENTIFIC_NAME']):
    # Get name and new save-path
    first_row = df.iloc[0]
    name = first_row['SCIENTIFIC_NAME'].replace('/', '_').replace(' ', '_')
    # Create a new folder to store the merged data
    save_dir = f'{SNES_TIF_path}/LIKELY_MAYBE_MERGED/SNES/{first_row["TAXON_GROUP"]}'
    save_path = f'{save_dir}/{name}.tif'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    def merge_save(in_df, to_path):
        arr = merge_arr(in_df)
        with rasterio.open(to_path, 'w', **ref_meta) as dst:
            dst.write(arr, 1)
            
    # Save the merged data
    tasks.append(delayed(merge_save)(df, save_path))
    save_paths = pd.concat([save_paths, pd.DataFrame([{'LISTED_TAXON_ID':first_row['LISTED_TAXON_ID'], 'TIF_PATH':save_path}])])

# Parallel processing to merge and save the SNES data   
for _ in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
    pass
    

# Save the metadata to csv
snes_meta_merged = snes_meta.copy()
snes_meta_merged = snes_meta_merged.groupby(['SCIENTIFIC_NAME']).aggregate('first').reset_index()
snes_meta_merged = snes_meta_merged.drop(columns=['PRESENCE_CATEGORY', 'PRESENCE_RANK','SHAPE_Length', 'SHAPE_Area', 'TIF_PATH'])
snes_meta_merged = snes_meta_merged.merge(save_paths, on='LISTED_TAXON_ID')
snes_meta_merged.to_csv(f'{SNES_TIF_path}/DCCEEW_SNES_meta_merged.csv', index=False)





# Merge ECNES data
ecnes_meta = pd.read_csv(f'{SNES_TIF_path}/DCCEEW_ECNES_meta.csv')

tasks = []
save_paths = pd.DataFrame()
for _,df in ecnes_meta.groupby(['COMMUNITY']):
    # Get name and new save-path
    first_row = df.iloc[0]
    name = first_row['COMMUNITY'].replace('/', '_').replace(' ', '_')
    # Create a new folder to store the merged data
    save_dir = f'{SNES_TIF_path}/LIKELY_MAYBE_MERGED/ECNES'
    save_path = f'{save_dir}/{name}.tif'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    def merge_save(in_df, to_path):
        arr = merge_arr(in_df)
        with rasterio.open(to_path, 'w', **ref_meta) as dst:
            dst.write(arr, 1)
            
    # Save the merged data
    tasks.append(delayed(merge_save)(df, save_path))
    save_paths = pd.concat([save_paths, pd.DataFrame([{'COM_ID':first_row['COM_ID'], 'TIF_PATH':save_path}])])


# Parallel processing to merge and save the ECNES data
for _ in tqdm(Parallel(n_jobs=n_workers, return_as='generator')(tasks), total=len(tasks)):
    pass


# Save the metadata to csv
ecnes_meta_merged = ecnes_meta.copy()
ecnes_meta_merged = ecnes_meta_merged.groupby(['COMMUNITY']).aggregate('first').reset_index()
ecnes_meta_merged = ecnes_meta_merged.drop(columns=['PRES_RANK', 'CATEGORY', 'SHAPE_Length', 'SHAPE_Area', 'TIF_PATH'])
ecnes_meta_merged = ecnes_meta_merged.merge(save_paths, on='COM_ID')
ecnes_meta_merged.to_csv(f'{SNES_TIF_path}/DCCEEW_ECNES_meta_merged.csv', index=False)




# ------------------- Masking GEOTIFFs and save SNES to NetCDF ------------------------------------------

# Read DCCEEW SNES GeoTIFF file paths
SNES_meta = pd.read_csv('N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/DCCEEW_SNES_meta.csv')

# Create an empty array to store the data
SNES_arr = xr.DataArray(
    np.zeros((SNES_meta['SCIENTIFIC_NAME'].nunique(), SNES_meta['PRESENCE_RANK'].nunique(), NLUM.sum().item()), dtype=np.int8),
    dims=['species', 'presence', 'cell'],
    coords={
        'species':SNES_meta['SCIENTIFIC_NAME'].unique(), 
        'presence':SNES_meta['PRESENCE_RANK'].unique(), 
        'cell':np.arange(NLUM.sum().item())}
)


# Parallel processing to put the data into the empty array
def get_arr(row):
    ds = rxr.open_rasterio(row['TIF_PATH']).sel(band=1).drop_vars('band')
    ds = xr.where(ds.isin([1, 2]), 1, 0).astype(np.bool_)                       # 1 is 'MAYBE', 2 is 'LIKELY'.
    ds = ds.values[np.nonzero(NLUM.values)]
    return row['SCIENTIFIC_NAME'], row['PRESENCE_RANK'], ds


tasks = (delayed(get_arr)(row) for _,row in SNES_meta.iterrows())
for species,rank,arr in tqdm(Parallel(n_jobs=20, return_as='generator')(tasks), total=len(SNES_meta)):
    SNES_arr.loc[species, rank] = arr
    
# Sum the 'LIKELY' and 'MAYBE' layers to get the full species distribution
SNES_arr_LIKELY_MAYBE_sum = SNES_arr.sum('presence').astype(np.bool_)           # 0 is 'NOT PRESENT', 1 is 'MAYBE AND LIKELY'
SNES_arr.loc[dict(presence=1)] = SNES_arr_LIKELY_MAYBE_sum.values
SNES_arr.coords['presence'] = ['LIKELY', 'LIKELY_AND_MAYBE']


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





# ------------------- Calculate the biodiversity score for SNES  ------------------------------------------
def get_area(arr):
    score_area_weighted_all_Australia = arr * zones['CELL_HA'].values
    score_area_weighted_in_LUTO = arr * idx_in_LUTO_natural * zones['CELL_HA'].values * biodiv_degrade_ly
    score_area_weighted_out_LUTO = arr * idx_out_LUTO_natural * zones['CELL_HA'].values
    return [{
        'ALL_HA':score_area_weighted_all_Australia.sum(), 
        'NATURAL_IN_LUTO_HA':score_area_weighted_in_LUTO.sum(),
        'NATURAL_OUT_LUTO_HA':score_area_weighted_out_LUTO.sum()
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
SNES_df['HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA'] = SNES_df['ALL_HA']
SNES_df['HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL'] = SNES_df['NATURAL_OUT_LUTO_HA']
SNES_df['HABITAT_SIGNIFICANCE_BASELINE_SCORE'] = SNES_df['NATURAL_IN_LUTO_HA'] + SNES_df['NATURAL_OUT_LUTO_HA']
SNES_df['HABITAT_SIGNIFICANCE_BASELINE_PERCENT'] = SNES_df['HABITAT_SIGNIFICANCE_BASELINE_SCORE'] / SNES_df['ALL_HA'] * 100

# Drop unneeded columns, and split the data into three dataframes based on the PRESENCE_RANK
SNES_df = SNES_df.drop(columns=['ALL_HA', 'NATURAL_IN_LUTO_HA', 'NATURAL_OUT_LUTO_HA'])
SNES_df_LIKELY = SNES_df.query('PRESENCE_RANK == "LIKELY"').copy().drop(columns=['PRESENCE_RANK'])
SNES_df_LIKELY_MAYBE = SNES_df.query('PRESENCE_RANK == "LIKELY_AND_MAYBE"').copy().drop(columns=['PRESENCE_RANK'])

# Append suffix to the columns for the LIKELY and MAYBE dataframes
SNES_df_LIKELY.columns = [f'{col}_LIKELY' if col != 'SCIENTIFIC_NAME' else 'SCIENTIFIC_NAME' for col in SNES_df_LIKELY.columns]
SNES_df_LIKELY_MAYBE.columns = [f'{col}_LIKELY_MAYBE' if col != 'SCIENTIFIC_NAME' else 'SCIENTIFIC_NAME' for col in SNES_df_LIKELY_MAYBE.columns]

# Add user defined columns to the LIKELY and MAYBE dataframes
SNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2100_LIKELY', np.nan)
SNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2050_LIKELY', np.nan)
SNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2030_LIKELY', np.nan)

SNES_df_LIKELY_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2100_LIKELY_MAYBE', np.nan)
SNES_df_LIKELY_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2050_LIKELY_MAYBE', np.nan)
SNES_df_LIKELY_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2030_LIKELY_MAYBE', np.nan)

# Merge the LIKELY and MAYBE dataframes, and append the shared attributes
SNES_df = SNES_df_LIKELY.merge(SNES_df_LIKELY_MAYBE, on='SCIENTIFIC_NAME', how='outer')
SNES_df = SNES_df.merge(SNES_meta_att, on='SCIENTIFIC_NAME')


# Reorder the columns
cols = ['SCIENTIFIC_NAME','VERNACULAR_NAME',
        
        'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2030_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2050_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2100_LIKELY',
         
        'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_LIKELY_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2030_LIKELY_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2050_LIKELY_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2100_LIKELY_MAYBE',
        
        'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_LIKELY',
        'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_LIKELY',
        'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_LIKELY_MAYBE',
        'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_LIKELY_MAYBE',

        'LISTED_TAXON_ID','MAP_TAXON_ID', 'THREATENED_STATUS',
        'MIGRATORY_STATUS', 'MARINE', 'CETACEAN', 'EXTRACT_DATE', 'TAXON_GROUP',
        'TAXON_FAMILY', 'TAXON_ORDER', 'TAXON_CLASS', 'TAXON_PHYLUM',
        'TAXON_KINGDOM', 'OTHER_IDS', 'CELL_SIZE', 'REGIONS', 'ATTRIBUTION',
        'SPRAT_PROFILE']

SNES_df = SNES_df[cols]
SNES_df.to_csv(f'{bio_DCCEEW_dir}/bio_DCCEEW_SNES_target1.csv', index=False)



# ------------------- Masking GEOTIFFs and save ECNES to NetCDF ------------------------------------------

# Read DCCEEW ECNES GeoTIFF file paths
ECNES_meta = pd.read_csv('N:/Data-Master/Biodiversity/DCCEEW/SNES_GEOTIFF/DCCEEW_ECNES_meta.csv')

# Create an empty array to store the data
ECNES_arr = xr.DataArray(
    np.zeros((ECNES_meta['COMMUNITY'].nunique(), ECNES_meta['PRES_RANK'].nunique(), NLUM.sum().item()), dtype=np.bool_),
    dims=['species', 'presence', 'cell'],
    coords={'species':ECNES_meta['COMMUNITY'].unique(), 'presence':ECNES_meta['PRES_RANK'].unique(), 'cell':np.arange(NLUM.sum().item())}
)

def get_arr(row):
    arr = rasterio.open(row['TIF_PATH']).read(1).astype('bool')
    arr = arr[np.nonzero(NLUM.values)]
    return row['COMMUNITY'], row['PRES_RANK'], arr

tasks = (delayed(get_arr)(row) for _,row in ECNES_meta.iterrows())
for species,rank,arr in tqdm(Parallel(n_jobs=20, return_as='generator')(tasks), total=len(ECNES_meta)):
    ECNES_arr.loc[species,rank] = arr
    
# Sum the 'LIKELY' and 'MAYBE' layers to get the full species distribution
ECNES_arr_LIKELY_MAYBE_sum = ECNES_arr.sum('presence').astype(np.int8)        # 0 is 'NOT PRESENT', 1 is 'MAYBE AND LIKELY'
ECNES_arr.loc[dict(presence=1)] = ECNES_arr_LIKELY_MAYBE_sum
ECNES_arr.coords['presence'] = ['LIKELY', 'LIKELY_AND_MAYBE']



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



# ------------------- Calculate the biodiversity score for ECNES  ------------------------------------------

def get_area(arr):
    score_area_weighted_all_Australia = arr * zones['CELL_HA'].values
    score_area_weighted_in_LUTO = arr * idx_in_LUTO_natural * zones['CELL_HA'].values * biodiv_degrade_ly
    score_area_weighted_out_LUTO = arr * idx_out_LUTO_natural * zones['CELL_HA'].values
    return [{
        'ALL_HA':score_area_weighted_all_Australia.sum(), 
        'NATURAL_IN_LUTO_HA':score_area_weighted_in_LUTO.sum(),
        'NATURAL_OUT_LUTO_HA':score_area_weighted_out_LUTO.sum()
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
ECNES_df['HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA'] = ECNES_df['ALL_HA']
ECNES_df['HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL'] = ECNES_df['NATURAL_OUT_LUTO_HA']
ECNES_df['HABITAT_SIGNIFICANCE_BASELINE_SCORE'] = ECNES_df['NATURAL_IN_LUTO_HA'] + ECNES_df['NATURAL_OUT_LUTO_HA']
ECNES_df['HABITAT_SIGNIFICANCE_BASELINE_PERCENT'] = ECNES_df['HABITAT_SIGNIFICANCE_BASELINE_SCORE'] / ECNES_df['ALL_HA'] * 100

# Fill the missing COMMUNITY and PRES_RANK with nan
re_index = pd.MultiIndex.from_product([ECNES_df['COMMUNITY'].unique(), ECNES_df['PRES_RANK'].unique()], names=['COMMUNITY', 'PRES_RANK'])
ECNES_df = ECNES_df.set_index(['COMMUNITY', 'PRES_RANK']).reindex(re_index).reset_index()
    
# Drop unneeded columns, and split the data into three dataframes based on the PRES_RANK
ECNES_df = ECNES_df.drop(columns=['ALL_HA', 'NATURAL_IN_LUTO_HA', 'NATURAL_OUT_LUTO_HA'])
ECNES_df_LIKELY = ECNES_df.query('PRES_RANK == "LIKELY"').copy().drop(columns=['PRES_RANK'])
ECNES_df_LIKELY_MAYBE = ECNES_df.query('PRES_RANK == "LIKELY_AND_MAYBE"').copy().drop(columns=['PRES_RANK'])

# Append suffix to the columns for the LIKELY and MAYBE dataframes
ECNES_df_LIKELY.columns = [f'{col}_LIKELY' if col != 'COMMUNITY' else 'COMMUNITY' for col in ECNES_df_LIKELY.columns]
ECNES_df_LIKELY_MAYBE.columns = [f'{col}_LIKELY_MAYBE' if col != 'COMMUNITY' else 'COMMUNITY' for col in ECNES_df_LIKELY_MAYBE.columns]

# Add user defined columns to the LIKELY and MAYBE dataframes
ECNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2100_LIKELY', np.nan)
ECNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2050_LIKELY', np.nan)
ECNES_df_LIKELY.insert(0, 'USER_DEFINED_TARGET_PERCENT_2030_LIKELY', np.nan)

ECNES_df_LIKELY_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2100_LIKELY_MAYBE', np.nan)
ECNES_df_LIKELY_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2050_LIKELY_MAYBE', np.nan)
ECNES_df_LIKELY_MAYBE.insert(0, 'USER_DEFINED_TARGET_PERCENT_2030_LIKELY_MAYBE', np.nan)

# Merge the LIKELY and MAYBE dataframes, and append the shared attributes
ECNES_df = ECNES_df_LIKELY.merge(ECNES_df_LIKELY_MAYBE, on='COMMUNITY', how='outer')
ECNES_df = ECNES_df.merge(ECNES_meta_att, on='COMMUNITY')

# Reorder the columns
cols = ['COMMUNITY',
        
        'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2030_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2050_LIKELY',
        'USER_DEFINED_TARGET_PERCENT_2100_LIKELY',
         
        'HABITAT_SIGNIFICANCE_BASELINE_PERCENT_LIKELY_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2030_LIKELY_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2050_LIKELY_MAYBE',
        'USER_DEFINED_TARGET_PERCENT_2100_LIKELY_MAYBE',
        
        'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_LIKELY',
        'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_LIKELY',
        'HABITAT_SIGNIFICANCE_BASELINE_ALL_AUSTRALIA_LIKELY_MAYBE',
        'HABITAT_SIGNIFICANCE_BASELINE_OUT_LUTO_NATURAL_LIKELY_MAYBE',
        
        'CATEGORY', 'COM_ID','EPBC', 'EXTRACTED', 'CELL_SIZE', 'REGIONS', 'CITATION', 'SPRAT']

ECNES_df = ECNES_df[cols]
ECNES_df.to_csv(f'{bio_DCCEEW_dir}/bio_DCCEEW_ECNES_target.csv', index=False)





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
    
    # Save xarray DataArray to NetCDF
    encoding = {'data': {"compression": "gzip", "compression_opts": 9,  "dtype": 'uint8'}}
    output_layer_name = layer_raster.replace('_ALB', '')
    
    # Option-1: use each separate group layer, which is the percentage [0-100], to represent the cell
    save_path = f'{os.path.dirname(gdb_path)}/{output_layer_name}_HIGH_SPATIAL_DETAIL.nc'
    dst_array_xr.name = 'data'
    dst_array_xr.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')
    
    # Optioin-2: Use the index of the largest group value to represent the cell
    dst_array_xr_argmax = dst_array_xr.argmax(dim='group')     
    
    save_path = f'{os.path.dirname(gdb_path)}/{output_layer_name}_LOW_SPATIAL_DETAIL.nc'
    dst_array_xr_argmax.name = 'data'
    dst_array_xr_argmax.to_netcdf(save_path, encoding=encoding, engine='h5netcdf')
  
  
  


# --------------- Get the sum of areas (ha) for pre-1750 ---------------

# Read NVIS data
NVIS_pre_mvg_xr_low_spatial_detail = xr.load_dataarray(f'{NVIS_PRE_1750_path}/NVIS7_0_AUST_PRE_MVG_LOW_SPATIAL_DETAIL.nc')
NVIS_pre_mvs_xr_low_spatial_detail = xr.load_dataarray(f'{NVIS_PRE_1750_path}/NVIS7_0_AUST_PRE_MVS_LOW_SPATIAL_DETAIL.nc')
NVIS_pre_mvg_xr_high_spatial_detail = xr.load_dataarray(f'{NVIS_PRE_1750_path}/NVIS7_0_AUST_PRE_MVG_HIGH_SPATIAL_DETAIL.nc') / 100  # Convert percentage to fraction
NVIS_pre_mvs_xr_high_spatial_detail = xr.load_dataarray(f'{NVIS_PRE_1750_path}/NVIS7_0_AUST_PRE_MVS_HIGH_SPATIAL_DETAIL.nc') / 100  # Convert percentage to fraction

# Get the NVIS names
NVIS_pre_mvg_names = NVIS_pre_mvg_xr_high_spatial_detail.coords['group'].values.tolist()
NVIS_pre_mvs_names = NVIS_pre_mvs_xr_high_spatial_detail.coords['group'].values.tolist()


# ================================== Total vegataion area (ha) pre-1750 ==================================

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

NVIS_pre_mvg_total_ha_low_spatial_detail_df = pd.DataFrame({'group':NVIS_pre_mvg_names, 'AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA': NVIS_pre_mvg_total_ha_low_spatial_detail})
NVIS_pre_mvs_total_ha_low_spatial_detail_df = pd.DataFrame({'group':NVIS_pre_mvs_names, 'AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA': NVIS_pre_mvs_total_ha_low_spatial_detail})




# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------
NVIS_pre_mvg_total_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail * zones['CELL_HA'].values[None, :]
NVIS_pre_mvs_total_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail * zones['CELL_HA'].values[None, :]
NVIS_pre_mvg_total_ha_high_spatial_detail_df = NVIS_pre_mvg_total_ha_high_spatial_detail.sum(dim='cell').to_dataframe('AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA').reset_index()
NVIS_pre_mvs_total_ha_high_spatial_detail_df = NVIS_pre_mvs_total_ha_high_spatial_detail.sum(dim='cell').to_dataframe('AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA').reset_index()





# ================================== Vegataion area outside the LUTO study area ==================================

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

NVIS_pre_mvg_outside_ha_low_spatial_detail_df = pd.DataFrame({'group':NVIS_pre_mvg_names,'AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA': NVIS_pre_mvg_outside_ha_low_spatial_detail})
NVIS_pre_mvs_outside_ha_low_spatial_detail_df = pd.DataFrame({'group':NVIS_pre_mvs_names,'AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA': NVIS_pre_mvs_outside_ha_low_spatial_detail})


# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------

NVIS_pre_mvg_outside_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail.sel(cell=idx_out_LUTO_natural) * zones['CELL_HA'].values[None, idx_out_LUTO_natural]     
NVIS_pre_mvs_outside_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail.sel(cell=idx_out_LUTO_natural) * zones['CELL_HA'].values[None, idx_out_LUTO_natural]     
NVIS_pre_mvg_outside_ha_high_spatial_detail_df = NVIS_pre_mvg_outside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA').reset_index()
NVIS_pre_mvs_outside_ha_high_spatial_detail_df = NVIS_pre_mvs_outside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA').reset_index()





# ================================== Vegataion area inside the LUTO study area ==================================

# ------------- NVIS_SPATIAL_DETAIL == 'LOW' -------------

NVIS_pre_mvg_inside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvg_xr_low_spatial_detail.sel(cell=idx_in_LUTO_natural).values, 
    weights = zones['CELL_HA'].values[idx_in_LUTO_natural] * biodiv_degrade_ly[idx_in_LUTO_natural],
    minlength = NVIS_pre_mvg_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvs_inside_ha_low_spatial_detail = np.bincount(
    NVIS_pre_mvs_xr_low_spatial_detail.sel(cell=idx_in_LUTO_natural).values,
    weights = zones['CELL_HA'].values[idx_in_LUTO_natural] * biodiv_degrade_ly[idx_in_LUTO_natural],
    minlength = NVIS_pre_mvs_xr_low_spatial_detail.max().values + 1
)

NVIS_pre_mvg_inside_ha_low_spatial_detail_df = pd.DataFrame({'group':NVIS_pre_mvg_names,'AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_NATURAL_HA': NVIS_pre_mvg_inside_ha_low_spatial_detail})
NVIS_pre_mvs_inside_ha_low_spatial_detail_df = pd.DataFrame({'group':NVIS_pre_mvs_names,'AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_NATURAL_HA': NVIS_pre_mvs_inside_ha_low_spatial_detail})


# ------------- NVIS_SPATIAL_DETAIL == 'HIGH' -------------
NVIS_pre_mvg_inside_ha_high_spatial_detail = NVIS_pre_mvg_xr_high_spatial_detail.sel(cell=idx_in_LUTO_natural) * zones['CELL_HA'].values[None, idx_in_LUTO_natural] * biodiv_degrade_ly[idx_in_LUTO_natural]
NVIS_pre_mvs_inside_ha_high_spatial_detail = NVIS_pre_mvs_xr_high_spatial_detail.sel(cell=idx_in_LUTO_natural) * zones['CELL_HA'].values[None, idx_in_LUTO_natural] * biodiv_degrade_ly[idx_in_LUTO_natural]
NVIS_pre_mvg_inside_ha_high_spatial_detail_df = NVIS_pre_mvg_inside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_NATURAL_HA').reset_index()
NVIS_pre_mvs_inside_ha_high_spatial_detail_df = NVIS_pre_mvs_inside_ha_high_spatial_detail.sum(dim='cell').to_dataframe('AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_NATURAL_HA').reset_index()





# ================================== Combine 'HIGH' and 'LOW' ==================================

# Concatenate the two dataframes
NVIS_pre_mvg_low_spatial_detail = NVIS_pre_mvg_total_ha_low_spatial_detail_df.merge(
    NVIS_pre_mvg_outside_ha_low_spatial_detail_df, on='group').merge(
    NVIS_pre_mvg_inside_ha_low_spatial_detail_df, on='group')
    
NVIS_pre_mvg_high_spatial_detail = NVIS_pre_mvg_total_ha_high_spatial_detail_df.merge(
    NVIS_pre_mvg_outside_ha_high_spatial_detail_df, on='group').merge(
    NVIS_pre_mvg_inside_ha_high_spatial_detail_df, on='group')

NVIS_pre_mvs_low_spatial_detail = NVIS_pre_mvs_total_ha_low_spatial_detail_df.merge(
    NVIS_pre_mvs_outside_ha_low_spatial_detail_df, on='group').merge(
    NVIS_pre_mvs_inside_ha_low_spatial_detail_df, on='group')
    
NVIS_pre_mvs_high_spatial_detail = NVIS_pre_mvs_total_ha_high_spatial_detail_df.merge(
    NVIS_pre_mvs_outside_ha_high_spatial_detail_df, on='group').merge(
    NVIS_pre_mvs_inside_ha_high_spatial_detail_df, on='group')


# Calculate the percentage of base-year biodiversity socre to pre-1750 level of the base year
NVIS_pre_mvg_low_spatial_detail.insert(1, 'BASE_YR_PERCENT', NVIS_pre_mvg_low_spatial_detail.eval(
    '(AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_NATURAL_HA + AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA) \
    / AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA * 100'))

NVIS_pre_mvg_high_spatial_detail.insert(1, 'BASE_YR_PERCENT', NVIS_pre_mvg_high_spatial_detail.eval(
    '(AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_NATURAL_HA + AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA) \
    / AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA * 100'))

NVIS_pre_mvs_low_spatial_detail.insert(1, 'BASE_YR_PERCENT', NVIS_pre_mvs_low_spatial_detail.eval(
    '(AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_NATURAL_HA + AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA) \
    / AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA * 100'))

NVIS_pre_mvs_high_spatial_detail.insert(1, 'BASE_YR_PERCENT', NVIS_pre_mvs_high_spatial_detail.eval(
    '(AREA_WEIGHTED_AND_LANDUSE_DEGRADE_SCORE_INSIDE_LUTO_NATURAL_HA + AREA_WEIGHTED_SCORE_OUTSIDE_LUTO_NATURAL_HA) \
    / AREA_WEIGHTED_SCORE_ALL_AUSTRALIA_HA * 100'))


# Append a user-defined target column
NVIS_pre_mvg_low_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2100', 30)
NVIS_pre_mvg_low_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2050', 30)
NVIS_pre_mvg_low_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2030', 30)

NVIS_pre_mvg_high_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2100', 30)
NVIS_pre_mvg_high_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2050', 30)
NVIS_pre_mvg_high_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2030', 30)

NVIS_pre_mvs_low_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2100', 30)
NVIS_pre_mvs_low_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2050', 30)
NVIS_pre_mvs_low_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2030', 30)

NVIS_pre_mvs_high_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2100', 30)
NVIS_pre_mvs_high_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2050', 30)
NVIS_pre_mvs_high_spatial_detail.insert(2, 'USER_DEFINED_TARGET_PERCENT_2030', 30)


# Combine all CSVs and save them to Excel
csv_files = {
    'NVIS_MVG_LOW_SPATIAL_DETAIL': NVIS_pre_mvg_low_spatial_detail,
    'NVIS_MVG_HIGH_SPATIAL_DETAIL': NVIS_pre_mvg_high_spatial_detail,
    'NVIS_MVS_LOW_SPATIAL_DETAIL': NVIS_pre_mvs_low_spatial_detail,
    'NVIS_MVS_HIGH_SPATIAL_DETAIL': NVIS_pre_mvs_high_spatial_detail
}

with pd.ExcelWriter(NVIS_PRE_1750_path + '/BIODIVERSITY_GBF3_SCORES_AND_TARGETS.xlsx') as writer:
    for sheet_name, df in csv_files.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

