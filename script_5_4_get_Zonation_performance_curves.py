
import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd



###############################################################################################
#                                  Global variables                                           #
###############################################################################################

# Paths
bio_Carla_EnviroSuit_dir = 'N:/Data-Master/Biodiversity/Environmental-suitability'
SNES_ECNES_dir = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES'


# Upstream data
zones = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5',
    key='cell_zones_df',
    columns=['X', 'Y', 'CELL_HA']
)
lumap = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5',
    key = 'cell_LU_mapping',
    columns=['LU_DESC','LU_ID_LUTO']
)


# Get real area for each cell
real_area_ha = zones['CELL_HA'].values

# Get the index of cells that are inside the LUTO study area
idx_in_LUTO = np.logical_not(np.isin(lumap['LU_DESC'], ['Non-agricultural land']))  # shape=6956407, sum=4218733




###############################################################################################
#                  Process Speciese Conservation Priority data (GBF2) with Xarray             #
###############################################################################################

'''
Rank-to-area performance curves for every zonation rank layer available.

This script only reads rank rasters that are already on disk:
  - SSP layers   {bio_Carla_EnviroSuit_dir}/Zonation/{ssp}/{ssp}_zonation_rank_1km.tif  (script 4)
  - NES layers   {SNES_ECNES_dir}/Processed/Zonation/{nes}_Priority/rankmap.tif         (script 5_3)

Run it after both producers. The whole workbook is rewritten in one pass, so all ten
sources must be computed together.
'''


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
