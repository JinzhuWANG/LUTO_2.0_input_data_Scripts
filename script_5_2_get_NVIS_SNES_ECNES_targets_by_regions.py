'''
Pre-computes biodiversity conservation scores and targets for NVIS vegetation and SNES/ECNES
species/communities across all region levels and resfactors (1–10). Each output contains a
`resfactor` column so LUTO can look up the appropriate pre-computed values at runtime.

Region levels:
  AUSTRALIA  — national totals
  NRM        — Natural Resource Management regions (NECMA/GBCMA have contractual targets)
  STATE      — Australian states and territories
  IBRA_REG   — IBRA biogeographic regions
  IBRA_SUB   — IBRA biogeographic subregions (NVIS only)

Outputs:
  BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv          — NVIS MVG + MVS, all regions × resfactors
  bio_DCCEEW_SNES_target_ALL_REGIONS.csv                  — SNES LIKELY + MAYBE merged, all regions × resfactors
  bio_DCCEEW_ECNES_target_ALL_REGIONS.csv                 — ECNES LIKELY + MAYBE merged, all regions × resfactors

Score columns (per dataset):
  ALL_HA               — total weighted habitat area in region
  IN_LUTO_HA           — habitat area inside LUTO, weighted by USER_DEFINED degradation
  NATURAL_OUT_LUTO_HA  — natural habitat area outside LUTO study area
  BASEYEAR_LEVEL       — (IN_LUTO_HA + NATURAL_OUT_LUTO_HA) / ALL_HA × 100
  ATTAINABLE_LEVEL     — maximum achievable score if all non-natural outside-LUTO is restored

Degradation weights from HABITAT_CONDITION.csv USER_DEFINED column (normalised to lu=23 = 1.0,
with policy overrides for lu=2, 6, 15 = 0.7). Area scaled by resfactor² so absolute HA values
are correct at all resolutions.
'''

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from joblib import Parallel, delayed
from tqdm.auto import tqdm
from script_5_0_SNES_ECNES_selected import NECMA_SNES, GBCMA_SNES, NECMA_ECNES, GBCMA_ECNES, SNES_AUSTRALIA, ECNES_AUSTRALIA



###############################################################################################
#                                  Global variables                                           #
###############################################################################################

N_JOBS = 32 # number of parallel workers



Unalloc_nat_code = 23
HCAS_condition  = 'N:/Data-Master/Habitat_condition_assessment_system/Data/Processed/HABITAT_CONDITION.csv'
SNES_ECNES_dir  = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES'
NVIS_SAVE_path  = 'N:/Data-Master/NVIS/Processed'
NECMA_NRM_NAMES = ['North East', 'Goulburn Broken']

# Get the land-use map as a template for 2D layers
NLUM       = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8')
NLUM_zero  = NLUM.copy() * 0
RESFACTORS = range(1, 11)  # only pre-compute resfactored averages for 1-10.


# Upstream data
zones = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5',
    key='cell_zones_df',
    columns=['X', 'Y', 'CELL_HA', 'NRM_CODE', 'NRM_NAME', 'STE_NAME11', 'IBRA_REG_NAME_7', 'IBRA_SUB_NAME_7']
)
bioph = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_biophysical_df.h5',
    key='cell_biophysical_df',
    columns=['NATURAL_AREA_INC_WATER', 'DCCEEW_NCI', 'NATURAL_AREA_CONNECTIVITY']
)
lumap = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5',
    key='cell_LU_mapping',
    columns=['LU_DESC', 'LU_ID_LUTO']
)


# Get real area for each cell
real_area_ha    = zones['CELL_HA'].values
real_area_ha_2D = NLUM_zero.copy().astype(np.float32)
np.place(real_area_ha_2D.values, NLUM.values, real_area_ha)

# Get the index of cells that are in natural state, and inside/outside the LUTO study area
natural_cells            = np.logical_not(bioph['NATURAL_AREA_INC_WATER'].values)               # flip: 1 = natural
idx_in_LUTO              = np.logical_not(np.isin(lumap['LU_DESC'], ['Non-agricultural land'])) # shape=6956407, sum=4218733
idx_out_LUTO             = np.isin(lumap['LU_DESC'], ['Non-agricultural land'])                 # shape=6956407, sum=2737674
idx_out_LUTO_natural     = idx_out_LUTO & natural_cells                                         # shape=6956407, sum=2677065
idx_out_LUTO_non_natural = idx_out_LUTO & np.logical_not(natural_cells)                         # shape=6956407, sum=60609


idx_in_LUTO_2D = NLUM_zero.copy()
np.place(idx_in_LUTO_2D.values, NLUM.values, idx_in_LUTO.astype('uint8'))

idx_out_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_out_LUTO_natural_2D.values, NLUM.values, idx_out_LUTO_natural.astype('uint8'))

idx_out_LUTO_non_natural_2D = NLUM_zero.copy()
np.place(idx_out_LUTO_non_natural_2D.values, NLUM.values, idx_out_LUTO_non_natural.astype('uint8'))


# Get land-use degradation data
biodiv_degrade_lookup = pd.read_csv(HCAS_condition).set_index(['lu'])['USER_DEFINED'].to_dict()
biodiv_degrade_lookup[-1] = 0
biodiv_degrade_ly = np.vectorize(biodiv_degrade_lookup.get, otypes=[np.float32])(lumap['LU_ID_LUTO'].values).astype(np.float32)
biodiv_degrade_ly[idx_out_LUTO_natural] = 1.0

biodiv_degrade_ly_2D = NLUM_zero.copy().astype(np.float32)
np.place(biodiv_degrade_ly_2D.values, NLUM.values, biodiv_degrade_ly)


# 2D masks
idx_in_LUTO_2D = NLUM_zero.copy()
np.place(idx_in_LUTO_2D.values, NLUM.values, idx_in_LUTO.astype('uint8'))
idx_out_LUTO_natural_2D = NLUM_zero.copy()
np.place(idx_out_LUTO_natural_2D.values, NLUM.values, idx_out_LUTO_natural.astype('uint8'))


# Region label arrays per cell (1D, length = n_cells)
region_Aus = np.full(len(zones), 'AUSTRALIA', dtype=object)
region_Aus_2D = NLUM_zero.copy().astype(object)
np.place(region_Aus_2D.values, NLUM.values, region_Aus)

region_NRM = zones['NRM_NAME'].values.astype(object)
region_NRM_2D = NLUM_zero.copy().astype(object)
np.place(region_NRM_2D.values, NLUM.values, region_NRM)

region_STATE = zones['STE_NAME11'].values.astype(object)
region_STATE_2D = NLUM_zero.copy().astype(object)
np.place(region_STATE_2D.values, NLUM.values, region_STATE)

region_IBRA_REG = zones['IBRA_REG_NAME_7'].values.astype(object)
region_IBRA_REG_2D = NLUM_zero.copy().astype(object)
np.place(region_IBRA_REG_2D.values, NLUM.values, region_IBRA_REG)

region_IBRA_SUB = zones['IBRA_SUB_NAME_7'].values.astype(object)
region_IBRA_SUB_2D = NLUM_zero.copy().astype(object)
np.place(region_IBRA_SUB_2D.values, NLUM.values, region_IBRA_SUB)

region_array = {
    'AUSTRALIA': region_Aus_2D.values,
    'NRM':       region_NRM_2D.values,
    'STATE':     region_STATE_2D.values,
    'IBRA_REG':  region_IBRA_REG_2D.values,
    'IBRA_SUB':  region_IBRA_SUB_2D.values,
}

# Cell-area weights for each score partition (1D, length = n_cells)
# float64 required: float32 groupby accumulation causes ~0.1% error on large groups
# (e.g. 130k ha error on 137M ha Hummock Grasslands). SNES/ECNES species arrays
# stay float32; multiplying f32 × f64 upcasts automatically.
cell_ha       = zones['CELL_HA'].values.astype(np.float64)
degrade_in_xr = xr.DataArray((biodiv_degrade_ly * idx_in_LUTO).astype(np.float64),   dims=['cell'])
nat_out_xr    = xr.DataArray(idx_out_LUTO_natural.astype(np.float64),                dims=['cell'])
nnat_out_xr   = xr.DataArray(idx_out_LUTO_non_natural.astype(np.float64),            dims=['cell'])



###############################################################################################
#                                   Helper Functions                                          #
###############################################################################################

def get_2D_mask(resfactor: int) -> np.ndarray:
    h, w = NLUM.shape
    # MASK_2D: True at the center pixel of each coarse block containing >=1 valid NLUM cell
    have_cells = xr.DataArray(NLUM.astype(np.float32), dims=NLUM.dims, coords=NLUM.coords).coarsen(x=resfactor, y=resfactor, boundary='pad').sum()
    have_cells_full = np.repeat(np.repeat(have_cells.values, resfactor, axis=0), resfactor, axis=1)[:h, :w]
    mask_2d = np.zeros((h, w), dtype=bool)
    mask_2d[resfactor//2::resfactor, resfactor//2::resfactor] = have_cells_full[resfactor//2::resfactor, resfactor//2::resfactor] > 0
    mask_2d &= NLUM.values.astype(bool)
    return mask_2d


def get_resfactored_average_fraction(arr: np.ndarray, resfactor: int, mask_2d: np.ndarray) -> np.ndarray:
    
    if resfactor == 1:
        return arr

    h, w = NLUM.shape
    arr_2d = np.zeros((h, w), dtype=np.float32)
    np.place(arr_2d, NLUM, arr.astype(np.float32))

    arr_2d_xr = xr.DataArray(arr_2d, dims=NLUM.dims, coords=NLUM.coords)
    arr_block_mean = arr_2d_xr.coarsen(x=resfactor, y=resfactor, boundary='pad').mean()
    arr_2d_fullres = np.repeat(np.repeat(arr_block_mean.values, resfactor, axis=1), resfactor, axis=0)
    arr_2d_fullres = arr_2d_fullres[:h, :w]

    return arr_2d_fullres[mask_2d]

def compute_region_scores(
    arr: xr.DataArray, 
    area: np.ndarray, 
    in_degrade: np.ndarray, 
    out_idx_nat: np.ndarray, 
    out_idx_non_nat: np.ndarray, 
    region_labels: np.ndarray
    ) -> pd.DataFrame:
    """
    Assign region as a non-dimension coordinate on 'cell', then groupby-sum to get
    the four score components per region. Works on any DataArray with a 'cell' dimension.
    """
    arr = arr.assign_coords({'region': ('cell', region_labels)})
    arr_df =  xr.Dataset({
        'ALL_HA':                  (arr * area * 1                ).groupby('region').sum('cell'),
        'IN_LUTO_HA':              (arr * area * in_degrade       ).groupby('region').sum('cell'),
        'NATURAL_OUT_LUTO_HA':     (arr * area * out_idx_nat      ).groupby('region').sum('cell'),
        'NON_NATURAL_OUT_LUTO_HA': (arr * area * out_idx_non_nat  ).groupby('region').sum('cell'),
    }).compute().to_dataframe()
    
    return arr_df.query('ALL_HA > 0').reset_index()  # only keep regions with >0 ha in the input array


def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df['BASEYEAR_SCORE']   = df['IN_LUTO_HA'] + df['NATURAL_OUT_LUTO_HA']
    df['BASEYEAR_LEVEL']   = df['BASEYEAR_SCORE']   / df['ALL_HA'] * 100
    df['ATTAINABLE_LEVEL'] = (1 - df['NON_NATURAL_OUT_LUTO_HA'] / df['ALL_HA']) * 100
    for col in ['TARGET_LEVEL_2030', 'TARGET_LEVEL_2050', 'TARGET_LEVEL_2100']:
        df[col] = np.nan
    return df[df['ALL_HA'] > 0].reset_index(drop=True)



rename_likely = {
    'ATTAINABLE_LEVEL':    'ATTAINABLE_LEVEL_LIKELY',
    'BASEYEAR_LEVEL':      'BASEYEAR_LEVEL_LIKELY',
    'TARGET_LEVEL_2030':   'TARGET_LEVEL_2030_LIKELY',
    'TARGET_LEVEL_2050':   'TARGET_LEVEL_2050_LIKELY',
    'TARGET_LEVEL_2100':   'TARGET_LEVEL_2100_LIKELY',
    'ALL_HA':              'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY',
    'IN_LUTO_HA':          'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY',
    'NATURAL_OUT_LUTO_HA': 'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY',
}

rename_likely_maybe = {
    'ATTAINABLE_LEVEL':    'ATTAINABLE_LEVEL_LIKELY_MAYBE',
    'BASEYEAR_LEVEL':      'BASEYEAR_LEVEL_LIKELY_MAYBE',
    'TARGET_LEVEL_2030':   'TARGET_LEVEL_2030_LIKELY_MAYBE',
    'TARGET_LEVEL_2050':   'TARGET_LEVEL_2050_LIKELY_MAYBE',
    'TARGET_LEVEL_2100':   'TARGET_LEVEL_2100_LIKELY_MAYBE',
    'ALL_HA':              'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY_MAYBE',
    'IN_LUTO_HA':          'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY_MAYBE',
    'NATURAL_OUT_LUTO_HA': 'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY_MAYBE',
}

# Internal columns not needed in final outputs — drop before merging LIKELY + LIKELY_MAYBE
_DROP_INTERNAL = ['NON_NATURAL_OUT_LUTO_HA', 'BASEYEAR_SCORE']







###############################################################################################
#        NVIS — Pre-1750 MVG and MVS weighted area by all region levels  (GBF3)               #
###############################################################################################
NVIS_df = pd.DataFrame()  # placeholder for consolidated output; rows appended in loop below

tasks = []
for sheet_name, nc_path in [
    ('NVIS_MVG', f'{NVIS_SAVE_path}/NVIS7_0_AUST_PRE_MVG.nc'),
    ('NVIS_MVS', f'{NVIS_SAVE_path}/NVIS7_0_AUST_PRE_MVS.nc'),
]:
    xr_pre = xr.load_dataarray(nc_path).astype(np.float32) / 100   # (group, cell), fraction [0-1]
    xr_pre = xr_pre.chunk({'group': 1, 'cell': -1})  # chunk by group for parallel processing; cell dimension must be unchunked for groupby in compute_region_scores
    for species in xr_pre.coords['group'].values:
        for reg, reg_arr in region_array.items():
            for resfactor in RESFACTORS:
                
                def wrapper(species=species, reg=reg, reg_arr=reg_arr, resfactor=resfactor, sheet_name=sheet_name, xr_pre=xr_pre) -> pd.DataFrame:
                    mask_2d         = get_2D_mask(resfactor)
                    arr             = xr.DataArray(get_resfactored_average_fraction(xr_pre.sel(group=species).values, resfactor, mask_2d), dims=['cell'])
                    area            = get_resfactored_average_fraction(cell_ha, resfactor, mask_2d) * resfactor**2
                    in_degrade      = get_resfactored_average_fraction((biodiv_degrade_ly * idx_in_LUTO).astype(np.float32), resfactor, mask_2d)
                    out_idx_nat     = get_resfactored_average_fraction(idx_out_LUTO_natural.astype(np.float32), resfactor, mask_2d)
                    out_idx_non_nat = get_resfactored_average_fraction(idx_out_LUTO_non_natural.astype(np.float32), resfactor, mask_2d)
                    region_labels   = reg_arr[mask_2d]

                    df = compute_region_scores(arr, area, in_degrade, out_idx_nat, out_idx_non_nat, region_labels)
                    return add_derived_cols(df).assign(species=species, region_level=reg, resfactor=resfactor, sheet_name=sheet_name)
                
                tasks.append(delayed(wrapper)())

# Get df and save to disk        
for df in tqdm(Parallel(n_jobs=N_JOBS, return_as='generator_unordered')(tasks), total=len(tasks)):
    NVIS_df = pd.concat([NVIS_df, df], ignore_index=True)

NVIS_df.to_csv(f'{NVIS_SAVE_path}/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS.csv', index=False)             
              


###############################################################################################
#         Load SNES and ECNES weighted arrays  (shared by NRM and IBRA sections)              #
###############################################################################################

SNES_raw = xr.open_dataarray(
    f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES.nc', chunks={'species': 50, 'presence': 1}
)
SNES_likely_arr = SNES_raw.sel(presence='LIKELY').astype(np.float32)
SNES_lm_arr     = SNES_raw.sel(presence='MAYBE').astype(np.float32)

ECNES_raw = xr.open_dataarray(
    f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES.nc', chunks={'species': 50, 'presence': 1}
)
ECNES_likely_arr = ECNES_raw.sel(presence='LIKELY').astype(np.float32)
ECNES_lm_arr     = ECNES_raw.sel(presence='MAYBE').astype(np.float32)

# Metadata (loaded once, reused by NRM and IBRA outputs)
SNES_meta = pd.read_csv(f'{SNES_ECNES_dir}/Processed/DCCEEW_SNES_meta.csv')
SNES_meta_att = (
    SNES_meta.groupby('SCIENTIFIC_NAME', observed=True).first()
    .drop(columns=['PRESENCE_CATEGORY', 'PRESENCE_RANK', 'SHAPE_Length', 'SHAPE_Area', 'TIF_PATH'], errors='ignore')
    .reset_index()
)

ECNES_meta = pd.read_csv(f'{SNES_ECNES_dir}/Processed/DCCEEW_ECNES_meta.csv')
ECNES_meta_att = (
    ECNES_meta.groupby('COMMUNITY', observed=True).first()
    .drop(columns=['PRES_RANK', 'SHAPE_Length', 'SHAPE_Area', 'TIF_PATH'], errors='ignore')
    .reset_index()
)




###############################################################################################
#         SNES — weighted area by presence × region × resfactor  (GBF4)                      #
###############################################################################################

snes_tasks = []
for presence, arr in [('LIKELY', SNES_likely_arr), ('MAYBE', SNES_lm_arr)]:
    for reg, reg_arr in region_array.items():
        for resfactor in RESFACTORS:
            def wrapper(presence=presence, arr=arr, reg=reg, reg_arr=reg_arr, resfactor=resfactor) -> pd.DataFrame:
                mask_2d         = get_2D_mask(resfactor)
                arr_c           = arr.compute()
                species_list    = arr_c.coords['species'].values
                rf_vals         = np.stack([get_resfactored_average_fraction(arr_c.sel(species=sp).values, resfactor, mask_2d) for sp in species_list])
                arr_rf          = xr.DataArray(rf_vals, dims=['species', 'cell'], coords={'species': species_list})
                area            = get_resfactored_average_fraction(cell_ha, resfactor, mask_2d) * resfactor**2
                in_degrade      = get_resfactored_average_fraction((biodiv_degrade_ly * idx_in_LUTO).astype(np.float32), resfactor, mask_2d)
                out_idx_nat     = get_resfactored_average_fraction(idx_out_LUTO_natural.astype(np.float32), resfactor, mask_2d)
                out_idx_non_nat = get_resfactored_average_fraction(idx_out_LUTO_non_natural.astype(np.float32), resfactor, mask_2d)
                region_labels   = reg_arr[mask_2d]
                df = compute_region_scores(arr_rf, area, in_degrade, out_idx_nat, out_idx_non_nat, region_labels)
                return add_derived_cols(df).assign(presence=presence, region_level=reg, resfactor=resfactor)
            snes_tasks.append(delayed(wrapper)())

snes_raw = pd.DataFrame()
for df in tqdm(Parallel(n_jobs=N_JOBS, return_as='generator_unordered')(snes_tasks), total=len(snes_tasks)):
    snes_raw = pd.concat([snes_raw, df], ignore_index=True)




###############################################################################################
#         ECNES — weighted area by presence × region × resfactor  (GBF4)                     #
###############################################################################################

ecnes_tasks = []
for presence, arr in [('LIKELY', ECNES_likely_arr), ('MAYBE', ECNES_lm_arr)]:
    for reg, reg_arr in region_array.items():
        for resfactor in RESFACTORS:
            def wrapper(presence=presence, arr=arr, reg=reg, reg_arr=reg_arr, resfactor=resfactor) -> pd.DataFrame:
                mask_2d         = get_2D_mask(resfactor)
                arr_c           = arr.compute()
                species_list    = arr_c.coords['species'].values
                rf_vals         = np.stack([get_resfactored_average_fraction(arr_c.sel(species=sp).values, resfactor, mask_2d) for sp in species_list])
                arr_rf          = xr.DataArray(rf_vals, dims=['species', 'cell'], coords={'species': species_list})
                area            = get_resfactored_average_fraction(cell_ha, resfactor, mask_2d) * resfactor**2
                in_degrade      = get_resfactored_average_fraction((biodiv_degrade_ly * idx_in_LUTO).astype(np.float32), resfactor, mask_2d)
                out_idx_nat     = get_resfactored_average_fraction(idx_out_LUTO_natural.astype(np.float32), resfactor, mask_2d)
                out_idx_non_nat = get_resfactored_average_fraction(idx_out_LUTO_non_natural.astype(np.float32), resfactor, mask_2d)
                region_labels   = reg_arr[mask_2d]
                df = compute_region_scores(arr_rf, area, in_degrade, out_idx_nat, out_idx_non_nat, region_labels)
                return add_derived_cols(df).assign(presence=presence, region_level=reg, resfactor=resfactor)
            ecnes_tasks.append(delayed(wrapper)())

ecnes_raw = pd.DataFrame()
for df in tqdm(Parallel(n_jobs=N_JOBS, return_as='generator_unordered')(ecnes_tasks), total=len(ecnes_tasks)):
    ecnes_raw = pd.concat([ecnes_raw, df], ignore_index=True)




###############################################################################################
#          Assemble: merge LIKELY + MAYBE, join metadata, apply targets, save                 #
###############################################################################################

'''
NRM targets: from NECMA/GBCMA contract with Deakin — >=50% by 2030, >=70% by 2050/2100
             in the specific NRM region where each species is listed.
Australia targets: same thresholds applied nationally for species in SNES_AUSTRALIA / ECNES_AUSTRALIA.
NECMA only specify LIKELY targets; same values applied to LIKELY_MAYBE to be precautionary.
'''

def assemble_species_df(raw: pd.DataFrame, name_col: str, meta_att: pd.DataFrame) -> pd.DataFrame:
    merge_on = [name_col, 'region', 'region_level', 'resfactor']
    likely = (raw[raw['presence'] == 'LIKELY']
              .drop(columns=['presence'] + _DROP_INTERNAL)
              .rename(columns={'species': name_col, **rename_likely}))
    maybe  = (raw[raw['presence'] == 'MAYBE']
              .drop(columns=['presence'] + _DROP_INTERNAL)
              .rename(columns={'species': name_col, **rename_likely_maybe}))
    return likely.merge(maybe, on=merge_on, how='outer').merge(meta_att, on=name_col, how='left')

snes_all  = assemble_species_df(snes_raw,  'SCIENTIFIC_NAME', SNES_meta_att)
ecnes_all = assemble_species_df(ecnes_raw, 'COMMUNITY',       ECNES_meta_att)

# NRM region targets
for df, key_col, region_lists in [
    (snes_all,  'SCIENTIFIC_NAME', [('North East', NECMA_SNES),  ('Goulburn Broken', GBCMA_SNES)]),
    (ecnes_all, 'COMMUNITY',       [('North East', NECMA_ECNES), ('Goulburn Broken', GBCMA_ECNES)]),
]:
    for region, species_list in region_lists:
        mask = (df['region_level'] == 'NRM') & (df['region'] == region) & (df[key_col].isin(species_list))
        df.loc[mask, 'TARGET_LEVEL_2030_LIKELY']       = 50
        df.loc[mask, 'TARGET_LEVEL_2050_LIKELY']       = 70
        df.loc[mask, 'TARGET_LEVEL_2100_LIKELY']       = 70
        df.loc[mask, 'TARGET_LEVEL_2030_LIKELY_MAYBE'] = 50
        df.loc[mask, 'TARGET_LEVEL_2050_LIKELY_MAYBE'] = 70
        df.loc[mask, 'TARGET_LEVEL_2100_LIKELY_MAYBE'] = 70

# Australia-wide targets
mask = (snes_all['region_level'] == 'AUSTRALIA') & snes_all['SCIENTIFIC_NAME'].isin(SNES_AUSTRALIA)
snes_all.loc[mask, 'TARGET_LEVEL_2030_LIKELY'] = 50
snes_all.loc[mask, 'TARGET_LEVEL_2050_LIKELY'] = 70
snes_all.loc[mask, 'TARGET_LEVEL_2100_LIKELY'] = 70

mask = (ecnes_all['region_level'] == 'AUSTRALIA') & ecnes_all['COMMUNITY'].isin(ECNES_AUSTRALIA)
ecnes_all.loc[mask, 'TARGET_LEVEL_2030_LIKELY'] = 50
ecnes_all.loc[mask, 'TARGET_LEVEL_2050_LIKELY'] = 70
ecnes_all.loc[mask, 'TARGET_LEVEL_2100_LIKELY'] = 70

snes_cols = [
    'SCIENTIFIC_NAME', 'region_level', 'region', 'resfactor',
    'ATTAINABLE_LEVEL_LIKELY', 'BASEYEAR_LEVEL_LIKELY',
    'TARGET_LEVEL_2030_LIKELY', 'TARGET_LEVEL_2050_LIKELY', 'TARGET_LEVEL_2100_LIKELY',
    'ATTAINABLE_LEVEL_LIKELY_MAYBE', 'BASEYEAR_LEVEL_LIKELY_MAYBE',
    'TARGET_LEVEL_2030_LIKELY_MAYBE', 'TARGET_LEVEL_2050_LIKELY_MAYBE', 'TARGET_LEVEL_2100_LIKELY_MAYBE',
    'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY',
    'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY', 'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY',
    'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY_MAYBE',
    'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY_MAYBE', 'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY_MAYBE',
    'LISTED_TAXON_ID', 'MAP_TAXON_ID', 'VERNACULAR_NAME', 'THREATENED_STATUS',
    'MIGRATORY_STATUS', 'MARINE', 'CETACEAN', 'EXTRACT_DATE', 'TAXON_GROUP',
    'TAXON_FAMILY', 'TAXON_ORDER', 'TAXON_CLASS', 'TAXON_PHYLUM',
    'TAXON_KINGDOM', 'OTHER_IDS', 'CELL_SIZE', 'REGIONS', 'ATTRIBUTION', 'SPRAT_PROFILE',
]
ecnes_cols = [
    'COMMUNITY', 'region_level', 'region', 'resfactor',
    'ATTAINABLE_LEVEL_LIKELY', 'BASEYEAR_LEVEL_LIKELY',
    'TARGET_LEVEL_2030_LIKELY', 'TARGET_LEVEL_2050_LIKELY', 'TARGET_LEVEL_2100_LIKELY',
    'ATTAINABLE_LEVEL_LIKELY_MAYBE', 'BASEYEAR_LEVEL_LIKELY_MAYBE',
    'TARGET_LEVEL_2030_LIKELY_MAYBE', 'TARGET_LEVEL_2050_LIKELY_MAYBE', 'TARGET_LEVEL_2100_LIKELY_MAYBE',
    'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY',
    'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY', 'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY',
    'BASELINE_LEVEL_ALL_AUSTRALIA_LIKELY_MAYBE',
    'BASEYEAR_SCORE_OUT_LUTO_NATURAL_LIKELY_MAYBE', 'BASEYEAR_SCORE_INSIDE_LUTO_NATURAL_LIKELY_MAYBE',
    'CATEGORY', 'COM_ID', 'EPBC', 'EXTRACTED', 'CELL_SIZE', 'REGIONS', 'CITATION', 'SPRAT',
]

snes_all[[c for c in snes_cols  if c in snes_all.columns ]].to_csv(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES_target_ALL_REGIONS.csv',  index=False)
ecnes_all[[c for c in ecnes_cols if c in ecnes_all.columns]].to_csv(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES_target_ALL_REGIONS.csv', index=False)
