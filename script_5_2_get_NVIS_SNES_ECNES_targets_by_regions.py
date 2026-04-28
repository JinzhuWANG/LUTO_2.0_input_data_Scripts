'''
Calculates biodiversity conservation targets for NVIS vegetation and SNES/ECNES species/communities,
broken down by two regional frameworks:

  NRM (Natural Resource Management) regions — e.g. NECMA / GBCMA targets set by Deakin/CMAs
  IBRA (Interim Biogeographic Regionalisation for Australia) regions and subregions

For each framework, three data products are produced:
  1. NVIS Pre-1750 MVG and MVS weighted area scores and targets    (GBF3)
  2. SNES species weighted area scores and targets                 (GBF4)
  3. ECNES ecological community weighted area scores and targets   (GBF4)

Column naming follows the national 5_1 convention, with an additional `region` column.
LIKELY and LIKELY_MAYBE presence variants are both included for SNES/ECNES.
'''

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from script_5_0_SNES_ECNES_selected import NECMA_SNES, GBCMA_SNES, NECMA_ECNES, GBCMA_ECNES



###############################################################################################
#                                  Global variables                                           #
###############################################################################################

Unalloc_nat_code = 23
HCAS_condition  = 'N:/Data-Master/Habitat_condition_assessment_system/Data/Processed/HABITAT_CONDITION.csv'
SNES_ECNES_dir  = 'N:/Data-Master/Biodiversity/DCCEEW/SNES_ECNES'
NVIS_SAVE_path  = 'N:/Data-Master/NVIS/Processed'
NECMA_NRM_NAMES = ['North East', 'Goulburn Broken']

# Get the land-use map as a template for 2D layers
NLUM      = rxr.open_rasterio('N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif').squeeze('band').drop_vars('band').astype('uint8')
NLUM_zero = NLUM.copy() * 0


# Upstream data
zones = pd.read_hdf(
    'N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5',
    key='cell_zones_df',
    columns=['X', 'Y', 'CELL_HA', 'NRM_CODE', 'NRM_NAME', 'IBRA_REG_NAME_7', 'IBRA_SUB_NAME_7']
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

# Get land-use degradation data
biodiv_degrade_lookup = pd.read_csv(HCAS_condition).set_index(['lu'])['PERCENTILE_50'].to_dict()
biodiv_degrade_lookup = {k: v * (1 / biodiv_degrade_lookup[Unalloc_nat_code]) for k, v in biodiv_degrade_lookup.items()}
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


# Region label arrays per cell — use np.asarray to avoid pandas StringArray,
# which causes infinite recursion in xarray's PandasExtensionArray.__getattr__
nrm_region_per_cell = np.asarray(zones['NRM_NAME'],         dtype=object)
ibra_reg_per_cell   = np.asarray(zones['IBRA_REG_NAME_7'],  dtype=object)
ibra_sub_per_cell   = np.asarray(zones['IBRA_SUB_NAME_7'],  dtype=object)

# Cell-area weights for each score partition (1D, length = n_cells)
cell_ha       = zones['CELL_HA'].values.astype(np.float32)
degrade_in_xr = xr.DataArray((biodiv_degrade_ly * idx_in_LUTO).astype(np.float32),   dims=['cell'])
nat_out_xr    = xr.DataArray(idx_out_LUTO_natural.astype(np.float32),                dims=['cell'])
nnat_out_xr   = xr.DataArray(idx_out_LUTO_non_natural.astype(np.float32),            dims=['cell'])



def compute_region_scores(arr: xr.DataArray, region_labels: np.ndarray) -> xr.Dataset:
    """
    Assign region as a non-dimension coordinate on 'cell', then groupby-sum to get
    the four score components per region. Works on any DataArray with a 'cell' dimension.
    """
    arr = arr.assign_coords({'region': ('cell', region_labels)})
    return xr.Dataset({
        'ALL_HA':                  (arr * cell_ha * 1            ).groupby('region').sum('cell'),
        'IN_LUTO_HA':              (arr * cell_ha * degrade_in_xr).groupby('region').sum('cell'),
        'NATURAL_OUT_LUTO_HA':     (arr * cell_ha * nat_out_xr   ).groupby('region').sum('cell'),
        'NON_NATURAL_OUT_LUTO_HA': (arr * cell_ha * nnat_out_xr  ).groupby('region').sum('cell'),
    }).compute()


def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
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
#                    NRM — NVIS Pre-1750 MVG and MVS weighted area  (GBF3)                    #
###############################################################################################

nrm_nvis_col_order = [
    'group', 'region',
    'ATTAINABLE_LEVEL',
    'BASEYEAR_LEVEL',
    'TARGET_LEVEL_2030',
    'TARGET_LEVEL_2050',
    'TARGET_LEVEL_2100',
    'ALL_HA',
    'NATURAL_OUT_LUTO_HA',
    'IN_LUTO_HA',
]

nrm_nvis_results = {}
for sheet_name, nc_path in [
    ('NVIS_MVG', f'{NVIS_SAVE_path}/NVIS7_0_AUST_PRE_MVG.nc'),
    ('NVIS_MVS', f'{NVIS_SAVE_path}/NVIS7_0_AUST_PRE_MVS.nc'),
]:
    xr_pre = xr.load_dataarray(nc_path).astype(np.float32) / 100   # (group, cell), fraction [0-1]

    df = compute_region_scores(xr_pre, nrm_region_per_cell).to_dataframe().reset_index()
    df = add_derived_cols(df)

    mask = df['region'].isin(NECMA_NRM_NAMES)
    df.loc[mask, 'TARGET_LEVEL_2030'] = 30
    df.loc[mask, 'TARGET_LEVEL_2050'] = 50
    df.loc[mask, 'TARGET_LEVEL_2100'] = 50

    nrm_nvis_results[sheet_name] = df[nrm_nvis_col_order]

with pd.ExcelWriter(f'{NVIS_SAVE_path}/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS_NRM.xlsx') as writer:
    for sheet, df in nrm_nvis_results.items():
        df.to_excel(writer, sheet_name=sheet, index=False)




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
#             NRM — SNES weighted area by region  (LIKELY and LIKELY_MAYBE)  (GBF4)           #
###############################################################################################

snes_likely_nrm = compute_region_scores(SNES_likely_arr, nrm_region_per_cell).to_dataframe().reset_index()
snes_likely_nrm = add_derived_cols(snes_likely_nrm)
snes_likely_nrm = snes_likely_nrm.drop(columns=_DROP_INTERNAL).rename(columns={'species': 'SCIENTIFIC_NAME', **rename_likely})

snes_lm_nrm = compute_region_scores(SNES_lm_arr, nrm_region_per_cell).to_dataframe().reset_index()
snes_lm_nrm = add_derived_cols(snes_lm_nrm)
snes_lm_nrm = snes_lm_nrm.drop(columns=_DROP_INTERNAL).rename(columns={'species': 'SCIENTIFIC_NAME', **rename_likely_maybe})

snes_df_nrm = snes_likely_nrm.merge(snes_lm_nrm, on=['SCIENTIFIC_NAME', 'region'], how='outer')
snes_df_nrm = snes_df_nrm.merge(SNES_meta_att, on='SCIENTIFIC_NAME', how='left')




###############################################################################################
#             NRM — ECNES weighted area by region  (LIKELY and LIKELY_MAYBE)  (GBF4)          #
###############################################################################################

ecnes_likely_nrm = compute_region_scores(ECNES_likely_arr, nrm_region_per_cell).to_dataframe().reset_index()
ecnes_likely_nrm = add_derived_cols(ecnes_likely_nrm)
ecnes_likely_nrm = ecnes_likely_nrm.drop(columns=_DROP_INTERNAL).rename(columns={'species': 'COMMUNITY', **rename_likely})

ecnes_lm_nrm = compute_region_scores(ECNES_lm_arr, nrm_region_per_cell).to_dataframe().reset_index()
ecnes_lm_nrm = add_derived_cols(ecnes_lm_nrm)
ecnes_lm_nrm = ecnes_lm_nrm.drop(columns=_DROP_INTERNAL).rename(columns={'species': 'COMMUNITY', **rename_likely_maybe})

ecnes_df_nrm = ecnes_likely_nrm.merge(ecnes_lm_nrm, on=['COMMUNITY', 'region'], how='outer')
ecnes_df_nrm = ecnes_df_nrm.merge(ECNES_meta_att, on='COMMUNITY', how='left')




###############################################################################################
#        IBRA — Biodiversity scores by region and subregion  (GBF3, moved from 5_1)           #
###############################################################################################

# A unit DataArray: each cell contributes 1 × its area-weight to the IBRA groupby sum.
# Mathematically identical to the pandas groupby in 5_1 but uses the shared xarray helper.
ones_arr = xr.DataArray(np.ones(len(cell_ha), dtype=np.float32), dims=['cell'])

ibra_col_order = [
    'Region',
    'ATTAINABLE_LEVEL',
    'BASEYEAR_LEVEL',
    'TARGET_LEVEL_2030', 'TARGET_LEVEL_2050', 'TARGET_LEVEL_2100',
    'ALL_HA',
    'NATURAL_OUT_LUTO_HA',
    'IN_LUTO_HA',
]

ibra_reg_df = compute_region_scores(ones_arr, ibra_reg_per_cell).to_dataframe().reset_index()
ibra_reg_df = add_derived_cols(ibra_reg_df)
ibra_reg_df = ibra_reg_df.rename(columns={'region': 'Region'})
ibra_reg_df['TARGET_LEVEL_2030'] = 30
ibra_reg_df['TARGET_LEVEL_2050'] = 50
ibra_reg_df['TARGET_LEVEL_2100'] = 50

ibra_sub_df = compute_region_scores(ones_arr, ibra_sub_per_cell).to_dataframe().reset_index()
ibra_sub_df = add_derived_cols(ibra_sub_df)
ibra_sub_df = ibra_sub_df.rename(columns={'region': 'Region'})
ibra_sub_df['TARGET_LEVEL_2030'] = 30
ibra_sub_df['TARGET_LEVEL_2050'] = 50
ibra_sub_df['TARGET_LEVEL_2100'] = 50

with pd.ExcelWriter(f'{NVIS_SAVE_path}/BIODIVERSITY_GBF3_NVIS_SCORES_AND_TARGETS_IBRA.xlsx') as writer:
    ibra_reg_df[ibra_col_order].to_excel(writer, sheet_name='NVIS_MVG', index=False)
    ibra_sub_df[ibra_col_order].to_excel(writer, sheet_name='NVIS_MVS', index=False)




###############################################################################################
#             IBRA — SNES weighted area by region  (LIKELY and LIKELY_MAYBE)                  #
###############################################################################################

snes_likely_ibra = compute_region_scores(SNES_likely_arr, ibra_reg_per_cell).to_dataframe().reset_index()
snes_likely_ibra = add_derived_cols(snes_likely_ibra)
snes_likely_ibra = snes_likely_ibra.drop(columns=_DROP_INTERNAL).rename(columns={'species': 'SCIENTIFIC_NAME', **rename_likely})

snes_lm_ibra = compute_region_scores(SNES_lm_arr, ibra_reg_per_cell).to_dataframe().reset_index()
snes_lm_ibra = add_derived_cols(snes_lm_ibra)
snes_lm_ibra = snes_lm_ibra.drop(columns=_DROP_INTERNAL).rename(columns={'species': 'SCIENTIFIC_NAME', **rename_likely_maybe})

snes_df_ibra = snes_likely_ibra.merge(snes_lm_ibra, on=['SCIENTIFIC_NAME', 'region'], how='outer')
snes_df_ibra = snes_df_ibra.merge(SNES_meta_att, on='SCIENTIFIC_NAME', how='left')
# Targets left as NaN — user fills per IBRA region as needed




###############################################################################################
#             IBRA — ECNES weighted area by region  (LIKELY and LIKELY_MAYBE)                 #
###############################################################################################

ecnes_likely_ibra = compute_region_scores(ECNES_likely_arr, ibra_reg_per_cell).to_dataframe().reset_index()
ecnes_likely_ibra = add_derived_cols(ecnes_likely_ibra)
ecnes_likely_ibra = ecnes_likely_ibra.drop(columns=_DROP_INTERNAL).rename(columns={'species': 'COMMUNITY', **rename_likely})

ecnes_lm_ibra = compute_region_scores(ECNES_lm_arr, ibra_reg_per_cell).to_dataframe().reset_index()
ecnes_lm_ibra = add_derived_cols(ecnes_lm_ibra)
ecnes_lm_ibra = ecnes_lm_ibra.drop(columns=_DROP_INTERNAL).rename(columns={'species': 'COMMUNITY', **rename_likely_maybe})

ecnes_df_ibra = ecnes_likely_ibra.merge(ecnes_lm_ibra, on=['COMMUNITY', 'region'], how='outer')
ecnes_df_ibra = ecnes_df_ibra.merge(ECNES_meta_att, on='COMMUNITY', how='left')
# Targets left as NaN — user fills per IBRA region as needed




###############################################################################################
#                     NRM — Set SNES/ECNES species targets  (GBF4)                            #
###############################################################################################

'''
These species come from the contract of NECMA with Deakin. They specified that for these species, 
the targets should be >=50% by 2030, and >=70% by 2050/2100, in the NRM region(s) where they occur. 

NECMA only specify targets for LIKELY, but we apply the same targets to LIKELY_AND_MAYBE to be precautionary. 
'''

for df, key_col, region_lists in [
    (snes_df_nrm,  'SCIENTIFIC_NAME', [('North East', NECMA_SNES),  ('Goulburn Broken', GBCMA_SNES)]),
    (ecnes_df_nrm, 'COMMUNITY',       [('North East', NECMA_ECNES), ('Goulburn Broken', GBCMA_ECNES)]),
]:
    for region, species_list in region_lists:
        mask = (df['region'] == region) & (df[key_col].isin(species_list))
        df.loc[mask, 'TARGET_LEVEL_2030_LIKELY'] = 50
        df.loc[mask, 'TARGET_LEVEL_2050_LIKELY'] = 70
        df.loc[mask, 'TARGET_LEVEL_2100_LIKELY'] = 70




###############################################################################################
#                         Save outputs                                                        #
###############################################################################################

snes_cols = [
    'SCIENTIFIC_NAME', 'region',
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
    'COMMUNITY', 'region',
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

snes_df_nrm[snes_cols].to_csv(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES_target_NRM.csv',  index=False)
ecnes_df_nrm[ecnes_cols].to_csv(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES_target_NRM.csv', index=False)

snes_df_ibra[snes_cols].to_csv(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_SNES_target_IBRA.csv',  index=False)
ecnes_df_ibra[ecnes_cols].to_csv(f'{SNES_ECNES_dir}/Processed/bio_DCCEEW_ECNES_target_IBRA.csv', index=False)
