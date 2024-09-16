import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil

# Set some options
pd.set_option('display.width', 6000)
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.float_format', '{:,.4f}'.format)

# Load 2010 land-use map (crops and livestock)
ludf_raw = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_LU_mapping.h5')[['CELL_ID', 'SA2_ID', 'SPREAD_ID', 'SPREAD_DESC', 'IRRIGATION', 'LU_ID', 'LU_DESC']]
ludf = ludf_raw.copy()

# Rename columns
ludf = ludf.rename(columns = {'SPREAD_ID': 'SPREAD_ID_OLD'}) 

# Calculate some new columns to match CSIRO naming convention
ludf['SPREAD_Commodity'] = ludf['LU_DESC'].astype(str)
ludf['SPREAD_ID'] = ludf['SPREAD_ID_OLD'].astype(int)

# Calculate values to match CSIRO naming convention
ludf.loc[ludf['SPREAD_Commodity'].str.contains('Unallocated'), 'SPREAD_Commodity'] = 'Unalllocated land'

idx = ludf['SPREAD_Commodity'].str.contains('Dairy')
ludf.loc[idx, 'SPREAD_Commodity'] = 'Dairy Cattle'
ludf.loc[idx, 'SPREAD_ID'] = 31

idx = ludf['SPREAD_Commodity'].str.contains('Beef')
ludf.loc[idx, 'SPREAD_Commodity'] = 'Beef Cattle'
ludf.loc[idx, 'SPREAD_ID'] = 32

idx = ludf['SPREAD_Commodity'].str.contains('Sheep')
ludf.loc[idx, 'SPREAD_Commodity'] = 'Sheep'
ludf.loc[idx, 'SPREAD_ID'] = 33

ludf['SPREAD_Commodity'] = ludf['SPREAD_Commodity'].str.title()

# Summarise crops and livestock by SA2, SPREAD_Commodity, and land management
SA2_lu_df = ludf.query('SPREAD_ID >= 5').groupby(['SA2_ID', 'SPREAD_ID', 'IRRIGATION'], observed = True, as_index = False).agg(
                    SPREAD_Commodity = ('SPREAD_Commodity', 'first')
                    ).sort_values(by = ['SA2_ID', 'SPREAD_ID', 'IRRIGATION'])


# Read in the NLUM SA2 template to check for NaNs (this table has every combination of LU, irr/dry, and SA2)
def_df = pd.read_hdf('N:/Data-Master/Profit_map/NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')

# Calculate some new columns to match CSIRO naming convention
def_df['SPREAD_Commodity'] = def_df['LU_DESC'].astype(str)
def_df['SPREAD_ID'] = def_df['LU_ID'].astype(int)

idx = def_df['LU_DESC'].str.contains('Dairy')
def_df.loc[idx, 'SPREAD_Commodity'] = 'Dairy Cattle'
def_df.loc[idx, 'SPREAD_ID'] = 31

idx = def_df['SPREAD_Commodity'].str.contains('Beef')
def_df.loc[idx, 'SPREAD_Commodity'] = 'Beef Cattle'
def_df.loc[idx, 'SPREAD_ID'] = 32

idx = def_df['SPREAD_Commodity'].str.contains('Sheep')
def_df.loc[idx, 'SPREAD_Commodity'] = 'Sheep'
def_df.loc[idx, 'SPREAD_ID'] = 33

def_df = def_df.drop(columns = ['LU_ID', 'LU_DESC'])
def_df.drop_duplicates(inplace = True, ignore_index = True)
def_df['SPREAD_Commodity'] = def_df['SPREAD_Commodity'].str.title()

# Ultimate land-use mapping truth according top NLUM and CSIRO livestock mapping
def_df.to_csv('N:/Data-Master/Profit_map/NLUM_SPREAD_CSIRO_Livestock_Mapped_Concordance.csv')



# Merge SA4 and STATE ID columns for gap filling
def_crops = def_df.query('5 <= LU_ID <= 25')
SA2_lu_crops = SA2_lu_df.query('5 <= SPREAD_ID <= 25')
crop_check = def_crops.merge(SA2_lu_crops, how = 'outer', left_on = ['SA2_ID', 'LU_ID', 'IRRIGATION'], right_on = ['SA2_ID', 'SPREAD_ID', 'IRRIGATION'])

print('Number of NaNs =', crop_check[crop_check.isna().any(axis=1)].shape[0])






############################################################################################################################################
# Assemble CROP GHG EMISSIONS
############################################################################################################################################


################# Calculate emissions

# Read agricultural emissions data
ghg_dataframe = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20231124/T_GHG_by_SPREAD_SA2_2010_NLUM_Navarroetal_fix_pears_nuts_othernoncereal.csv')

# Read in the NLUM SA2 template to check for NaNs (this table has every combination of LU, irr/dry, and SA2)
def_df = pd.read_csv('N:/Data-Master/Profit_map/NLUM_SPREAD_CSIRO_Livestock_Mapped_Concordance.csv')

# Join GHG table to concordance template and check for nodata. 
ghg_nans = def_df.merge(ghg_dataframe, how = 'left', left_on = ['SA2_ID', 'SPREAD_ID', 'IRRIGATION'], right_on = ['SA2_ID', 'SPREAD_ID', 'irrigation'])
print('Number of NaNs =', ghg_nans[ghg_nans.isna().any(axis=1)].shape[0]) # Number of NaNs = 402

ghg_nans.to_csv('N:/Data-Master/Profit_map/ghg_nans.csv')





# Select and order columns
ghg_df = ghg_dataframe[['SA2_ID', 'SPREAD_ID', 'irrigation', 
                        'kgco2_chem_app',
                        'kgco2_crop_management',
                        'kgco2_cult',
                        'kgco2_fert',
                        'kgco2_harvest',
                        'kgco2_irrig',
                        'kgco2_pest',
                        'kgco2_soil',
                        'kgco2_sowing',
                        'kgco2_transport']]

# Rename columns
ghg_df = ghg_df.rename(columns = {'SPREAD_ID': 'LU_ID', 
                                  'SPREAD_Commodity': 'LU_DESC',
                                  'irrigation': 'IRRIGATION', 
                                  'kgco2_fert': 'CO2E_KG_HA_FERT_PROD',
                                  'kgco2_pest': 'CO2E_KG_HA_PEST_PROD',
                                  'kgco2_irrig': 'CO2E_KG_HA_IRRIG',
                                  'kgco2_chem_app': 'CO2E_KG_HA_CHEM_APPL', 
                                  'kgco2_crop_management': 'CO2E_KG_HA_CROP_MGT', 
                                  'kgco2_cult': 'CO2E_KG_HA_CULTIV', 
                                  'kgco2_harvest': 'CO2E_KG_HA_HARVEST', 
                                  'kgco2_sowing': 'CO2E_KG_HA_SOWING', 
                                  'kgco2_soil': 'CO2E_KG_HA_SOIL', 
                                  'kgco2_transport': 'CO2E_KG_HA_TRANSPORT'})

# Read in the NLUM SA2 template to check for NaNs (this table has every combination of LU, irr/dry, and SA2)
def_df = pd.read_hdf('N:/Data-Master/Profit_map/NLUM_SPREAD_LU_ID_Mapped_Concordance.h5')

# Read cell_df file from disk to a new data frame for ag data with just the relevant columns
cell_df = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5')
cell_df = cell_df[['CELL_ID', 'CELL_HA', 'SA2_ID', 'SA4_CODE11', 'STE_CODE11', 'STE_NAME11']]
cell_df.rename(columns = {'STE_CODE11': 'STE_ID', 'SA4_CODE11': 'SA4_ID'}, inplace = True)
lut = cell_df.groupby(['SA2_ID'], observed = True, as_index = False).agg(
                    SA4_ID = ('SA4_ID', 'first'),
                    STATE_ID = ('STE_ID', 'first')
                    ).sort_values(by = 'SA2_ID')

# Merge SA4 and STATE ID columns for gap filling
def_df = def_df.query('5 <= LU_ID <= 25').merge(lut, how = 'left', on = 'SA2_ID')

# Join GHG table to concordance template and check for nodata. 
c_ghg = def_df.merge(ghg_df, how = 'left', on = ['SA2_ID', 'LU_ID', 'IRRIGATION'])
print('Number of NaNs =', c_ghg[c_ghg.isna().any(axis=1)].shape[0]) # Number of NaNs = 402

# Summarise mean of commodity and irrigation status by SA4 to fill in data gaps and count NaNs
SA4 = c_ghg.groupby(['SA4_ID', 'LU_ID', 'IRRIGATION'], observed = True, as_index = False).agg(
                    CO2E_KG_HA_CHEM_APPL_SA4 = ('CO2E_KG_HA_CHEM_APPL', 'mean'), 
                    CO2E_KG_HA_CROP_MGT_SA4 = ('CO2E_KG_HA_CROP_MGT', 'mean'), 
                    CO2E_KG_HA_CULTIV_SA4 = ('CO2E_KG_HA_CULTIV', 'mean'), 
                    CO2E_KG_HA_FERT_PROD_SA4 = ('CO2E_KG_HA_FERT_PROD', 'mean'), 
                    CO2E_KG_HA_HARVEST_SA4 = ('CO2E_KG_HA_HARVEST', 'mean'), 
                    CO2E_KG_HA_IRRIG_SA4 = ('CO2E_KG_HA_IRRIG', 'mean'), 
                    CO2E_KG_HA_PEST_PROD_SA4 = ('CO2E_KG_HA_PEST_PROD', 'mean'), 
                    CO2E_KG_HA_SOIL_SA4 = ('CO2E_KG_HA_SOIL', 'mean'),
                    CO2E_KG_HA_SOWING_SA4 = ('CO2E_KG_HA_SOWING', 'mean'), 
                    CO2E_KG_HA_TRANSPORT_SA4 = ('CO2E_KG_HA_TRANSPORT', 'mean')
                    ).sort_values(by = ['SA4_ID', 'LU_ID', 'IRRIGATION'])
print('Number of NaNs =', SA4[SA4.isna().any(axis=1)].shape[0]) # Number of NaNs = 80

# Summarise mean of commodity and irrigation status by STATE to fill in data gaps and count NaNs
STATE = c_ghg.groupby(['STATE_ID', 'LU_ID', 'IRRIGATION'], observed = True, as_index = False).agg(
                    CO2E_KG_HA_CHEM_APPL_STATE = ('CO2E_KG_HA_CHEM_APPL', 'mean'), 
                    CO2E_KG_HA_CROP_MGT_STATE = ('CO2E_KG_HA_CROP_MGT', 'mean'), 
                    CO2E_KG_HA_CULTIV_STATE = ('CO2E_KG_HA_CULTIV', 'mean'), 
                    CO2E_KG_HA_FERT_PROD_STATE = ('CO2E_KG_HA_FERT_PROD', 'mean'), 
                    CO2E_KG_HA_HARVEST_STATE = ('CO2E_KG_HA_HARVEST', 'mean'), 
                    CO2E_KG_HA_IRRIG_STATE = ('CO2E_KG_HA_IRRIG', 'mean'), 
                    CO2E_KG_HA_PEST_PROD_STATE = ('CO2E_KG_HA_PEST_PROD', 'mean'), 
                    CO2E_KG_HA_SOIL_STATE = ('CO2E_KG_HA_SOIL', 'mean'),
                    CO2E_KG_HA_SOWING_STATE = ('CO2E_KG_HA_SOWING', 'mean'), 
                    CO2E_KG_HA_TRANSPORT_STATE = ('CO2E_KG_HA_TRANSPORT', 'mean')
                    ).sort_values(by = ['STATE_ID', 'LU_ID', 'IRRIGATION'])
print('Number of NaNs =', STATE[STATE.isna().any(axis=1)].shape[0]) # Number of NaNs = 10

# Summarise mean of commodity and irrigation status by NATIONAL to fill in data gaps and count NaNs
AUS = c_ghg.groupby(['LU_ID', 'IRRIGATION'], observed = True, as_index = False).agg(
                    CO2E_KG_HA_CHEM_APPL_AUS = ('CO2E_KG_HA_CHEM_APPL', 'mean'), 
                    CO2E_KG_HA_CROP_MGT_AUS = ('CO2E_KG_HA_CROP_MGT', 'mean'), 
                    CO2E_KG_HA_CULTIV_AUS = ('CO2E_KG_HA_CULTIV', 'mean'), 
                    CO2E_KG_HA_FERT_PROD_AUS = ('CO2E_KG_HA_FERT_PROD', 'mean'), 
                    CO2E_KG_HA_HARVEST_AUS = ('CO2E_KG_HA_HARVEST', 'mean'), 
                    CO2E_KG_HA_IRRIG_AUS = ('CO2E_KG_HA_IRRIG', 'mean'), 
                    CO2E_KG_HA_PEST_PROD_AUS = ('CO2E_KG_HA_PEST_PROD', 'mean'), 
                    CO2E_KG_HA_SOIL_AUS = ('CO2E_KG_HA_SOIL', 'mean'),
                    CO2E_KG_HA_SOWING_AUS = ('CO2E_KG_HA_SOWING', 'mean'), 
                    CO2E_KG_HA_TRANSPORT_AUS = ('CO2E_KG_HA_TRANSPORT', 'mean')
                    ).sort_values(by = ['LU_ID', 'IRRIGATION'])
print('Number of NaNs =', AUS[AUS.isna().any(axis=1)].shape[0]) # Number of NaNs = 0

# Join GHG summary tables to concordance template and check for nodata. 
c_ghg = c_ghg.merge(SA4, how = 'left', on = ['SA4_ID', 'LU_ID', 'IRRIGATION'])
c_ghg = c_ghg.merge(STATE, how = 'left', on = ['STATE_ID', 'LU_ID', 'IRRIGATION'])
c_ghg = c_ghg.merge(AUS, how = 'left', on = ['LU_ID', 'IRRIGATION'])

# Set up lists of GHG sources and columns suffixes
cols = ['CO2E_KG_HA_CHEM_APPL', 'CO2E_KG_HA_CROP_MGT', 'CO2E_KG_HA_CULTIV', 'CO2E_KG_HA_FERT_PROD', 'CO2E_KG_HA_HARVEST', 'CO2E_KG_HA_IRRIG', 'CO2E_KG_HA_PEST_PROD', 'CO2E_KG_HA_SOIL', 'CO2E_KG_HA_SOWING', 'CO2E_KG_HA_TRANSPORT']
suffixes = ['_SA4', '_STATE', '_AUS']

# Fill in data gaps, if SA2 is NaN use SA4 average, if SA4 value is NaN use State average, if State is NaN use national average.
for col in cols:
    for suf in suffixes: 
        tf = c_ghg[col] != c_ghg[col]
        c_ghg.loc[tf, col] = c_ghg.loc[tf, col + suf]

# Drop unwanted columns
c_ghg = c_ghg.drop(columns = [col for col in c_ghg.columns for suf in suffixes if suf in col])

# Check that all gaps are filled
print('Number of NaNs =', c_ghg[c_ghg.isna().any(axis=1)].shape[0]) # Number of NaNs should = 0

# Check original output data format
# c_ghg_orig = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_crop_GHG_data_OLD.h5')

# Convert LU_DESC to Sentence case
c_ghg['LU_DESC'] = c_ghg['LU_DESC'].str.capitalize()

# Downcast
downcast(c_ghg)

# Drop STATE_ID
c_ghg = c_ghg.drop(columns = 'SA4_ID')

# Save output to file
c_ghg.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_crop_GHG_data.h5', key = 'SA2_crop_GHG_data', mode = 'w', format = 't')



##################### Check total crop GHG emissions

c_ghg = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_crop_GHG_data.h5')

# Read in the PROFIT MAP table as provided by CSIRO
ag_df = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20210817/pfe_table_13082021.csv', low_memory = False).drop(columns = 'rev_notes')
ag_df = ag_df.iloc[:, 2:19].drop(columns = ['SPREAD_ID_original'])

# Select crops only
crops_df = ag_df.query('5 <= SPREAD_ID <= 25').copy()

# Rename columns
crops_df = crops_df.rename(columns = {'SPREAD_ID': 'LU_ID', 
                                      'SPREAD_Commodity': 'LU_DESC',
                                      'irrigation': 'IRRIGATION'
                                     })

# Aggregate commodity-level to SPREAD-level based on area of Commodity within SA2
crops_sum_df = crops_df.groupby(['SA2_ID', 'LU_ID', 'IRRIGATION'], as_index = False).agg(
                    SPREAD_Commodity = ('LU_DESC', 'first'),
                    area = ('area', 'sum'),
                    prod = ('prod', 'sum'),
                    )

# Join the emissions table to the profit map table and drop uneccesary columns
crops_ghg_df = c_ghg.merge(crops_sum_df, how = 'left', on = ['SA2_ID', 'LU_ID', 'IRRIGATION'])

# Check for NaNs - note that there are 13 rows with NaNs in kgCO2e_pest
print('Number of NaNs =', crops_ghg_df[crops_ghg_df.isna().any(axis = 1)].shape[0])

crops_ghg_df['TCO2E_SUM'] = crops_ghg_df.eval('(CO2E_KG_HA_CHEM_APPL + CO2E_KG_HA_CROP_MGT + CO2E_KG_HA_CULTIV + CO2E_KG_HA_FERT_PROD + CO2E_KG_HA_HARVEST + CO2E_KG_HA_IRRIG + CO2E_KG_HA_PEST_PROD + CO2E_KG_HA_SOIL + CO2E_KG_HA_SOWING + CO2E_KG_HA_TRANSPORT) * area / 1000')

# Sum GHG sources by crop
crops_sum_df = crops_ghg_df.groupby('LU_ID', as_index = False).agg(
                    SPREAD_Commodity = ('LU_DESC', 'first'),
                    TCO2E_SUM = ('TCO2E_SUM', 'sum')
                    )

# Sum crop GHG emissions - Crop GHG emissions =  21,317,454 tCO2e
print('Crop GHG emissions = ', crops_sum_df['TCO2E_SUM'].sum(), 'tCO2e')





############################################################################################################################################
# Assemble LIVESTOCK GHG EMISSIONS
############################################################################################################################################


##################  FIRST DATASET SUPPLIED BY CSIRO

# Read in the livestock emissions table and drop unwanted columns
ghg_ls = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20231113/T_GHG_by_SPREAD_SA2_livestock_2010_NLUM_per_head.csv')
ghg_ls.drop(columns = ['track', 'irrigation'], inplace = True)
ghg_ls.drop_duplicates(inplace = True, ignore_index = True)



lmap = pd.read_hdf('N:/Data-Master/Profit_map/From_CSIRO/20210910/lmap.h5')
lmap = lmap[['SA2_ID', 'SPREAD_id_mapped', 'SPREAD_mapped']].copy()
tmp = lmap.query('SPREAD_id_mapped >= 31').merge(ghg_ls, how = 'left', left_on = ['SA2_ID', 'SPREAD_mapped'], right_on = ['SA2_ID', 'SPREAD_Commodity'])
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0]) # Zero NaNs



ghg_ls.loc[ghg_ls.query('SPREAD_Commodity == "Dairy Cattle"').index, 'SPREAD_Commodity'] = 'DAIRY'
ghg_ls.loc[ghg_ls.query('SPREAD_Commodity == "Beef Cattle"').index, 'SPREAD_Commodity'] = 'BEEF'
ghg_ls.loc[ghg_ls.query('SPREAD_Commodity == "Sheep"').index, 'SPREAD_Commodity'] = 'SHEEP'

# Re-order and rename columns
ls_ghg = ghg_ls[['SA2_ID', 'SPREAD_Commodity', 'GHG_enteric_perHead', 'GHG_manure management_perHead', 'GHG_indirect leaching and runoff_perHead', 'GHG_dung and urine_perHead', 'GHG_Seed emissions_perHead', 'GHG_Fodder emissions_perHead', 'GHG_Fuel_perHead', 'GHG_Electricity_perHead']]
ls_ghg = ls_ghg.rename(columns = {'irrigation': 'IRRIGATION', 
                                  'GHG_enteric_perHead': 'CO2E_KG_HEAD_ENTERIC',
                                  'GHG_manure management_perHead': 'CO2E_KG_HEAD_MANURE_MGT',
                                  'GHG_indirect leaching and runoff_perHead': 'CO2E_KG_HEAD_IND_LEACH_RUNOFF',
                                  'GHG_dung and urine_perHead': 'CO2E_KG_HEAD_DUNG_URINE', 
                                  'GHG_Seed emissions_perHead': 'CO2E_KG_HEAD_SEED', 
                                  'GHG_Fodder emissions_perHead': 'CO2E_KG_HEAD_FODDER', 
                                  'GHG_Fuel_perHead': 'CO2E_KG_HEAD_FUEL', 
                                  'GHG_Electricity_perHead': 'CO2E_KG_HEAD_ELEC'
                                  })


# Calculate pivot table
lvstk_ghg_sources = ['CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_MANURE_MGT', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_SEED', 'CO2E_KG_HEAD_FODDER', 'CO2E_KG_HEAD_FUEL', 'CO2E_KG_HEAD_ELEC'].sort()
p_ls = pd.pivot_table(ls_ghg, 
                      values = lvstk_ghg_sources, 
                      index = 'SA2_ID', 
                      columns = 'SPREAD_Commodity', 
                      aggfunc = 'first'
                     ).sort_values(by = 'SA2_ID')

# Rearrange pivot table
p_ls.columns = p_ls.columns.rename('GHG Source', level = 0)
p_ls = p_ls.reorder_levels(['SPREAD_Commodity', 'GHG Source'], axis = 1)

# Flatten multiindex dataframe
p_ls.columns = p_ls.columns.to_flat_index()


# Merge livestock GHG to ludf dataframe by SA2 to check nodata
tmp = ludf.query('LU_ID >= 31').merge(p_ls, how = 'left', on = 'SA2_ID')
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])

# Recreate multiindex dataframe
p_ls.columns = pd.MultiIndex.from_tuples(p_ls.columns, names=['Livestock type','GHG Source'])

# Sort columns
p_ls.sort_index(axis = 1, level = 0, inplace = True)

# Downcast and save file
downcast(p_ls)
p_ls.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_livestock_GHG_data.h5', key = 'SA2_livestock_GHG_data', mode = 'w', format = 't')





##################  SECOND DATASET SUPPLIED BY CSIRO

# Load original data for comparison
# irr_pasture_ghg_orig = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_irrigated_pasture_GHG_data.h5')
# p_ls_orig = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_livestock_GHG_data.h5')

# Read agricultural emissions data
ghg_dataframe = pd.read_csv('N:/Data-Master/Profit_map/From_CSIRO/20231124/T_GHG_by_SPREAD_SA2_2010_NLUM_Navarroetal_fix_pears_nuts_othernoncereal.csv')

# Read in the livestock emissions table and select only the livestock rows and necessary columns
l_ghg = ghg_dataframe.query('SPREAD_ID >= 31')
l_ghg = l_ghg[['SA2_ID', 'SPREAD_Commodity', 
               'kgco2_livestock_n2o_dung_urine',
               'kgco2_livestock_electricity',
               'kgco2_livestock_enteric',
               'kgco2_livestock_fodder',
               'kgco2_livestock_fuel',
               'kgco2_livestock_n2o_leaching_runoff',
               'kgco2_livestock_manure_management',
               'kgco2_livestock_pasture_seeds'
              ]]

# Drop duplicate rows as irrigation and dryland are the same for 'biogenic' emissions
l_ghg = l_ghg.drop_duplicates(ignore_index = True)

lmap = pd.read_hdf('N:/Data-Master/Profit_map/From_CSIRO/20210910/lmap.h5')
lmap = lmap[['SA2_ID', 'SPREAD_id_mapped', 'SPREAD_mapped']].copy()
tmp = lmap.query('SPREAD_id_mapped >= 31').merge(l_ghg, how = 'left', left_on = ['SA2_ID', 'SPREAD_mapped'], right_on = ['SA2_ID', 'SPREAD_Commodity'])
print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0]) # THERE ARE MANY NaNs

# Rename livestock
l_ghg.loc[l_ghg.query('SPREAD_Commodity == "Dairy Cattle"').index, 'SPREAD_Commodity'] = 'DAIRY'
l_ghg.loc[l_ghg.query('SPREAD_Commodity == "Beef Cattle"').index, 'SPREAD_Commodity'] = 'BEEF'
l_ghg.loc[l_ghg.query('SPREAD_Commodity == "Sheep"').index, 'SPREAD_Commodity'] = 'SHEEP'

# Re-order and rename columns
l_ghg = l_ghg.rename(columns = {'kgco2_livestock_n2o_dung_urine': 'CO2E_KG_HEAD_DUNG_URINE', 
                                  'kgco2_livestock_electricity': 'CO2E_KG_HEAD_ELEC',
                                  'kgco2_livestock_enteric': 'CO2E_KG_HEAD_ENTERIC',
                                  'kgco2_livestock_fodder': 'CO2E_KG_HEAD_FODDER', 
                                  'kgco2_livestock_fuel': 'CO2E_KG_HEAD_FUEL', 
                                  'kgco2_livestock_n2o_leaching_runoff': 'CO2E_KG_HEAD_IND_LEACH_RUNOFF',
                                  'kgco2_livestock_manure_management': 'CO2E_KG_HEAD_MANURE_MGT',
                                  'kgco2_livestock_pasture_seeds': 'CO2E_KG_HEAD_SEED'
                                 })

# Calculate pivot table
lvstk_ghg_sources = ['CO2E_KG_HEAD_DUNG_URINE', 'CO2E_KG_HEAD_ELEC', 'CO2E_KG_HEAD_ENTERIC', 'CO2E_KG_HEAD_FODDER', 'CO2E_KG_HEAD_FUEL', 'CO2E_KG_HEAD_IND_LEACH_RUNOFF', 'CO2E_KG_HEAD_MANURE_MGT', 'CO2E_KG_HEAD_SEED'].sort()
p_ls = pd.pivot_table(l_ghg, 
                      values = lvstk_ghg_sources, 
                      index = 'SA2_ID', 
                      columns = 'SPREAD_Commodity', 
                      aggfunc = 'first'
                     ).sort_values(by = 'SA2_ID')

# Rearrange pivot table
p_ls.columns = p_ls.columns.rename('GHG Source', level = 0)
p_ls = p_ls.reorder_levels(['SPREAD_Commodity','GHG Source'], axis = 1)

# Flatten multiindex dataframe
# p_ls.columns = p_ls.columns.to_flat_index()

"""We have elected to omit GHG from livestock drinking water and assume that livestock predominantly drink from natural waterbodies 
   or from gravity fed sources, leading to zero emissions from drinking water. We include emissions from For irrigated pasture below.
"""
   
# # Merge livestock GHG to ludf dataframe by SA2 to check nodata
# ludf = pd.read_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_livestock_data.h5')
# tmp = ludf.query('LU_ID >= 31').merge(p_ls, how = 'left', on = 'SA2_ID')
# print('Number of NaNs =', tmp[tmp.isna().any(axis=1)].shape[0])

# Recreate multiindex dataframe
# p_ls.columns = pd.MultiIndex.from_tuples(p_ls.columns, names=['Livestock type','GHG Source'])

# Sort columns
p_ls.sort_index(axis = 1, level = 0, inplace = True)

# Downcast and save file
downcast(p_ls)
p_ls.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_livestock_GHG_data.h5', key = 'SA2_livestock_GHG_data', mode = 'w', format = 't')


# p_ls can then be joined to NLUM based on SA2_ID only. Values are per head so can be applied across livestock types on different MODL/NATL land and irrigation status.


"""
# Read in the livestock emissions table and drop unwanted columns
l_ghg = ghg_dataframe[['SA2_ID', 'SPREAD_ID', 'SPREAD_Commodity', 'irrigation', 
                       'kgco2_chem_app',
                       'kgco2_crop_management',
                       'kgco2_cult',
                       'kgco2_fert',
                       'kgco2_harvest',
                       'kgco2_irrig',
                       'kgco2_pest',
                       'kgco2_soil',
                       'kgco2_sowing',
                       'kgco2_transport'
                     ]].query('SPREAD_ID >= 31')
"""
# Calculate emissions from irrigated sown pastures as the same as from Winter cereals production (Hay has dodgy zero values for soil N20)
crop_ghg_sources = ['CO2E_KG_HA_CHEM_APPL', 
                    'CO2E_KG_HA_CROP_MGT', 
                    'CO2E_KG_HA_CULTIV', 
                    'CO2E_KG_HA_FERT_PROD', 
                    'CO2E_KG_HA_HARVEST', 
                    'CO2E_KG_HA_IRRIG', 
                    'CO2E_KG_HA_PEST_PROD', 
                    'CO2E_KG_HA_SOIL', 
                    'CO2E_KG_HA_SOWING']

# Rearrange the table structure
pivot_ghg = pd.pivot_table(c_ghg_with_STATE_ID, 
                      values = crop_ghg_sources, 
                      index = ['SA2_ID', 'SA4_ID', 'STATE_ID'],
                      columns = ['LU_DESC', 'IRRIGATION'],
                      aggfunc = 'first'
                     ).sort_values(by = 'SA2_ID')

# Select irrigated hay to represent emissions from irrigated sown pasture
irr_pasture_ghg = pivot_ghg.loc[:, (slice(None), 'Winter cereals', 1)]

# Flatten the MultiIndex to dataframe
irr_pasture_ghg = irr_pasture_ghg.droplevel(['IRRIGATION', 'LU_DESC'], axis = 1).reset_index(level = [0, 1, 2])

# Calculate and join summary tables by SA4 and STATE to fill in data gaps
tmp_SA4 = irr_pasture_ghg.groupby('SA4_ID', observed = True, as_index = False).agg(
                                    CO2E_KG_HA_CHEM_APPL_SA4 = ('CO2E_KG_HA_CHEM_APPL', 'mean'),
                                    CO2E_KG_HA_CROP_MGT_SA4  = ('CO2E_KG_HA_CROP_MGT', 'mean'),
                                    CO2E_KG_HA_CULTIV_SA4    = ('CO2E_KG_HA_CULTIV', 'mean'),
                                    CO2E_KG_HA_FERT_PROD_SA4 = ('CO2E_KG_HA_FERT_PROD', 'mean'),
                                    CO2E_KG_HA_HARVEST_SA4   = ('CO2E_KG_HA_HARVEST', 'mean'),
                                    CO2E_KG_HA_IRRIG_SA4     = ('CO2E_KG_HA_IRRIG', 'mean'),
                                    CO2E_KG_HA_PEST_PROD_SA4 = ('CO2E_KG_HA_PEST_PROD', 'mean'),
                                    CO2E_KG_HA_SOIL_SA4      = ('CO2E_KG_HA_SOIL', 'mean'),
                                    CO2E_KG_HA_SOWING_SA4    = ('CO2E_KG_HA_SOWING', 'mean'),
                                    )

tmp_STATE = irr_pasture_ghg.groupby('STATE_ID', observed = True, as_index = False).agg(
                                    CO2E_KG_HA_CHEM_APPL_STE = ('CO2E_KG_HA_CHEM_APPL', 'mean'),
                                    CO2E_KG_HA_CROP_MGT_STE  = ('CO2E_KG_HA_CROP_MGT', 'mean'),
                                    CO2E_KG_HA_CULTIV_STE    = ('CO2E_KG_HA_CULTIV', 'mean'),
                                    CO2E_KG_HA_FERT_PROD_STE = ('CO2E_KG_HA_FERT_PROD', 'mean'),
                                    CO2E_KG_HA_HARVEST_STE   = ('CO2E_KG_HA_HARVEST', 'mean'),
                                    CO2E_KG_HA_IRRIG_STE     = ('CO2E_KG_HA_IRRIG', 'mean'),
                                    CO2E_KG_HA_PEST_PROD_STE = ('CO2E_KG_HA_PEST_PROD', 'mean'),
                                    CO2E_KG_HA_SOIL_STE      = ('CO2E_KG_HA_SOIL', 'mean'),
                                    CO2E_KG_HA_SOWING_STE    = ('CO2E_KG_HA_SOWING', 'mean'),
                                    )

# Create dataframe with all SA2_ID, SA4_ID, and STE_ID
sa2_sa4_ste = ludf.groupby(['SA2_ID'], observed = True, as_index = False)[['SA4_ID', 'STE_ID']].first().sort_values(by = 'SA2_ID')

irr_pasture_ghg = sa2_sa4_ste.merge(irr_pasture_ghg.drop(columns = ['SA4_ID', 'STATE_ID']), how = 'left', on = 'SA2_ID')
irr_pasture_ghg = irr_pasture_ghg.merge(tmp_SA4, how = 'left', on = 'SA4_ID')
irr_pasture_ghg = irr_pasture_ghg.merge(tmp_STATE, how = 'left', left_on = 'STE_ID', right_on = 'STATE_ID')

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CHEM_APPL != CO2E_KG_HA_CHEM_APPL').index, 'CO2E_KG_HA_CHEM_APPL'] = irr_pasture_ghg['CO2E_KG_HA_CHEM_APPL_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CHEM_APPL != CO2E_KG_HA_CHEM_APPL').index, 'CO2E_KG_HA_CHEM_APPL'] = irr_pasture_ghg['CO2E_KG_HA_CHEM_APPL_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CHEM_APPL != CO2E_KG_HA_CHEM_APPL').index, 'CO2E_KG_HA_CHEM_APPL'] = irr_pasture_ghg['CO2E_KG_HA_CHEM_APPL'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CROP_MGT != CO2E_KG_HA_CROP_MGT').index, 'CO2E_KG_HA_CROP_MGT'] = irr_pasture_ghg['CO2E_KG_HA_CROP_MGT_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CROP_MGT != CO2E_KG_HA_CROP_MGT').index, 'CO2E_KG_HA_CROP_MGT'] = irr_pasture_ghg['CO2E_KG_HA_CROP_MGT_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CROP_MGT != CO2E_KG_HA_CROP_MGT').index, 'CO2E_KG_HA_CROP_MGT'] = irr_pasture_ghg['CO2E_KG_HA_CROP_MGT'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CULTIV != CO2E_KG_HA_CULTIV').index, 'CO2E_KG_HA_CULTIV'] = irr_pasture_ghg['CO2E_KG_HA_CULTIV_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CULTIV != CO2E_KG_HA_CULTIV').index, 'CO2E_KG_HA_CULTIV'] = irr_pasture_ghg['CO2E_KG_HA_CULTIV_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_CULTIV != CO2E_KG_HA_CULTIV').index, 'CO2E_KG_HA_CULTIV'] = irr_pasture_ghg['CO2E_KG_HA_CULTIV'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_FERT_PROD != CO2E_KG_HA_FERT_PROD').index, 'CO2E_KG_HA_FERT_PROD'] = irr_pasture_ghg['CO2E_KG_HA_FERT_PROD_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_FERT_PROD != CO2E_KG_HA_FERT_PROD').index, 'CO2E_KG_HA_FERT_PROD'] = irr_pasture_ghg['CO2E_KG_HA_FERT_PROD_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_FERT_PROD != CO2E_KG_HA_FERT_PROD').index, 'CO2E_KG_HA_FERT_PROD'] = irr_pasture_ghg['CO2E_KG_HA_FERT_PROD'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_IRRIG != CO2E_KG_HA_IRRIG').index, 'CO2E_KG_HA_IRRIG'] = irr_pasture_ghg['CO2E_KG_HA_IRRIG_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_IRRIG != CO2E_KG_HA_IRRIG').index, 'CO2E_KG_HA_IRRIG'] = irr_pasture_ghg['CO2E_KG_HA_IRRIG_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_IRRIG != CO2E_KG_HA_IRRIG').index, 'CO2E_KG_HA_IRRIG'] = irr_pasture_ghg['CO2E_KG_HA_IRRIG'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_PEST_PROD != CO2E_KG_HA_PEST_PROD').index, 'CO2E_KG_HA_PEST_PROD'] = irr_pasture_ghg['CO2E_KG_HA_PEST_PROD_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_PEST_PROD != CO2E_KG_HA_PEST_PROD').index, 'CO2E_KG_HA_PEST_PROD'] = irr_pasture_ghg['CO2E_KG_HA_PEST_PROD_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_PEST_PROD != CO2E_KG_HA_PEST_PROD').index, 'CO2E_KG_HA_PEST_PROD'] = irr_pasture_ghg['CO2E_KG_HA_PEST_PROD'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOIL != CO2E_KG_HA_SOIL').index, 'CO2E_KG_HA_SOIL'] = irr_pasture_ghg['CO2E_KG_HA_SOIL_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOIL != CO2E_KG_HA_SOIL').index, 'CO2E_KG_HA_SOIL'] = irr_pasture_ghg['CO2E_KG_HA_SOIL_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOIL != CO2E_KG_HA_SOIL').index, 'CO2E_KG_HA_SOIL'] = irr_pasture_ghg['CO2E_KG_HA_SOIL'].mean()

irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOWING != CO2E_KG_HA_SOWING').index, 'CO2E_KG_HA_SOWING'] = irr_pasture_ghg['CO2E_KG_HA_SOWING_SA4']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOWING != CO2E_KG_HA_SOWING').index, 'CO2E_KG_HA_SOWING'] = irr_pasture_ghg['CO2E_KG_HA_SOWING_STE']
irr_pasture_ghg.loc[irr_pasture_ghg.query('CO2E_KG_HA_SOWING != CO2E_KG_HA_SOWING').index, 'CO2E_KG_HA_SOWING'] = irr_pasture_ghg['CO2E_KG_HA_SOWING'].mean()

# Set HARVEST emissions to zero becasue pasture is not harvested
irr_pasture_ghg['CO2E_KG_HA_HARVEST'] = 0

# Whittle down the columns 
irr_pasture_ghg = irr_pasture_ghg[['SA2_ID'] + crop_ghg_sources]


print('Number of NaNs =', irr_pasture_ghg[irr_pasture_ghg.isna().any(axis=1)].shape[0])

# Downcast and save to file
downcast(irr_pasture_ghg)
irr_pasture_ghg.to_hdf('N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/SA2_irrigated_pasture_GHG_data.h5', key = 'SA2_irrigated_pasture_GHG_data', mode = 'w', format = 't')



