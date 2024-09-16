import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
from scipy import ndimage as nd
import rasterio, matplotlib
from rasterio import features
import lidario as lio

# Set some options
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 200)
pd.set_option('display.float_format', '{:,.2f}'.format)

infile = r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_mask.tif'
outgpkg = r'N:\Planet-A\Data-Master\Profit_map\ag_spatial_data.gpkg'


# Set file path
in_cell_df_path = 'N:/Planet-A/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.pkl'
 
# Read cell_df file from disk to a new data frame for ag data with just the relevant columns
cell_df = pd.read_pickle(in_cell_df_path)[['CELL_ID', 'CELL_HA', 'SA2_ID', 'COMMODITIES', 'COMMODITIES_DESC', 'IRRIGATION']]



################################ Open NLUM_ID as mask raster and get metadata, create some helper functions

with rasterio.open(infile) as rst:
    NLUM_mask = rst.read(1) # Loads a 2D masked array with nodata masked out
    
    # Get metadata and update parameters
    NLUM_transform = rst.transform
    meta = rst.meta.copy()
    meta.update(compress='lzw', driver='GTiff') # dtype='int32', nodata='-99')
    [meta.pop(key) for key in ['dtype', 'nodata']] # Need to add dtype and nodata manually when exporting GeoTiffs
        
    # Set some data structures to enable conversion on 1D arrays to 2D
    array_2D = np.zeros(NLUM_mask.shape) - 99
    xy = np.nonzero(NLUM_mask)
    
# Convert 1D column to 2D spatial array
def conv_1D_to_2D(in_1D_array):
    array_2D[xy] = np.array(in_1D_array)
    return array_2D.astype(in_1D_array.dtype)

# Convert 1D column to 2D spatial array and plot map
def map_in_2D(col, data): # data = 'continuous' or 'categorical'
    a2D = conv_1D_to_2D(col)
    if data == 'categorical':
        n = col.nunique()
        cmap = matplotlib.colors.ListedColormap(np.random.rand(n,3))
        plt.imshow(a2D, cmap=cmap, resample=False)
    elif data == 'continuous':
        plt.imshow(a2D, cmap = 'pink', resample = False)
    plt.show()

# Convert object columns to categories and downcast int64 columns to save memory and space
def downcast(dframe):
    obj_cols = dframe.select_dtypes(include = ["object"]).columns
    dframe[obj_cols] = dframe[obj_cols].astype('category')
    int64_cols = dframe.select_dtypes(include = ["int64"]).columns
    dframe[int64_cols] = dframe[int64_cols].apply(pd.to_numeric, downcast = 'integer')






################################ Join PROFIT MAP table to the cell_df dataframe

# =============================================================================
# Read in the PROFIT MAP table as provided by CSIRO to dataframe
ag_df = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/20210609/T_pfe_per_product_09062021.csv').drop(columns = 'rev_notes')

# Select crops only
crops_df = ag_df.query("SPREAD_ID >= 5 and SPREAD_ID <= 25")

# Rename columns to avoid python built-in naming
crops_df.rename(columns = {'yield': 'Yield', 
                        'prod': 'Production', 
                        'irrigation': 'Irrigation',
                        'irrig_factor': 'Prod_factor',
                        'SPREAD_ID_original': 'SPREAD_ID_orig',
                        'area': 'Area'}, inplace = True) 

# Calculate costs for checking post-aggregation cost calculation
crops_df.eval('Costs_ha_TRUE = (AC_rel + FDC_rel + FOC_rel + FLC_rel) + (QC_rel * Yield) + (WR * WP)', inplace = True)

######## Aggregate the ABS Commodity-level data to SPREAD class

# Define a lambda function to compute the weighted mean
area_weighted_mean = lambda x: np.average(x, weights = crops_df.loc[x.index, 'area_ABS'])  
prod_weighted_mean = lambda x: np.average(x, weights = crops_df.loc[x.index, 'prod_ABS'])  

# Aggregate commodity-level to SPREAD-level using weighted mean based on area of Commodity within SA2
# crops_df = crops_df[crops_df['SA2_Name'] == 'Yass Region']

crops_sum_df = crops_df.groupby(['SA2_ID', 'SPREAD_ID', 'SPREAD_ID_orig', 'Irrigation'], as_index = False).agg(
                    STATE_ID = ('STATE_ID', 'first'),
                    SA2_Name = ('SA2_Name', 'first'),
                    SPREAD_Name = ('SPREAD_Commodity', 'first'),
                    Count = ('Commodity', 'size'),
                    Area_ABS = ('area_ABS', 'sum'),
                    Prod_ABS = ('prod_ABS', 'sum'),
                    Area = ('Area', 'sum'),
                    Production = ('Production', 'sum'), 
                    Prop_irrig = ('pct_irrig', area_weighted_mean), 
                    Prod_factor = ('Prod_factor', area_weighted_mean),
                    Yield = ('Yield', area_weighted_mean),
                    P1 = ('P1', prod_weighted_mean),
                    Rev_ha_TRUE = ('rev', area_weighted_mean),
                    Costs_ha_TRUE = ('Costs_ha_TRUE', area_weighted_mean),
                    AC_rel = ('AC_rel', area_weighted_mean),
                    QC_rel = ('QC_rel', prod_weighted_mean),
                    FDC_rel = ('FDC_rel', area_weighted_mean),
                    FLC_rel = ('FLC_rel', area_weighted_mean),
                    FOC_rel = ('FOC_rel', area_weighted_mean),
                    WR = ('WR', area_weighted_mean),
                    WP = ('WP', area_weighted_mean)
                    )

# Sort in place
crops_sum_df.sort_values(by = ['SA2_ID', 'SPREAD_ID', 'SPREAD_ID_orig', 'Irrigation'], ascending = True, inplace = True)

# Save file
# crops_sum_df.to_csv('N:/Planet-A/Data-Master/Profit_map/SPREAD_aggregated.csv')
# crops_sum_df.to_pickle('N:/Planet-A/Data-Master/Profit_map/SPREAD_aggregated.pkl')
# crops_sum_df.to_hdf('N:/Planet-A/Data-Master/Profit_map/SPREAD_aggregated.h5', key = 'SPREAD_aggregated', mode = 'w')

# Calculate revenue and costs
crops_sum_df.eval('Yield_ha_TRUE = Production / Area', inplace = True)
crops_sum_df.eval('Rev_ha = Yield * P1', inplace = True)
crops_sum_df.eval('Rev_tot = Production * P1', inplace = True)

crops_sum_df.eval('Costs_ha = (AC_rel + FDC_rel + FOC_rel + FLC_rel) + (QC_rel * Yield) + (WR * WP)', inplace = True)
crops_sum_df.eval('Costs_t = Costs_ha / Yield', inplace = True)

crops_sum_df[['SPREAD_ID', 'SPREAD_ID_orig', 'Irrigation', 'SA2_Name', 'SPREAD_Name', 
           'Count', 'Production', 'Area', 'Prod_factor', 'Yield_ha_TRUE', 'Yield', 'Rev_ha_TRUE', 
           'Rev_ha', 'Rev_tot', 'Costs_ha_TRUE', 'Costs_ha', 'Costs_t', 'P1']].head(50)


# , 'F1', 'Q1', 'P1'
crops_sum_df.drop(columns = ['Costs_PCT_ABS'], inplace = True)

# Join the table to the cell_df dataframe and drop uneccesary columns
adf = cell_df.merge(crops_sum_df, how = 'left', left_on = ['SA2_ID', 'SPREAD_ID', 'IRRIGATION'], right_on = ['SA2_ID', 'SPREAD_ID', 'Irrigation']) 
adf2 = adf.query("SPREAD_ID >= 5 and SPREAD_ID <= 25")

#join the PFE table to NLUM_SA2_gdf and check for NaNs 
NLUM_SA2_gdf2 = NLUM_SA2_gdf.merge(tmp, how='left', left_on=['SA2_ID', 'SPREAD_ID', 'IRRIGATION'], right_on=['SA2_ID', 'SPREAD_ID', 'irrigation'])
# NLUM_SA2_gdf2 = NLUM_SA2_gdf2.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
print('Number of NaNs =', adf2[adf2.isna().any(axis=1)].shape[0])
# =============================================================================



################################ Join EMISSIONS tables to the cell_df dataframe

# =============================================================================
# # Read in the CROPS EMISSIONS table, join to cell_df, and drop unwanted columns
# tmp = pd.read_csv(r'N:\Planet-A\Data-Master\Profit_map\emissions-maps\T_emissions_by_SPREAD_SLA_crops.csv')
# cell_df = cell_df.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['SLA_code_2006', 'SPREAD_ID', 'irrigation']) # SLA_code_2006 needs to change to SA2 2011 code
# cell_df = cell_df.drop(columns=['kwhfert', 'kwhpest', 'kwhirrig', 'kwh_chemapp', 'kwh_cropmanagement', 'kwhcult', 'kwhharvest', 'kwhsowing'])
# 
# #join the CROPS EMISSIONS table to NLUM_SA2_gdf and check for NaNs 
# NLUM_SA2_gdf2 = NLUM_SA2_gdf.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['SLA_code_2006', 'SPREAD_ID', 'irrigation']) # Need to fix PFE table first
# # NLUM_SA2_gdf2 = NLUM_SA2_gdf2.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# print('Number of NaNs =', NLUM_SA2_gdf2[NLUM_SA2_gdf2.isna().any(axis=1)].shape[0])
# 
# # Read in the LIVESTOCK EMISSIONS table, join to cell_df, and drop unwanted columns
# tmp = pd.read_csv(r'N:\Planet-A\Data-Master\Profit_map\emissions-maps\T_emissions_by_SPREAD_SLA_livestock.csv')
# cell_df = cell_df.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['sla06_id', 'spread_id', 'irrigation']) # sla06_id needs to change to SA2 2011 code, no 'irrigation' code?
# # cell_df = cell_df.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# 
# #join the LIVESTOCK EMISSIONS table to NLUM_SA2_gdf and check for NaNs 
# NLUM_SA2_gdf2 = NLUM_SA2_gdf.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['sla06_id', 'SPREAD_ID', 'irrigation']) # Need to fix PFE table first
# # NLUM_SA2_gdf2 = NLUM_SA2_gdf2.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# print('Number of NaNs =', NLUM_SA2_gdf2[NLUM_SA2_gdf2.isna().any(axis=1)].shape[0])
# 
# =============================================================================



################################ Join TOXICITY table to the cell_df dataframe

# =============================================================================
# # Read in the TOXICITY table
# tmp = pd.read_csv(r'N:\Planet-A\Data-Master\Profit_map\pesticides-nutrients-maps\T_USETOX_CFvalue_by_SPREAD_SA2.csv')
# 
# # Join the table to the dataframe
# cell_df = cell_df.merge(tmp, how = 'left', left_on = ['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on = ['SA211_id', 'SPREAD_ID', 'irrigation'])
# # cell_df = cell_df.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# 
# #join the TOXICITY table to NLUM_SA2_gdf and check for NaNs 
# NLUM_SA2_gdf2 = NLUM_SA2_gdf.merge(tmp, how='left', left_on=['SA2_ID', 'COMMODITIES', 'IRRIGATION'], right_on=['SA211_id', 'SPREAD_ID', 'irrigation']) # Need to fix PFE table first
# # NLUM_SA2_gdf2 = NLUM_SA2_gdf2.drop(columns=['ALBERS_SQM', 'Rowid', 'VALUE', 'COUNT'])
# print('Number of NaNs =', NLUM_SA2_gdf2[NLUM_SA2_gdf2.isna().any(axis=1)].shape[0])
# 
# =============================================================================




# cell_df.query('SA2_ID' == 702011054 & 'SPREAD_ID' == 25), ('SA2_ID', 'SA2_Name', 'SPREAD_ID', 'SPREAD_Commodity', 'product', 'ha_weight_SPREAD')]


# tmp.groupby(['SPREAD'], as_index=False)[['SPREAD_ID']].first().sort_values(by=['SPREAD_ID'])
cell_df.groupby(['COMMODITIES'])[['COMMODITIES_DESC']].nunique() 
cell_df.groupby(['COMMODITIES'])[['COMMODITIES_DESC']].first()



cell_df.groupby(['PROT_AREAS'], as_index=False)[['PROT_AREAS_DESC']].first().sort_values(by=['PROT_AREAS'])

cell_df[cell_df['SA2_ID'] == 101011002].groupby(['COMMODITIES', 'IRRIGATION'])[['COMMODITIES_DESC']].size()

ag_df[(ag_df['area_ABS'] < 1) & (ag_df['SPREAD_ID'] <= 25) & (ag_df['SPREAD_ID'] >= 5)].iloc[:,3:16]

################################ Summarise Javi's profit map data table and join it to the cell_df dataframe


# Read in the profit map table to dataframe
javi = pd.read_csv('N:/Planet-A/Data-Master/Profit_map/From_CSIRO/T_pfe_per_product_07052021.csv')
javi.rename(columns={'yield': 'Yield', 'prod': 'Production', 'irrig_factor': 'Prod_factor'}, inplace = True)

javi.info()


javi['Commodity'].unique()
javi['SPREAD_Commodity'].unique()

javi.groupby(['SPREAD_Commodity'])[['SPREAD_ID']].nunique() 
javi.groupby(['SPREAD_Commodity'])[['SPREAD_ID']].first()
javi.groupby(['Commodity'])[['SPREAD_Commodity']].nunique()

javi[javi['SA2_Name'] == 'Adelaide Hills'].groupby(['SPREAD_Commodity', 'irrigation', 'SA2_Name'])[['pfe']].mean()

javi.groupby(['SPREAD_Commodity', 'irrigation', 'SPREAD_ID_original'])[['pfe']].mean()

pd.pivot_table(javi, index = 'SA2_ID', columns = 'SPREAD_ID', values = 'SPREAD_Commodity', aggfunc = 'count')

# Define a lambda function to compute the weighted mean
wm = lambda x: np.average(x, weights = javi.loc[x.index, 'area'])  

# Summarise ag data by SPREAD Commodity
j1 = javi[javi['SA2_Name'].isin(['Goulburn'])]

j = javi.groupby(['SA2_ID', 'SPREAD_ID', 'SPREAD_ID_original', 'irrigation'], as_index = False).agg(
                    SA2_Name = ('SA2_Name', 'first'),
                    SPREAD_Commodity = ('SPREAD_Commodity', 'first'),
                    Production = ('Production', wm), 
                    Area_crops = ('area', 'sum'),
                    Area_livestock = ('area', 'first'),
                    Prod_factor = ('Prod_factor', wm),
                    Yield = ('Yield', wm),
                    F1 = ('F1', wm),
                    Q1 = ('Q1', wm),
                    P1 = ('P1', wm),
                    F2 = ('F2', wm),
                    Q2 = ('Q2', wm),
                    P2 = ('P2', wm),
                    F3 = ('F3', wm),
                    Q3 = ('Q3', wm),
                    P3 = ('P3', wm),
                    Rev = ('rev', wm),
                    Cost_pct = ('cost_pct', wm),
                    AC = ('AC', wm),
                    QC = ('QC', wm),
                    FDC = ('FDC', wm),
                    FLC = ('FLC', wm),
                    FOC = ('FOC', wm),
                    AC_rel = ('AC_rel', wm),
                    QC_rel = ('QC_rel', wm),
                    FDC_rel = ('FDC_rel', wm),
                    FLC_rel = ('FLC_rel', wm),
                    FOC_rel = ('FOC_rel', wm),
                    WR = ('WR', wm),
                    WP = ('WP', wm),
                    PFE = ('pfe', wm),
                    Trace = ('trace', wm)
                    )

j.sort_values(by=['SA2_ID', 'SPREAD_ID', 'SPREAD_ID_original', 'irrigation'], ascending = True, inplace = True)
cols = list(range(1, 5)) + list(range(-5, -1))
j[j.columns[cols]]

j.to_csv('N:/Planet-A/Data-Master/Profit_map/SPREAD_aggregated.csv')
j.to_pickle('N:/Planet-A/Data-Master/Profit_map/SPREAD_aggregated.pkl')


adf.loc[adf.duplicated(subset=['CELL_ID']), ('CELL_ID', 'CELL_HA', 'SA2_ID', 'COMMODITIES', 'COMMODITIES_DESC', 'IRRIGATION', 'SPREAD_ID', 'SPREAD_ID_original', 'Irrigation', 'SA2_Name', 'SPREAD_Name', 'Production', 'Area', 'Prod_factor', 'Yield')]

# Check the sum of area of commmodities ABS vs NLUM
tmp = crops_sum_df.groupby(['SPREAD_Name', 'SA2_ID'])[['Area_crops']].first()
tmp.groupby(['SPREAD_Name'])[['Area_crops']].sum()

cell_df.groupby(['COMMODITIES_DESC'])[['CELL_HA']].sum()


""" Javi's equations
( (prod/area) + ((prod/area) * (irrig_factor-1)) ) * 
(F1*Q1*P1)+(F2*Q2*P2)+(F3*Q3*P3))) - (AC_rel+FDC_rel+FOC_rel+FLC_rel) - (QC_rel * ((prod/area)+((prod/area)*(irrig_factor-1)))) - (WR*WP)
rev = prod/area * irrig_factor * ((F1 * Q1 * P1) + (F2 * Q2 * P2) + (F3 * Q3 * P3))
costs = (AC_rel + FDC_rel + FOC_rel + FLC_rel) + (QC_rel * prod/area * irrig_factor) + (WR * WP)
"""



javi[['SA2_ID', 'Commodity', 'SPREAD_Commodity', 'Yield', 'F1', 'Q1', 'P1', 'irrig_factor', 'irrigation']]

# Calculate revenue and costs
javi['Revenue_BB'] = javi.eval('Production * irrig_factor / area * ((F1 * Q1 * P1) + (F2 * Q2 * P2) + (F3 * Q3 * P3))')
javi['Costs_PCT'] = javi.eval('rev * cost_pct')
javi['Costs_RAW'] = javi.eval('(AC_rel + FDC_rel + FOC_rel + FLC_rel) + (QC_rel * Production / area * irrig_factor) + (WR * WP)')
javi[javi['SPREAD_Commodity'].isin(['Beef Cattle','Sheep'])] \
    [['SPREAD_Commodity', 'irrigation', 'irrig_factor', 'Costs_PCT', 'Costs_RAW']]

javi[javi['SPREAD_Commodity'].isin(['Beef Cattle','Sheep'])] \
    [['SA2_ID', 'Commodity', 'SPREAD_Commodity', 'irrigation', 'rev', 'Revenue_BB']] \
    .sort_values(by=['SPREAD_Commodity', 'Commodity'], ascending = False)

javi.loc[(javi['SA2_ID'] == 801041042) & (javi['SPREAD_Commodity'] == 'Beef Cattle'), ('SA2_ID', 'Commodity', 'SPREAD_Commodity', 'Production', 'area', 'irrig_factor', 'Yield', 'rev', 'Revenue_BB')].sort_values(by=['SPREAD_Commodity', 'Commodity'])

    
    
# Calculate costs
javi['Costs_BB'] = javi.eval('AC + QC + FDC + FLC + FOC')
javi['Costs_JN'] = javi.eval('gross_revenue * mean_cost_pct')
javi['Costs_DIFF_%'] = javi.eval('abs(100 * (Costs_BB - Costs_JN) / Costs_JN)')
javi[['SA2_ID', 'product', 'Commodity', 'SPREAD_Commodity', 'gross_revenue', 'Costs_BB', 'Costs_JN', 'Costs_DIFF_%']].sort_values(by=['Costs_DIFF_%', 'Commodity'], ascending=False).head(100)

# Using the mean_cost_pct method the costs per tonne should be constant for all Commodities
javi['PFE_Tonne'] = javi.eval('PFE_dry / yield_sa2_avg')
javi[javi['Commodity'] == 'Canola']['PFE_Tonne'].describe()


# Quick comparison of costs 
javi.eval('Costs_BB - Costs_JN').describe()


# Calculate PFE
javi['PFE_BB'] = javi.eval('Revenue_BB - Costs_BB') 
javi[['SA2_ID', 'yield', 'Commodity', 'SPREAD_Commodity', 'rev', 'pfe']].head(100)




# Groupby and aggregate with namedAgg
wm = lambda x: np.average(x, weights=javi.loc[x.index, "ha_weight_SPREAD"])  # Define a lambda function to compute the weighted mean
j = javi.groupby(['SA2_ID', 'SPREAD_ID'], as_index=False).agg(SPREAD_Name = ('SPREAD_Commodity', 'first'),
                                                              sum_ha_weights = ('ha_weight_SPREAD', 'sum'), 
                                                              Count = ('SPREAD_Commodity', 'size'),
                                                              PFE_Dry_WM = ('PFE_dry', wm)
                                                              )



# Join the table to the dataframe
dfm = cell_df.merge(javi, how='left', left_on=['SA2_MAIN11', 'COMMODITIES', 'IRRIGATION'], right_on=['SA2_ID', 'SPREAD_ID', 'irrigation'])




# Check weighted sum calculations
j.loc[j['sum_ha_weights'] < 0.8]
javi.loc[(javi['SA2_ID'] == 702011054) & (javi['SPREAD_ID'] == 25), ('SA2_ID', 'SA2_Name', 'SPREAD_ID', 'SPREAD_Commodity', 'product', 'ha_weight_SPREAD')]


index = cell_df.query("(C18_DESCRIPTION == 'Dryland cropping (3.3)' or \
                        C18_DESCRIPTION == 'Grazing modified pastures (3.2)' or \
                        C18_DESCRIPTION == 'Grazing native vegetation (2.1)' or \
                        C18_DESCRIPTION == 'Other minimal use (1.3)') and \
                       (FOREST_TYPE_DESC == 'Non-forest or no data')").index
                 
index = cell_df[cell_df['C18_DESCRIPTION'].isin(["Dryland cropping (3.3)", "Grazing modified pastures (3.2)", "Grazing native vegetation (2.1)", "Other minimal use (1.3)"]) & \
                (cell_df['FOREST_TYPE_DESC'] == 'Non-forest or no data')].index
    

# Select rows which satisfy the query
index = cell_df.query("C18_DESCRIPTION in ['Dryland cropping (3.3)', 'Grazing modified pastures (3.2)', 'Grazing native vegetation (2.1)', 'Other minimal use (1.3)'] and \
                      FOREST_TYPE_DESC == 'Non-forest or no data'").index

# Add a new field ad calculate values for selected rows
cell_df['MASK'] = 0
cell_df['MASK'] = cell_df['MASK'].astype(np.uint8)
cell_df.loc[index,'MASK'] = 1










# Add new column in cell_df 
pos = cell_df.columns.get_loc('VEG_COND_DESC') + 1
cell_df.insert(loc = pos, column = 'NATURAL_AREAS', value = 0, allow_duplicates = True)

    
    
    
######################## Handy crosstab queries for exploring the NLUM categorisations

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'VEG_MASK', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'C18_DESCRIPTION', columns = 'VEG_MASK', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'C18_DESCRIPTION', columns = 'COMMODITIES', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'C18_DESCRIPTION', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'PRIMARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'SECONDARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'SECONDARY_V7', columns = 'PRIMARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'SECONDARY_V7', columns = 'VEG_MASK', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'PRIMARY_V7', columns = 'VEG_MASK', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'PRIMARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'VEG_COND_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'PRIMARY_V7', columns = 'VEG_COND_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'PRIMARY_V7', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'FOREST_TYPE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'SECONDARY_V7', columns = 'FOREST_TYPE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'PRIMARY_V7', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')

pd.pivot_table(cell_df, index = 'PRIMARY_V7', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')



cell_df.groupby(['COMMODITIES'])[['COMMODITIES_DESC']].first()



# Knock out 
Mining and waste (5.8, 5.9)
No data
Plantation forestry (3.1, 4.1)
Rural residential and farm infrastructure (5.4...
Urban intensive uses (5.3, 5.4, 5.4.1, 5.5, 5.6...
Water (6.0)

Tenure - water, defence, ocean, aboriginal?


SECONDARY_V7                                                                                                                                                                                           
3.1 Plantation forestry
4.1 Irrigated plantation forestry




np.sort(cell_df['COMMODITIES'].unique())
np.sort(javi['SPREAD_ID'].unique())
np.sort(javi['Commodity'].unique())


# Print mapping of 'product' vs 'SPREAD Commodity' vs 'Commodity' classifications
javi.groupby(['product', 'Commodity', 'SPREAD_Commodity'], as_index=False)[['SPREAD_ID']].first().sort_values(by=['SPREAD_ID', 'Commodity', 'product'])


# Pivot table
pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', values = 'COMMODITIES', aggfunc='mean').sort_values(by=['COMMODITIES'])
pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', values = ['COMMODITIES', 'ALBERS_SQM'], aggfunc= ['sum', 'first']).sort_values(by=[('first', 'COMMODITIES')])
pd.pivot_table(javi, index = 'SPREAD_Commodity', values = 'SPREAD_ID', aggfunc='first').sort_values(by=['SPREAD_ID'])

gdf = gpd.read_file(r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.gpkg')
# df = pd.read_csv(r'N:\Planet-A\Data-Master\National_Landuse_Map\NLUM_2010-11_clip.tif.csv')
gdf["AREA"] = gdf['geometry'].area
pd.pivot_table(gdf, index = 'COMMODITIES_DESC', columns = 'VEG_COND_DESC', values = 'AREA', aggfunc = 'sum')

pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'VEG_COND_DESC', values = 'CELL_HA', aggfunc = 'sum')
pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'VEG_COND_DESC', values = 'CELL_HA', aggfunc = 'count')
pd.pivot_table(cell_df, index = 'COMMODITIES_DESC', columns = 'TENURE_DESC', values = 'CELL_HA', aggfunc = 'count')


# Dissolve grid cells to 
attrs = ["attr0", "attr1", "attr2"]
SA2_gdf_dissolved = SA2_gdf.dissolve(by=attrs, as_index=False)



