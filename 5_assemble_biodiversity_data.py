import numpy as np

# The script uses `xarray` to process data, therefore has different methods.

# Author:		Jinzhu WANG
# Email: 		wangjinzhulala@gmail.com
# Last update: 	22 Nov, 2024


'''
The benefits of using xarray is its multi-dimension labeling for NDArray, inherent parallelizing,
and raster processing (with rioxarray) capabilities.


The key technical consideration here is how to deal with the ~10k layers in a reasonable time.
We choose to use a 5km spatial resolution to reduce data size, and leverage the parallelising 
of xarray to speed up the processing.
'''




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


