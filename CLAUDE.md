# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains a sequential data processing pipeline for assembling LUTO 2.0 (Land Use Trade-Offs) input data. The scripts process geospatial, agricultural, biodiversity, and biophysical data to create standardized datasets for land use modeling in Australia.

## Data Processing Pipeline Architecture

The scripts follow a numbered sequence that must be executed in order:

1. **1_assemble_zones_data.py** - Creates base spatial zones from NLUM (National Land Use Map) raster data, converts to vector format, and establishes the foundational cell-based spatial framework
2. **2_assemble_agricultural_data.py** - Processes agricultural profit mapping data from CSIRO, livestock data, and creates agricultural commodity datasets
3. **3_agriculture_climate_damage.py** - Calculates climate damage impacts on agricultural productivity
4. **4_assemble_biophysical_data.py** - Processes biophysical environmental data including soils, climate, and terrain variables
5. **5_assemble_biodiversity_data.py** - Assembles biodiversity and conservation data including species suitability and environmental condition indices
6. **6_water_yield_modelling.py** - Calculates water yield using InVEST model under various climate scenarios
7. **7_assemble_additional_land_use_sieve_data.py** - Creates land use suitability constraints and sieve layers
8. **8_assemble_ag_yield_gap_data.py** - Processes agricultural yield gap analysis data
9. **9_reforestation_carbon_data.py** - Assembles carbon sequestration data for reforestation scenarios

## Common Data Patterns

### Spatial Framework
All scripts use a consistent spatial framework based on:
- **NLUM mask raster**: `N:/Data-Master/National_Landuse_Map/NLUM_2010-11_mask.tif`
- **Cell dataframe**: `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/cell_zones_df.h5`
- Standard helper functions for 1D/2D array conversion and spatial operations

### Common Helper Functions
Most scripts implement similar utility functions:
- `conv_1D_to_2D()` - Converts 1D data arrays to 2D spatial arrays
- `map_in_2D()` - Visualizes data as spatial maps
- `downcast()` - Optimizes dataframe memory usage by downcasting data types

### Data Sources
Key external data dependencies include:
- NLUM (National Land Use Map) spatial data
- CSIRO Profit Map agricultural data
- Biodiversity and environmental suitability data
- Climate and biophysical raster datasets
- Various government statistical and spatial datasets

## Development Commands

**Running Scripts:**
```bash
python <script_name>.py
```

**Data Validation:**
Scripts typically output processed data to:
- `N:/Data-Master/LUTO_2.0_input_data/Input_data/2D_Spatial_Snapshot/`
- `Intermediate_data_outputs/`

## Architecture Notes

### Memory Management
- Scripts use pandas HDF5 format for efficient large dataset storage
- Rasterio for geospatial raster processing with memory-mapped access
- Downcast functions to optimize memory usage of dataframes

### Spatial Data Processing
- All spatial operations maintain consistent CRS and spatial extent
- Uses rasterio for raster operations and geopandas for vector processing
- xarray integration for multi-dimensional climate/biodiversity datasets

### Data Pipeline Dependencies
Scripts have strict sequential dependencies - later scripts require outputs from earlier ones. The `cell_zones_df.h5` file created by script 1 is fundamental to all subsequent processing.

### Error Handling
Scripts typically include manual data validation and quality checking rather than automated error handling. Visual plotting functions help verify spatial data integrity.