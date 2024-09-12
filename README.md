# RUSLE R-Factor PyData Paris 2024 Sprint

## Introduction

Erosion is the process by which natural forces like wind, water, ice, or gravity wear away rocks, soil, and other materials from the Earth's surface. Over time, these materials are transported and deposited in new locations, reshaping landscapes. Erosion can be caused by various factors, including rainfall, river flow, ocean waves, glaciers, and human activities such as deforestation or agriculture.

There are different types of erosion, such as Water erosion, caused by rain, rivers, and waves.

## The objetives of this sprint are:

### 1. Develop Python routines for imputing missing daily precipitation data values. All ideas are welcome from AI, ML, statistical, and others. [Precipitation data](https://github.com/ccalvocm/Erosion_24/blob/main/data/precipitation/example_data/Pp_day_86-16_Chile.csv) is used for R-Factor computing. The results must have physical meaning, such as avoiding negative values. 

### 2. Code a Python package for Geostatistical interpolation (like Co-Kriging) of R-Factor raster data, subject to high resolution DEM and raster color depth values. The results must have physical meaning, such as avoiding negative values. Inputs for Co-Kriging are:
####  2.1 [Average precipitation data gauges to interpolate across the mask](https://github.com/ccalvocm/Erosion_24/tree/main/data/precipitation/gauges_data).  
####  2.2 Digital Elevation Model.
####  2.3 R-Factor mask for the whole country.
