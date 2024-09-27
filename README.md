# RUSLE R-Factor PyData Paris 2024 Sprint

## Introduction

Erosion is the process by which natural forces like wind, water, ice, or gravity wear away rocks, soil, and other materials from the Earth's surface. Over time, these materials are transported and deposited in new locations, reshaping landscapes. Erosion can be caused by various factors, including rainfall, river flow, ocean waves, glaciers, and human activities such as deforestation or agriculture.

There are different types of erosion, such as Water erosion, caused by rain, rivers, and waves.

## The objectives of this sprint are:

### 1. Develop Python routines for imputing missing daily precipitation data values. All ideas are welcome from AI, ML, Multilinear Regression, OLS, and Python packages like pandas, geopandas, scikit-learn, among others. [Precipitation data](https://github.com/ccalvocm/Erosion_24/blob/main/data/precipitation/gauges_data/est_DMC_2024-05-23.xlsx) is used for R-Factor computing. The results must have physical meaning, such as avoiding negative values. See [rusleData.py](https://github.com/ccalvocm/Erosion_24/blob/main/src/rusleData.py). The future roadmap will include time series trend analysis.

### 2. Code a Python package for Geostatistical interpolation (like Co-Kriging) of R-Factor raster data, subject to high resolution DEM and raster color depth values. The results must have physical meaning, such as avoiding negative values. Inputs for Co-Kriging are:
####  2.1 [Average precipitation data gauges to interpolate across the mask](https://github.com/ccalvocm/Erosion_24/tree/main/data/precipitation/gauges_data).  
####  2.2 [Digital Elevation Model](https://www.dropbox.com/scl/fi/rmcbngua9kkyymnq78xa3/DEM_Chile90m19s.zip?rlkey=x58t1hi8nlmp710b5fn2f2dp8&st=dc0ei95i&dl=0).
####  2.3 [R-Factor mask for the whole country](https://www.dropbox.com/scl/fi/dnrcisgfjfpujjpmbimtj/ODEPA_FACTOR_R.zip?rlkey=qpknxgrjkj76edibn8d3violx&st=z7pe6tut&dl=0).

![](https://raw.githubusercontent.com/ccalvocm/Erosion_24/main/imgs/thumbnail_pp.png)

## 3. [Environment](https://github.com/ccalvocm/Erosion_24/blob/main/env/environment.yaml): 
#### 3.1 pandas
#### 3.2 scikit-learn
#### 3.3 geopandas
#### 3.4 openpyxl
