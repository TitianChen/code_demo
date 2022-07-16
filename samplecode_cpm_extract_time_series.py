# -*- coding: utf-8 -*-
"""
sample script to extract time series from CPM2.2 (UKCP18 data format)

@author: Yuting Chen

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

products_path = "E:/temp/"

ensNo = 5
year = 1989

# london
lat_0 = 51.5072
lon_0 = 0.1276

variable = "pr"
model_collection = "UKCP18"
emission_scenario = "rcp85"
product = "cpm_uk_2.2km"
area = "UK"
cm_fn_pat = products_path + "%02d/%s_%s_land-%s_%02d_day_%04d1201-%04d1130.nc"
cm_fn = (cm_fn_pat) % (
    ensNo,
    variable,
    emission_scenario,
    product,
    ensNo,
    year,
    year + 1,
)


ds = xr.open_dataset(cm_fn)

d_lat = ds.latitude - lat_0
d_lon = ds.longitude - lon_0
r2_requested = d_lat**2 + d_lon**2
(loc_lat, loc_lon) = np.where(r2_requested == np.min(r2_requested))

ds_sub = ds.isel(grid_latitude=loc_lat, grid_longitude=loc_lon)
ds.close()

plt.plot(ds_sub.pr.squeeze())

