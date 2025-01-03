"""
Created on Mon Aug 31 17:04:50 2020

@author: Tim Busker 

This is a support script, not used in the paper. It was used to check the return period rainfall maps downloaded from Copernicus.
"""

import os
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import xarray as xr
from netCDF4 import Dataset as netcdf_dataset
from pylab import *
import sys
import os
from os import listdir
import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask
from dask.diagnostics import ProgressBar
import xskillscore as xs

pbar = ProgressBar()
pbar.register()

os.chdir('/scistor/ivm/tbr910/precip_analysis/scripts')
from func_Fval import *

############################################ paths #############################################
path_obs= '/scistor/ivm/tbr910/precip_analysis/obs'
path_forecasts='/scistor/ivm/tbr910/precip_analysis/forecasts'
path_obs_for= '/scistor/ivm/tbr910/precip_analysis/obs_for'
path_q='/scistor/ivm/tbr910/precip_analysis/quantiles'
path_cont='/scistor/ivm/tbr910/precip_analysis/cont_metrics'
path_efi='/scistor/ivm/tbr910/ECMWF/files_efi/'
path_verif= '/scistor/ivm/tbr910/precip_analysis/verif_files'
path_obs_for_new= '/scistor/ivm/tbr910/precip_analysis/obs_for_new_LT'
path_return_periods = "/scistor/ivm/tbr910/precip_analysis/return_periods_europe"

"""
lon lat boxes 
"[2.5, 14, 47.5, 55] --> large area Western Europe (used till now)
[3.95,7.8,49.3,51.3] --> Affected by 2021 floods
[-10, 20, 39, 55] --> much larger (rondom?) area
[1,7.8,48,52] --> area based on many events
[3.5,7.8,48,52] --> area based on many events (excluding coastal area of france)
"""

lon_lat_box = [3.5, 7.8, 48, 52]  # [lon_min, lon_max, lat_min, lat_max]
lon_slice = slice(lon_lat_box[0], lon_lat_box[1])  # in case of area selection
lat_slice = slice(lon_lat_box[2], lon_lat_box[3])  # in case of area selection


############################################ Load observations, select time of analysis #############################################
# load obs
os.chdir(path_obs)
precip=xr.open_dataset('rr_ens_mean_0.25deg_reg_v28.0e.nc') # 025 deg
precip=precip.sel(time=slice("2016-03-08", "2023-12-31"))
precip=precip.sel(longitude=lon_slice, latitude=lat_slice)
print(os.listdir(path_return_periods))

############################################ Load return period maps  #############################################
precip_5RP = xr.open_dataset(path_return_periods+ f"/precipitation-at-fixed-return-period_europe_e-obs_30-year_5-yrs_1989-2018_v1.nc").rename({'r5yrrp':'rr'})
precip_5RP=precip_5RP.interp_like(precip)
precip_10RP= xr.open_dataset(path_return_periods+ f"/precipitation-at-fixed-return-period_europe_e-obs_30-year_10-yrs_1989-2018_v1.nc").rename({'r10yrrp':'rr'})
precip_10RP=precip_10RP.interp_like(precip)
precip_20RP= xr.open_dataset(path_return_periods+ f"/precipitation-at-fixed-return-period_europe_e-obs_30-year_25-yrs_1989-2018_v1.nc").rename({'r25yrrp':'rr'})
precip_20RP=precip_20RP.interp_like(precip)

precip_5RP['rr'].plot(vmin=0, vmax=100, cmap='viridis')
plt.title('5-year return period, interpolated') # visually checked and interpolation went OK 
plt.show()


precip_10RP['rr'].plot(vmin=0, vmax=100, cmap='viridis')
plt.title('10-year return period, interpolated') # visually checked and interpolation went OK
plt.show()

precip_20RP['rr'].plot(vmin=0, vmax=100, cmap='viridis')
plt.title('25-year return period, interpolated') # visually checked and interpolation went OK
plt.show()

############################################ exceendance maps #############################################
precip_5RP_exc=precip.where(precip.rr > precip_5RP.rr, drop=True) # keep only the months that have a extreme events (>threshold)
precip_5RP_exc.rr['time.month']
precip_5RP_exc.time[0:10]
precip_10RP_exc=precip.where(precip.rr > precip_10RP.rr, drop=True) # keep only the months that have a extreme events (>threshold)
precip_20RP_exc=precip.where(precip.rr > precip_20RP.rr, drop=True) # keep only the months that have a extreme events (>threshold)
# now, do the following: extract all the months present in precip_5RP_exc, and then make a histogram that shows how often the months are represented in the exceedance map
# return the bins of the hist plot 
precip_5RP_exc['time.month'].plot.hist(bins=12)
plt.show()
plt.close() 
precip_10RP_exc['time.month'].plot.hist(bins=12)
plt.show()
plt.close()
precip_20RP_exc['time.month'].plot.hist(bins=12)
plt.show()
plt.close()

# get relative frequency of the december jan february months 
winter_months = len((precip_5RP_exc.where(precip_5RP_exc['time.month']==12, drop=True)).time)  + len((precip_5RP_exc.where(precip_5RP_exc['time.month']==1, drop=True)).time) + len((precip_5RP_exc.where(precip_5RP_exc['time.month']==2, drop=True)).time)
winter_months_percentage= winter_months/len(precip_5RP_exc.time)
print(winter_months_percentage)

# summer months percentage
summer_months = len((precip_5RP_exc.where(precip_5RP_exc['time.month']==6, drop=True)).time)  + len((precip_5RP_exc.where(precip_5RP_exc['time.month']==7, drop=True)).time) + len((precip_5RP_exc.where(precip_5RP_exc['time.month']==8, drop=True)).time)
summer_months_percentage= summer_months/len(precip_5RP_exc.time)
print(summer_months_percentage)

# spring months percentage
spring_months = len(precip_5RP_exc.where(precip_5RP_exc['time.month'].isin([3, 4, 5]), drop=True).time)
spring_months_percentage= spring_months/len(precip_5RP_exc.time)
print(spring_months_percentage)

# feb march april
aut_months = len((precip_5RP_exc.where(precip_5RP_exc['time.month']==9, drop=True)).time)  + len((precip_5RP_exc.where(precip_5RP_exc['time.month']==10, drop=True)).time) + len((precip_5RP_exc.where(precip_5RP_exc['time.month']==11, drop=True)).time)
fma_months_percentage= aut_months/len(precip_5RP_exc.time)
print(fma_months_percentage)

