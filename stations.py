"""
Created on Mon Aug 31 17:04:50 2020

@author: Tim Busker 

This script loads the stations from the KNMI database and plots them on a map. These represent the stations from the European Climate Assessment and Datasets (ECA&D)

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
from lat_lon_parser import parse 
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
# set dask to split large chunks 
dask.config.set({"array.slicing.split_large_chunks": True})


############################################ Load observationsn  #############################################
# load obs
os.chdir(path_obs)
#precip=xr.open_dataset('rr_ens_mean_0.25deg_reg_v28.0e.nc') # 025 deg

############################################ Load coordinates of stations #############################################
# load stations from online .txt file https://knmi-ecad-assets-prd.s3.amazonaws.com/download/stations.txt
stations_raw=pd.read_csv('https://knmi-ecad-assets-prd.s3.amazonaws.com/download/stations.txt', sep=",", header=13,index_col=False, decimal='.', encoding='latin-1')
# lat is in degrees:minutes:seconds (+: North, -: South)
# lon is in degrees:minutes:seconds (+: East, -: West)
# delete spaces in the left and right side of the string of the headers
for column_name in stations_raw.columns:
      stations_raw.rename(columns={column_name: column_name.strip()}, inplace=True)

stations=stations_raw.copy()

# Define the substrings to search for
substrings = ['180', '181', '184', '186', '190']

# Create a regex pattern to match any of the substrings
pattern = '|'.join(substrings)

# Filter out rows where the LON column contains any of the substrings
stations = stations[~stations['LON'].str.contains(pattern)]
stations['LAT']=stations['LAT'].apply(parse)
stations['LON']=stations['LON'].apply(parse)

############################################ Plot stations #############################################
# filter lon lats for europe
stations=stations.where(stations['LON'] > -10)
stations=stations.where(stations['LON'] < 40)
stations=stations.where(stations['LAT'] > 30)
stations=stations.where(stations['LAT'] < 70)


# select same lon lats in precip
# precip=precip.where(precip['longitude'] > -10, drop=True)
# precip=precip.where(precip['longitude'] < 40, drop=True)
# precip=precip.where(precip['latitude'] > 30, drop=True)
# precip=precip.where(precip['latitude'] < 70, drop=True)
# precip_plot=precip.isel(time=20010).rr

fig=plt.figure(figsize=(20,10))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.plot(stations['LON'],stations['LAT'], 'bo', markersize=0.15, transform=ccrs.PlateCarree(), color='darkblue')
# plot precip without colorbar, no title
#precip_plot.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues', add_colorbar=False, vmin=0, vmax=20)
plt.show()

# save the plot
fig.savefig('/scistor/ivm/tbr910/precip_analysis/figures/stations_precip_no_background.pdf', dpi=300, bbox_inches='tight')

