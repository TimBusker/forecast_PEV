%reset
# %%
import sys
import os
from os import listdir
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import dask
from dask.diagnostics import ProgressBar
import regionmask
import geopandas as gpd

pbar = ProgressBar()
pbar.register()
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import colour
from colour import Color
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import pandas as pd



# import all functions from function.py (in home dir)
#sys.path.append("/scistor/ivm/tbr910/")
#from functions import *
#season='aut'
#rename_files("/scistor/ivm/tbr910/precip_analysis/verif_files", f"S1{season}",f"S1_{season}", season)
# %%
###############################################################################################################################
##################################################### Setup ###################################################################
###############################################################################################################################

##################################### Config  #####################################

home = "/scistor/ivm/tbr910/"  # \\scistor.vu.nl\shares\BETA-IVM-BAZIS\tbr910\ # /scistor/ivm/tbr910/
path_base = home + "precip_analysis"
path_obs = home + "precip_analysis/obs"
path_forecasts = home + "precip_analysis/forecasts"
path_obs_for = home + "precip_analysis/obs_for"
path_q = home + "precip_analysis/quantiles"
path_cont = home + "precip_analysis/cont_metrics"
path_efi = home + "ECMWF/files_efi/"
path_verif = home + "precip_analysis/verif_files"
path_obs_for_new = home + "precip_analysis/obs_for_new_LT"
path_figs = home + "precip_analysis/figures/revisions/"
path_return_periods = "/scistor/ivm/tbr910/precip_analysis/return_periods_europe"

seasons=['summer', 'winter', 'aut', 'spring']

#seasons= [i+'_WHOLE_PERIOD' for i in seasons]

day_month='26_08'
CL_config= "minor"
lead_times = ["1 days", "2 days", "3 days", "4 days", "5 days"]
shift = 1  # then 95/2. was 1?
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
lat_slice = slice(lon_lat_box[3], lon_lat_box[2])  # in case of area selection



"""
Select the precipitation threshold. The following options are supported in this version: 

- Fixed percentiles: 0.95, 0.98, 0.99 (method var: quantile_extremes)
- Fixed rainfall amounts (mm): 40, 60, 90 (method var: threshold_method, not implemented yet?)
- Fixed return periods: 5RP, 10RP, 20RP (method var: return_periods)


"""
p_thresholds = ["5RP"] # 10RP
indicator = "sot" # ES, efi or sot --> ES is combined efi+sot, if you need efi + sot seperately, the script needs to run twice.

os.chdir(path_base)

quality_mask = xr.open_dataset(
    path_base + "/quality_mask_025.nc" #025 degree
)  # includes land-sea and X% nan criteria


# loop over seasons and load+merge cont metrics 
for p_threshold in p_thresholds:
    
    ################################################ Merge cont metrics of different seasons ################################################
    cont_metrics_merged=xr.Dataset()
    Fval_max_merged=xr.Dataset()
    Fval_area_merged=xr.Dataset()
    n_events_area_merged=xr.Dataset()
    for season in seasons: 
        file_accessor = f'{indicator}_{day_month}_{str(p_threshold).replace(".","")}_S{shift}_{season}.nc'  # file accessor from the 'save_string' variable in PEV.py
        
        cont = xr.open_dataset(path_verif+"/cont_metrics_merged_%s" %(file_accessor))
        Fval = xr.open_dataset(path_verif+"/Fval_merged_%s" %(file_accessor))
        Fval_area = xr.open_dataset(path_verif+"/Fval_area_merged_%s" %(file_accessor))
        n_events_area=Fval_area.n_events

        #Calculate Fval_max 
        Fval_max=Fval.max(dim=("ew_threshold"), keep_attrs=True)
        Fval_area_max=Fval_area.max(dim=("ew_threshold"),keep_attrs=True)

        # Add the season dimension
        #cont = cont.expand_dims({'season': [season]})
        Fval_max = Fval_max.expand_dims({'season': [season]})
        Fval_max_merged=xr.merge([Fval_max, Fval_max_merged])
        Fval_area_max = Fval_area_max.expand_dims({'season': [season]})
        Fval_area_max_merged=xr.merge([Fval_area_max, Fval_area_merged])
        cont=cont.expand_dims({'season': [season]})
        cont_metrics_merged=xr.merge([cont, cont_metrics_merged])
        
        n_events_area=n_events_area.expand_dims({'season': [season]})
        n_events_area_merged=xr.merge([n_events_area, n_events_area_merged])

    print('all seasons loaded and merged')

    # n_events map 

    # calculate sum of cont metrics
    cont_metrics_merged=cont_metrics_merged.sum(dim='season')
    #cont_metrics_merged=cont_metrics_merged.where(cont_metrics_merged.n_events > 0, np.nan)
    
    # n_events map
    n_events=cont_metrics_merged.isel(ew_threshold=0).n_events.drop_vars('ew_threshold')
    
    # calculate the mean of the Fval_max and Fval_area_max
    Fval_max_seasonal=Fval_max_merged.mean(dim='season')
    Fval_area_max_seasonal=Fval_area_max_merged.mean(dim='season', keep_attrs=True)
    
    # filter areas with less 0 events
    Fval_max_seasonal = Fval_max_seasonal.where(quality_mask.rr == 0, np.nan)
    #Fval_max_seasonal=Fval_max_seasonal.where(n_events> 0, np.nan)


    n_events_area_merged=n_events_area_merged.sum(dim='season')
    #Fval_area_max_seasonal=Fval_area_max_seasonal.where(n_events_area_merged.n_events > 0, np.nan)
    
    # quality mask 
    #cont_metrics_merged=cont_metrics_merged.where(cont_metrics_merged.n_events > 0, np.nan)
    
    #save_name= f"{indicator}_{file_accessor}"  # save name for the figures
    
    #
    
    # save
    # save the merged season files
    save_string=f"_{indicator}_max_{day_month}_{str(p_threshold).replace('.','')}_S{shift}_seasonal.nc"
    cont_metrics_merged.to_netcdf(path_verif+"/cont_metrics_merged%s" %(save_string))
    Fval_max_seasonal.to_netcdf(path_verif+"/Fval_merged%s" %(save_string))
    Fval_area_max_seasonal.to_netcdf(path_verif+"/Fval_area_merged%s" %(save_string))
    print('seasonal files saved')

        # sum cont and cont_metrics_merged if season is not the first in the list of variable seasons
        #if season == seasons[0]: # if first season, just load the cont metrics
        #    cont_metrics_merged = cont.copy()
        #    Fval_merged = Fval.copy()
        #    Fval_area_merged = Fval_area.copy()
        #    print('first season, %s , loaded'%(season))
        #else: # if not the first season, sum cont and cont_metrics_merged 
        #    cont_metrics_merged = cont_metrics_merged.fillna(0) + cont.fillna(0)
        #    Fval_merged=Fval_merged
        #    print('season %s added'%(season))
    
    
    
    