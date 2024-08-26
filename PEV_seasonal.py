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


indicator='sot'
seasons=['summer', 'winter', 'aut', 'spring']

#seasons= [i+'_WHOLE_PERIOD' for i in seasons]

p_thresholds=["5RP"] # ","10RP"
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

#%%
##################################################################################### Seasonal PEV Calculation #####################################################################################

os.chdir(path_base)

quality_mask = xr.open_dataset(
    path_base + "/quality_mask_025.nc" #025 degree
)  # includes land-sea and X% nan criteria

############################################################# START LOOP  ############################################################


# loop over seasons and load+merge cont metrics 
for p_threshold in p_thresholds:
    
    ################################################ Merge cont metrics of different seasons ################################################
    cont_metrics_merged=xr.Dataset()
    for season in seasons: 
        file_accessor = f'{indicator}_{day_month}_{str(p_threshold).replace(".","")}_S{shift}_{season}.nc'  # file accessor from the 'save_string' variable in PEV.py
        save_name= f"{indicator}_{file_accessor}"  # save name for the figures
        cont = xr.open_dataset(path_verif+"/cont_metrics_merged_%s" %(file_accessor))
        # sum cont and cont_metrics_merged if season is not the first in the list of variable seasons
        if season == seasons[0]: # if first season, just load the cont metrics
            cont_metrics_merged = cont.copy()
            print('first season, %s , loaded'%(season))
        else: # if not the first season, sum cont and cont_metrics_merged 
            cont_metrics_merged = cont_metrics_merged.fillna(0) + cont.fillna(0)
            print('season %s added'%(season))
    # quality mask 
    cont_metrics_merged=cont_metrics_merged.where(cont_metrics_merged.n_events > 0, np.nan)
    cont_metrics_merged = cont_metrics_merged.where(quality_mask.rr == 0, np.nan)
    # save
    print(f"cont_metrics_merged_seasonal_{file_accessor} saved")
    Fval_merged = xr.Dataset()
    Fval_area_merged = xr.Dataset()

    for lt in lead_times:
        print("Start LT %s" % (lt[0]))

        cont_metrics_merged_lead=cont_metrics_merged.sel(lead=lt)

        
        
        hits=cont_metrics_merged_lead.hits
        misses=cont_metrics_merged_lead.misses
        false_alarms=cont_metrics_merged_lead.false_alarms
        correct_negatives=cont_metrics_merged_lead.correct_negatives
        
        # event count and clim frequency
        time_steps_total = hits+misses+false_alarms+correct_negatives
        event_count = hits+misses



        #time_steps_total = time_steps_total.where(event_count > 0, np.nan)
        Climfreq = (event_count / time_steps_total)  # Ratio of event time steps compared to all time steps in rainfall dataset

        #print("average climfreq is %s" % (Climfreq.mean().values.round(5)))

        # mask areas with 0 obs
        #hits = hits.where(event_count > 0, np.nan)
        #false_alarms= false_alarms.where(event_count > 0, np.nan)
        #misses= misses.where(event_count > 0, np.nan)
        #correct_negatives= correct_negatives.where(event_count > 0, np.nan)
        #Climfreq = Climfreq.where(event_count > 0, np.nan)

        ew_thresholds=hits.ew_threshold.values



        c_counter = 0
        Fval_merged_CL = xr.Dataset()
        Fval_area_merged_CL = xr.Dataset()


        # create mask over cont metrics for pixels with 0 events

        ## Hits, misses, false alarms --> attention: masks area with 0 events
        #hits = hits.where(quality_mask.rr == 0, np.nan)
        #misses = misses.where(quality_mask.rr == 0, np.nan)
        #false_alarms = false_alarms.where(quality_mask.rr == 0, np.nan)
        #correct_negatives = correct_negatives.where(quality_mask.rr == 0, np.nan)

        # n_events = hits + misses
        # n_events = n_events.where(quality_mask.rr == 0, np.nan)
        # n_events = n_events.where(event_count > 0, np.nan)
        # n_events = n_events.rename("n_events")


        ## fraction of hits, misses, false alarms
        hit_fraction = hits / time_steps_total  # not the same as hit rate!
        miss_fraction = misses / time_steps_total
        false_alarm_fraction = (
            false_alarms / time_steps_total
        )  # EQUAL TO FAR! is that true?
        ## hit rate
        hit_rate = hits/(hits + misses)
        hit_rate = hit_rate#.where(event_count > 0, np.nan)

        ## False alarm rate
        FAR = false_alarms / (false_alarms + correct_negatives)
        FAR = FAR#.where(event_count > 0, np.nan)

        # rename the hits, misses, false alarms, correct negatives data-arrays
        hits = hits.rename("hits")
        misses = misses.rename("misses")
        false_alarms = false_alarms.rename("false_alarms")
        correct_negatives = correct_negatives.rename("correct_negatives")

        # save the contingency metrics (hits, misses, false alarms, correct negatives, false alarm rate, false alarm ratio, hit rate) to one xarray dataset


        ###################################################### PEV ##########################################################

        # set costs and losses
        if CL_config == "major":
            C_L_ratios = np.round(
                np.arange(0.1, 1.00, 0.01), 2
            )  # list of C/L ratios
            C_L_ratios = np.insert(C_L_ratios, 0, 0.09)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.085)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.08)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.075)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.07)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.065)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.06)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.055)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.05)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.045)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.04)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.035)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.03)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.025)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.02)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.015)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.01)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.005)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.004)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.003)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.002)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.001)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.0001)

        if CL_config == "minor":
            C_L_ratios = np.round(np.arange(0.1, 1.00, 0.2), 2)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.05)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.03)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.01)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.001)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.0001)
            C_L_ratios = np.insert(C_L_ratios, 0, 0.00001)
        L = 150  # protectable loss, in euro

        C_protection = (
            L * C_L_ratios
        )  ## list of protection costs to create C_L ratios above

        # determine forecast value
        print("start C/L ")
        for C, C_L in zip(C_protection, C_L_ratios):
            L_no_action = Climfreq * L  # note: loss the same for each cell
            latitude = L_no_action.latitude.values
            longitude = L_no_action.longitude.values

            ## create Fval 
            Fval = (
                np.minimum(C_L, Climfreq)
                - (FAR * (1 - Climfreq) * C_L)
                + (hit_rate * Climfreq * (1 - C_L))
                - Climfreq
            ) / (np.minimum(C_L, Climfreq) - (Climfreq * C_L))

            ## process Fval ## otherwise the merge operation below doesn't function
            Fval = (
                Fval.to_dataset(name="Fval")
                .assign_coords(C_L=C_L)
                .expand_dims("lead")
                .expand_dims("C_L")
            )  

            

            ##############################################################################################
            ##################################### Spatial aggregation ####################################
            ##############################################################################################

            event_count_a = event_count.sel(longitude=lon_slice, latitude=lat_slice)
            # calculate contingency metri
            hits_a = hits.sel(longitude=lon_slice, latitude=lat_slice)
            #hits_a = hits_a.where(event_count_a > 0, np.nan)
            misses_a = misses.sel(longitude=lon_slice, latitude=lat_slice)
            #misses_a = misses_a.where(event_count_a > 0, np.nan)
            false_alarms_a = false_alarms.sel(
                longitude=lon_slice, latitude=lat_slice
            )
            #false_alarms_a = false_alarms_a.where(event_count_a > 0, np.nan)
            correct_negatives_a = correct_negatives.sel(
                longitude=lon_slice, latitude=lat_slice
            )
            correct_negatives_a = correct_negatives_a.where(
                event_count_a > 0, np.nan
            )
            ts_total_a = time_steps_total.sel(
                longitude=lon_slice, latitude=lat_slice
            )
            #ts_total_a = ts_total_a.where(event_count_a > 0, np.nan)
            Climfreq_a = Climfreq.sel(longitude=lon_slice, latitude=lat_slice)
            #Climfreq_a = Climfreq_a.where(event_count_a > 0, np.nan)

            # sum stats over area
            hits_t = hits_a.sum(dim=["latitude", "longitude"])
            misses_t = misses_a.sum(dim=["latitude", "longitude"])
            false_alarms_t = false_alarms_a.sum(dim=["latitude", "longitude"])
            correct_negatives_t = correct_negatives_a.sum(
                dim=["latitude", "longitude"]
            )
            ts_total_t = ts_total_a.sum(dim=["latitude", "longitude"])
            Climfreq_t = (hits_t + misses_t) / ts_total_t
            print("climfreq=%s"%(Climfreq_t))
        
            FAR_a = false_alarms_t / (
                false_alarms_t + correct_negatives_t
            )  # area-aggregated FAR. false alarms+correct_negatives=ts_total
            #FAR_a = FAR_a.where(event_count_a > 0, np.nan)
            hit_rate_a = hits_t / (hits_t + misses_t)  # area-aggregated hit rate.
            #hit_rate_a = hit_rate_a.where(event_count_a > 0, np.nan)
            Fval_area = (
                np.minimum(C_L, Climfreq_t)
                - (FAR_a * (1 - Climfreq_t) * C_L)
                + (hit_rate_a * Climfreq_t * (1 - C_L))
                - Climfreq_t
            ) / (np.minimum(C_L, Climfreq_t) - (Climfreq_t * C_L))
            Fval_area = (
                Fval_area.to_dataset(name="Fval")
                .assign_coords(C_L=C_L)
                .expand_dims("lead")
                .expand_dims("C_L")
            )  ## otherwise the merge operation below doesn't function

            #######################  CL Merger ################################
            if c_counter == 0:
                Fval_merged_CL = xr.merge([Fval_merged_CL, Fval])
                Fval_area_merged_CL = xr.merge([Fval_area_merged_CL, Fval_area])

            else:
                Fval_merged_CL = xr.combine_by_coords(
                    [Fval_merged_CL, Fval]
                )  ## skill specific days (all leads).
                Fval_area_merged_CL = xr.combine_by_coords(
                    [Fval_area_merged_CL, Fval_area]
                )  ## skill specific days (all leads).
            c_counter = c_counter + 1

        #######################  ew_threshold merger  ################################
        #Fval_merged_CL = Fval_merged_CL.assign_coords(
        #    ew_threshold=ew_threshold
        #).expand_dims("ew_threshold")
        #Fval_area_merged_CL = Fval_area_merged_CL.assign_coords(
        #    ew_threshold=ew_threshold
        #).expand_dims("ew_threshold")




        # if p_counter == 0:
        #     Fval_merged_p = xr.merge([Fval_merged_p, Fval_merged_CL])
        #     Fval_area_merged_p = xr.merge([Fval_area_merged_p, Fval_area_merged_CL])

        # else:
        #     Fval_merged_p = xr.combine_by_coords(
        #         [Fval_merged_p, Fval_merged_CL]
        #     )  ## skill specific days (all leads).
        #     Fval_area_merged_p = xr.combine_by_coords(
        #         [Fval_area_merged_p, Fval_area_merged_CL]
        #     )  ## skill specific days (all leads).

        # p_counter = p_counter + 1

        #######################  LT merger  ################################
        if lt[0] == "1":  # 1 day lead time
            Fval_merged = xr.merge([Fval_merged, Fval_merged_CL])
            Fval_area_merged = xr.merge([Fval_area_merged, Fval_area_merged_CL])


        else:
            Fval_merged = xr.combine_by_coords(
                [Fval_merged, Fval_merged_CL]
            )  ## skill specific days (all leads).
            Fval_area_merged = xr.combine_by_coords(
                [Fval_area_merged, Fval_area_merged_CL]
            )  ## skill specific days (all leads).


    # attach n_events as map as variable
    n_events_a = event_count.sel(longitude=lon_slice, latitude=lat_slice)
    Fval_area_merged = Fval_area_merged.assign(n_events=n_events_a)
    # attach n_events as attribute
    Fval_area_merged.attrs["n_events"] = (hits_t + misses_t).values
    Fval_area_merged.attrs["lon_slice"] = str(lon_slice)
    Fval_area_merged.attrs["lat_slice"] = str(lat_slice)


    ########################################### only keep pixels with an event ############################################
    Fval_merged = Fval_merged.where(event_count > 0, np.nan)
    Fval_area_merged = Fval_area_merged.where(event_count_a > 0, np.nan)
    #cont_metrics_merged = cont_metrics_merged.where(event_count > 0, np.nan) --> event count filer can only be done on Fval maps! Otherwise the cont metrics will be wrong.




    os.chdir(path_verif)

    ########################################### Save all files (Fval whole Europe, Fval selected area, and contingency metrics) ############################################
    save_string = f'{indicator}_{day_month}_{str(p_threshold).replace(".","")}_S{shift}_seasonal_threshold.nc'
    
    if 'WHOLE_PERIOD' in season:# add whole period to the end of the save string. dont replace but add
        save_string=save_string[:-3]
        save_string=save_string+'_WHOLE_PERIOD.nc'
    print(f"saving nc files with string {save_string}")
    # save files
    Fval_merged.to_netcdf(f"Fval_merged_{save_string}")
    Fval_area_merged.to_netcdf(f"Fval_area_merged_{save_string}")
    cont_metrics_merged.to_netcdf(f"cont_metrics_merged_{save_string}")


print("done")




