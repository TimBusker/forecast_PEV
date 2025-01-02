"""
Created on Mon Aug 31 17:04:50 2020

@author: Tim Busker

This scripts calculates the PEV for both the European files as the regional files for the selected ROI. The PEV is calculated over all the C/L ratios and early-warning thresholds. Two different PEV equations are supported. 

if "summer", "aut", "winter" or "spring" are present in the save_annotation, the script will only calculate the PEV for these seasons. If all seasons are run, the merger_seasons.py script will be used to calculate the average PEV over the whole year.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import xarray as xr
from netCDF4 import Dataset as netcdf_dataset
from pylab import *
import geopandas
import regionmask
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
import cfgrib
import geopandas as gpd

pbar = ProgressBar()
pbar.register()


############################################ paths #############################################
path_base = "/scistor/ivm/tbr910/precip_analysis"
path_obs = "/scistor/ivm/tbr910/precip_analysis/obs"
path_forecasts = "/scistor/ivm/tbr910/precip_analysis/forecasts"
path_obs_for = "/scistor/ivm/tbr910/precip_analysis/obs_for"
path_q = "/scistor/ivm/tbr910/precip_analysis/quantiles"
path_cont = "/scistor/ivm/tbr910/precip_analysis/cont_metrics"
path_efi = "/scistor/ivm/tbr910/ECMWF/files_efi/"
path_verif = "/scistor/ivm/tbr910/precip_analysis/verif_files"
path_obs_for_new = "/scistor/ivm/tbr910/precip_analysis/obs_for_new_LT"
path_return_periods = "/scistor/ivm/tbr910/precip_analysis/return_periods_europe"
# set dask to split large chunks
dask.config.set({"array.slicing.split_large_chunks": True})


############################################ Config variables #############################################
resolution = "025"  # 0.125 or 0.25
file_indicator = "ES"  # The files are named ES. This contains ES, sot and efi dims. Used to load obs_for files --> for 1 degree, select 'ES_1'
save_annotation = (
    "aut_FINAL_major"  # _season_description: this selects the season, and is also the string used to save the file. 
)

seasons = {
    "summer": [6, 7, 8],
    "aut": [9, 10, 11],
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
}

# Initialize selected_months to 'all'
i=0
for season, months in seasons.items():
    print(season)
    if season in save_annotation:
        i=1
        selected_months = months
        print(selected_months)
    else: 
        if i==0:
            selected_months = 'all'
        
print(selected_months)

start_date = "2016-03-08"  # implementation of cycle CY41R2 from 32km to 16km res
limit_t = True
CL_config = "major"  # 'minor' or 'major'
EW_config = "major"  # 'minor' or 'major'
q_method = "seasonal"  # 'seasonal' or 'daily'
method = "return_periods"  # 'quantile_extremes' or 'return_periods'
filter_2021=False
alternative_Fval = True  ## Chose between two different Fval Equations.
lead_times = ["1 days", "2 days", "3 days", "4 days", "5 days"]
shift = 1  # then 95/2. was 1?


"""
The different options for lat/lon boxes (ROI) are:
"[2.5, 14, 47.5, 55] --> large area Western Europe 
[3.95,7.8,49.3,51.3] --> Affected by 2021 floods (small) --> this one was used in first submission
[3.5,7.8,48,52] --> Affected by 2021 floods (large) --> this one is used in revisions for paper 
[-10, 20, 39, 55] --> much larger (rondom?) area
[1,7.8,48,52] --> area based on many events
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

p_thresholds = ["5RP","10RP"] # 10yr return period is also ran (not used in paper)
indicators = ["efi", "sot", "ES"]  # ES, efi or sot --> ES is combined efi+sot (not used in paper), if you need efi + sot seperately, the script needs to run twice.

############################################ Start run #############################################
for indicator in indicators:
    print(
        f"start run for {indicator}, {selected_months} months and warning thresholds {p_thresholds} with temporal shift {shift}, {save_annotation}"
    )


    #################################### Set thresholds for EFI, ES, SOT ####################################
    if indicator == "efi":
        if EW_config == "minor":
            ew_thresholds = [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                # 0.91,
                # 0.92,
                # 0.93,
                # 0.94,
                0.95,
                # 0.96,
                # 0.97,
                # 0.98,
                0.99,
                # 1.0,
            ]
        if EW_config == "major":
            ew_thresholds = [0.001, 0.003, 0.005, 0.007, 0.009] + list(
                np.round(np.arange(0.01, 1.00, 0.01), 2)
            )

    if indicator == "ES":
        if EW_config == "minor":
            ew_thresholds = [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.9,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
                5,
            ]
        if EW_config == "major":
            ew_thresholds = (
                [
                    0.000005,
                    0.00001,
                    0.00002,
                    0.00005,
                    0.0001,
                    0.0002,
                    0.0005,
                    0.0007,
                    0.001,
                    0.003,
                    0.005,
                    0.007,
                    0.009,
                ]
                + list(np.round(np.arange(0.01, 4.00, 0.01), 2))
                + [4, 5]
            )

    if indicator == "sot":
        if EW_config == "minor":
            ew_thresholds = [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                1.7,
                1.8,
                1.85,
                1.9,
                1.95,
                2.0,
                2.05,
                2.1,
                2.2,
                2.3,
                2.4,
                2.5,
                2.6,
                2.7,
                2.8,
                2.9,
                3.0,
                3.5,
                4.0,
                4.5,
                5,
            ]
        if EW_config == "major":
            ew_thresholds = (
                [
                    0.000005,
                    0.00001,
                    0.00002,
                    0.00005,
                    0.0001,
                    0.0002,
                    0.0005,
                    0.0007,
                    0.001,
                    0.003,
                    0.005,
                    0.007,
                    0.009,
                ]
                + list(np.round(np.arange(0.01, 4.00, 0.01), 2))
                + [4, 5]
            )

    ########################################### Load precipitation and precipitation threshold datasets ############################################

    # Load precipitation
    os.chdir(path_obs)
    precip = xr.open_dataset("precip_%s_V2.nc" % (resolution))  # made in regrid.py
    precip = precip.sel(longitude=slice(-11, precip.longitude.max().values))

    # keep only wet days
    precip = precip.where(precip.rr > 1, drop=True)

    # Calculate (wet-days) quantile thresholds
    if method == "quantile_extremes":
        if q_method == "seasonal":
            precip = precip.assign_coords(season=precip["time.season"])
            precip["season"] = precip["time"].dt.season
            precip_q = precip.groupby("season").quantile(p_thresholds, dim="time")
        else:
            precip_q = precip.quantile(p_thresholds, dim="time")

        # save to netcdf
        precip_q.to_netcdf(path_q + "/precip_q_%s_%s.nc" % (resolution, q_method))

    if selected_months != 'all': 
        precip=precip.where(precip['time.month'].isin(selected_months), drop=True) # select only the months that are in the selected_months list

    if filter_2021==True:
        days_2021_event=pd.date_range(start='2021-07-14', end='2021-7-16', freq='D').values
        precip=precip.where(~precip.time.isin(days_2021_event), drop=True)
      
    ########################################### Load quality mask  ############################################
    os.chdir(path_base)

    quality_mask = xr.open_dataset(
        path_base + "/quality_mask_%s.nc" % resolution
    )  # includes land-sea and X% nan criteria


    ############################################################# START precipitation threshold loop  ############################################################

    for p_threshold in p_thresholds:
        os.chdir(path_obs_for_new)
        Fval_merged = xr.Dataset()
        cont_metrics_merged = xr.Dataset()
        Fval_area_merged = xr.Dataset()
        for lt in lead_times:
            print("Start LT %s" % (lt[0]))

            if shift != 0:
                obs_for = xr.open_mfdataset(
                    "obs_for_ES_%s_L%s_S%s.nc" % (resolution, lt[0], str(shift)),
                    parallel=True,
                ).load()
            else:
                obs_for = xr.open_mfdataset(
                    "obs_for_ES_%s_L%s_S%s.nc" % (resolution, lt[0], str(shift)),
                    parallel=True,
                ).load()

            if limit_t == True:
                obs_for = obs_for.sel(valid_time=slice(start_date, "2023-12-31"))

            if selected_months != 'all':
                obs_for = obs_for.where(obs_for["valid_time.month"].isin(selected_months), drop=True)

            if filter_2021==True:
                days_2021_event=pd.date_range(start='2021-07-14', end='2021-7-16', freq='D').values
                obs_for=obs_for.where(~obs_for.valid_time.isin(days_2021_event), drop=True)
            # obs_for=obs_for.where(quality_mask.rr==0, np.nan)

            # delete and drop time steps in obs_for that are fully nan in one of the variables (efi,sot,tp_obs). also if tp_obs got values, still, delete the valid_time step if one of the variables is nan over the whole lat/lon dimensions.
            obs_for = obs_for.dropna(dim="valid_time", how="all", subset=["efi"])
            obs_for = obs_for.dropna(dim="valid_time", how="all", subset=["sot"])
            obs_for = obs_for.dropna(dim="valid_time", how="all", subset=["tp_obs"])

            obs_for = obs_for.assign_coords(lead=lt)

            #print(obs_for)

            # select obs event (>Xth percentile)
            if method == "quantile_extremes":
                precip_q = precip_q.interp_like(
                    quality_mask, method="nearest"
                )  # quality mask
                if q_method == "daily":
                    extreme_mm = precip_q.sel(quantile=p_threshold).rr.drop("quantile")

                    obs_event = obs_for.tp_obs.where(obs_for["tp_obs"] > extreme_mm, 0)
                    obs_event = obs_event.where(obs_event == 0, 1)

                if q_method == "seasonal":
                    test = obs_for.assign_coords(season=obs_for["valid_time.season"])
                    test["season"] = test["valid_time"].dt.season
                    extreme_mm = precip_q.sel(quantile=p_threshold).rr.drop("quantile")
                    obs_event = test.groupby("valid_time.season") > extreme_mm
                    obs_event = obs_event.tp_obs
                    obs_event = obs_event.where(obs_event == True, 0)
                    obs_event = obs_event.where(obs_event == 0, 1)

            if method == "return_periods":
                # load return period maps
                precip_q = xr.open_dataset(
                    path_return_periods
                    + f"/precipitation-at-fixed-return-period_europe_e-obs_30-year_{p_threshold[:-2]}-yrs_1989-2018_v1.nc"
                )  # not used for threshold method, but still given to make the c_mask
                precip_q = precip_q.interp_like(
                    quality_mask, method="nearest"
                )  # precip quality mask

                extreme_mm = precip_q[f"r{p_threshold[:-2]}yrrp"]
                obs_event = obs_for.tp_obs.where(obs_for["tp_obs"] > extreme_mm, 0)
                obs_event = obs_event.where(obs_event == 0, 1)

            # event count and clim frequency
            time_steps_total = obs_event.count(
                dim="valid_time"
            )  # how many time steps do we have in the data?
            time_steps_total = time_steps_total.where(
                quality_mask.rr == 0, np.nan
            )  # mask using a land mask

            event_count = obs_event.sum(
                dim="valid_time"
            )  ## how many times do we have extreme precip?
            event_count = event_count.where(quality_mask.rr == 0, np.nan)

            # print unique dates of events
            
            if lt=='1 days':
                obs_event_area=obs_event.sel(longitude=lon_slice, latitude=lat_slice)
                obs_event_area=obs_event_area.max(dim=['latitude', 'longitude'])
                obs_event_area=obs_event_area.where(obs_event_area>0, drop=True)
                obs_event_area_array=np.unique(obs_event_area.valid_time)
                dates_df=pd.DataFrame(obs_event_area_array, columns=['dates with extreme precip'])
                dates_df.to_excel(f'{path_verif}/dates_events_{p_threshold}_{save_annotation}.xlsx')
                print(dates_df)


            #time_steps_total = time_steps_total.where(event_count > 0, np.nan)
            Climfreq = (
                event_count / time_steps_total
            )  # Ratio of drought time steps compared to all time steps in rainfall dataset
            print (Climfreq)
            
            #print("average climfreq is %s" % (Climfreq.mean().values.round(5)))
            # calculate contingency metrics using xs (xskillscore)
            p_counter = 0
            Fval_merged_p = xr.Dataset()
            Fval_area_merged_p = xr.Dataset()
            cont_metrics_merged_p = xr.Dataset()

            if indicator == "ES":
                obs_for["ES"] = obs_for.efi + obs_for.sot

            # mask areas with 0 obs
            #obs_for = obs_for.where(event_count > 0, np.nan)
            #obs_event = obs_event.where(event_count > 0, np.nan)

            for ew_threshold in ew_thresholds:

                c_counter = 0
                Fval_merged_CL = xr.Dataset()
                Fval_area_merged_CL = xr.Dataset()
                cont_metrics = xs.Contingency(
                    obs_event,
                    obs_for[indicator],
                    observation_category_edges=np.array([0, 0.999, 1]),
                    forecast_category_edges=np.array([-10, ew_threshold, 20]),
                    dim="valid_time",
                )  ## this triggers with efi threshold of 33%.

                

                # create mask over cont metrics for pixels with 0 events

                ## Hits, misses, false alarms --> attention: masks area with 0 events
                hits = cont_metrics.hits()
                hits = hits.where(quality_mask.rr == 0, np.nan)
                misses = cont_metrics.misses()
                misses = misses.where(quality_mask.rr == 0, np.nan)
                false_alarms = cont_metrics.false_alarms()
                false_alarms = false_alarms.where(quality_mask.rr == 0, np.nan)
                correct_negatives = cont_metrics.correct_negatives()
                correct_negatives = correct_negatives.where(quality_mask.rr == 0, np.nan)
                
                n_events = hits + misses
                n_events = n_events.where(quality_mask.rr == 0, np.nan)
                #n_events = n_events.where(event_count > 0, np.nan)
                n_events = n_events.rename("n_events")

                # check with manual calculation
                # hits_manual= obs_for[indicator].where(((obs_for[indicator]>=ew_threshold) & (obs_event==1)), 0)
                # hits_manual=hits_manual.where(hits_manual==0, 1)
                # hits_manual=hits_manual.sum(dim='valid_time')
                # hits_manual=hits_manual.where(quality_mask.rr==0, np.nan)

                # misses_manual= obs_for[indicator].where(((obs_for[indicator]<ew_threshold) & (obs_event==1)), 0)
                # misses_manual=misses_manual.where(misses_manual==0, 1)
                # misses_manual=misses_manual.sum(dim='valid_time')
                # misses_manual=misses_manual.where(quality_mask.rr==0, np.nan)

                # false_alarms_manual= obs_for[indicator].where(((obs_for[indicator]>=ew_threshold) & (obs_event==0)), 0)
                # false_alarms_manual=false_alarms_manual.where(false_alarms_manual==0, 1)
                # false_alarms_manual=false_alarms_manual.sum(dim='valid_time')
                # false_alarms_manual=false_alarms_manual.where(quality_mask.rr==0, np.nan)

                # correct_negatives_manual= obs_for[indicator].where(((obs_for[indicator]<ew_threshold) & (obs_event==0)), 0)
                # correct_negatives_manual=correct_negatives_manual.where(correct_negatives_manual==0, 1)
                # correct_negatives_manual=correct_negatives_manual.sum(dim='valid_time')
                # correct_negatives_manual=correct_negatives_manual.where(quality_mask.rr==0, np.nan)

                ## fraction of hits, misses, false alarms
                hit_fraction = hits / time_steps_total  # not the same as hit rate!
                miss_fraction = misses / time_steps_total
                false_alarm_fraction = (
                    false_alarms / time_steps_total
                )  # EQUAL TO FAR! is that true?
                ## hit rate
                hit_rate = cont_metrics.hit_rate()  # hit rate= POD= hits/ (hits+misses)
                #hit_rate = hit_rate.where(event_count > 0, np.nan)
                # manual hit rate
                hit_rate2 = hits / (hits + misses)
                #hit_rate2 = hit_rate2.where(event_count > 0, np.nan)
                ## False alarm rate
                FAR = cont_metrics.false_alarm_rate()  # False alarm rate = FA / (FA+CN)
                #FAR = FAR.where(event_count > 0, np.nan)
                ## False alarm ratio
                FAR_ratio = (
                    cont_metrics.false_alarm_ratio()
                )  # False alarm ratio = FA / (FA+HITS)
                #FAR_ratio = FAR_ratio.where(event_count > 0, np.nan)
                # rename the hits, misses, false alarms, correct negatives data-arrays
                hits = hits.rename("hits")
                misses = misses.rename("misses")
                false_alarms = false_alarms.rename("false_alarms")
                correct_negatives = correct_negatives.rename("correct_negatives")

                # save the contingency metrics (hits, misses, false alarms, correct negatives, false alarm rate, false alarm ratio, hit rate) to one xarray dataset
                cont_metrics_xr = xr.merge(
                    [hits, misses, false_alarms, correct_negatives, n_events]
                )
                # add the ew_threshold
                cont_metrics_xr = cont_metrics_xr.assign_coords(
                    ew_threshold=ew_threshold
                ).expand_dims("ew_threshold")
                # add the lead time
                cont_metrics_xr = cont_metrics_xr.assign_coords(lead=lt).expand_dims("lead")

                ###################################################### PEV ##########################################################

                # set costs and losses
                if CL_config == "major":
                    C_L_ratios = np.round(
                        np.arange(0.1, 1.01, 0.01), 2
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
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.008)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.005)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.004)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0035)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.003)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0025)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.002)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0015)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.001)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0008)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0005)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0003)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0001)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.00008)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.00005)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.00003)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.00001)

                if CL_config == "minor":
                    C_L_ratios = np.round(np.arange(0.1, 1.1, 0.1), 2)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.08)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.05)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.03)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.01)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.008)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.005)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.003)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.001)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0008)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0005)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0003)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.0001)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.00008)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.00005)
                    C_L_ratios = np.insert(C_L_ratios, 0, 0.00003)
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

                    # Step 1: Climatological expense (Ec)
                    Ec = np.minimum(
                        (L_no_action.values), C
                    )  ## climatological expense for the x percentile, calculated as numpy array

                    Ec = xr.DataArray(  ## convert np array back to xarray
                        data=Ec,
                        dims=["latitude", "longitude"],
                        coords=dict(
                            longitude=(longitude),
                            latitude=(latitude),
                        ),
                        attrs=dict(
                            description="Ec",
                        ),
                    )

                    # Step 2: perfect forecast expense (Epf)
                    Epf = Climfreq * C

                    # Step 3: realistic forecast expense (Ef)
                    Ef = (
                        (hit_fraction * C)
                        + (false_alarm_fraction * C)
                        + (miss_fraction * L)
                    )

                    # Step 4: forecast value calculation
                    Fval = (Ec - Ef) / (
                        Ec - Epf
                    )  # ec is climfreq*L for high C/L ratios. Ef= miss_fraction*L for high efi thresholds. Ec-Ef gives small value (max 0.35).

                    # ALTERNATIVE PEV CALCULATION
                    if alternative_Fval == True:

                        Ec = xr.DataArray(  ## convert np array back to xarray
                            data=Ec,
                            dims=["latitude", "longitude"],
                            coords=dict(
                                longitude=(longitude),
                                latitude=(latitude),
                            ),
                            attrs=dict(
                                description="Ec",
                            ),
                        )

                        Fval = (
                            np.minimum(C_L, Climfreq)
                            - (FAR * (1 - Climfreq) * C_L)
                            + (hit_rate * Climfreq * (1 - C_L))
                            - Climfreq
                        ) / (np.minimum(C_L, Climfreq) - (Climfreq * C_L))

                    Fval = (
                        Fval.to_dataset(name="Fval")
                        .assign_coords(C_L=C_L)
                        .expand_dims("lead")
                        .expand_dims("C_L")
                    )  ## otherwise the merge operation below doesn't function

                    ##############################################################################################
                    ##################################### Spatial aggregation ####################################
                    ##############################################################################################

                    event_count_a = event_count.sel(longitude=lon_slice, latitude=lat_slice)
                    # calculate contingency metrics
                    hits_a = hits.sel(longitude=lon_slice, latitude=lat_slice)
                    hits_a = hits_a.where(event_count_a > 0, np.nan)
                    misses_a = misses.sel(longitude=lon_slice, latitude=lat_slice)
                    misses_a = misses_a.where(event_count_a > 0, np.nan)
                    false_alarms_a = false_alarms.sel(
                        longitude=lon_slice, latitude=lat_slice
                    )
                    false_alarms_a = false_alarms_a.where(event_count_a > 0, np.nan)
                    correct_negatives_a = correct_negatives.sel(
                        longitude=lon_slice, latitude=lat_slice
                    )
                    correct_negatives_a = correct_negatives_a.where(
                        event_count_a > 0, np.nan
                    )
                    ts_total_a = time_steps_total.sel(
                        longitude=lon_slice, latitude=lat_slice
                    )
                    ts_total_a = ts_total_a.where(event_count_a > 0, np.nan)
                    Climfreq_a = Climfreq.sel(longitude=lon_slice, latitude=lat_slice) # redundant, but checked and same as Climfreq_t. 
                    Climfreq_a = Climfreq_a.where(event_count_a > 0, np.nan)

                    # sum stats over area
                    hits_t = hits_a.sum(dim=["latitude", "longitude"])
                    misses_t = misses_a.sum(dim=["latitude", "longitude"])
                    false_alarms_t = false_alarms_a.sum(dim=["latitude", "longitude"])
                    correct_negatives_t = correct_negatives_a.sum(
                        dim=["latitude", "longitude"]
                    )
                    ts_total_t = ts_total_a.sum(dim=["latitude", "longitude"])
                    Climfreq_t = (hits_t + misses_t) / ts_total_t

                    

                    FAR_a = false_alarms_t / (
                        false_alarms_t + correct_negatives_t
                    )  # area-aggregated FAR. false alarms+correct_negatives=ts_total
                    #FAR_a = FAR_a.where(event_count_a > 0, np.nan) # dont activate! adds lats/lons
                    hit_rate_a = hits_t / (hits_t + misses_t)  # area-aggregated hit rate.
                    #hit_rate_a = hit_rate_a.where(event_count_a > 0, np.nan) # dont activate! adds lats/lons 
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
                Fval_merged_CL = Fval_merged_CL.assign_coords(
                    ew_threshold=ew_threshold
                ).expand_dims("ew_threshold")
                Fval_area_merged_CL = Fval_area_merged_CL.assign_coords(
                    ew_threshold=ew_threshold
                ).expand_dims("ew_threshold")

                if p_counter == 0:
                    Fval_merged_p = xr.merge([Fval_merged_p, Fval_merged_CL])
                    Fval_area_merged_p = xr.merge([Fval_area_merged_p, Fval_area_merged_CL])
                    cont_metrics_merged_p = xr.merge(
                        [cont_metrics_merged_p, cont_metrics_xr]
                    )
                else:
                    Fval_merged_p = xr.combine_by_coords(
                        [Fval_merged_p, Fval_merged_CL]
                    )  ## skill specific days (all leads).
                    Fval_area_merged_p = xr.combine_by_coords(
                        [Fval_area_merged_p, Fval_area_merged_CL]
                    )  ## skill specific days (all leads).
                    cont_metrics_merged_p = xr.combine_by_coords(
                        [cont_metrics_merged_p, cont_metrics_xr]
                    )  ## skill specific days (all leads).

                p_counter = p_counter + 1

            #######################  LT merger  ################################
            if lt[0] == "1":  # 1 day lead time
                Fval_merged = xr.merge([Fval_merged, Fval_merged_p])
                Fval_area_merged = xr.merge([Fval_area_merged, Fval_area_merged_p])
                cont_metrics_merged = xr.merge([cont_metrics_merged, cont_metrics_merged_p])

            else:
                Fval_merged = xr.combine_by_coords(
                    [Fval_merged, Fval_merged_p]
                )  ## skill specific days (all leads).
                Fval_area_merged = xr.combine_by_coords(
                    [Fval_area_merged, Fval_area_merged_p]
                )  ## skill specific days (all leads).
                cont_metrics_merged = xr.combine_by_coords(
                    [cont_metrics_merged, cont_metrics_merged_p]
                )  ## skill specific days (all leads).

        # attach n_events as map as variable
        n_events_a = n_events.sel(longitude=lon_slice, latitude=lat_slice) # from start of the script 

        Fval_area_merged = Fval_area_merged.assign(n_events=n_events_a)
        # attach n_events as attribute
        Fval_area_merged.attrs["n_events"] = (hits_t + misses_t).values
        Fval_area_merged.attrs["lon_slice"] = str(lon_slice)
        Fval_area_merged.attrs["lat_slice"] = str(lat_slice)

        ########################################### only keep pixels with an event ############################################
        #Fval_merged = Fval_merged.where(event_count > 0, np.nan)
        #Fval_area_merged = Fval_area_merged.where(event_count_a > 0, np.nan)
        cont_metrics_merged = cont_metrics_merged.where(event_count > 0, np.nan) # --> event count filer can only be done on Fval maps! Otherwise the cont metrics will be wrong.

        os.chdir(path_verif)

        ########################################### Save all files (Fval whole Europe, Fval selected area, and contingency metrics) ############################################
        day_month = pd.Timestamp.now().strftime("%d_%m")
        save_string = f'{indicator}_{day_month}_{str(p_threshold).replace(".","")}_S{shift}{save_annotation}.nc'
        print(f"saving nc files with string {save_string}")

        # save files
        Fval_merged.to_netcdf(f"Fval_merged_{save_string}")
        Fval_area_merged.to_netcdf(f"Fval_area_merged_{save_string}")
        cont_metrics_merged.to_netcdf(f"cont_metrics_merged_{save_string}")


    print("done")







