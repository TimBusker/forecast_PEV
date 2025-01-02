# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:33:21 2022

@author: Tim Busker 

This script: 

1) Reads the .nc files of E-OBS, and ECMWF EFI and/or SOT
2) Creates obs_for files for each lead time (1-5 days)
3) Saves them as obs_for files for a specific resolution, lead time and shift value: e.g. obs_for_ES_1_L5_S1.nc for 1 degree res, lead time 5 days and shift value 1 day.

Note: mind the 1-day timeshift between ECMWF and E-OBS: 

ECMWF EFI and SOT datestamps represent rainfall over that day. In ECMWF words:"Accumulations are over the hour (the processing period) ending at the validity date/time".
E-OBS datestamps mean rainfall over the next day.

Take rainfall fallen on 2017-04-04: 
- ECMWF will indicate this as 2017-04-04
- E-OBS will indicate this as 2017-04-03

So, to compare the two, we need to shift the E-OBS data by 1 day.

Only Denmark doesnt need a shift. E-OBS datestamps indicate the rainfall fallen on that day. 

"""
import os
import xarray as xr
from pylab import *
#import cfgrib
#import metview as mv
# Library
import os
#import iris
import numpy as np
import pandas as pd
import dask
from dask.diagnostics import ProgressBar
from func_Fval import *
import geopandas as gpd
import regionmask

pbar = ProgressBar()
pbar.register()

os.chdir('/scistor/ivm/tbr910/precip_analysis/scripts')
from func_Fval import *

path_base='/scistor/ivm/tbr910/precip_analysis'
path_obs= '/scistor/ivm/tbr910/precip_analysis/obs'
path_forecasts='/scistor/ivm/tbr910/precip_analysis/forecasts'
path_obs_for= '/scistor/ivm/tbr910/precip_analysis/obs_for'
path_q='/scistor/ivm/tbr910/precip_analysis/quantiles'
path_cont='/scistor/ivm/tbr910/precip_analysis/cont_metrics'
path_efi='/scistor/ivm/tbr910/ECMWF/files_efi/'
path_verif= '/scistor/ivm/tbr910/precip_analysis/verif_files'
path_obs_for_new= '/scistor/ivm/tbr910/precip_analysis/obs_for_new_LT'
path_return_periods='/scistor/ivm/tbr910/precip_analysis/return_periods_europe'
# set dask to split large chunks 
dask.config.set({"array.slicing.split_large_chunks": True})


########################################## set ini vars #############################################
shift_value=1 
resolution='025' # 0125 or 025 or 1

############################################ EFI #############################################
os.chdir(path_base)
efi=xr.open_dataset(path_base+'/efi_merged_%s.nc'%(resolution))
efi=efi.sel(longitude=slice(-11,efi.longitude.max().values))
############################################ SOT #############################################
os.chdir(path_base)
sot=xr.open_dataset(path_base+'/sot_merged_%s.nc'%(resolution))
sot=sot.sel(longitude=slice(-11,sot.longitude.max().values))
############################################ Rainfall #############################################
os.chdir(path_obs)
precip=xr.open_dataset('precip_%s_V2.nc'%(resolution))
precip=precip.sel(longitude=slice(-11,precip.longitude.max().values))
############################################ Quality mask #############################################
precip_spread= xr.open_dataset('rr_ens_spread_0.25deg_reg_v28.0e.nc')

# precip=precip.interp_like(precip_spread, method='linear')
# average_spread=precip_spread.rr.mean(dim='time')
# average_p=precip.rr.mean(dim='time')

# drop based on uncertainty indicator
# uncertainty_indicator=average_spread/average_p
# uncertainty_indicator=uncertainty_indicator.where(uncertainty_indicator<1, 9999) # everything >10 gets 9999
# uncertainty_indicator=uncertainty_indicator.where(uncertainty_indicator==9999, 0) # keep everything ==9999, else 0
# uncertainty_indicator=uncertainty_indicator.where(uncertainty_indicator<9999, 1) # keep everything ==9999, else 0
# uncertainty_indicator.plot()

# drop based on amount of nan time steps 
start_date=precip.time[0].values
end_date=precip.time[-1].values
timesteps=len(pd.date_range(start=start_date, end=end_date, freq='D'))
nan_timesteps=precip.rr.isnull().sum(dim='time')
quality_mask=nan_timesteps/timesteps
quality_mask=quality_mask.where(quality_mask<0.1, 9999) # everything >0.2 gets 9999
quality_mask=quality_mask.where(quality_mask==9999, 0) # keep everything ==9999, else 0
quality_mask=quality_mask.where(quality_mask<9999, 1) # keep everything ==9999, else 0
quality_mask.plot()

quality_mask=quality_mask.sel(longitude=slice(-11,quality_mask.longitude.max().values))
quality_mask.to_netcdf(path_base+'/quality_mask_%s.nc'%(resolution))

############################################ Mask #############################################
os.chdir(path_base)
# create mask where all values that are always nan are set to 0, else 1
ls_mask=precip.isnull().all(dim='time')
ls_mask=ls_mask.sel(longitude=slice(-11,ls_mask.longitude.max().values)) # INTEGRATE IN FORECAST_PREPROC.PY
ls_mask.to_netcdf(path_base+'/land_sea_mask_%s.nc'%(resolution))

############################ country mask ##########################################
countries=gpd.read_file(path_base+'/support_files/eur_countries/world-administrative-boundaries.shp')
country_names=countries.name.to_list()

# make a mask of the countries
c_mask= regionmask.mask_geopandas(countries, precip.longitude.values, precip.latitude.values)

# find index position of United Kingdom 
c_mask=c_mask.rename({'lon': 'longitude','lat': 'latitude'})

########################################### Start Loop ###########################################
lead_times= ['1 days', '2 days', '3 days', '4 days', '5 days']

for lt in lead_times: 
    print ('start lead %s'%(lt[0]))
    # select lead time
    efi_lt=efi.sel(step=lt)
    efi_lt=efi_lt.drop('step')
    efi_lt=efi_lt.swap_dims({'time':'valid_time'})
    efi_lt=efi_lt.drop('time')

    sot_lt=sot.sel(step=lt)
    sot_lt=sot_lt.drop('step')
    sot_lt=sot_lt.swap_dims({'time':'valid_time'})
    sot_lt=sot_lt.drop('time')
    
    # rename precip
    precip_m=precip.rename({'rr':'tp_obs'}) # rename rr to tp_obs
    precip_m=precip_m.rename({'time':'valid_time'}) # rename time to valid_time

    # shift precip by 1 day
    precip_s=precip_m.shift(valid_time=shift_value, fill_value=np.nan).dropna(dim='valid_time', how='all')
    
    ##################### Denmark surgery #####################
    # remove Denmark
    precip_s=precip_s.where(c_mask!=country_names.index('Denmark'), np.nan)

    # isolate Denmark from non-shifted dataset 
    precip_dk=precip_m.where(c_mask==country_names.index('Denmark'), np.nan)
    precip_dk=precip_dk.sel(valid_time=slice(precip_s.valid_time.min(), precip_s.valid_time.max()))

    # merge precip_dk with precip_s
    precip_s=xr.merge([precip_s, precip_dk])

    # take starting date of SOT
    precip_s=precip_s.sel(valid_time=slice(sot_lt.valid_time.min().values, sot_lt.valid_time.max().values)) # select same valid times as efi_lt
        
    # interpolate efi_lt to precip 
    #efi_lt=efi_lt.interp_like(precip_s, method='linear')
    
    # land-sea mask
    efi_lt_masked= efi_lt.where(ls_mask.rr==0, np.nan)
    sot_lt_masked= sot_lt.where(ls_mask.rr==0, np.nan)
    precip_s= precip_s.where(ls_mask.rr==0, np.nan)

    # quality mask 
    efi_lt_masked= efi_lt_masked.where(quality_mask==0, np.nan)
    sot_lt_masked= sot_lt_masked.where(quality_mask==0, np.nan)
    precip_s= precip_s.where(quality_mask==0, np.nan)

    # change dtype to float32 
    #efi_lt_masked=efi_lt_masked.astype('float32') # does this change the res? 
    
    # save to .nc for specific lt 
    obs_for= xr.merge([efi_lt_masked,sot_lt_masked, precip_s])
    obs_for=obs_for.sel(valid_time=slice(sot_lt.valid_time.min().values, sot_lt.valid_time.max().values)) # select same valid times as efi_lt
    # drop if all nan 
    obs_for=obs_for.sel(longitude=slice(-11,obs_for.longitude.max().values))
    obs_for.to_netcdf(path_obs_for_new+'/obs_for_ES_%s_L%s_S%s.nc'%(resolution,lt[0], str(shift_value)))

    print('saved lead %s'%(lt[0]))

print ('done')

