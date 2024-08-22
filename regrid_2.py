import os
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import xarray as xr
from netCDF4 import Dataset as netcdf_dataset
from pylab import *
#import cfgrib
import sys
#import metview as mv

# Library
import os
#import iris
from os import listdir
import numpy as np
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import dask
from dask.diagnostics import ProgressBar
import xskillscore as xs
from func_Fval import *
from cfgrib.xarray_to_grib import to_grib
import cfgrib
import psutil 
import metview as mv
from metview import *

# see 

pbar = ProgressBar()
pbar.register()

os.chdir('/scistor/ivm/tbr910/precip_analysis/scripts')
from func_Fval import *

path_base='/scistor/ivm/tbr910/precip_analysis'
path_obs= '/scistor/ivm/tbr910/precip_analysis/obs'
path_forecasts_efi='/scistor/ivm/tbr910/precip_analysis/forecasts_efi'
path_forecasts_sot='/scistor/ivm/tbr910/precip_analysis/forecasts_sot'
path_obs_for= '/scistor/ivm/tbr910/precip_analysis/obs_for'
path_q='/scistor/ivm/tbr910/precip_analysis/quantiles'
path_cont='/scistor/ivm/tbr910/precip_analysis/cont_metrics'
path_efi='/scistor/ivm/tbr910/ECMWF/files_efi/'
path_sot='/scistor/ivm/tbr910/ECMWF/files_sot/'
path_verif= '/scistor/ivm/tbr910/precip_analysis/verif_files'
path_obs_for_new= '/scistor/ivm/tbr910/precip_analysis/obs_for_new_LT'
# set dask to split large chunks 
dask.config.set({"array.slicing.split_large_chunks": True})

target_res=0.25 # 0.125 or 0.25, 1 or 2
grib_nc_conversion=True # True or False

# load all forecasts 


############################################ convert .grb to .nc #############################################
if grib_nc_conversion==True:
    # for efi 
    os.chdir(path_efi)
    file_names= sorted([f for f in os.listdir(path_efi) if f.endswith('.grb')])

    # remove all .nc files in folder for efi and sot    
    for f in os.listdir(path_forecasts_efi):
        if f.endswith(".nc"):
            os.remove(path_forecasts_efi+'/'+f)

    for f in os.listdir(path_forecasts_sot):
        if f.endswith(".nc"):
            os.remove(path_forecasts_sot+'/'+f)
            

    for i in file_names:
        forecast=xr.open_dataset(i, engine='cfgrib')
        # drop surface and number variable 
        forecast=forecast.drop('surface')
        forecast=forecast.drop('number')
        forecast.to_netcdf(path_forecasts_efi+"/%s.nc" %(i[:-4]))

    #for sot 
    os.chdir(path_sot)
    file_names= sorted([f for f in os.listdir(path_sot) if f.endswith('.grb')])
    for i in file_names:
        forecast=xr.open_dataset(i, engine='cfgrib')
        # drop surface and number variable 
        forecast=forecast.drop('surface')
        forecast=forecast.drop('number')
        forecast=forecast.rename({'tpi':'sot'})
        forecast.to_netcdf(path_forecasts_sot+"/%s.nc" %(i[:-4]))



############################################# load ref grid #################################
# load 0.125 degree file
ref_grid= xr.open_dataset(path_base+'/ref_grid_0125.grib', engine='cfgrib')
ref_grid=ref_grid.isel(step=0).isel(number=0).drop('number').drop('time').drop('step').drop('surface').drop('valid_time')

############################################ load precipitation #############################################  
os.chdir(path_obs)
lead_times= ['1 days', '2 days', '3 days', '4 days', '5 days']

# precip=xr.open_dataset('rr_ens_mean_0.1deg_reg_v28.0e.nc')#, chunks={'time': 1})

# #precip=precip.load()

# print ('precip resolution is:'+ str(compute_res(precip))) --> not used?? 


############################################ EFI #############################################
os.chdir(path_forecasts_efi)
file_names= sorted([f for f in os.listdir(path_forecasts_efi) if f.endswith('.nc')])
efi_merged= xr.open_mfdataset(path_forecasts_efi+'/*.nc', parallel=True, chunks={'time':1}) # 0.14 degree lat/lon resolution 
#efi_merged=efi_merged.load()


########################################### SOT #############################################
os.chdir(path_forecasts_sot)
file_names= sorted([f for f in os.listdir(path_forecasts_sot) if f.endswith('.nc')])
sot_merged= xr.open_mfdataset(path_forecasts_sot+'/*.nc', parallel=True, chunks={'time':1}) # 0.14 degree lat/lon resolution


############################################ INTERPOLATE TO TARGET RES 0.25 degrees (equals +_ 17.5km on 50 longitude) #############################################
#https://www.opendem.info/arc2meters.html
#https://metview.readthedocs.io/en/latest/examples/advanced_regrid.html
#https://github.com/ecmwf/cfgrib/issues/18



############################################ efi #############################################
mv_efi=mv.read(path_base+'/efi_merged_int.nc')
#efi_netcdf= xr.open_dataset(path_base+'/efi_merged_int.nc') # OPENING THE SAME FILE AS BOTH MV AND XARRAY DOES RETURN ERROR? 
regrid_efi = mv.regrid(
    grid          = [target_res,target_res],
    interpolation = "linear",
    data          = mv_efi.to_dataset(),
    accuracy      = 40, 
    area          = 73.5/-27/33/45, #  (equivalent to area=E) --> latmax, lonmin, latmin, lonmax
)

# repair regridded file 
regrid_efi=regrid_efi.to_dataset()
regrid_efi= regrid_efi.rename({'tpi':'efi'})
regrid_efi= regrid_efi.drop('surface').drop('number')
# drop attributes
regrid_efi.efi.attrs = {}
regrid_efi.latitude.attrs = {}
regrid_efi.longitude.attrs = {}

# check re-gridding
regrid_efi.isel(time=200).isel(step=1).efi.plot()
# efi_netcdf.isel(time=200).isel(step=0).tpi.plot()

# save regrid_efi to .nc
regrid_efi.to_netcdf(path_base+'/efi_merged_%s.nc'%(str(target_res).replace('.','')))

print ('efi resolution is:'+ str(compute_res(regrid_efi)))



############################################ SOT #############################################
mv_sot=mv.read(path_base+'/sot_merged_int.nc')
#sot_netcdf= xr.open_dataset(path_base+'/sot_merged_int.nc') OPENING THE SAME FILE AS BOTH MV AND XARRAY DOES RETURN ERROR? 
regrid_sot = mv.regrid(
    grid          = [target_res,target_res],
    interpolation = "linear",
    data          = mv_sot.to_dataset(),
    accuracy      = 40, 
    area          = 73.5/-27/33/45, #  (equivalent to area=E) --> latmax, lonmin, latmin, lonmax
)

# repair regridded file 
regrid_sot=regrid_sot.to_dataset()
regrid_sot= regrid_sot.rename({'tpi':'sot'})
regrid_sot= regrid_sot.drop('surface').drop('number')

# drop attributes
regrid_sot.sot.attrs = {}
regrid_sot.latitude.attrs = {}
regrid_sot.longitude.attrs = {}

# check re-gridding
regrid_sot.isel(time=200).isel(step=0).sot.plot()
#sot_netcdf.isel(time=200).isel(step=0).sot.plot()

# save regrid_efi to .nc
regrid_sot.to_netcdf(path_base+'/sot_merged_%s.nc'%(str(target_res).replace('.','')))

print ('sot resolution is:'+ str(compute_res(regrid_sot)))





############################################ RAINFALL #############################################

efi_merged_025=xr.open_dataset(path_base+'/efi_merged_025.nc')
rainfall_025=xr.open_dataset(path_obs+'/rr_ens_mean_0.25deg_reg_v28.0e.nc')
rainfall_025=rainfall_025.sel(time=slice('2000-01-01', '2023-12-31'))
ref_grid= efi_merged_025.isel(step=0).isel(time=0).drop('time').drop('step').drop('valid_time')
rainfall_rg=rainfall_025.interp_like(ref_grid, method='nearest')

rainfall_rg.to_netcdf(path_obs+'/precip_025_V2.nc')
