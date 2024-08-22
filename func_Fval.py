# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:33:21 2022

@author: tbr910
"""

import os
import matplotlib.pyplot as plt
from netCDF4 import Dataset as netcdf_dataset
import numpy as np
import cartopy
from cartopy import config
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import cartopy.mpl.geoaxes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as cm

import xarray as xr
import urllib.request
import requests
import re
import glob
#from bs4 import BeautifulSoup
import datetime as dtmod
import re
import cfgrib
from zipfile import ZipFile 
path = '/scistor/ivm/tbr910/Forecast_action_analysis'

def unzip(file, target_path):
    # loading the temp.zip and creating a zip object 
    with ZipFile(file, 'r') as zObject: 
        # Extracting all the members of the zip  
        # into a specific location. 
        zObject.extractall(path=target_path) 



def compute_res(dataset): 
    xr_lats=dataset.latitude
    xr_lons=dataset.longitude
    lat_res=abs(xr_lats.values[1]-xr_lats.values[0])
    lon_res=abs(xr_lons.values[1]-xr_lons.values[0])
    return lat_res, lon_res

#%% Download data from URL 
def download_data(clim_start_year, clim_end_year, rfe_url, file_path): 

        
        # Define climatological period
        clim_years = np.arange(clim_start_year, clim_end_year + 1, 1)

        # Create empty array to store precip file names


        for yr in clim_years:
            server_filenames=[]
            soup = BeautifulSoup(requests.get(rfe_url + '/%s'%(yr)).text)
            
            for a in soup.findAll(href = re.compile(".tif$")):
                server_filenames.append(a['href'])
            
            for server_filename in server_filenames: 
                
                fname= file_path + "/" + server_filename
                if os.path.isfile(fname) == False:
                    
                    url= rfe_url+"/"+str(yr)+'/'+server_filename
                    print (url, fname)
                    urllib.request.urlretrieve(url, fname)
                
            print ("CHIRPS download in progress...... year:%s" %(yr))            
            
#%%% Mask areas 
def P_mask(p_input,month, resample, p_thres):
    
    ## vars needed to sel months 
    months= (range(1,13))
    months_z= []
    for i in months: 
        j=f"{i:02d}"
        months_z.append(j)  
        
    def month_selector(month_select):
        return (month_select == month_of_interest)  ## can also be a range
                
    mask= p_input.resample(time=resample).sum()
    month_of_interest= int(months_z[month]) ## oct ## convert to int to select month, doesnt work with string
    mask = mask.sel(time=month_selector(mask['time.month']))
    mask=mask.mean(dim='time')
    mask= mask.where(mask.tp>p_thres, 0)
    mask=mask.where(mask.tp ==0, 1)
    
    return mask
    
    

#%% plot netCDF dataset
def plot_xarray_dataset(dataset, var, lat, lon, title, unit_factor, time_aggregation_method): 
    
    ###################################### read data #############################################
    #os.chdir(folder)
    dataset=dataset#make dataset a callable variable
    dataset=dataset.where((dataset.latitude > -4.5) & (dataset.latitude<14.9) & (dataset.longitude > 33.0) &(dataset.longitude < 51) , drop=True)
    if time_aggregation_method=='mean': 
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var]#.mean(dim='step')
        else:
            value = dataset.variables[var]#.mean(dim='time')            
    elif time_aggregation_method=='sum':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].sum(dim='step')
        else:
            value = dataset.variables[var].sum(dim='time')
    elif time_aggregation_method=='median':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].median(dim='step')
        else:
            value = dataset.variables[var].median(dim='time')
    elif time_aggregation_method=='max':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].max(dim='step')
        else:
            value = dataset.variables[var].max(dim='time')
    else: 
        value=dataset.variables[var]
        
    ######################################## plot ################################################   
    fig=plt.figure(figsize=(12,12))
    lats = dataset.variables[lat][:]
    lons = dataset.variables[lon][:]
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plt.contourf(lons,lats, value*unit_factor,30,
                 transform=ccrs.PlateCarree(), vmin=0, vmax=200,cmap=cm.Blues)
    #ax.plot(37.568358,0.341406, w'bo', markersize=7, transform=ccrs.PlateCarree())
    m = plt.cm.ScalarMappable(cmap=cm.Blues)
    m.set_array(value*unit_factor)
    m.set_clim(0., 200.)
    cbar=plt.colorbar(m, boundaries=np.linspace(0, 200, 11))
    cbar.ax.tick_params(labelsize=20) 
    ax.coastlines()
    plt.title(title, size=20)
    
    ax.add_feature(cartopy.feature.BORDERS)
    plt.show()


#%% Facet plot from netCDF   
def plot_xr_facet_seas5(dataset, var, vmin, vmax, TOI, cmap, cbar_label, title):
    dataset=dataset
    country_borders = cfeature.NaturalEarthFeature(
      category='cultural',
      name='‘admin_0_boundary_lines_land',
      scale='50m',
      facecolor='none')
    proj0=ccrs.PlateCarree(central_longitude=0)
    fig=plt.figure(figsize=(20,20))# (W,H)
    
    gs=fig.add_gridspec(3,3,wspace=0,hspace=0.1)
    ax1=fig.add_subplot(gs[0,0],projection=proj0) # 2:,2:
    ax2=fig.add_subplot(gs[0,1],projection=proj0)
    ax3=fig.add_subplot(gs[0,2],projection=proj0)
    ax4=fig.add_subplot(gs[1,0],projection=proj0)
    ax5=fig.add_subplot(gs[1,1],projection=proj0)
    ax6=fig.add_subplot(gs[1,2],projection=proj0)
    ax7=fig.add_subplot(gs[2,0],projection=proj0)

    lead0=dataset.isel(lead=0)[var].plot.pcolormesh(ax=ax1,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead1=dataset.isel(lead=1)[var].plot.pcolormesh(ax=ax2,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead2=dataset.isel(lead=2)[var].plot.pcolormesh(ax=ax3,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead3=dataset.isel(lead=3)[var].plot.pcolormesh(ax=ax4,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead4=dataset.isel(lead=4)[var].plot.pcolormesh(ax=ax5,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead5=dataset.isel(lead=5)[var].plot.pcolormesh(ax=ax6,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
    lead6=dataset.isel(lead=6)[var].plot.pcolormesh(ax=ax7,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
       
 
    
    #%%%ax1 
    lead=str(int(round((float(dataset.isel(lead=0).lead.values)/(2592000000000000)),0)))
    ax1.set_title('lead=%s months'%(lead), size=20)
    gl = ax1.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels=False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS)
    

    #%%%ax2 
    lead=str(int(round((float(dataset.isel(lead=1).lead.values)/(2592000000000000)),0)))
    ax2.set_title('lead=%s months'%(lead), size=20)
    gl = ax2.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels=False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS)
    
    #%%%ax3
    lead=str(int(round((float(dataset.isel(lead=2).lead.values)/(2592000000000000)),0)))
    ax3.set_title('lead=%s months'%(lead), size=20)
    gl = ax3.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.left_labels = False
    gl.bottom_labels=False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax3.coastlines()
    ax3.add_feature(cfeature.BORDERS)
 
    #%%%ax4
    lead=str(int(round((float(dataset.isel(lead=3).lead.values)/(2592000000000000)),0)))
    ax4.set_title('lead=%s months'%(lead), size=20)
    gl = ax4.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels=False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax4.coastlines()
    ax4.add_feature(cfeature.BORDERS)
    #%%%ax5
    lead=str(int(round((float(dataset.isel(lead=4).lead.values)/(2592000000000000)),0)))
    ax5.set_title('lead=%s months'%(lead), size=20)
    
    gl = ax5.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = False
    gl.bottom_labels=True
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax5.coastlines()
    ax5.add_feature(cfeature.BORDERS)
    #%%%ax6
    lead=str(int(round((float(dataset.isel(lead=5).lead.values)/(2592000000000000)),0)))
    ax6.set_title('lead=%s months'%(lead),size=20)
    
    gl = ax6.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.left_labels = False
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax6.coastlines()
    ax6.add_feature(cfeature.BORDERS) 
    #%%%ax7
    lead=str(int(round((float(dataset.isel(lead=6).lead.values)/(2592000000000000)),0)))
    ax7.set_title('lead=%s months'%(lead), size=20)
    gl = ax7.gridlines(crs=proj0, draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = True
    gl.left_labels = True
    gl.xlabel_style = {'color': 'gray'}
    gl.ylabel_style = {'color': 'gray'}
    ax7.coastlines()
    ax7.add_feature(cfeature.BORDERS)
    
    #%%% formatting 
    cax1= fig.add_axes([0.5,0.2,0.3,0.02]) #[left, bottom, width, height]
    cbar=plt.colorbar(lead6,pad=0.05,cax=cax1,orientation='horizontal',cmap=cmap)
    cbar.set_label(label=cbar_label, size='large', weight='bold')
    cbar.ax.tick_params(labelsize=15) 

    plt.suptitle(title, fontsize=15, fontweight='bold')#x=0.54, y=0.1
    fig.tight_layout()
    plt.show()
    fig.savefig(os.path.join(path, 'plots/python/facet_%s_%s.pdf'%(var,TOI)), bbox_inches='tight')






#%% Do we use these plots? 
######################## function to plot netcdf data ##############################
def plot_xarray_array(dataset, lat, lon, title, folder, unit_factor, time_aggregation_method): 
    
    ###################################### read data #############################################
    os.chdir(folder)
    dataset=dataset#make dataset a callable variable
    dataset=dataset.where((dataset.latitude > -4.5) & (dataset.latitude<14.9) & (dataset.longitude > 33.0) &(dataset.longitude < 51) , drop=True)
    if time_aggregation_method=='mean': 
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.mean(dim='step')
        else:
            value = dataset.mean(dim='time')            
    elif time_aggregation_method=='sum':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.sum(dim='step')
        else:
            value = dataset.sum(dim='time')
    elif time_aggregation_method=='median':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.median(dim='step')
        else:
            value = dataset.median(dim='time')
    elif time_aggregation_method=='max':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.max(dim='step')
        else:
            value = dataset.max(dim='time')
    else: 
        value=dataset
        
    ######################################## plot ################################################   
    lats = dataset.latitude
    lons = dataset.longitude
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plt.contourf(lons,lats, value*unit_factor,
                 transform=ccrs.PlateCarree())
    ax.plot(37.568358,0.341406, 'bo', markersize=7, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.title(title, size=10)
    cbar = plt.colorbar()
    ax.add_feature(cartopy.feature.BORDERS)
    plt.show()
 
def plot_xarray_dataset_ICPAC(dataset, var, lat, lon, title, folder, unit_factor, time_aggregation_method): 
    
    ###################################### read data #############################################
    os.chdir(folder)
    dataset=dataset#make dataset a callable variable
    dataset=dataset.where((dataset.lat > -4.5) & (dataset.lat<14.9) & (dataset.lon > 33.0) &(dataset.lon < 51) , drop=True)
    if time_aggregation_method=='mean': 
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].mean(dim='step')
        else:
            value = dataset.variables[var].mean(dim='time')            
    elif time_aggregation_method=='sum':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].sum(dim='step')
        else:
            value = dataset.variables[var].sum(dim='time')
    elif time_aggregation_method=='median':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].median(dim='step')
        else:
            value = dataset.variables[var].median(dim='time')
    elif time_aggregation_method=='max':
        if 'number' in dataset.dims:
            dataset=dataset.median(dim='number')
            value = dataset.variables[var].max(dim='step')
        else:
            value = dataset.variables[var].max(dim='time')
    else: 
        value=dataset.variables[var]
        
    ######################################## plot ################################################   
    lats = dataset.variables[lat][:]
    lons = dataset.variables[lon][:]
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    plt.contourf(lons,lats, value*unit_factor,
                 transform=ccrs.PlateCarree())
    ax.plot(37.568358,0.341406, 'bo', markersize=7, transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.title(title, size=10)
    cbar = plt.colorbar()
    
    ax.add_feature(cartopy.feature.BORDERS)
    plt.show()    
 

