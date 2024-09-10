
#%% packages and paths 

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os
import regionmask
from geopy import geocoders
import geopy.distance
from geopy.geocoders import Nominatim # openstreetmap geocoder
import geoplot as gplt
import geoplot.crs as gcrs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import time
import openpyxl
import seaborn as sns
import colour
from matplotlib.collections import LineCollection
import matplotlib.ticker as mticker
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
import geopandas as gpd

geolocator = Nominatim(user_agent="hornofafrica")

############################################ paths #############################################
home='/scistor/ivm/tbr910/' #\\scistor.vu.nl\shares\BETA-IVM-BAZIS\tbr910\ # /scistor/ivm/tbr910/
path_base= home+'precip_analysis'
path_obs= home+'precip_analysis/obs'
path_forecasts=home+'precip_analysis/forecasts'

path_obs_for= home+'precip_analysis/obs_for'
path_q=home+'precip_analysis/quantiles'
path_cont=home+'precip_analysis/cont_metrics'
path_efi= home+'ECMWF/files_efi/'
path_verif= home+'precip_analysis/verif_files'
path_obs_for_new= home+'precip_analysis/obs_for_new_LT'
path_impact= home+'precip_analysis/risk_impact'
path_figures= home+'/precip_analysis/figures/revisions'
path_support=home+'/precip_analysis/support_files'

#%% 
############################################ Flood event #############################################
flood_event= 'WE' # 'WE', 'IT, 'SL', 'HALLEIN'
plot_europe=False


#%% Load Ahr catchment 
german_catchments= gpd.read_file(path_impact+'//Flussgebietsgrenzen_und_Flusseinzugsgebiete_2963387439123051134/Flusseinzugsgebiet_DE.shp')
ahr_catchment=german_catchments.loc[german_catchments['NAME_500']=='Ahr']

#%% load impacts

##########################################################################################################
############################################ Impact datasets #############################################
##########################################################################################################

# 1. WE floods 2021

if flood_event=='WE':
    # check if emdat_flood.csv already exists in the impact folder
    if os.path.exists(path_impact+'/emdat_flood.csv'):
        emdat_flood=pd.read_csv(path_impact+'/emdat_flood.csv')
    else:
        ################################################ Prepare EMDAT ########################################################
        os.chdir(path_impact)
        emdat= pd.read_excel('emdat.xlsx')

        # select only some columns
        emdat=emdat[['Country', 'Disaster Type', 'Subregion', 'Region', 'Location', 'Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day','No. Affected', 'Total Affected', 'Total Deaths']]
        # types Flood and Storm 
        #emdat=emdat[(emdat['Disaster Type']=='Flood') | (emdat['Disaster Type']=='Storm')]


        # nan to 0 
        emdat=emdat.replace('nan', np.nan)
        emdat['Start Year']=emdat['Start Year'].fillna('01')
        emdat['Start Month']=emdat['Start Month'].fillna('01')
        emdat['Start Day']=emdat['Start Day'].fillna('01')


        # convert 'Start Year', 'Start Month', 'Start Day' to datetime
        emdat['Start Year']=emdat['Start Year'].astype(str)
        emdat['Start Month']=emdat['Start Month'].astype(str)
        emdat['Start Day']=emdat['Start Day'].astype(str)
        emdat['Start Day']=emdat['Start Day'].str.replace('.0', '')
        emdat['Start Month']=emdat['Start Month'].str.replace('.0', '')

        emdat['Start Date']=emdat['Start Year']+'-'+emdat['Start Month']+'-'+emdat['Start Day']

        emdat['Start Date']=pd.to_datetime(emdat['Start Date'], format='%Y-%m-%d')


        ################################################ Select flood event ########################################################
        # SL: 3rd August at around 20.00h and night of 3-4 august



        # select the top 100 largest flood events based on no. affected
        recent_dis= emdat[(emdat['Start Date']>= '2016-01-01') & (emdat['Start Date']<= '2023-12-31')]
        largest_no_aff=recent_dis.sort_values(by=['No. Affected'], ascending=False).head(50)
        largest_no_deaths=recent_dis.sort_values(by=['Total Deaths'], ascending=False).head(50)

        if flood_event=='WE':
            # select only 2021 rainfall events (12-15 july 2021)
            emdat_flood=emdat[(emdat['Start Date']>= '2021-07-11') & (emdat['Start Date']<= '2021-07-16')]
            emdat_flood=emdat_flood[(emdat_flood['Country']=='Germany') | (emdat_flood['Country']=='Belgium') | (emdat_flood['Country']=='Netherlands (Kingdom of the)') | (emdat_flood['Country']=='Luxembourg')] # | (emdat_flood['Country']=='France') | (emdat_flood['Country']=='Austria') (emdat_flood['Country']=='Switzerland') | 

            # rename 'Berchtesgaden (Bavaria); Heilbronn (Baden-Württemberg); Saxony' to comma seperated locations
            emdat_flood['Location']=emdat_flood['Location'].str.replace('Berchtesgaden (Bavaria); Heilbronn (Baden-Württemberg); Saxony', 
                                                                        'Berchtesgaden (Bavaria), Heilbronn (Baden-Württemberg), Saxony')
            
            emdat_flood['Location']=emdat_flood['Location'].str.replace('Saxony-Anhalt; Ahrweiler',
                                                                        'Saxony-Anhalt, Ahrweiler')
            emdat_flood['Location']=emdat_flood['Location'].str.replace('Köln (Rheinland-Pfalz); Märkischer Kreis',
                                                                        'Köln (Rheinland-Pfalz), Märkischer Kreis')
            emdat_flood['Location']=emdat_flood['Location'].str.replace('Rhein-Erf (Nordrhein-Westfalen); Hessen',
                                                                        'Rhein-Erft-Kreis, Hessen')
            emdat_flood['Location']=emdat_flood['Location'].str.replace('La Moselle Departments; Châtel de Houx',
                                                                        'La Moselle') 
            emdat_flood['Location']=emdat_flood['Location'].str.replace('Le Field (Jura); Plainfaing (Vosges); Villers la Chèvre',
                                                                        'Jura, Plainfaing, Villers-la-Chèvre')
            
            emdat_flood['Location']=emdat_flood['Location'].str.replace('Villette (Meurthe en Moselle); Bras-sur-Meuse (Meuse)',
                                                                        'Villette, Bras-sur-Meuse')
            emdat_flood['Location']=emdat_flood['Location'].str.replace(' Limbourg',
                                                                        'Limbourg Belgium')
            emdat_flood['Location']=emdat_flood['Location'].str.replace(' Luxembourg', 
                                                                        'Luxembourg Belgium')
                                                                        
            
            emdat_flood['Location']=emdat_flood['Location'].str.replace('Schwyz and Uri Cantons', 'Schwyz')


        elif flood_event=='SL':
            # select all events from 2023-08-01 to present 
            emdat_flood=emdat[(emdat['Start Date']>= '2023-08-01') & (emdat['Start Date']<= '2024-08-15')]
            emdat_flood= emdat.loc[emdat['Country'].str.contains('Slovenia')]
            emdat_flood=emdat[(emdat['Start Date']>= '2023-08-02') & (emdat['Start Date']<= '2023-08-15')]

            emdat_flood=emdat[(emdat['Start Date']>= '2021-07-13') & (emdat['Start Date']<= '2021-07-15')]

        # put all the Locations in the location column in a new row (duplicate). The locations are seperated by a comma
        emdat_flood['Location']=emdat_flood['Location'].str.split(',')
        emdat_flood=emdat_flood.explode('Location')

        if flood_event=='WE':
            delete_locs= ['Rolleng','La Marne', 'La Meuse', 'La Moselle', 'Thüringen', 'Saxony-Anhalt', 'Berchtesgaden', 'Saxony'] # La Marne flood happened couple of days after (other event). Rolleng cannot be located. 'Thüringen' is now included 
            #delete rows for which delete_locs are in the location string
            emdat_flood=emdat_flood.loc[~emdat_flood['Location'].str.contains('|'.join(delete_locs))]
            # rename 'Schwyz and Uri Cantons' to Schwyz
            

            


        # get lat lon of the 'Location' column, add to emdat_2021. geolocator is the geopy Nominatim geocoder
        emdat_flood['Location']=emdat_flood['Location'].astype(str)       
        emdat_flood['lat']=np.nan
        emdat_flood['lon']=np.nan


        ################################################ Geolocate impacts ########################################################

        # reset index
        emdat_flood=emdat_flood.reset_index(drop=True)
        for i in range(len(emdat_flood)):
            print (i)
            row=emdat_flood.iloc[i]
            if geolocator.geocode(row['Location']) is not None:
                    
                    print (row['Location'], 'not none')
                    location = geolocator.geocode(row['Location'])
                    lat=location.latitude
                    lon=location.longitude
                    emdat_flood.loc[i,'lat']=lat
                    emdat_flood.loc[i,'lon']=lon
            

            else:
                print ('none')
                print (row['Location'])

        

        # save emdat_flood 
        os.chdir(path_impact)
        emdat_flood.to_csv('emdat_flood.csv')

#2. IT floods 2023
IT_floods= gpd.read_file(path_impact+'/IT_flood_damage.shp')

#%% 
################################################################################################################
############################################ LOAD & PROCESS EFI + SOT ##########################################
################################################################################################################

os.chdir(path_obs_for_new)

def get_data(file_name, lon_slice, lat_slice, valid_time):
    data = xr.open_dataset(file_name)
    if plot_europe==False:
        data = data.sel(longitude=lon_slice, latitude=lat_slice)
    efi = data.efi.sel(valid_time=valid_time)
    sot = data.sot.sel(valid_time=valid_time)


    return efi, sot

if flood_event=='WE':
    lon_lat_box_plot= [2.5, 12.5, 46.5, 54.5]   # [lon_min, lon_max, lat_min, lat_max]  --> 2021 floods [2.5, 14, 47.5, 55]  [-10, 20, 39, 55]
    lon_lat_box_analysis= [3.5, 7.8, 48, 52] # large analysis box, covering all impacts: [3.95,7.8,49.3,51.3]. Small analysis box, based on SOT indicator: [5.5,7.8,49.3,51.3] --> was 5.5,8.4,49.3,51.7 before revisions

    lon_slice_plot=slice(lon_lat_box_plot[0], lon_lat_box_plot[1]) # in case of area selection
    lat_slice_plot=slice(lon_lat_box_plot[3], lon_lat_box_plot[2]) # in case of area selection

    


    valid_time = '2021-07-15' # 15 july    --> # 2018-06-02 also very high efi/sot values. 2021-07-15

if flood_event=='IT':
    lon_slice_plot = slice(8.5, 13.5)
    lat_slice_plot = slice(46.5, 42.5)
    valid_time = '2023-05-17' # 16 may 

if flood_event=='HALLEIN':
    lon_slice_plot = slice(12, 14)
    lat_slice_plot = slice(49, 46)
    valid_time = '2021-07-17' # 17 july






efi_L1_event, sot_L1_event = get_data('obs_for_ES_025_L1_S1.nc', lon_slice_plot, lat_slice_plot, valid_time) # 1 day lead
efi_L2_event, sot_L2_event = get_data('obs_for_ES_025_L2_S1.nc', lon_slice_plot, lat_slice_plot, valid_time) # 2 day lead
efi_L3_event, sot_L3_event = get_data('obs_for_ES_025_L3_S1.nc', lon_slice_plot, lat_slice_plot, valid_time) # 3 day lead
efi_L4_event, sot_L4_event = get_data('obs_for_ES_025_L4_S1.nc', lon_slice_plot, lat_slice_plot, valid_time) # 4 day lead
efi_L5_event, sot_L5_event = get_data('obs_for_ES_025_L5_S1.nc', lon_slice_plot, lat_slice_plot, valid_time) # 5 day lead

# attach initialization times 
efi_L1_event['init_time'] = '2021-07-14 00:00:00'
efi_L2_event['init_time'] = '2021-07-13 00:00:00'
efi_L3_event['init_time'] = '2021-07-12 00:00:00'
efi_L4_event['init_time'] = '2021-07-11 00:00:00'
efi_L5_event['init_time'] = '2021-07-10 00:00:00'

sot_L1_event['init_time'] = '2021-07-14 00:00:00'
sot_L2_event['init_time'] = '2021-07-13 00:00:00'
sot_L3_event['init_time'] = '2021-07-12 00:00:00'
sot_L4_event['init_time'] = '2021-07-11 00:00:00'
sot_L5_event['init_time'] = '2021-07-10 00:00:00'



############################################################ Check with Raw ECMWF forecasts straight from MARS ############################################
# efi_L1_event_check=xr.open_dataset(path_support+'/check_forecasts/efi_14july.nc') 
# efi_L2_event_check=xr.open_dataset(path_support+'/check_forecasts/efi_13july.nc') 
# sot_L1_event_check=xr.open_dataset(path_support+'/check_forecasts/sot_14july.nc')
# sot_L2_event_check=xr.open_dataset(path_support+'/check_forecasts/sot_13july.nc')

# # rename tpi to efi and sot
# efi_L1_event_check=efi_L1_event_check.rename({'tpi':'efi'})
# efi_L2_event_check=efi_L2_event_check.rename({'tpi':'efi'})
# sot_L1_event_check=sot_L1_event_check.rename({'tpi':'sot'})
# sot_L2_event_check=sot_L2_event_check.rename({'tpi':'sot'})


# # select area
# efi_L1_event_check= efi_L1_event_check.sel(longitude=lon_slice_plot, latitude=lat_slice_plot)
# efi_L2_event_check= efi_L2_event_check.sel(longitude=lon_slice_plot, latitude=lat_slice_plot)
# sot_L1_event_check= sot_L1_event_check.sel(longitude=lon_slice_plot, latitude=lat_slice_plot)
# sot_L2_event_check= sot_L2_event_check.sel(longitude=lon_slice_plot, latitude=lat_slice_plot)

# efi_L1_event_check=efi_L1_event_check.efi.squeeze('time')
# efi_L2_event_check=efi_L2_event_check.efi.squeeze('time')
# sot_L1_event_check=sot_L1_event_check.sot.squeeze('time')
# sot_L2_event_check=sot_L2_event_check.sot.squeeze('time')

# # substitute
# if check==True:
#     efi_L1_event=efi_L1_event_check.copy()
#     efi_L2_event=efi_L2_event_check.copy()
#     sot_L1_event=sot_L1_event_check.copy()
#     sot_L2_event=sot_L2_event_check.copy()

#     # attach initialization times
#     efi_L1_event['init_time'] = '2021-07-14 00:00:00'
#     efi_L2_event['init_time'] = '2021-07-13 00:00:00'
#     sot_L1_event['init_time'] = '2021-07-14 00:00:00'
#     sot_L2_event['init_time'] = '2021-07-13 00:00:00'







#%%
################################################################################################################
############################################ EFI/SOT PLOT WITH IMPACTS##########################################
################################################################################################################

############################################## Plot EFI+SOT #####################################################

vmin_efi=0
vmax_efi=1

a_e_WE=0.5
a_e_IT=0.0025
a_ahr=0.03
labelsize=17
cmap_sot=plt.cm.get_cmap('YlOrRd', 16)
cmap_efi=plt.cm.get_cmap('YlOrRd', 10)

if flood_event=='WE':
    vmin_sot=0
    vmax_sot=8

if flood_event=='IT':
    vmin_sot=0
    vmax_sot=4

if flood_event=='HALLEIN':
    vmin_sot=0
    vmax_sot=4

fig=plt.figure(figsize=(20,7))# (W,H)
proj0=ccrs.PlateCarree(central_longitude=0)
# 2 rows 5 cols
gs=fig.add_gridspec(2,5,wspace=0.05,hspace=0.2)
ax1=fig.add_subplot(gs[0,0],projection=proj0) # 2:,2:
ax2=fig.add_subplot(gs[0,1],projection=proj0)
ax3=fig.add_subplot(gs[0,2],projection=proj0)
ax4=fig.add_subplot(gs[0,3],projection=proj0)
ax5=fig.add_subplot(gs[0,4],projection=proj0)
ax6=fig.add_subplot(gs[1,0],projection=proj0)
ax7=fig.add_subplot(gs[1,1],projection=proj0)
ax8=fig.add_subplot(gs[1,2],projection=proj0)
ax9=fig.add_subplot(gs[1,3],projection=proj0)
ax10=fig.add_subplot(gs[1,4],projection=proj0)


axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

efi_data = [efi_L1_event, efi_L2_event, efi_L3_event, efi_L4_event, efi_L5_event]
sot_data = [sot_L1_event, sot_L2_event, sot_L3_event, sot_L4_event, sot_L5_event]

plots = []

for i in range(5):
    plot = efi_data[i].plot.pcolormesh(ax=axes[i], transform=ccrs.PlateCarree(central_longitude=0), add_colorbar=False, vmin=vmin_efi, vmax=vmax_efi, cmap=cmap_efi)
    plots.append(plot)
    # title: initialization time: (X days lead)
    if i==0:
        axes[i].set_title(r"$\bf{"+ str(i+1) + '\ day\ lead' + "}$" + '\n' + 'initialization : '+(str(efi_data[i].init_time.values)[:-8]) + ' 00 UTC')
    else:
        axes[i].set_title(r"$\bf{"+ str(i+1) + '\ days\ lead' + "}$" + '\n' + 'initialization : '+(str(efi_data[i].init_time.values)[:-8]) + ' 00 UTC')

    axes[i].coastlines()
    axes[i].add_feature(cfeature.BORDERS)

    # plot lon-lat box analysis as a bounding box
    #axes[i].plot([lon_lat_box_analysis[0], lon_lat_box_analysis[0], lon_lat_box_analysis[1], lon_lat_box_analysis[1], lon_lat_box_analysis[0]], [lon_lat_box_analysis[2], lon_lat_box_analysis[3], lon_lat_box_analysis[3], lon_lat_box_analysis[2], lon_lat_box_analysis[2]], color='black', linewidth=2, transform=ccrs.PlateCarree(central_longitude=0))

for i in range(5, 10):
    plot = sot_data[i-5].plot.pcolormesh(ax=axes[i], transform=ccrs.PlateCarree(central_longitude=0), add_colorbar=False, vmin=vmin_sot, vmax=vmax_sot, cmap=cmap_sot)
    plots.append(plot)
    axes[i].set_title('')
    axes[i].coastlines()
    axes[i].add_feature(cfeature.BORDERS)

    # plot lon-lat box analysis as a bounding box
    #axes[i].plot([lon_lat_box_analysis[0], lon_lat_box_analysis[0], lon_lat_box_analysis[1], lon_lat_box_analysis[1], lon_lat_box_analysis[0]], [lon_lat_box_analysis[2], lon_lat_box_analysis[3], lon_lat_box_analysis[3], lon_lat_box_analysis[2], lon_lat_box_analysis[2]], color='black', linewidth=2, transform=ccrs.PlateCarree(central_longitude=0))


# color_bar 1
cax1= fig.add_axes([0.07,0.55,0.01,0.3]) #[left, bottom, width, height]
cbar=plt.colorbar(plots[0],pad=0.00,cax=cax1,orientation='vertical',cmap=cmap_efi)
cbar.set_label(label=r'EFI$_{precipitation}$', size='20', weight='bold')
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks(np.round(cbar.get_ticks(),2))

# Now you can refer to plot6 as plots[5]
cax2= fig.add_axes([0.07,0.15,0.01,0.3]) #[left, bottom, width, height]
cbar=plt.colorbar(plots[5],pad=0.00,cax=cax2,orientation='vertical',cmap=cmap_sot)
cbar.set_label(label=r'SOT$_{precipitation}$', size='20', weight='bold')
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks(np.round(cbar.get_ticks(),2))


##################################################### PLOT IMPACTS #######################################################
if flood_event=='WE':
    
    ########################### ACTIVATE EM-DAT ################################
    for i in range(len(emdat_flood)):
        print(i)
        row = emdat_flood.iloc[i]
        for ax in axes:
            #ax.scatter(row['lon'], row['lat'], s=100, c='black', marker="x", alpha=a_e_WE, transform=ccrs.PlateCarree(central_longitude=0))
            #blue empty circles
            if plot_europe==False: 
                ax.scatter(row['lon'], row['lat'], s=40, edgecolors='#3f3fd4', facecolors='none', marker="o", linewidths=1,alpha=1, transform=ccrs.PlateCarree(central_longitude=0))
                
            ax.scatter(7.118, 50.544, s=100, facecolors='none', edgecolors='yellow', marker='D', alpha=1, linewidths=1, transform=ccrs.PlateCarree(central_longitude=0))

            #location of Bad neuenahr ahrweiler 
            #ax.scatter(7.1, 50.5, s=100, c='Green', marker='D', alpha=a_ahr, transform=ccrs.PlateCarree(central_longitude=0))

if flood_event=='IT':
    ########################### ACTIVATE COPERNICUS EMERGENCY MANAGEMENT SERVICE ################################
    for ax in axes:
        if plot_europe==False:
            IT_floods.plot(ax=ax, color='yellow', alpha=a_e_IT, transform=ccrs.PlateCarree(central_longitude=0), marker="x", markersize=100)
        #ax.scatter(7.1, 50.5, s=50, c='Green', marker='D', alpha=a_ahr, transform=ccrs.PlateCarree(central_longitude=0))

if flood_event=='HALLEIN':
    for ax in axes:
        if plot_europe==False:
            ax.scatter(13.1, 47.7, s=100, c='Green', marker='D', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))



# Add gridlines and labels to the plots
for ax in axes:
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=mticker.MultipleLocator(3), ylocs=mticker.MultipleLocator(3),  linewidth=0)
    gl.top_labels = False
    gl.left_labels = False
    if ax in [ax6, ax7, ax8, ax9, ax10]:  # second row
        gl.bottom_labels = True
    else:
        gl.bottom_labels = False
    if ax in [ax5, ax10]:  # last column
        gl.right_labels = True
    else:
        gl.right_labels = False



plt.savefig(path_figures+'/efi_sot_impacts_%s.pdf'%flood_event,bbox_inches='tight', dpi=800)
plt.show()


#%% 
##########################################################################################################################
####################################################### CDF PLOTS ########################################################
##########################################################################################################################

lead_time_cdf='1'
sample='pixel' # 'box' or 'pixel'



lon_slice_analysis=slice(lon_lat_box_analysis[0], lon_lat_box_analysis[1]) # in case of area selection
lat_slice_analysis=slice(lon_lat_box_analysis[3], lon_lat_box_analysis[2]) # in case of area selection


############################ calculate CDF ############################

# load forecasts for lead time X
efi_L1=xr.open_dataset(path_obs_for_new+'/obs_for_ES_025_L%s_S1.nc'%(lead_time_cdf)).efi
sot_L1=xr.open_dataset(path_obs_for_new+'/obs_for_ES_025_L%s_S1.nc'%(lead_time_cdf)).sot


############################ country mask ##########################################
countries=gpd.read_file(path_base+'/support_files/eur_countries/world-administrative-boundaries.shp')
country_names=countries.name.to_list()

# make a mask of the countries
c_mask= regionmask.mask_geopandas(countries, efi_L1.longitude.values, efi_L1.latitude.values)

# find index position of United Kingdom 
c_mask=c_mask.rename({'lon': 'longitude','lat': 'latitude'})


# select area 
if sample=='box':
    efi_L1_box= efi_L1.sel(longitude=lon_slice_analysis, latitude=lat_slice_analysis)
    sot_L1_box= sot_L1.sel(longitude=lon_slice_analysis, latitude=lat_slice_analysis)

    # calculate average over area
    efi_L1_roi=efi_L1_box.mean(dim=('longitude', 'latitude'))
    sot_L1_roi=sot_L1_box.mean(dim=('longitude', 'latitude'))

else:
    #efi_L1=efi_L1.where((c_mask==(country_names.index('Belgium')))|(c_mask==(country_names.index('Germany')))|(c_mask==(country_names.index('Netherlands')))|(c_mask==(country_names.index('Luxembourg'))), np.nan)
    #sot_L1=sot_L1.where((c_mask==(country_names.index('Belgium')))|(c_mask==(country_names.index('Germany')))|(c_mask==(country_names.index('Netherlands')))|(c_mask==(country_names.index('Luxembourg'))), np.nan)
    efi_L1_roi= efi_L1.sel(longitude=lon_slice_plot, latitude=lat_slice_plot)
    sot_L1_roi= sot_L1.sel(longitude=lon_slice_plot, latitude=lat_slice_plot)



# flatten
efi_L1_roi_np=efi_L1_roi.values.flatten()
sot_L1_roi_np=sot_L1_roi.values.flatten()

# remove nans
efi_L1_roi_np=efi_L1_roi_np[~np.isnan(efi_L1_roi_np)]
sot_L1_roi_np=sot_L1_roi_np[~np.isnan(sot_L1_roi_np)]

# sort
efi_L1_roi_np=np.sort(efi_L1_roi_np)
sot_L1_roi_np=np.sort(sot_L1_roi_np)

# cdf
cdf_efi=np.arange(len(efi_L1_roi_np))/float(len(efi_L1_roi_np)) # same plot as sns.ecdfplot
cdf_sot=np.arange(len(sot_L1_roi_np))/float(len(sot_L1_roi_np)) # same plot as sns.ecdfplot

############################ calculate event value  ############################
efi_L1_event_roi, sot_L1_event_roi=get_data('obs_for_ES_025_L%s_S1.nc'%(lead_time_cdf), lon_slice=lon_slice_analysis, lat_slice=lat_slice_analysis, valid_time=valid_time)

# find max sot and efi values in the analysis area 
max_sot=sot_L1_event.where(sot_L1_event_roi==sot_L1_event_roi.max(dim=('longitude', 'latitude')).values,drop=True)
max_lon_sot, max_lat_sot=float(max_sot.longitude.values), float(max_sot.latitude.values)


if sample=='pixel': 
    efi_L1_event_roi=efi_L1_event_roi.sel(longitude=max_lon_sot, latitude=max_lat_sot, method='nearest') # 7.118, 50.544
    sot_L1_event_roi=sot_L1_event_roi.sel(longitude=max_lon_sot, latitude=max_lat_sot, method='nearest') # 7.118, 50.544

# flatten
efi_L1_event_roi=efi_L1_event_roi.values.flatten()
sot_L1_event_roi=sot_L1_event_roi.values.flatten()

# remove nans
efi_L1_event_roi=efi_L1_event_roi[~np.isnan(efi_L1_event_roi)]
sot_L1_event_roi=sot_L1_event_roi[~np.isnan(sot_L1_event_roi)]

# mean
if sample=='box':
    efi_event=np.mean(efi_L1_event_roi)
    sot_event=np.mean(sot_L1_event_roi)

if sample=='pixel':
    efi_event=efi_L1_event_roi.copy()
    sot_event=sot_L1_event_roi.copy()

print (f"EFI event value : {efi_event}")
print (f"SOT event value : {sot_event}")

############################ Calculate how extreme these values where ############################
############################# EFI ########################################
# Get the average for 15th July
#average_july_15 = efi_L1_average.sel(valid_time='2021-07-15')

# Count the number of time steps with a higher average
count_higher = (efi_L1_roi > efi_event).sum().values
print(f"Number of time steps with a efi > 14th July: {count_higher}")

# Get the valid_time datestamps where the average is higher than 14th July
dates_higher = efi_L1_roi.where(efi_L1_roi > efi_event, True, np.nan).dropna(dim='valid_time').valid_time.values
print(f"Time steps with a higher EFI average than 14th July: {dates_higher}")

############################# SOT ########################################
# Get the average for 15th July
#average_july_15 = sot_L1_average.sel(valid_time='2021-07-15')

count_higher = (sot_L1_roi > sot_event).sum().values
print(f"Number of time steps or pixels with a sot > 14th July: {count_higher}")

# Get the valid_time datestamps where the average is higher than 14th July
dates_higher = sot_L1_roi.where(sot_L1_roi > sot_event, True, np.nan).dropna(dim='valid_time').valid_time.values
print(f"Time steps with a higher sot average than 14th July: {dates_higher}")

############################ plot ############################################
def gradient_fill(c,x, y, cmap, ax=None, downsampling=1):
    """
    Plot a line with a linear alpha gradient filled above it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    cmap : a matplotlib colormap
        The colormap for the fill.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    downsampling : int
        The factor by which to downsample the data. Only every nth point will be used.
    """
    if ax is None:
        ax = plt.gca()


    
    # Downsample the data
    x = x[::downsampling]
    y = y[::downsampling]

    norm = plt.Normalize(0, c.max())
    # Create a set of polygons and color them according to the colormap
    for i in range(len(x) - 1):
        polygon = Polygon([[x[i], 1], [x[i+1], 1], [x[i+1], y[i+1]], [x[i], y[i]]])
        color = cmap(norm(x[i]))
        ax.add_patch(polygon)
        polygon.set_facecolor(color)
        polygon.set_edgecolor(color)



# # plot cdf's in 1x2 subplot
# fig, ax = plt.subplots(2,1, figsize=(3,2)) 
# # width space between subplots
# fig.subplots_adjust(hspace=0.8, wspace=0.3)

# ####################################################### CMAP ######################################################
# # Define the colors for the colormap
# colors = ["#f7f5a1", "#FFA500", "#FF4500", "#8B0000"]  # darker yellow to dark red

# # Create the colormap
# cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

# ###################################################### plot CDF lines (uncoloured) ######################################################
# ax[0].plot(efi_L1_roi_np, cdf_efi, label='EFI')
# ax[1].plot(sot_L1_roi_np, cdf_sot, label='SOT')

# # xlim
# ax[0].set_xlim(-1, 1)
# ax[1].set_xlim(sot_L1_roi.min()-0.1, sot_L1_roi.max()+0.1)

# # ylim
# ax[0].set_ylim(-0.1, 1.1)
# ax[1].set_ylim(-0.1, 1.1)

# ####################################################### Straight lines on event EFI and SOT ############################################
# norm_efi_mean = plt.Normalize(0, vmax_efi)  # vmax from spatial plot
# color_efi_mean = cmap(norm_efi_mean(efi_event))
# norm_sot_mean = plt.Normalize(0, vmax_sot) # vmax from spatial plot 
# color_sot_mean = cmap(norm_sot_mean(sot_event))


# ax[0].axvline(x=efi_event, color=color_efi_mean, linestyle='--', alpha=1)
# ax[1].axvline(x=sot_event, color=color_sot_mean, linestyle='--', alpha=1)

# ######################################################## Colored lines for all EFI and SOT ############################################
# def interpolate(x, y, num_points):
#     x_new = np.linspace(x.min(), x.max(), num_points)
#     y_new = np.interp(x_new, x, y)
#     return x_new, y_new

# # colored line ax 0
# x, y = ax[0].get_lines()[0].get_data()
# x_new, y_new = interpolate(x, y, len(x) * 2)  # Increase the number of points by 2
# segments = np.array([x_new[:-1], y_new[:-1], x_new[1:], y_new[1:]]).T.reshape(-1, 2, 2)

# lc = LineCollection(segments, cmap=cmap, norm=norm_efi_mean,antialiaseds=False)
# lc.set_array(x_new[:-1])
# lc.set_linewidth(1)
# ax[0].get_lines()[0].remove()
# line = ax[0].add_collection(lc)

# # colored line ax 1
# x, y = ax[1].get_lines()[0].get_data()
# x_new, y_new = interpolate(x, y, len(x) * 2)  # Increase the number of points by 2
# segments = np.array([x_new[:-1], y_new[:-1], x_new[1:], y_new[1:]]).T.reshape(-1, 2, 2)

# lc = LineCollection(segments, cmap=cmap, norm=norm_sot_mean,antialiaseds=False)
# lc.set_array(x_new[:-1])
# lc.set_linewidth(1)
# ax[1].get_lines()[0].remove()
# line = ax[1].add_collection(lc)

######################################################## Gradient fill ############################################
# if sample=='box':
#     gradient_fill(efi_L1_roi,efi_L1_roi_np, cdf_efi, cmap=cmap, ax=ax[0], downsampling=10)
#     gradient_fill(sot_L1_roi,sot_L1_roi_np, cdf_sot, cmap=cmap, ax=ax[1], downsampling=10)
# if sample=='pixel':
#     gradient_fill(efi_L1,efi_L1_roi_np, cdf_efi, cmap=cmap, ax=ax[0], downsampling=100)
#     gradient_fill(sot_L1,sot_L1_roi_np, cdf_sot, cmap=cmap, ax=ax[1], downsampling=100)


# add legend for both subplots at once
#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fancybox=True, shadow=True)

# add black lines 
#ax[0].plot(efi_L1_np, cdf_efi, color='black', linewidth=2)
#ax[1].plot(sot_L1_np, cdf_sot, color='black', linewidth=2)


# add labels
# ax[0].set_xlabel('EFI')
# ax[0].set_ylabel('CDF')
# ax[1].set_xlabel('SOT')
# ax[1].set_ylabel('CDF')

# ax[1].set_xticks([-3,-2, -1, 0, 1, 2,3,4,5,6,7,8])

# # save
# plt.savefig(path_figures+'/cdf_efi_sot_%s.png'%(lead_time_cdf), dpi=1000, bbox_inches='tight')
# plt.show()




#%%
##################################################################################################################################
####################################################### EFI/SOT PLOT WITH ACTION TRIGGERS  ################################################
##################################################################################################################################

##################################################### Process ACTION TRIGGERS #######################################################


# Define the list of C_L values
C_L_values = [0.08,0.18] 
season = 'summer_FINAL_major'
colors_cl = ['black', 'green'] # 'gray']
p_threshold = "5RP"
loader="_FINAL_major"

# Initialize an empty DataFrame to store the combined data
combined_triggers = pd.DataFrame()

# Loop over each C_L value
for C_L in C_L_values:
    # Read the corresponding Excel file
    triggers = pd.read_excel(f'{path_figures}/ew_thresholds_{C_L}_{p_threshold}_{loader}.xlsx', index_col=0)
    
    # Filter the DataFrame for the specified season
    triggers = triggers.loc[triggers['season'] == season]
    
    # Add a column for the C_L value
    triggers['C_L'] = C_L
    
    # Append the filtered DataFrame to the combined DataFrame
    combined_triggers = pd.concat([combined_triggers, triggers])

# Reset the index of the combined DataFrame
combined_triggers.reset_index(inplace=True)

# select season 
combined_triggers=combined_triggers.loc[combined_triggers['season']==season]
# Print the combined DataFrame
print(combined_triggers)



# set extra threshold on PEV value to be displayed (<0 is already not displayed)
# set threshold values of pev<0.2 to np.nan
combined_triggers.loc[combined_triggers['PEV_efi']<0.2, 'ew_threshold_efi'] = np.nan
combined_triggers.loc[combined_triggers['PEV_sot']<0.2, 'ew_threshold_sot'] = np.nan

lead_times=list(combined_triggers['lead'].unique())

############################################## Plot EFI+SOT #####################################################

vmin_efi=0
vmax_efi=1

a_e_WE=0.5
a_e_IT=0.0025
a_ahr=0.03
labelsize=17
cmap_sot=plt.cm.get_cmap('YlOrRd', 16)
cmap_efi=plt.cm.get_cmap('YlOrRd', 10)

if flood_event=='WE':
    vmin_sot=0
    vmax_sot=8

if flood_event=='IT':
    vmin_sot=0
    vmax_sot=4

if flood_event=='HALLEIN':
    vmin_sot=0
    vmax_sot=4

fig=plt.figure(figsize=(20,7))# (W,H)
proj0=ccrs.PlateCarree(central_longitude=0)
# 2 rows 5 cols
gs=fig.add_gridspec(2,5,wspace=0.05,hspace=0.2)
ax1=fig.add_subplot(gs[0,0],projection=proj0) # 2:,2:
ax2=fig.add_subplot(gs[0,1],projection=proj0)
ax3=fig.add_subplot(gs[0,2],projection=proj0)
ax4=fig.add_subplot(gs[0,3],projection=proj0)
ax5=fig.add_subplot(gs[0,4],projection=proj0)
ax6=fig.add_subplot(gs[1,0],projection=proj0)
ax7=fig.add_subplot(gs[1,1],projection=proj0)
ax8=fig.add_subplot(gs[1,2],projection=proj0)
ax9=fig.add_subplot(gs[1,3],projection=proj0)
ax10=fig.add_subplot(gs[1,4],projection=proj0)


axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

efi_data = [efi_L1_event, efi_L2_event, efi_L3_event, efi_L4_event, efi_L5_event]
sot_data = [sot_L1_event, sot_L2_event, sot_L3_event, sot_L4_event, sot_L5_event]

plots = []

for i in range(5):
    plot = efi_data[i].plot.pcolormesh(ax=axes[i], transform=ccrs.PlateCarree(central_longitude=0), add_colorbar=False, vmin=vmin_efi, vmax=vmax_efi, cmap=cmap_efi)
    plots.append(plot)
    # title: initialization time: (X days lead)
    if i==0:
        axes[i].set_title(r"$\bf{"+ str(i+1) + '\ day\ lead' + "}$" + '\n' + 'initialization : '+(str(efi_data[i].init_time.values)[:-8]) + ' 00 UTC')
    else:
        axes[i].set_title(r"$\bf{"+ str(i+1) + '\ days\ lead' + "}$" + '\n' + 'initialization : '+(str(efi_data[i].init_time.values)[:-8]) + ' 00 UTC')

    axes[i].coastlines()
    axes[i].add_feature(cfeature.BORDERS)


    ################################ plot action triggers ################################
    lead_time_triggers = combined_triggers.loc[combined_triggers.lead == lead_times[i]]


    # plot 3 triggers for three C_L values 
    for cl in C_L_values:
        
        ############# plot action triggers ################
        trigger_value=float(lead_time_triggers.loc[lead_time_triggers.C_L == cl].ew_threshold_efi.values[0])


        # Create a mask for pixels greater than the trigger value
        mask = efi_data[i] > trigger_value


        masked_data = efi_data[i].where(mask)
        #masked_data.plot.pcolormesh(ax=axes[i], transform=ccrs.PlateCarree(), cmap='Blues', alpha=1, vmin=vmin_efi, vmax=vmin_efi, add_colorbar=False)
        if cl==0.18:
            line_width=1
        else:
            line_width=1.5
        contour = axes[i].contour(masked_data.longitude, masked_data.latitude, mask, levels=[0.5], colors=colors_cl[C_L_values.index(cl)], linewidths=line_width, transform=ccrs.PlateCarree())




    # plot lon-lat box analysis as a bounding box
    #axes[i].plot([lon_lat_box_analysis[0], lon_lat_box_analysis[0], lon_lat_box_analysis[1], lon_lat_box_analysis[1], lon_lat_box_analysis[0]], [lon_lat_box_analysis[2], lon_lat_box_analysis[3], lon_lat_box_analysis[3], lon_lat_box_analysis[2], lon_lat_box_analysis[2]], color='black', linewidth=2, transform=ccrs.PlateCarree(central_longitude=0))

for i in range(5, 10):
    print(i)
    plot = sot_data[i-5].plot.pcolormesh(ax=axes[i], transform=ccrs.PlateCarree(central_longitude=0), add_colorbar=False, vmin=vmin_sot, vmax=vmax_sot, cmap=cmap_sot)
    plots.append(plot)
    axes[i].set_title('')
    axes[i].coastlines()
    axes[i].add_feature(cfeature.BORDERS)

    ################################ plot action triggers ################################


    ################################ plot action triggers ################################
    lead_time_triggers = combined_triggers.loc[combined_triggers.lead == lead_times[i-5]]


    # plot 3 triggers for three C_L values 
    for cl in C_L_values:
        
        ############# plot action triggers ################
        trigger_value=float(lead_time_triggers.loc[lead_time_triggers.C_L == cl].ew_threshold_sot.values[0])


        # Create a mask for pixels greater than the trigger value
        mask = sot_data[i-5] > trigger_value


        masked_data = sot_data[i-5].where(mask)
        #masked_data.plot.pcolormesh(ax=axes[i], transform=ccrs.PlateCarree(), cmap='Blues', alpha=1, vmin=vmin_efi, vmax=vmin_efi, add_colorbar=False)
        if cl==0.18:
            line_width=1
        else:
            line_width=1.5
        contour = axes[i].contour(masked_data.longitude, masked_data.latitude, mask, levels=[0.5], colors=colors_cl[C_L_values.index(cl)], linewidths=line_width, transform=ccrs.PlateCarree())




    # plot lon-lat box analysis as a bounding box
    #axes[i].plot([lon_lat_box_analysis[0], lon_lat_box_analysis[0], lon_lat_box_analysis[1], lon_lat_box_analysis[1], lon_lat_box_analysis[0]], [lon_lat_box_analysis[2], lon_lat_box_analysis[3], lon_lat_box_analysis[3], lon_lat_box_analysis[2], lon_lat_box_analysis[2]], color='black', linewidth=2, transform=ccrs.PlateCarree(central_longitude=0))


# color_bar 1
cax1= fig.add_axes([0.07,0.55,0.01,0.3]) #[left, bottom, width, height]
cbar=plt.colorbar(plots[0],pad=0.00,cax=cax1,orientation='vertical',cmap=cmap_efi)
cbar.set_label(label=r'EFI$_{precipitation}$', size='20', weight='bold')
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks(np.round(cbar.get_ticks(),2))

# Now you can refer to plot6 as plots[5]
cax2= fig.add_axes([0.07,0.15,0.01,0.3]) #[left, bottom, width, height]
cbar=plt.colorbar(plots[5],pad=0.00,cax=cax2,orientation='vertical',cmap=cmap_sot)
cbar.set_label(label=r'SOT$_{precipitation}$', size='20', weight='bold')
cbar.ax.tick_params(labelsize=15)
cbar.set_ticks(np.round(cbar.get_ticks(),2))

# Create custom legend handles for C/L values
legend_handles = [mlines.Line2D([], [], color=color, linewidth=2, label=f'C/L={cl}') for color, cl in zip(colors_cl, C_L_values)]

# Add the legend to the figure
fig.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(1.05, 0.1), title='C/L values', title_fontsize='15', fontsize='15')

##################################################### PLOT ACTION TRIGGERS #######################################################

##################################################### PLOT ACTION TRIGGERS #######################################################
if flood_event=='WE':
    for ax in axes:
        print(ax)
        #ax.scatter(row['lon'], row['lat'], s=100, c='black', marker="x", alpha=a_e_WE, transform=ccrs.PlateCarree(central_longitude=0))
        #blue empty circles

        #ax.scatter(7.118, 50.544, s=100, facecolors='none', edgecolors='yellow', marker='D', alpha=1, linewidths=1, transform=ccrs.PlateCarree(central_longitude=0))
        
        ############# plot ahr valley ################
        ahr_catchment.plot(ax=ax, edgecolors='yellow', facecolor='none', alpha=1, linewidths=1.2, transform=ccrs.Mercator(), zorder=100)
        

        ########################### ACTIVATE EM-DAT ################################
        for i in range(len(emdat_flood)):
            print(i)
            row = emdat_flood.iloc[i]
    
            if plot_europe==False: 
                ax.scatter(row['lon'], row['lat'], s=40, edgecolors='#3f3fd4', facecolors='none', marker="o", linewidths=1,alpha=1, transform=ccrs.PlateCarree(central_longitude=0))
                
        
        #location of Bad neuenahr ahrweiler 
        #ax.scatter(7.1, 50.5, s=100, c='Green', marker='D', alpha=a_ahr, transform=ccrs.PlateCarree(central_longitude=0))

if flood_event=='IT':
    ########################### ACTIVATE COPERNICUS EMERGENCY MANAGEMENT SERVICE ################################
    for ax in axes:
        if plot_europe==False:
            IT_floods.plot(ax=ax, color='yellow', alpha=a_e_IT, transform=ccrs.PlateCarree(central_longitude=0), marker="x", markersize=100)
        #ax.scatter(7.1, 50.5, s=50, c='Green', marker='D', alpha=a_ahr, transform=ccrs.PlateCarree(central_longitude=0))

if flood_event=='HALLEIN':
    for ax in axes:
        if plot_europe==False:
            ax.scatter(13.1, 47.7, s=100, c='Green', marker='D', alpha=1, transform=ccrs.PlateCarree(central_longitude=0))



# Add gridlines and labels to the plots
for ax in axes:
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=mticker.MultipleLocator(3), ylocs=mticker.MultipleLocator(3),  linewidth=0)
    gl.top_labels = False
    gl.left_labels = False
    if ax in [ax6, ax7, ax8, ax9, ax10]:  # second row
        gl.bottom_labels = True
    else:
        gl.bottom_labels = False
    if ax in [ax5, ax10]:  # last column
        gl.right_labels = True
    else:
        gl.right_labels = False



plt.savefig(path_figures+'/efi_sot_action_%s_%s.pdf'%(flood_event,p_threshold),bbox_inches='tight', dpi=800)
plt.show()

