
# #%% SPATIAL PEVmax SUBPLOT 
# os.chdir(path_verif)

# p_threshold=0.9 #(0.99, 0.95, or .99)
# Fval_merged=xr.open_dataset('Fval_merged_1L_%s_S%s.nc'%(str(p_threshold).replace('.',''), shift)).load()
# Fval_max=Fval_merged.max(dim=('ew_threshold', 'C_L'))

# vmin=0
# vmax=1
# # cmap from red to green
# cmap=plt.cm.get_cmap('BuGn', 5)
# cbar_label='PEVmax'
# title='PEV max for quantile %s'%(str(p_threshold)[-2:])
# var='Fval'

# dataset=Fval_max.copy()

# country_borders = cfeature.NaturalEarthFeature(
#     category='cultural',
#     name='‘admin_0_boundary_lines_land',
#     scale='50m',
#     facecolor='none')
# proj0=ccrs.PlateCarree(central_longitude=0)

# fig=plt.figure(figsize=(20,20))# (W,H)
# gs=fig.add_gridspec(3,3,wspace=0,hspace=0.1)
# ax1=fig.add_subplot(gs[0,0],projection=proj0) # 2:,2:
# ax2=fig.add_subplot(gs[0,1],projection=proj0)
# ax3=fig.add_subplot(gs[0,2],projection=proj0)
# ax4=fig.add_subplot(gs[1,0],projection=proj0)
# ax5=fig.add_subplot(gs[1,1],projection=proj0)


# lead0=dataset.isel(lead=0)[var].plot.pcolormesh(ax=ax1,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
# lead1=dataset.isel(lead=1)[var].plot.pcolormesh(ax=ax2,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
# lead2=dataset.isel(lead=2)[var].plot.pcolormesh(ax=ax3,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
# lead3=dataset.isel(lead=3)[var].plot.pcolormesh(ax=ax4,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
# lead4=dataset.isel(lead=4)[var].plot.pcolormesh(ax=ax5,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)


# #%%%ax1 
# lead=str(dataset.isel(lead=0).lead.values)
# ax1.set_title('lead=%s '%(lead), size=20)
# gl = ax1.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = True
# gl.bottom_labels=False
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax1.coastlines()
# ax1.add_feature(cfeature.BORDERS)


# #%%%ax2 
# lead=str(dataset.isel(lead=1).lead.values)
# ax2.set_title('lead=%s '%(lead), size=20)
# gl = ax2.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = False
# gl.bottom_labels=False
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax2.coastlines()
# ax2.add_feature(cfeature.BORDERS)

# #%%%ax3
# lead=str(dataset.isel(lead=2).lead.values)
# ax3.set_title('lead=%s '%(lead), size=20)
# gl = ax3.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = True
# gl.left_labels = False
# gl.bottom_labels=False
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax3.coastlines()
# ax3.add_feature(cfeature.BORDERS)

# #%%%ax4
# lead=str(dataset.isel(lead=3).lead.values)
# ax4.set_title('lead=%s'%(lead), size=20)
# gl = ax4.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = True
# gl.bottom_labels=False
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax4.coastlines()
# ax4.add_feature(cfeature.BORDERS)
# #%%%ax5
# lead=str(dataset.isel(lead=4).lead.values)
# ax5.set_title('lead=%s'%(lead), size=20)

# gl = ax5.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = False
# gl.bottom_labels=True
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax5.coastlines()
# ax5.add_feature(cfeature.BORDERS)

# #%%% formatting 
# cax1= fig.add_axes([0.5,0.2,0.3,0.02]) #[left, bottom, width, height]
# cbar=plt.colorbar(lead4,pad=0.05,cax=cax1,orientation='horizontal',cmap=cmap)
# cbar.set_label(label=cbar_label, size='large', weight='bold')
# cbar.ax.tick_params(labelsize=15) 

# plt.suptitle(title, fontsize=15, fontweight='bold')#x=0.54, y=0.1
# fig.tight_layout()
# plt.show()
#fig.savefig(os.path.join(path, 'plots/python/facet_%s_%s.pdf'%(var,TOI)), bbox_inches='tight')







# ###################################################### plot module ##########################################################

# # plot quantiles
# precip_q.isel(quantile=2).rr.plot.imshow(ax=ax, cmap= "Blues", vmin =1, vmax = 50, transform=ccrs.PlateCarree())
# plt.show()
# plt.close()




# add to master



# stacker_master= xr.Dataset()


#             stacker= xr.merge([stacker, obs_for])

#       #save stacker to .nc 
#       stacker.to_netcdf('/scistor/ivm/tbr910/precip_analysis/obs_for/obs_for_%s.nc'%(ini_date))



# large_dataset= xr.open_mfdataset('/scistor/ivm/tbr910/precip_analysis/obs_for/*.nc', combine='nested', concat_dim='time',chunks={'time':1, 'longitude':705, 'latitude':465, 'valid_time':5})

# large_dataset.to_netcdf('/scistor/ivm/tbr910/precip_analysis/obs_for/obs_for_merged.nc')

# to .nc 




#             tp_obs=P_UP.sel(time=valid_date) ## select tamsat/chirps rainfall for specific days 
#             tp_obs=tp_obs.assign_coords({"valid_time":valid_date})
#             tp_obs=tp_obs.expand_dims('valid_time')
#             tp_obs=tp_obs.drop('time')
#             tp_obs=tp_obs.rename(tp='tp_obs')## create variable for tp observed
        
#             obs_for= xr.merge([seas5_vt,tp_obs])
#             obs_for= obs_for.expand_dims('time')
#             merged_obs_for= xr.merge([merged_obs_for,obs_for])



        









#%%%% obs-for loop 
# for y in years_for_nc: 
#     merged_obs_for_master=xr.Dataset()
#     os.chdir(os.path.join(path, 'ECMWF/SEAS5/DA'))
#     for m in months_z: # select forecast initiation  
#         merged_obs_for=xr.Dataset()
#         print ('Forecast date is: %s %s'%(y,m))
#         seas5=xr.open_dataset('FINAL_ecmwf_seas5_%s-%s-01_DA.nc'%(y, m)) 
#         seas5=seas5*1000 ## rainfall (tp) unit is m! https://apps.ecmwf.int/codes/grib/param-db/?id=228
    
#         # wet days 
#         seas5_wet_days=seas5.where(seas5['tp'] >1, 0)
#         seas5_wet_days=seas5_wet_days.where(seas5_wet_days['tp'] ==0, 1)       
#         if r_indicator=='wet_days':  
#             seas5=seas5_wet_days 
            
    
#         seas5=seas5.resample(valid_time=time_interval).sum() ## resample seas5 to monthly
#         seas5_mean=seas5.drop('surface')#.mean(dim='number') ## select or de-selct this one to switch between deterministic or probabilistic  ## .isel(valid_time=slice(0, 7)) --> possible to select first time steps in lead
#         seas5_mean=seas5_mean.rename(tp='tp_for') ## create variable for tp forecasted
    
#         for i in range(0,len(seas5_mean.valid_time)): ## loop over all leads
#             valid_date=seas5_mean.valid_time[i].values
#             print(valid_date)
#             initiation_date= seas5_mean.time.values
#             lead_time= int(round((float(valid_date-initiation_date)/(2592000000000000)),0))
        
#             seas5_vt=seas5_mean.sel(valid_time=str(valid_date)[0:10]).expand_dims('valid_time')## select seas5 rainfall for specific days        
#             tp_obs=P_UP.sel(time=valid_date) ## select tamsat/chirps rainfall for specific days 
#             tp_obs=tp_obs.assign_coords({"valid_time":valid_date})
#             tp_obs=tp_obs.expand_dims('valid_time')
#             tp_obs=tp_obs.drop('time')
#             tp_obs=tp_obs.rename(tp='tp_obs')## create variable for tp observed
        
#             obs_for= xr.merge([seas5_vt,tp_obs])
#             obs_for= obs_for.expand_dims('time')
#             merged_obs_for= xr.merge([merged_obs_for,obs_for])

#         merged_obs_for_master= xr.merge([merged_obs_for_master,merged_obs_for])

#     os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981') 

#     if r_indicator=='wet_days': 
#         os.chdir(os.path.join(path, 'ECMWF/SEAS5/validation/monthly_1981_wd'))
#     merged_obs_for_master.to_netcdf('obs_for_dataset_%s_%s_ENS.nc'%(r_indicator,y))    


























######################################## plot single lead ########################################
# vmin=0
# vmax=1

# # make variable with the file names, sorted
# file_names= sorted([f for f in os.listdir(path_all_forecasts) if f.endswith('.grb')])

# for i in file_names:
#       forecast=xr.open_dataset(i, engine='cfgrib')

#       forecast_plot=forecast.isel(time=0).isel(step=0).tpi
#       date= str(forecast_plot['time'].values)[:-19]
#       lead= str(int(forecast_plot['step'])/86400000000000)
    
#       fig=plt.figure(figsize=(20,10))

#       lats = forecast_plot['latitude'][:]
#       lons = forecast_plot['longitude'][:]
#       ax = plt.axes(projection=ccrs.PlateCarree())

#       plt.pcolormesh(lons,lats, forecast_plot,
#                   transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax,cmap=cm.Reds)

#       lat_valkenburg= 50.8655
#       lon_valkenburg= 5.7639
#       ax.plot(lon_valkenburg,lat_valkenburg, 'bo', markersize=7, transform=ccrs.PlateCarree())

#       m = plt.cm.ScalarMappable(cmap=cm.Reds)
#       #m.set_array(forecast_plot) add nothings? 
#       m.set_clim(vmin,vmax)
#       cbar=plt.colorbar(m, boundaries=np.linspace(vmin, vmax, 11))
#       #cbar.ax.tick_params(labelsize=20) 
#       ax.coastlines()
#       plt.title('ew_threshold for %s on lead of %s day'%(date, lead), size=20)

#       ax.add_feature(cartopy.feature.BORDERS)

#       plt.show()
#       plt.close()

######################################## plot multiple lead ########################################

# ini_date='2021-07-10'
# vmin=0
# vmax=1
# cmap=cm.Reds


# forecast_ini= forecast.sel(time='%sT00:00:00.000000000'%(ini_date)) # select forecast for initial date
# forecast_ini=forecast_ini.rename({'step':'lead'}) # change name of step to lead
# forecast_ini['lead']= forecast_ini['lead'].astype('timedelta64[D]') # change lead to days
# forecast_ini=forecast_ini.tpi   # select ew_threshold


# country_borders = cfeature.NaturalEarthFeature(
#       category='cultural',
#       name='‘admin_0_boundary_lines_land',
#       scale='50m',
#       facecolor='none')


# fig=plt.figure(figsize=(20,10))
# proj0=ccrs.PlateCarree(central_longitude=0)


# gs=fig.add_gridspec(2,3,wspace=0.1,hspace=0.01)
# ax1=fig.add_subplot(gs[0,0],projection=proj0) # 2:,2:
# ax2=fig.add_subplot(gs[0,1],projection=proj0)
# ax3=fig.add_subplot(gs[0,2],projection=proj0)
# ax4=fig.add_subplot(gs[1,0],projection=proj0)
# ax5=fig.add_subplot(gs[1,1],projection=proj0)


# lead0=forecast_ini.isel(lead=0).plot.pcolormesh(ax=ax1,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
# lead1=forecast_ini.isel(lead=1).plot.pcolormesh(ax=ax2,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
# lead2=forecast_ini.isel(lead=2).plot.pcolormesh(ax=ax3,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
# lead3=forecast_ini.isel(lead=3).plot.pcolormesh(ax=ax4,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)
# lead4=forecast_ini.isel(lead=4).plot.pcolormesh(ax=ax5,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap)



# ### ax1
# lead=str(int(forecast_ini.isel(lead=0)['lead'])/86400000000000)
# ax1.set_title('lead=%s days'%(lead), size=20)
# gl = ax1.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = True
# gl.bottom_labels=False
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax1.coastlines()
# ax1.add_feature(cfeature.BORDERS)

# ### ax2
# lead=str(int(forecast_ini.isel(lead=1)['lead'])/86400000000000)
# ax2.set_title('lead=%s days'%(lead), size=20)
# gl = ax2.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = False
# gl.bottom_labels=False
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax2.coastlines()
# ax2.add_feature(cfeature.BORDERS)

# ### ax3
# lead=str(int(forecast_ini.isel(lead=2)['lead'])/86400000000000)
# ax3.set_title('lead=%s days'%(lead), size=20)
# gl = ax3.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = True
# gl.left_labels = False
# gl.bottom_labels=True
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax3.coastlines()
# ax3.add_feature(cfeature.BORDERS)

# ### ax4
# lead=str(int(forecast_ini.isel(lead=3)['lead'])/86400000000000)
# ax4.set_title('lead=%s days'%(lead), size=20)
# gl = ax4.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')

# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = True
# gl.bottom_labels=True
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax4.coastlines()
# ax4.add_feature(cfeature.BORDERS)


# ### ax5
# lead=str(int(forecast_ini.isel(lead=4)['lead'])/86400000000000)
# ax5.set_title('lead=%s days'%(lead), size=20)
# gl = ax5.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')

# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = False
# gl.bottom_labels=True
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax5.coastlines()
# ax5.add_feature(cfeature.BORDERS)

# m = plt.cm.ScalarMappable(cmap=cm.Reds)
# m.set_clim(vmin, vmax)

# cax1= fig.add_axes([0.4,0.1,0.3,0.02]) #[left, bottom, width, height]
# cbar=plt.colorbar(m,pad=0.05,cax=cax1,orientation='horizontal',cmap=cmap) # cbar=plt.colorbar(m, boundaries=np.linspace(-1, 1, 11))
# cbar.set_label(label='test', size='large', weight='bold')
# cbar.ax.tick_params(labelsize=15) 


# plt.suptitle('ew_threshold for %s'%(ini_date), size=20)

# plt.show()
# #fig.savefig(os.path.join(path, 'plots/python/facet_%s_%s.pdf'%(var,TOI)), bbox_inches='tight')

# #%%% ROC sub-graphs

# #%%%% plot params 
# #1:1 benchmark line 
# x= np.arange(0,1.1,0.1)
# y=np.arange(0,1.1,0.1)

# label_x= (r'$FA_{rate}$')
# label_y= 'Hit rate' 
# label_x_size=25
# label_y_size= 25
# label_fontsize=25

# x_ticks=[0.2,0.8]
# y_ticks=[0.2,0.8]
# tick_size= 25
# title_size=60
# linewidth=5

# AUC_textsize= '17'


# #%%%% ax11
# #%%%%% ROC stats 
# ROC_region=ROC_merged.mean(dim=('latitude','longitude'))
# ROC_lead=ROC_region.isel(lead=0).sel(quantile=selected_quantile) ## select quantile
# TPR=ROC_lead.sel(metric='true positive rate').ROC.values
# FPR=ROC_lead.sel(metric='false positive rate').ROC.values
# prob_labels=np.round(ROC_lead.probability_bin.values, 2)
# #%%%%% plot 
# ax11.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
# ax11.set_xticks(x_ticks)
# ax11.set_yticks(y_ticks)
# ax11.plot(FPR, TPR)
# #%%%%% adjust labels 
# for i, txt in enumerate(prob_labels):
#     ax11.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)
# lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
# ax11.set_xlabel(label_x, size=label_x_size)
# ax11.set_ylabel(label_y, size=label_y_size)    
# ax11.yaxis.set_label_position("right")

# #%%%%% 1:1 benchmark line 
# ax11.plot(x,y, linestyle='--', color='red')

# #%%%%% AUC score 
# AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
# ax11.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

# #%%%% ax22
# #%%%%% ROC stats
# ROC_lead=ROC_region.isel(lead=1).sel(quantile=selected_quantile)
# TPR=ROC_lead.sel(metric='true positive rate').ROC.values
# FPR=ROC_lead.sel(metric='false positive rate').ROC.values
# prob_labels=np.round(ROC_lead.probability_bin.values, 2)
# #%%%%% plot 
# ax22.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
# ax22.set_xticks(x_ticks)
# ax22.set_yticks(y_ticks)
# ax22.plot(FPR, TPR)
# #%%%%% adjust labels 
# for i, txt in enumerate(prob_labels):
#     ax22.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right')     
# lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
# ax22.set_xlabel(label_x, size=label_x_size)
# ax22.set_ylabel(label_y, size=label_y_size) 
# ax22.yaxis.set_label_position("right")
# #%%%%% 1:1 benchmark line  
# ax22.plot(x,y, linestyle='--', color='red')        

# #%%%%% AUC score 
# AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
# ax22.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))       

# #%%%% ax33
# #%%%%% ROC stats
# ROC_lead=ROC_region.isel(lead=2).sel(quantile=selected_quantile)
# TPR=ROC_lead.sel(metric='true positive rate').ROC.values
# FPR=ROC_lead.sel(metric='false positive rate').ROC.values
# prob_labels=np.round(ROC_lead.probability_bin.values, 2)
# #%%%%% plot
# ax33.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
# ax33.set_xticks(x_ticks)
# ax33.set_yticks(y_ticks)
# ax33.plot(FPR, TPR)
# #%%%%% adjust labels
# for i, txt in enumerate(prob_labels):
#     ax33.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)   
# lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
# ax33.set_xlabel(label_x, size=label_x_size)
# ax33.set_ylabel(label_y, size=label_y_size)      
# ax33.yaxis.set_label_position("right")
# #%%%%% 1:1 benchmark line  
# ax33.plot(x,y, linestyle='--', color='red')
# #%%%%% AUC score 
# AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
# ax33.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))          

# #%%%% ax44
# #%%%%% ROC stats
# ROC_lead=ROC_region.isel(lead=3).sel(quantile=selected_quantile)
# TPR=ROC_lead.sel(metric='true positive rate').ROC.values
# FPR=ROC_lead.sel(metric='false positive rate').ROC.values
# prob_labels=np.round(ROC_lead.probability_bin.values, 2)

# #%%%%% plot
# ax44.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
# ax44.set_xticks(x_ticks)
# ax44.set_yticks(y_ticks)
# ax44.plot(FPR, TPR)
# #%%%%% adjust labels 
# for i, txt in enumerate(prob_labels):
#     ax44.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)     
# lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
# ax44.set_xlabel(label_x, size=label_x_size)
# ax44.set_ylabel(label_y, size=label_y_size) 
# ax44.yaxis.set_label_position("right") 
# #%%%%% 1:1 benchmark line  
# ax44.plot(x,y, linestyle='--', color='red')


# #%%%%% AUC scores 
# AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
# ax44.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))         

# #%%%% ax55
# #%%%%% ROC stats
# ROC_lead=ROC_region.isel(lead=4).sel(quantile=selected_quantile)
# TPR=ROC_lead.sel(metric='true positive rate').ROC.values
# FPR=ROC_lead.sel(metric='false positive rate').ROC.values
# prob_labels=np.round(ROC_lead.probability_bin.values, 2)
# #%%%%% plot
# ax55.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
# ax55.set_xticks(x_ticks)
# ax55.set_yticks(y_ticks)
# ax55.plot(FPR, TPR)
# #%%%%% adjust labels 
# for i, txt in enumerate(prob_labels):
#     ax55.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)  
# lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0))) 
# ax55.set_xlabel(label_x, size=label_x_size)
# ax55.set_ylabel(label_y, size=label_y_size)   
# ax55.yaxis.set_label_position("right")
# #%%%%% 1:1 benchmark line  
# ax55.plot(x,y, linestyle='--', color='red')
# #%%%%% AUC scores 
# AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
# ax55.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

# #%%%% ax66
# #%%%%% ROC stats
# ROC_lead=ROC_region.isel(lead=5).sel(quantile=selected_quantile)
# TPR=ROC_lead.sel(metric='true positive rate').ROC.values
# FPR=ROC_lead.sel(metric='false positive rate').ROC.values
# prob_labels=np.round(ROC_lead.probability_bin.values, 2)
# #%%%%% plot
# ax66.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
# ax66.set_xticks(x_ticks)
# ax66.set_yticks(y_ticks)
# ax66.plot(FPR, TPR)
# #%%%%% adjust labels 
# for i, txt in enumerate(prob_labels):
#     ax66.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8) 
# lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
# ax66.set_xlabel(label_x, size=label_x_size)
# ax66.set_ylabel(label_y, size=label_y_size)   
# ax66.yaxis.set_label_position("right") 
# #%%%%% 1:1 benchmark line 
# ax66.plot(x,y, linestyle='--', color='red')    

# #%%%%% AUC scores 
# AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
# ax66.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

# #%%%% ax77
# #%%%%% ROC stats
# ROC_lead=ROC_region.isel(lead=6).sel(quantile=selected_quantile)
# TPR=ROC_lead.sel(metric='true positive rate').ROC.values
# FPR=ROC_lead.sel(metric='false positive rate').ROC.values
# prob_labels=np.round(ROC_lead.probability_bin.values, 2)
# #%%%%% plot 
# ax77.tick_params(left = False, right = True , labelleft = False, labelright = True, labelsize=tick_size)
# ax77.set_xticks(x_ticks)
# ax77.set_yticks(y_ticks)
# ax77.plot(FPR, TPR)
# #%%%%% adjust labels 
# for i, txt in enumerate(prob_labels):
#     ax77.annotate(txt, (FPR[i], TPR[i]+0.02),va="center", ha='right', size=8)         
# lead=str(int(round((float(ROC_lead.lead.values)/(2592000000000000)),0)))
# ax77.set_xlabel(label_x, size=label_x_size)
# ax77.set_ylabel(label_y, size=label_y_size) 
# ax77.yaxis.set_label_position("right")
# #%%%%% 1:1 benchmark line 
# ax77.plot(x,y, linestyle='--', color='red')    

# #%%%%% AUC score 
# AUC_value=str(ROC_lead.sel(metric='area under curve').isel(probability_bin=4).ROC.values.round(2))
# ax77.text(0.6, 0.1, 'AUC=%s'%(AUC_value), horizontalalignment='center', verticalalignment='center', size=AUC_textsize,bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))



# plot country borders
# #%% SPATIAL ROC PLOT 
# ROC_merged=ROC_merged.sel(quantile=selected_quantile).sel(metric='area under curve').isel(probability_bin=1) ## select random probability bin, as the AUC is the same for every bin
# ROC_merged=ROC_merged.where(ROC_merged.ROC>=0.5,np.nan)
# if validation_length=='seasonal': 
#     plot_xr_facet_seas5(ROC_merged,'ROC', 0.5,1, TOI, plt.cm.get_cmap('Reds',5), 'ROC-AUC', 'ROC curve for %s and severity threshold of %s'%(TOI, (str(selected_quantile*100)[:-2]+' %')))
# else: 
#     plot_xr_facet_seas5(ROC_merged,'ROC', 0,1, TOI, plt.cm.get_cmap('coolwarm', 6), 'ROC-AUC', 'ROC curve for %s and severity threshold of %s'%((month_names[int(TOI)]), (str(selected_quantile*100)[:-2]+' %')))

# plt.close()   





############################################## OLD PLOT WITH TWO UPPER FVAL AND LOWER PRECENTILE ########################################
# # open Fval and calculate PEVmax
# os.chdir(path_verif)
# Fval_merged=xr.open_dataset(file_name).load() 
# ew_thresholds=Fval_merged.ew_threshold.values
# ew_thresholds=ew_thresholds.tolist()

# Fval_max=Fval_merged.max(dim=('C_L', 'ew_threshold')).Fval
# precip_q_plot=precip_q.sel(quantile=p_threshold).rr

# # plot params 
# title='test_plot'
# lead_time='1 days'
# vmin=0
# vmax=1
# # red to green colormap
# cmap_F=plt.cm.get_cmap('BrBG', 10)
# cmap_P=plt.cm.get_cmap('Blues', 10)

# # costum ECMWF colormap 
# hex_ecmwf=['#0c0173', '#226be5', '#216305', '#39af07','#edf00f', '#f0ce0f', '#f08c0f', '#f0290f', '#e543e7', '#601a61', '#c0c0c0']
# colors = [matplotlib.colors.to_rgb(i) for i in hex_ecmwf]
# n_bin=len(hex_ecmwf)
# cmap_P = LinearSegmentedColormap.from_list('cmap_ECMWF', colors, N=n_bin)

# # initiate plot 
# fig=plt.figure(figsize=(20,12))# (W,H)
# proj0=ccrs.PlateCarree(central_longitude=0)
# gs=fig.add_gridspec(2,2,wspace=0.5,hspace=0.1)
# ax1=fig.add_subplot(gs[0,0],projection=proj0) # 2:,2:
# ax2=fig.add_subplot(gs[0,1],projection=proj0)
# ax3=fig.add_subplot(gs[1:, :], projection=proj0)

# # ax1
# plot1=Fval_max.isel(lead=0).plot.pcolormesh(ax=ax1,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap_F)
# lead=Fval_max.isel(lead=0).lead.values
# ax1.set_title('lead=%s '%(lead), size=20)
# gl = ax1.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = False
# gl.left_labels = True
# gl.bottom_labels=True
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax1.coastlines()
# ax1.add_feature(cfeature.BORDERS)

# #ax2
# Fval_max.isel(lead=3).plot.pcolormesh(ax=ax2,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,vmin=vmin, vmax=vmax,cmap=cmap_F)
# lead=Fval_max.isel(lead=3).lead.values
# ax2.set_title('lead=%s '%(lead), size=20)
# gl = ax2.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = False
# gl.right_labels = True
# gl.left_labels = False
# gl.bottom_labels=True
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax2.coastlines()
# ax2.add_feature(cfeature.BORDERS)


# #ax3
# if q_method=='seasonal':
#     precip_q_plot=precip_q_plot.mean(dim='season')

# plot2=precip_q_plot.plot.contourf(ax=ax3,transform=ccrs.PlateCarree(central_longitude=0),add_colorbar=False,cmap=cmap_P, vmin=0.5, vmax=50)
# ax3.set_title('%s percentile climatology'%(str(int(p_threshold*100))+'th'), size=20)
# gl = ax3.gridlines(crs=proj0, draw_labels=True,
#                     linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.top_labels = True
# gl.right_labels = True
# gl.left_labels = True
# gl.bottom_labels=True
# gl.xlabel_style = {'color': 'gray'}
# gl.ylabel_style = {'color': 'gray'}
# ax3.coastlines(linewidth=0.5)
# ax3.add_feature(cfeature.BORDERS)


# # add bounding box 
# ax1.add_patch(mpatches.Rectangle(xy=[lon_lat_box[0], lon_lat_box[2]],width=lon_lat_box[1]-lon_lat_box[0],height=lon_lat_box[3]-lon_lat_box[2],linewidth=2,edgecolor='black',facecolor='none',transform=ccrs.PlateCarree(central_longitude=0)))



# # color_bar 1
# cax1= fig.add_axes([0.5,0.55,0.01,0.3]) #[left, bottom, width, height]
# cbar=plt.colorbar(plot1,pad=0.00,cax=cax1,orientation='vertical',cmap=cmap_F)
# cbar.set_label(label='PEV', size='20', weight='bold')
# cbar.ax.tick_params(labelsize=15)
# cbar.set_ticks(np.round(cbar.get_ticks(),2))

# # color bar 2
# cax2= fig.add_axes([0.75,0.15,0.01,0.3]) #[left, bottom, width, height]
# cbar=plt.colorbar(plot2,pad=0.00,cax=cax2,orientation='vertical',cmap=cmap_P)
# cbar.set_label(label='Precipitation (mm/24h)', size='20', weight='bold')
# cbar.ax.tick_params(labelsize=15) 

# # round the colorbar ticks to 1 decimal
# cbar.set_ticks(np.round(cbar.get_ticks(),0))

# #plt.suptitle(title, fontsize=15, fontweight='bold')#x=0.54, y=0.1
# fig.tight_layout()

# plt.show()



# # color bar 2
# cax2= fig.add_axes([0.75,0.15,0.01,0.3]) #[left, bottom, width, height]
# cbar=plt.colorbar(plot2,pad=0.00,cax=cax2,orientation='vertical',cmap=cmap_P)
# cbar.set_label(label='Precipitation (mm/24h)', size='20', weight='bold')
# cbar.ax.tick_params(labelsize=15) 

