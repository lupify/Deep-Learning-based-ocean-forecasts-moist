import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer

import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
# import netCDF4 as nc
from count_trainable_params import count_parameters
import hdf5storage
import gc
import scipy.signal as signal

torch.manual_seed(0)
np.random.seed(0)

from fourier2D_two_step_moist import FNO2d, \
                                     spectral_loss_channels_sqr,\
                                     spectral_loss_channels_og,\
                                     huber_loss,\
                                     RK4step,\
                                     Eulerstep,\
                                     PECstep,\
                                     directstep

import plotting
import data_utilities as datau
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
import os
from time import time
import yaml
import pprint
from importlib import reload
import itertools
from scipy import fft  

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

integration_methods = {"directstep" : directstep,
  "RK4step" : RK4step,
  "PECstep" : PECstep, 
  "Eulerstep" : Eulerstep}

channels = {
            "psi1" : ["mean", "std"],
            "psi2" : ["mean", "std"],
            "m" : ["mean", "std"],
           }
channel_names = ["psi1","psi2", "m"]

nn_dir = "/media/volume/qgm1/lenny_outputs/models/singleSteps_4-8-24/FNO2D_stepMethod-directstep_lambda-0p05_dataPrep-singleStep/"
moist_dir = '/media/volume/qgm1'
tacs_dir = '/media/volume/qgm1'
moist_loc_151 = f"{moist_dir}/moist_5_daily/151/output.3d.nc"

with open(f"{nn_dir}/model_params.yml", 'r') as h:
  model_params = yaml.load(h, Loader = yaml.Loader)
  
with open(f"{nn_dir}/epoch_losses.pkl", 'rb') as h:
  epoch_losses = pickle.load(h)

data_loc = model_params["data_loc"]
data_prep = model_params["data_prep"]
data_prep_args = model_params["data_prep_args"]
data_mod_loc = model_params["data_mod_loc"]
# nn_dir = model_params["model_dir"]
nn_loc = model_params["model_loc"]
model_name = model_params["model_name"]
num_epochs = model_params["num_epochs"]
lambda_fft = model_params["lambda_fft"]
wavenum_init = model_params["wavenum_init"]
wavenum_init_ydir = model_params["wavenum_init_ydir"]
modes1 = model_params["modes1"]
modes2 = model_params["modes2"]
width = model_params["width"]
batch_size = model_params["batch_size"]
learning_rate = model_params["learning_rate"]
#optimizer_name = model_params["optimizer"]
step_method = model_params["step_method"]
lossFunction = model_params["lossFunction"]

nn_loc = nn_loc.replace("sdc","qgm1")
net = FNO2d(modes1, modes2, width, channels = 3, channelsout = 3).to("cuda")
net.load_state_dict(torch.load(nn_loc))
net = net.eval()

pred_plots_dir = "/media/volume/qgm1/lenny_outputs/models/singleSteps_4-8-24/FNO2D_stepMethod-directstep_lambda-0p05_dataPrep-singleStep/pred_plots_noises/long3"

if not os.path.exists(pred_plots_dir):
  os.mkdir(pred_plots_dir)
  
ts_start = data_prep_args["ts_in"]
ts_start = 1000
autoregsteps = 40000


pred_pkl_loc = f"{pred_plots_dir}/pred.pkl"
actual_pkl_loc = f"{pred_plots_dir}/actual.pkl"
data_loc_mod = data_loc.replace('/home/exouser/nimrodxl1_mymount', tacs_dir)
moists_keep_fno, moists_keep_fno_timestamps, moists_info =  datau.data_load(data_loc_mod)

del moists_keep_fno

if False:
  actual = moists_keep_fno[151][ts_start:autoregsteps+ts_start+1]
  tstamp_start = moists_keep_fno_timestamps[151][ts_start]
  ## singlestep to be saved to autoreg_pred, to compare to actual later on
  autoreg_pred = actual[[0]] ## unseen data

  print(f"Running autoregression {data_prep}...")

  endactual = 8000
  actual = moists_keep_fno[151][0:endactual]
  tstamp_start = 0
  autoreg_pred = np.zeros(shape = (autoregsteps, 128, 128, 3))
  autoreg_pred[0,...] = actual[[0]]

  ## autoregression
  for step in range(1, autoregsteps):
      # grid = net.get_grid(previnput.shape, previnput.device)
      # previnput = torch.cat((previnput, grid), dim=-1)
      output = integration_methods[step_method](net, torch.tensor(autoreg_pred[[step-1],...]).cuda().float()).cpu().detach().numpy()
      autoreg_pred[step,...] = output[0,...]
      if step%1000 == 0:
        print(step, nn_loc)

  if not os.path.exists(pred_plots_dir):
      os.makedirs(pred_plots_dir)

  with open(f"{pred_plots_dir}/pred.pkl","wb") as h:
      pickle.dump(autoreg_pred, h)

  with open(f"{pred_plots_dir}/actual.pkl","wb") as h:
      pickle.dump(actual, h)

else:
  with open(f"{pred_plots_dir}/pred.pkl","rb") as h:
      autoreg_pred = pickle.load(h)

  with open(f"{pred_plots_dir}/actual.pkl","rb") as h:
      actual = pickle.load(h)

def get_accs(preds, truths):
    d2c = truths.mean(axis = 0)
    d1t = preds
    d2t = truths
    num = np.nansum((d1t - d2c)*(d2t - d2c), axis = (1,2))
    den = np.sqrt(np.nansum((d1t - d2c)**2, axis = (1,2)))*np.sqrt(np.nansum((d2t - d2c)**2, axis = (1,2)))
    accs = num/den
    
    return accs

def get_rmse(preds, truths):
    return np.sqrt(np.mean((preds-truths)**2,axis=(1,2)))

## Haley autoregression prediction/truth
preds = autoreg_pred[:actual.shape[0],...]
truths = actual
matfiledata = {}
matfiledata[u'prediction'] = preds
matfiledata[u'Truth'] = truths

accs = get_accs(preds,truths)
rmses = get_rmse(preds,truths)
    
matfiledata[u'ACC'] = accs
matfiledata[u'RMSE'] = rmses
#hdf5storage.write(matfiledata, '.', path_outputs+'predicted_FNO_2D_two_step_loss_eulerstep_3var_level_ocean_spectral_loss_5day_modes_'+str(modes)+'train_wavenumber'+str(wavenum_init)+'lead'+str(lead)+'lambda_'+str(lamda_reg)+'.mat', matlab_compatible=True)
hdf5storage.write(matfiledata, '.', nn_dir+'/autoreg_steps-8000_pred_truth_acc_rmse.mat', matlab_compatible=True)

print("Saved predictions")


gs_dir = f"{pred_plots_dir}/grid_spectrum"
if not os.path.exists(gs_dir):
    os.makedirs(gs_dir)

actual_spectrum = np.abs(fft.rfft(actual[:,:,:,:], axis = 2)[:,:,2:64]).mean(axis = 1).mean(axis = 0)
# tsteps_pred = autoreg_pred.shape[0]
steps_save = [0,1,2,5,20,50,200,500,1000,5000,10000,39999]
for step in steps_save:
    plotting.plot_2d_grid_spectrum(autoreg_pred,
                                   actual_spectrum = actual_spectrum,
                                   channels = channel_names,
                                   frame=step,
                                   savename = f"pred_step-{step}",
                                   output_dir = gs_dir,
                                   title = f"{model_name} autoregressive predictions; moist {151} init",
                                   begframe = tstamp_start)

## velocity/temperature plots...need to be denormalized!! 

Ly = 96
N2 = 128
lats = np.linspace( -Ly / 2, Ly / 2, N2 ,endpoint=False)
ixmin, ixmax = 40, -40

actual_denorm = actual.copy()
autoreg_pred_denorm = autoreg_pred.copy()
actual_denorm[...,0] = actual_denorm[...,0]*moists_info[151]["psi1"]["std"] + moists_info[151]["psi1"]["mean"]
actual_denorm[...,1] = actual_denorm[...,1]*moists_info[151]["psi2"]["std"] + moists_info[151]["psi2"]["mean"]
actual_denorm[...,2] = actual_denorm[...,2]*moists_info[151]["m"]["std"] + moists_info[151]["m"]["mean"]
autoreg_pred_denorm[...,0] = autoreg_pred_denorm[...,0]*moists_info[151]["psi1"]["std"] + moists_info[151]["psi1"]["mean"]
autoreg_pred_denorm[...,1] = autoreg_pred_denorm[...,1]*moists_info[151]["psi2"]["std"] + moists_info[151]["psi2"]["mean"]
autoreg_pred_denorm[...,2] = autoreg_pred_denorm[...,2]*moists_info[151]["m"]["std"] + moists_info[151]["m"]["mean"]


plotting.plot_zonal_velocity_norm_mean(actual_denorm[:,ixmin:ixmax,:,:],  
                                    autoreg_pred_denorm[:,ixmin:ixmax,:,:],
                                    lats = lats[ixmin:ixmax],
                                    channels = ["psi1","psi2"],
                                    dt = .25,
                                    loc = f"{pred_plots_dir}/vel_norm_lat_1.png")

plotting.plot_zonal_velocity_component_mean(actual_denorm[:,ixmin:ixmax,:,:],  
                                    autoreg_pred_denorm[:,ixmin:ixmax,:,:],
                                    lats = lats[ixmin:ixmax],
                                    channels = ["psi1","psi2"],
                                    dt = .25,
                                    loc = f"{pred_plots_dir}/vel_component_lat_1.png")
# actual_temp = (actual_denorm[...,0] - actual_denorm[...,1]).mean(axis=(2,0))
# autoreg_pred_temp = (autoreg_pred_denorm[...,0] - autoreg_pred_denorm[...,1]).mean(axis=(2,0))

plotting.plot_zonal_temp_mean(actual_denorm[:,ixmin:ixmax,:,:],  
                                    autoreg_pred_denorm[:,ixmin:ixmax,:,:],
                                    lats = lats[ixmin:ixmax],
                                    dt = .25,
                                    loc = f"{pred_plots_dir}/temp_lat_1.png")

plotting.plot_grid_temp_mean(actual_denorm,  
                                    autoreg_pred_denorm,
                                    dt = .25,
                                    loc = f"{pred_plots_dir}/temp_grid_1.png")


"""
EOF: grid for each timestep (128x128x40000)-> zonal mean (128x40000) -> SVD U \Sigma V^T(128 x 128)x(128x128)x(128x40000) -> first column of U is mean, 2nd and 3rd
plot first column U should match truth mean SVD
second/third
"""
svd_us = {}
ich = 0
autoreg_pred_denorm_svdForm = autoreg_pred_denorm.transpose(3,1,2,0)[ich,...].mean(axis=1)
Up,Sp,Vhp = np.linalg.svd(autoreg_pred_denorm_svdForm)

actual_denorm_svdForm = actual_denorm.transpose(3,1,2,0)[ich,...].mean(axis=1)
Ua,Sa,Vha = np.linalg.svd(actual_denorm_svdForm)

plotting.plot_eofs(Ua,Up,lats,channel="psi1",loc=f"{pred_plots_dir}/eofs_psi1.png")
svd_us["psi1"] = [Up,Ua]

ich = 1
autoreg_pred_denorm_svdForm = autoreg_pred_denorm.transpose(3,1,2,0)[ich,...].mean(axis=1)
Up,Sp,Vhp = np.linalg.svd(autoreg_pred_denorm_svdForm)

actual_denorm_svdForm = actual_denorm.transpose(3,1,2,0)[ich,...].mean(axis=1)
Ua,Sa,Vha = np.linalg.svd(actual_denorm_svdForm)

plotting.plot_eofs(Ua,Up,lats,channel="psi2",loc=f"{pred_plots_dir}/eofs_psi2.png")
svd_us["psi2"] = [Up,Ua]

ich = 2
autoreg_pred_denorm_svdForm = autoreg_pred_denorm.transpose(3,1,2,0)[ich,...].mean(axis=1)
Up,Sp,Vhp = np.linalg.svd(autoreg_pred_denorm_svdForm)

actual_denorm_svdForm = actual_denorm.transpose(3,1,2,0)[ich,...].mean(axis=1)
Ua,Sa,Vha = np.linalg.svd(actual_denorm_svdForm)

plotting.plot_eofs(Ua,Up,lats,channel="m",loc=f"{pred_plots_dir}/eofs_m.png")
svd_us["m"] = [Up,Ua]

"""
So make PDF for both psi1 and temperature. For that, take the predictions and then remove the time mean of truth from each of the snapshots for both truth and predictions. Basically the snapshots you used to calculate ACC, remember ? Then just take all those snapshots in time, turn them into one big vector and plot histogram. Truth and prediction one on top of the other
4:33
So in the pdf you loose all information about time because you take all the temporal snapshots and convert it into a vector.
"""

autoreg_pred_denorm2 = autoreg_pred_denorm - autoreg_pred_denorm.mean(axis=0)
actual_denorm2 = actual_denorm - autoreg_pred_denorm.mean(axis=0)
autoreg_pred_denorm2_hist = np.histogram(autoreg_pred_denorm2[...,0], bins = 800)
actual_denorm2_hist = np.histogram(actual_denorm2[...,0], bins = 800)

sum = np.sum(autoreg_pred_denorm2_hist[0]*(autoreg_pred_denorm2_hist[1][1:]-autoreg_pred_denorm2_hist[1][:-1]))
autoreg_pred_denorm2_hist_norm = autoreg_pred_denorm2_hist[0]/sum

sum = np.sum(actual_denorm2_hist[0]*(actual_denorm2_hist[1][1:]-actual_denorm2_hist[1][:-1]))
actual_denorm2_hist_norm = actual_denorm2_hist[0]/sum

loc = f"{pred_plots_dir}/pdf_psi1.png"
fig, ax = plt.subplots(1, 1, dpi = 200, figsize = (6,4))
ax.plot((autoreg_pred_denorm2_hist[1][1:]+autoreg_pred_denorm2_hist[1][:-1])/2,autoreg_pred_denorm2_hist_norm, color = "blue", linestyle="-")
ax.plot((actual_denorm2_hist[1][1:]+actual_denorm2_hist[1][:-1])/2,actual_denorm2_hist_norm, color = "black", linestyle="--")
plt.suptitle("psi1")
# ax.set_xlabel("Lattitude")
ax.set_ylabel("Density")
ax.set_xlim(-2,2)
ax.set_yscale("log")
ax.grid(alpha = .8)
plt.savefig(fname=loc, bbox_inches='tight')
plt.close()


autoreg_pred_denorm_temp = autoreg_pred_denorm[...,0] - autoreg_pred_denorm[...,1]
actual_denorm_temp = actual_denorm[...,0] - actual_denorm[...,1]
autoreg_pred_denorm_temp2 = autoreg_pred_denorm_temp - actual_denorm_temp.mean(axis=0)
actual_denorm_temp2 = actual_denorm_temp - actual_denorm_temp.mean(axis=0)

autoreg_pred_denorm_temp2_hist = np.histogram(autoreg_pred_denorm_temp2, bins = 800)
actual_denorm_temp2_hist = np.histogram(actual_denorm_temp2, bins = 800)

sum = np.sum(autoreg_pred_denorm_temp2_hist[0]*(autoreg_pred_denorm_temp2_hist[1][1:]-autoreg_pred_denorm_temp2_hist[1][:-1]))
autoreg_pred_denorm2_hist_norm = autoreg_pred_denorm_temp2_hist[0]/sum

sum = np.sum(actual_denorm_temp2_hist[0]*(actual_denorm_temp2_hist[1][1:]-actual_denorm_temp2_hist[1][:-1]))
actual_denorm2_hist_norm = actual_denorm_temp2_hist[0]/sum

loc = f"{pred_plots_dir}/pdf_temp.png"
fig, ax = plt.subplots(1, 1, dpi = 200, figsize = (6,4))
ax.plot((autoreg_pred_denorm_temp2_hist[1][1:]+autoreg_pred_denorm_temp2_hist[1][:-1])/2,autoreg_pred_denorm2_hist_norm, color = "blue", linestyle="-")
ax.plot((actual_denorm_temp2_hist[1][1:]+actual_denorm_temp2_hist[1][:-1])/2,actual_denorm2_hist_norm, color = "black", linestyle="--")
plt.suptitle(f"$\psi_1 - \psi_2$")
# ax.set_xlabel("")
ax.set_ylabel("Density")
ax.set_xlim(-2,2)
ax.set_yscale("log")
ax.grid(alpha = .8)
plt.savefig(fname=loc, bbox_inches='tight')
plt.close()






long_tsteps = endactual
plotting.plot_rmse(autoreg_pred[:long_tsteps], actual[:long_tsteps], channels = channel_names, loc = f"{pred_plots_dir}/mseVtime_full.png")
plotting.plot_acc(autoreg_pred[:long_tsteps], actual[:long_tsteps], channels = channel_names, loc = f"{pred_plots_dir}/accVtime_full.png")
plotting.plot_spectrums(autoreg_pred[:long_tsteps], actual[:long_tsteps], tsteps = [1,10,100,1000,7999], channels = channel_names, loc = f"{pred_plots_dir}/spectrum_graphs_full.png")
plotting.plot_spectrums2(autoreg_pred[:long_tsteps], actual[:long_tsteps], tsteps = [1,10,100,1000,7999], channels = channel_names, loc = f"{pred_plots_dir}/spectrum_graphs2_full.png")

# 2 weeks
short_tsteps = 14*4
plotting.plot_rmse(autoreg_pred[:short_tsteps], actual[:short_tsteps], channels = channel_names, loc = f"{pred_plots_dir}/mseVtime_short.png")
plotting.plot_acc(autoreg_pred[:short_tsteps], actual[:short_tsteps], channels = channel_names, loc = f"{pred_plots_dir}/accVtime_short.png")
plotting.plot_spectrums(autoreg_pred[:short_tsteps], actual[:short_tsteps], tsteps = np.arange(0, short_tsteps, 8), channels = channel_names, loc = f"{pred_plots_dir}/spectrum_graphs_short.png")
plotting.plot_spectrums2(autoreg_pred[:short_tsteps], actual[:short_tsteps], tsteps = np.arange(0, short_tsteps, 8), channels = channel_names, loc = f"{pred_plots_dir}/spectrum_graphs2_short.png")




## plot saving for animation, predictions
gsp_dir = f"{pred_plots_dir}/pred_pngs"
if not os.path.exists(gsp_dir):
    os.mkdir(gsp_dir)

max_tstep_animation = 40000
tstamp_start = 0
for istep, step in enumerate(np.arange(0,max_tstep_animation,10),0):
    str_step = "0"*(6-len(str(istep)))+str(istep)
    plotting.plot_2d_grid_spectrum(autoreg_pred,
                                   actual_spectrum = actual_spectrum,
                                   channels = channel_names,
                                   frame=step,
                                   savename = f"pred_{str_step}",
                                   output_dir = gsp_dir,
                                   title = f"{model_name} autoregressive predictions; moist {151} init",
                                   cmap = cm.viridis,
                                   begframe = tstamp_start)
    if istep%100:
      print("grid prediction", istep, step)
                                   
os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {gsp_dir}/pred_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {pred_plots_dir}/pred_long.mp4')

## plot saving for animation, predictions - actual
gspa_dir = f"{pred_plots_dir}/pred-actual_pngs"
if not os.path.exists(gspa_dir):
    os.mkdir(gspa_dir)

## plot saving for animation
for istep, step in enumerate(np.arange(0,endactual,10),0):
    str_step = "0"*(6-len(str(istep)))+str(istep)
    plotting.plot_2d_grid_spectrum(autoreg_pred[:endactual]-actual[:endactual],
                                   actual = None,
                                   channels = channel_names,
                                   frame=step,
                                   savename = f"pred-actual_{str_step}",
                                   output_dir = gspa_dir,
                                   title = f"{model_name} autoregressive predictions-actual; moist {151} init",
                                   cmap = cm.bwr,
                                   begframe = tstamp_start)
    if istep%100:
      print("grid prediction", istep, step)
      
os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {gspa_dir}/pred-actual_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {pred_plots_dir}/pred-act_long.mp4')


## plot saving for animation, predictions
gsp_dir = f"{pred_plots_dir}/pred_pngs_short"
if not os.path.exists(gsp_dir):
    os.mkdir(gsp_dir)

max_tstep_animation = 400
for istep, step in enumerate(np.arange(0,max_tstep_animation,1),0):
    str_step = "0"*(6-len(str(istep)))+str(istep)
    plotting.plot_2d_grid_spectrum(autoreg_pred,
                                   actual_spectrum = actual_spectrum,
                                   channels = channel_names,
                                   frame=step,
                                   savename = f"pred_{str_step}",
                                   output_dir = gsp_dir,
                                   title = f"{model_name} autoregressive predictions; moist {151} init",
                                   cmap = cm.viridis,
                                   begframe = tstamp_start)
    if istep%100:
      print("grid prediction", istep, step)
                                   
os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {gsp_dir}/pred_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {pred_plots_dir}/pred_short.mp4')

## plot saving for animation, predictions - actual
gspa_dir = f"{pred_plots_dir}/pred-actual_pngs_short"
if not os.path.exists(gspa_dir):
    os.mkdir(gspa_dir)

## plot saving for animation
for istep, step in enumerate(np.arange(0,max_tstep_animation,1),0):
    str_step = "0"*(6-len(str(istep)))+str(istep)
    plotting.plot_2d_grid_spectrum(autoreg_pred[:endactual]-actual[:endactual],
                                   actual = None,
                                   channels = channel_names,
                                   frame=step,
                                   savename = f"pred-actual_{str_step}",
                                   output_dir = gspa_dir,
                                   title = f"{model_name} autoregressive predictions-actual; moist {151} init",
                                   cmap = cm.bwr,
                                   begframe = tstamp_start)
    if istep%100:
      print("grid prediction", istep, step)
      
os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {gspa_dir}/pred-actual_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {pred_plots_dir}/pred-act_short.mp4')

## should implement a spectrum loss method to approximate differece between predicted and actual mean (right now it does actual snapshot)

gspa_dir = f"{pred_plots_dir}/actual_long"
if not os.path.exists(gspa_dir):
    os.mkdir(gspa_dir)
    
max_tstep_animation = endactual

for istep, step in enumerate(np.arange(0,max_tstep_animation,10),0):
    str_step = "0"*(6-len(str(istep)))+str(istep)
    plotting.plot_2d_grid_spectrum(actual[:],
                                   actual = None,
                                   channels = channel_names,
                                   frame=step,
                                   savename = f"actual_{str_step}",
                                   output_dir = gspa_dir,
                                   title = f"actual data, moist {151}",
                                   cmap = cm.viridis,
                                   begframe = tstamp_start)
    if istep%100:
      print("grid prediction", istep, step)
      
os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {gspa_dir}/actual_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {pred_plots_dir}/actual_long.mp4')


gspa_dir = f"{pred_plots_dir}/actual_short"
if not os.path.exists(gspa_dir):
    os.mkdir(gspa_dir)
    
max_tstep_animation = 14*4

for istep, step in enumerate(np.arange(0,max_tstep_animation,1),0):
    str_step = "0"*(6-len(str(istep)))+str(istep)
    plotting.plot_2d_grid_spectrum(actual[:],
                                   actual = None,
                                   channels = channel_names,
                                   frame=step,
                                   savename = f"actual_{str_step}",
                                   output_dir = gspa_dir,
                                   title = f"actual data, moist {151}",
                                   cmap = cm.viridis,
                                   begframe = tstamp_start)
    if istep%100:
      print("grid prediction", istep, step)
      
os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {gspa_dir}/actual_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {pred_plots_dir}/actual_short.mp4')