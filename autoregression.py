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

nn_dir = "/media/volume/sdc/lenny_outputs/models/singleSteps_4-8-24/FNO2D_stepMethod-directstep_lambda-0p05_dataPrep-singleStep/"
moist_dir = '/media/volume/sdb'
tacs_dir = '/media/volume/sdc'
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

net = FNO2d(modes1, modes2, width, channels = 3, channelsout = 3).to("cuda")
net.load_state_dict(torch.load(nn_loc))
net = net.eval()

pred_plots_dir = "/media/volume/sdc/lenny_outputs/models/singleSteps_4-8-24/FNO2D_stepMethod-directstep_lambda-0p05_dataPrep-singleStep/pred_plots_noises/long2"

if not os.path.exists(pred_plots_dir):
  os.mkdir(pred_plots_dir)
  
ts_start = data_prep_args["ts_in"]
ts_start = 1000
autoregsteps = 40000

data_loc_mod = data_loc.replace('/home/exouser/nimrodxl1_mymount', tacs_dir)
moists_keep_fno, moists_keep_fno_timestamps, moists_info =  datau.data_load(data_loc_mod)

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