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
import netCDF4 as nc
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

################################################################
# configs
################################################################
# path_outputs = '/home/exouser/ocean_reanalysis_daily_other_baselines/outputs/'
## make sure that nimrod xl 1 is mounted correctly
qgm_sim_dir = "/media/volume/sdc/qgm_sim/"

qgm_names_noise = [q for q in os.listdir(qgm_sim_dir) if "gauss" in q]
qgm_data_locs = {q.split("_")[-1] : os.path.join(qgm_sim_dir,q,"output.3d.nc") for q in qgm_names_noise}

qgm_names_noNoise_loc = os.path.join(qgm_sim_dir, 'load_resDir-151_noise-none_seed-0', "output.3d.nc")

channels = {
            "psi1" : ["mean", "std"],
            "psi2" : ["mean", "std"],
            "m" : ["mean", "std"],
           }
channel_names = ["psi1","psi2", "m"]

data_noise = None
tsteps = 400
ensemble_num = 100
data_noise = np.empty((ensemble_num,tsteps,128,128,3))

iq = 0
for q in qgm_data_locs:
    try:
        data = nc.Dataset(qgm_data_locs[q])
    except:
        print(f"{q} nc not loading. Continuing...")
        continue
        
    data_use = np.array(np.stack([data.variables["psi1"][:], data.variables["psi2"][:], data.variables["m"][:]],axis=3)).astype(float)
    
    ## normalization
    for ich, ch in enumerate(channel_names, 0):
    
        if "std" in channels[ch]:
            std = np.std(data_use[:,:,:,ich])
        else:
            std = 1.0
        if "mean" in channels[ch]:
            mean = np.mean(data_use[:,:,:,ich])
        else:
            mean = 0.0
    
        data_use[:,:,:,ich] = (data_use[:,:,:,ich] - mean)/std
    
    data_noise[iq,...] = data_use
    iq += 1
    
## actual without noise
## normalization
data = nc.Dataset(qgm_names_noNoise_loc)
actual = np.array(np.stack([data.variables["psi1"][:], data.variables["psi2"][:], data.variables["m"][:]],axis=3)).astype(float)

for ich, ch in enumerate(channel_names, 0):
    if "std" in channels[ch]:
        std = np.std(actual[:,:,:,ich])
    else:
        std = 1.0
    if "mean" in channels[ch]:
        mean = np.mean(actual[:,:,:,ich])
    else:
        mean = 0.0

    actual[:,:,:,ich] = (actual[:,:,:,ich] - mean)/std

  
dt = .25

for steps in [50, 100, 400]:
  ## dke
  actual_steps = actual[:steps,...]
  [dpsidt, dpsidy, dpsidx, _dpsidchannel] = np.gradient(actual_steps[:,10:-10,:,:])
  uactual, vactual = -dpsidy, dpsidx
  
  totE_actual_time = uactual**2+vactual**2
  totE_actual_time = np.mean(totE_actual_time, axis = (1,2))
  
  fig, ax = plt.subplots(3, actual_steps.shape[3], dpi = 200, figsize = (14,8))
  
  for i in range(data_noise.shape[0]):
  # for i in range(1):
    print(f"numerical ensemble noise: {i}")
    pred = data_noise[i,:steps,...]
    
    
    for ich, ch in enumerate(channel_names,0):
        sqerrors = []
        accs = []
        tsteps = np.arange(steps)
        #d1c = data1[:,:,:,ich].mean(axis = 0)
        ## time mean computed
        d2c = actual_steps[:,:,:,ich].mean(axis = 0)
        d2c = np.tile(d2c, (steps,1,1))

        # for tstep in tsteps:
            # d1t = pred[tstep,:,:,ich]
            # d2t = actual_steps[tstep,:,:,ich]
            # sqerrors.append(np.sqrt(np.mean((d1t-d2t)**2)))
        
        # for tstep in tsteps:
            # d1t = pred[tstep,:,:,ich]
            # d2t = actual_steps[tstep,:,:,ich]
            # num = np.sum((d1t - d2c)*(d2t - d2c))
            # den = np.sqrt(np.sum((d1t - d2c)**2))*np.sqrt(np.sum((d2t - d2c)**2))
            # accs.append(num/den)
        
        d1t = pred[:,:,:,ich]
        d2t = actual_steps[:,:,:,ich]
        
        sqerrors = np.sqrt(np.mean((d1t-d2t)**2, axis = (1,2)))
        
        d1t = pred[:,:,:,ich]
        d2t = actual_steps[:,:,:,ich]
        num = np.sum((d1t - d2c)*(d2t - d2c), axis = (1,2))
        den = np.sqrt(np.sum((d1t - d2c)**2, axis = (1,2)))*np.sqrt(np.sum((d2t - d2c)**2, axis = (1,2)))
        accs = num/den
        
        ax[0,ich].plot(tsteps*dt, 
                     sqerrors, 
                     color = "blue",
                     alpha = 0.3)
        
        ax[1,ich].plot(tsteps*dt, 
                     accs, 
                     color = "blue",
                     alpha = 0.3)
        
      # for ich, ch in enumerate(channel_names[:2],0):
        # first lat, second long
        if ch == "m":
          ax[2,ich].axis('off')
          continue
        
        ## update with this calculation for u and v
        # u2 = fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_2[1], workers =nworker) # -i * k * psic2
        # v2 = fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_2[1], workers =nworker)
        
        [dpsidt, dpsidy, dpsidx] = np.gradient(pred[:,10:-10,:,ich])
        upred, vpred = -dpsidy, dpsidx
        
        totE_pred_time = upred**2+vpred**2
        totE_pred_time = np.mean(totE_pred_time, axis = (1,2))
        
        ax[2,ich].plot(tsteps*dt, 
                       totE_pred_time - totE_actual_time[...,ich], 
                       color = "blue", 
                       alpha = .3,
                       zorder = 20)
            
        ax[2, ich].grid()
    
  
  for ich, ch in enumerate(channels,0):
    ax[0, ich].set_title(f"{ch}")
    ax[2, ich].set_xlabel(f"days")
    ax[0, ich].grid(alpha = .5)
    ax[1,ich].grid(alpha = .5)
    ax[2, ich].grid(alpha = .5)
    
  for im, metric in enumerate(["RMSE", "ACC", "DKE"],0):
    ax[im,0].set_ylabel(f"{metric}")

  plt.suptitle("Numerical Ensembles")
  save_loc = f"/home/exouser/lenny_scripts/Deep-Learning-based-ocean-forecasts-moist/outputs/ic_noise/numericalEnsembles_steps-{steps}_metricsNew.png"
  print(f"Saving fig: {save_loc}")
  plt.savefig(save_loc)
  plt.close()

  clear_mem()

## from numerical ensembles