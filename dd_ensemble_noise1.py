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
import pickle

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

import scipy.fft as fft

def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

################################################################
# configs
################################################################
# path_outputs = '/home/exouser/ocean_reanalysis_daily_other_baselines/outputs/'
## make sure that nimrod xl 1 is mounted correctly
moist_dir = '/media/volume/sdb'

# save_dir = "/home/exouser/nimrodxl1_mymount/"
save_dir = "/media/volume/sdc"
#save_fno_step_loc = f"/media/volume/sdb/lenny_outputs/save_fno_step_10-31-2023_meanNormalizedOnly.data"
output_dir = f"{save_dir}/lenny_outputs"
data_dir = f"{output_dir}/model_data"
data_loc = f"{data_dir}/save_fno_step_4-8-24_allNorm.pkl"
models_dir = f"{output_dir}/models/singleSteps_4-8-24/"

channels = {
            "psi1" : ["mean", "std"],
            "psi2" : ["mean", "std"],
            "m" : ["mean", "std"],
           }
channel_names = ["psi1","psi2", "m"]

## previous load from saved data
# moists_keep_fno, moists_keep_fno_timestamps, moists_info =  datau.data_load(data_loc)
# ## only keep 151
# for i in [153, 155, 156, 157, 158, 159, 162, 163, 164, 701, 702, 703, 704, 705, 706, 707]:
  # moists_keep_fno.pop(i, None)
  # moists_keep_fno_timestamps.pop(i, None)
  # moists_info.pop(i, None)
# ts_start = 1000
# autoregsteps = 400
# actual = moists_keep_fno[151][ts_start:autoregsteps+ts_start+1]

qgm_sim_dir = "/media/volume/sdc/qgm_sim/"
qgm_names_noNoise_dir = os.path.join(qgm_sim_dir, 'load_resDir-151_noise-none_seed-0')+"/res_init"
qgm_names_noNoise_loc = os.path.join(qgm_sim_dir, 'load_resDir-151_noise-none_seed-0', "output.3d.nc")

def load_res_file( filename ):

    fpsic1 = np.load( filename + "_psic1.npz")
    psic1 = fpsic1['u'][:]
    fpsic2 = np.load( filename + "_psic2.npz")
    psic2 = fpsic2['u'][:]
    fqc1 = np.load( filename + "_qc1.npz")
    qc1 = fqc1['u'][:]
    fqc2 = np.load( filename + "_qc2.npz")
    qc2 = fqc2['u'][:]
    fmc = np.load( filename + "_mc.npz")
    mc = fmc['u'][:]
    ft0 = np.load( filename + "_t0.npz")
    t0 = ft0['u']

    return psic1, psic2, qc1, qc2, mc, t0
    
## now loading from separate file from hb numerical ensemble runs...this is for the first time step
psic_1, psic_2, qc_1, qc_2, mc, t0_init = load_res_file(  qgm_names_noNoise_dir )
psi1 = fft.irfft2( psic_1[1], workers=4 ).reshape(1,128,128)
psi2 = fft.irfft2( psic_1[1], workers=4 ).reshape(1,128,128)
m = fft.irfft2( mc[1], workers=4 ).reshape(1,128,128)
actualic = np.stack([psi1,psi2,m], axis = 3)

data = nc.Dataset(qgm_names_noNoise_loc)
actual = np.array(np.stack([data.variables["psi1"][:], data.variables["psi2"][:], data.variables["m"][:]],axis=3)).astype(float)

# actual = np.concatenate([actualic, actual], axis = 0)

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

## loading numerical ensembles for initial noise

qgm_names_noise = [q for q in os.listdir(qgm_sim_dir) if "gauss" in q]
qgm_data_locs = {q.split("_")[-1] : os.path.join(qgm_sim_dir,q,"output.3d.nc") for q in qgm_names_noise}
ensemble_num = 100
tsteps = 400
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


ts_start = 0
autoregsteps = 400
actual = actual[ts_start:autoregsteps+ts_start+1]

integration_methods = {"directstep" : directstep,
  "RK4step" : RK4step,
  "PECstep" : PECstep, 
  "Eulerstep" : Eulerstep}

losses = \
  {
    "spectral_loss_channels_og" : spectral_loss_channels_og,
    "spectral_loss_channels_sqr" : spectral_loss_channels_sqr,
    "huber_loss" : huber_loss,
  }
  
nn_dir_home ="/media/volume/sdc/lenny_outputs/models/singleSteps_4-8-24/"

nn_names = [
 'FNO2D_stepMethod-directstep_lambda-0p05_dataPrep-singleStep',
 'FNO2D_stepMethod-directstep_lambda-0p0_dataPrep-singleStep',
 'FNO2D_stepMethod-directstep_lambda-0p15_dataPrep-singleStep',
 'FNO2D_stepMethod-directstep_lambda-0p1_dataPrep-singleStep',
 'FNO2D_stepMethod-directstep_lambda-0p2_dataPrep-singleStep',
 'FNO2D_stepMethod-directstep_lambda-0p4_dataPrep-singleStep',
 'FNO2D_stepMethod-Eulerstep_lambda-0p05_dataPrep-singleStep',
 'FNO2D_stepMethod-Eulerstep_lambda-0p0_dataPrep-singleStep',
 'FNO2D_stepMethod-Eulerstep_lambda-0p15_dataPrep-singleStep',
 'FNO2D_stepMethod-Eulerstep_lambda-0p1_dataPrep-singleStep',
 'FNO2D_stepMethod-Eulerstep_lambda-0p2_dataPrep-singleStep',
 'FNO2D_stepMethod-Eulerstep_lambda-0p4_dataPrep-singleStep',
 'FNO2D_stepMethod-PECstep_lambda-0p05_dataPrep-singleStep',
 'FNO2D_stepMethod-PECstep_lambda-0p0_dataPrep-singleStep',
 'FNO2D_stepMethod-PECstep_lambda-0p15_dataPrep-singleStep',
 'FNO2D_stepMethod-PECstep_lambda-0p1_dataPrep-singleStep',
 'FNO2D_stepMethod-PECstep_lambda-0p2_dataPrep-singleStep',
 'FNO2D_stepMethod-PECstep_lambda-0p4_dataPrep-singleStep',
 'FNO2D_stepMethod-RK4step_lambda-0p05_dataPrep-singleStep',
 'FNO2D_stepMethod-RK4step_lambda-0p0_dataPrep-singleStep',
 'FNO2D_stepMethod-RK4step_lambda-0p15_dataPrep-singleStep',
 'FNO2D_stepMethod-RK4step_lambda-0p1_dataPrep-singleStep',
 'FNO2D_stepMethod-RK4step_lambda-0p2_dataPrep-singleStep',
 'FNO2D_stepMethod-RK4step_lambda-0p4_dataPrep-singleStep',]


dt = .25
for nn_dir in [os.path.join(nn_dir_home, nn_name) for nn_name in nn_names]:
  # if not os.path.exists(f"{nn_dir}/model_params.yml"):
      # continue
  
  print(f"{nn_dir} noise plots...")
  with open(f"{nn_dir}/model_params.yml", 'r') as h:
      model_params = yaml.load(h, Loader = yaml.Loader)
      
  with open(f"{nn_dir}/epoch_losses.pkl", 'rb') as h:
      epoch_losses = pickle.load(h)
      
  nn_dir = model_params["model_dir"]
  nn_loc = model_params["model_loc"]
  model_name = model_params["model_name"]

  net = FNO2d(modes1, modes2, width, channels = 3, channelsout = 3).to("cuda")
  nn_loc = nn_loc.replace("/home/exouser/nimrodxl1_mymount//", "/media/volume/sdc/")
  net.load_state_dict(torch.load(nn_loc))
  net = net.eval()

  previnput = actual[[0]]
  autoreg_pred_og = actual[[0]]

  nseed_num = 100
  autoreg_preds_noise = np.zeros(shape = (nseed_num, autoregsteps, 128, 128, 3))
  nfact = .2

  for nseed in np.arange(data_noise.shape[0]):
    if nseed%20==0:
      print(f"model: {model_name}, nseed: {nseed}")
      
    ## seed start for predictions
    autoreg_preds_noise[nseed,0,...] = data_noise[nseed,[0],...]
    
    for step in range(1, autoregsteps):
      output = integration_methods[step_method](net, torch.tensor(autoreg_preds_noise[nseed,[step-1],...]).cuda().float()).cpu().detach().numpy()
      
      ## change shape
      autoreg_preds_noise[nseed,step,...] = output[0,...]
      if step % 100 == 0:
        specloss = losses[lossFunction](torch.tensor(output).cuda(),
                                           torch.tensor(actual[[step]]).cuda(),
                                           wavenum_init,
                                           wavenum_init_ydir,
                                           lambda_fft = lambda_fft,
                                           grid_valid_size = output.shape[1]*output.shape[2],
                                           channels = output.shape[3])[0]
        
  
  ## not enough space...
  # autoreg_preds_noise_loc = os.path.join("/media/volume/sdc/lenny_outputs/model_output_noise/singleSteps_4-8-24/", f"{model_name}_preds_noise.pkl")
  # with open(autoreg_preds_noise_loc, "wb") as h:
    # pickle.dump(autoreg_preds_noise, h)
  
  for steps in [40, 100, 399]:

    ## dke
    
    actual_steps = actual[:steps,...]
    [dpsidt, dpsidy, dpsidx, _dpsidchannel] = np.gradient(actual_steps[:,10:-10,:,:])
    uactual, vactual = -dpsidy, dpsidx
    
    totE_actual_time = uactual**2+vactual**2
    totE_actual_time = np.mean(totE_actual_time, axis = (1,2))
    
    
    fig, ax = plt.subplots(3, actual_steps.shape[3], dpi = 200, figsize = (11,6))
    if False:
      for nseed in range(data_noise.shape[0]):
      # for i in range(1):
        if nseed % 10 == 0:
            print(f"ACC RMSE DKE, dd pred: {nseed}")
            
        dd_noise_nseed = autoreg_preds_noise[nseed,:steps]
        num_noise_nseed = data_noise[nseed,:steps]
        for ich, ch in enumerate(channel_names,0):
            for ensemble, ensemble_name, ensemble_col in zip([dd_noise_nseed, num_noise_nseed], 
                                               ["dd_ensemble", "num_ensemble"],
                                               ["black", "blue"]):
              pred = ensemble
              sqerrors = []
              accs = []
              tsteps = np.arange(actual_steps.shape[0])
              #d1c = data1[:,:,:,ich].mean(axis = 0)
              ## time mean computed
              d2c = actual[:,:,:,ich].mean(axis = 0)
              
              d1t = pred[:,:,:,ich]
              d2t = actual_steps[:,:,:,ich]
              
              sqerrors = np.sqrt(np.mean((d1t-d2t)**2, axis = (1,2)))
              
              d1t = pred[:,:,:,ich]
              d2t = actual_steps[:,:,:,ich]
              num = np.sum((d1t - d2c)*(d2t - d2c), axis = (1,2))
              den = np.sqrt(np.sum((d1t - d2c)**2, axis = (1,2)))*np.sqrt(np.sum((d2t - d2c)**2, axis = (1,2)))
              accs = num/den
            
              ax[0,ich].plot(tsteps[1:]*dt, 
                           sqerrors[1:], 
                           color = ensemble_col,
                           alpha = 0.3)
              
              ax[1,ich].plot(tsteps[1:]*dt, 
                           accs[1:], 
                           color = ensemble_col,
                           alpha = 0.3)
              
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
              
              dke = totE_pred_time - totE_actual_time[...,ich]
              ax[2,ich].plot(tsteps[1:]*dt, 
                             dke[1:], 
                             color = ensemble_col, 
                             alpha = .3,
                             zorder = 20)
                  
              ax[2, ich].grid()
              
              
      for ich, ch in enumerate(channel_names,0):
        ax[0, ich].set_title(f"{ch}")
        ax[2, ich].set_xlabel(f"days")
        ax[0, ich].grid(alpha = .5)
        ax[1,ich].grid(alpha = .5)
        ax[2, ich].grid(alpha = .5)

      for im, metric in enumerate(["RMSE", "ACC", "DKE"],0):
        ax[im,0].set_ylabel(f"{metric}")

      plt.suptitle(f"{model_name}\nblack: data driven model, blue: numerical model")
      plt.savefig(f"/media/volume/sdc/lenny_outputs/model_output_noise/singleSteps_4-8-24_2/{model_name}_numNoise_steps-{steps}_nfact-{str(nfact).replace('.','p')}.png")
      plt.close()
    
    
    dkes = np.empty((data_noise.shape[0],steps,3))
    dkes2 = np.empty((data_noise.shape[0],steps,3))
    if True:
      for nseed in range(data_noise.shape[0]):
      # for i in range(1):
        if nseed % 10 == 0:
            print(f"ACC RMSE DKE, dd pred: {nseed}")
            
        dd_noise_nseed = autoreg_preds_noise[nseed,:steps]
        num_noise_nseed = data_noise[nseed,:steps]
        for ich, ch in enumerate(channel_names,0):
          
          for ensemble, ensemble_name, ensemble_col in zip([dd_noise_nseed, num_noise_nseed], 
                                             ["dd_ensemble", "num_ensemble"],
                                             ["black", "blue"]):
            pred = ensemble
            sqerrors = []
            accs = []
            tsteps = np.arange(actual_steps.shape[0])
            #d1c = data1[:,:,:,ich].mean(axis = 0)
            ## time mean computed
            d2c = actual[:,:,:,ich].mean(axis = 0)
            
            d1t = pred[:,:,:,ich]
            d2t = actual_steps[:,:,:,ich]
            
            sqerrors = np.sqrt(np.mean((d1t-d2t)**2, axis = (1,2)))
            
            d1t = pred[:,:,:,ich]
            d2t = actual_steps[:,:,:,ich]
            num = np.sum((d1t - d2c)*(d2t - d2c), axis = (1,2))
            den = np.sqrt(np.sum((d1t - d2c)**2, axis = (1,2)))*np.sqrt(np.sum((d2t - d2c)**2, axis = (1,2)))
            accs = num/den
            
            ax[0,ich].plot(tsteps[1:]*dt, 
                         sqerrors[1:], 
                         color = ensemble_col,
                         alpha = 0.3)
            
            ax[1,ich].plot(tsteps[1:]*dt, 
                         accs[1:], 
                         color = ensemble_col,
                         alpha = 0.3)
            
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
            
            dke = totE_pred_time - totE_actual_time[...,ich]
            
            dkes[nseed,:,ich] = dke
            # ax[2,ich].plot(tsteps[1:]*dt, 
                           # dke[1:], 
                           # color = ensemble_col, 
                           # alpha = .3,
                           # zorder = 20)
                
            # ax[2, ich].grid()
            
            ## spectrum averaging method
            
            
      dkes = dkes[:,:tsteps.shape[0],:]
      ax[2,0].plot(tsteps[1:]*dt, 
                           np.std(dkes[:,1:,0], axis = 0), 
                           color = "black", 
                           zorder = 20)
      
      ax[2,1].plot(tsteps[1:]*dt, 
                           np.std(dkes[:,1:,1], axis = 0), 
                           color = "black", 
                           zorder = 20)

      
      for ich, ch in enumerate(channel_names,0):
        ax[0, ich].set_title(f"{ch}")
        ax[2, ich].set_xlabel(f"days")
        ax[0, ich].grid(alpha = .5)
        ax[1,ich].grid(alpha = .5)
        ax[2, ich].grid(alpha = .5)

      for im, metric in enumerate(["RMSE", "ACC", r"$\sigma$(DKE)"],0):
        ax[im,0].set_ylabel(f"{metric}")

      plt.suptitle(f"{model_name}\nblack: data driven model, blue: numerical model")
      plt.savefig(f"/media/volume/sdc/lenny_outputs/model_output_noise/singleSteps_4-8-24_2/{model_name}_numNoise_steps-{steps}_nfact-{str(nfact).replace('.','p')}.png")
      plt.close()
  
    ## spectrum at time for noise ensembles 
    if True:
    
      ## actual spectrum plot
      actual_fft = np.abs(fft.rfft(actual, axis = 2))
      actual_fft = np.mean(actual_fft, axis = 1)
      
      ## multiple methods, compute inverse fft for single step
      ## fft, then fft space averging
      autoreg_preds_noise_fft = np.abs(fft.rfft(autoreg_preds_noise, axis = 3))
      autoreg_preds_noise_fft = np.mean(autoreg_preds_noise_fft, axis = 2)
      autoreg_preds_noise_fft = np.mean(autoreg_preds_noise_fft, axis = 0)
      
      for steps in [1, 16, 32]:

        fig, ax = plt.subplots(2, 3, dpi = 200, figsize = (12,5))
        
        ax[0,0].plot(actual_fft[steps, 1:, 0], linestyle = "--", color = "black")
        ax[0,1].plot(actual_fft[steps, 1:, 1], linestyle = "--", color = "black")
        ax[0,2].plot(actual_fft[steps, 1:, 2], linestyle = "--", color = "black")
        
        ax[0,0].plot(autoreg_preds_noise_fft[steps, 1:, 0], color = "blue", zorder = 20)
        ax[0,1].plot(autoreg_preds_noise_fft[steps, 1:, 1], color = "blue", zorder = 20)
        ax[0,2].plot(autoreg_preds_noise_fft[steps, 1:, 2], color = "blue", zorder = 20)
        
        deltafft = autoreg_preds_noise_fft[steps, 1:, :] - actual_fft[steps, 1:, :]
        ax[1,0].plot(deltafft[...,0], linestyle = "--", color = "black")
        ax[1,1].plot(deltafft[...,1], linestyle = "--", color = "black")
        ax[1,2].plot(deltafft[...,2], linestyle = "--", color = "black")
        
        ax[1,0].set_ylim(-1, 1)
        ax[1,1].set_ylim(-1, 1)
        ax[1,2].set_ylim(-1, 1)
        
        ax[0,0].set_ylabel("Amplitude")
        ax[1,0].set_ylabel("Delta")
        ax[1,0].set_xlabel("psi1")
        ax[1,1].set_xlabel("psi2")
        ax[1,2].set_xlabel("m")
           
        ax[0,0].grid(alpha = .3)
        ax[0,1].grid(alpha = .3)
        ax[0,2].grid(alpha = .3)
        ax[1,0].grid(alpha = .3)
        ax[1,1].grid(alpha = .3)
        ax[1,2].grid(alpha = .3)
        
        plt.suptitle(f"{model_name}\nfft -> latitude mean -> ensemble mean")
        plt.savefig(f"/media/volume/sdc/lenny_outputs/model_output_noise/singleSteps_4-8-24_2/{model_name}_numNoise_steps-{steps}_meanSpectrumPlot1.png")
        plt.close()
        
      
      
      
## from numerical ensembles
autoreg_preds_noise_loc = "/media/volume/sdc/lenny_outputs/model_output_noise/singleSteps_4-8-24/FNO2D_stepMethod-directstep_lambda-0p05_dataPrep-singleStep_preds_noise.pkl"
with open(autoreg_preds_noise_loc, "rb") as h:
  autoreg_preds_noise = pickle.load(h)
  

dt = .25
for nn_dir in [os.path.join(nn_dir_home, nn_name) for nn_name in nn_names]:
  # if not os.path.exists(f"{nn_dir}/model_params.yml"):
      # continue
  
  print(f"{nn_dir} noise plots...")
  with open(f"{nn_dir}/model_params.yml", 'r') as h:
      model_params = yaml.load(h, Loader = yaml.Loader)
      
  with open(f"{nn_dir}/epoch_losses.pkl", 'rb') as h:
      epoch_losses = pickle.load(h)
      
  nn_dir = model_params["model_dir"]
  nn_loc = model_params["model_loc"]
  model_name = model_params["model_name"]

  net = FNO2d(modes1, modes2, width, channels = 3, channelsout = 3).to("cuda")
  nn_loc = nn_loc.replace("/home/exouser/nimrodxl1_mymount//", "/media/volume/sdc/")
  net.load_state_dict(torch.load(nn_loc))
  net = net.eval()

  previnput = actual[[0]]
  autoreg_pred_og = actual[[0]]

  nseed_num = 100
  autoreg_preds_noise = np.zeros(shape = (nseed_num, autoregsteps, 128, 128, 3))
  nfact = .2

  for nseed in np.arange(data_noise.shape[0]):
    if nseed%20==0:
      print(f"model: {model_name}, nseed: {nseed}")
      
    ## seed start for predictions
    autoreg_preds_noise[nseed,0,...] = data_noise[nseed,[0],...]
    
    for step in range(1, autoregsteps):
      output = integration_methods[step_method](net, torch.tensor(autoreg_preds_noise[nseed,[step-1],...]).cuda().float()).cpu().detach().numpy()
      
      ## save new output to prediction array
      autoreg_preds_noise[nseed,step,...] = output[0,...]
      if step % 100 == 0:
        specloss = losses[lossFunction](torch.tensor(output).cuda(),
                                           torch.tensor(actual[[step]]).cuda(),
                                           wavenum_init,
                                           wavenum_init_ydir,
                                           lambda_fft = lambda_fft,
                                           grid_valid_size = output.shape[1]*output.shape[2],
                                           channels = output.shape[3])[0]
  
  ## spectrum at time for noise ensembles 
  if True:
  
    ## actual spectrum plot
    actual_fft = np.abs(fft.rfft(actual, axis = 2))
    actual_fft = np.mean(actual_fft, axis = 1)
    
    ## multiple methods, compute inverse fft for single step
    ## fft, then fft space averging
    autoreg_preds_noise_fft = np.abs(fft.rfft(autoreg_preds_noise, axis = 3))
    autoreg_preds_noise_fft = np.mean(autoreg_preds_noise_fft, axis = 2)
    autoreg_preds_noise_fft = np.mean(autoreg_preds_noise_fft, axis = 0)
    
    for steps in [1, 16, 32]:

      fig, ax = plt.subplots(2, 3, dpi = 200, figsize = (12,5))
      
      ax[0,0].plot(actual_fft[steps, 1:, 0], linestyle = "--", color = "black")
      ax[0,1].plot(actual_fft[steps, 1:, 1], linestyle = "--", color = "black")
      ax[0,2].plot(actual_fft[steps, 1:, 2], linestyle = "--", color = "black")
      
      ax[0,0].plot(autoreg_preds_noise_fft[steps, 1:, 0], color = "blue", zorder = 20)
      ax[0,1].plot(autoreg_preds_noise_fft[steps, 1:, 1], color = "blue", zorder = 20)
      ax[0,2].plot(autoreg_preds_noise_fft[steps, 1:, 2], color = "blue", zorder = 20)
      
      deltafft = autoreg_preds_noise_fft[steps, 1:, :] - actual_fft[steps, 1:, :]
      ax[1,0].plot(deltafft[...,0], linestyle = "--", color = "black")
      ax[1,1].plot(deltafft[...,1], linestyle = "--", color = "black")
      ax[1,2].plot(deltafft[...,2], linestyle = "--", color = "black")
      
      ax[1,0].set_ylim(-1, 1)
      ax[1,1].set_ylim(-1, 1)
      ax[1,2].set_ylim(-1, 1)
      
      ax[0,0].set_ylabel("Amplitude")
      ax[1,0].set_ylabel("Delta")
      ax[1,0].set_xlabel("psi1")
      ax[1,1].set_xlabel("psi2")
      ax[1,2].set_xlabel("m")
         
      ax[0,0].grid(alpha = .3)
      ax[0,1].grid(alpha = .3)
      ax[0,2].grid(alpha = .3)
      ax[1,0].grid(alpha = .3)
      ax[1,1].grid(alpha = .3)
      ax[1,2].grid(alpha = .3)
      
      plt.suptitle(f"{model_name}\nfft -> latitude mean -> ensemble mean")
      plt.savefig(f"/media/volume/sdc/lenny_outputs/model_output_noise/singleSteps_4-8-24_2/{model_name}_numNoise_steps-{steps}_meanSpectrumPlot1.png")
      plt.close()
      
    # ## grid space, the fft on axis
    # ##  (100, 400, 128, 128, 3)
    # autoreg_preds_noise_rfft2 = np.abs(fft.rfft(autoreg_preds_noise, axis = 3))
    # autoreg_preds_noise_rfft_mean = np.mean(autoreg_preds_noise_rfft, axis = 2)
    # autoreg_preds_noise_mean_rfft_mean = np.mean(autoreg_preds_noise_rfft_mean, axis = 0)
    
    # for steps in [1, 16, 32]:

      # fig, ax = plt.subplots(2, 3, dpi = 200, figsize = (12,5))
      
      # ax[0,0].plot(actual_fft[steps, 1:, 0], linestyle = "--", color = "black")
      # ax[0,1].plot(actual_fft[steps, 1:, 1], linestyle = "--", color = "black")
      # ax[0,2].plot(actual_fft[steps, 1:, 2], linestyle = "--", color = "black")
      
      # ax[0,0].plot(autoreg_preds_noise_mean_rfft_mean[steps, 1:, 0], color = "blue", zorder = 20)
      # ax[0,1].plot(autoreg_preds_noise_mean_rfft_mean[steps, 1:, 1], color = "blue", zorder = 20)
      # ax[0,2].plot(autoreg_preds_noise_mean_rfft_mean[steps, 1:, 2], color = "blue", zorder = 20)
      
      # deltafft = autoreg_preds_noise_mean_rfft_mean[steps, 1:, :] - actual_fft[steps, 1:, :]
      # ax[1,0].plot(deltafft[...,0], linestyle = "--", color = "black")
      # ax[1,1].plot(deltafft[...,1], linestyle = "--", color = "black")
      # ax[1,2].plot(deltafft[...,2], linestyle = "--", color = "black")
      
      # ax[1,0].set_ylim(-1, 1)
      # ax[1,1].set_ylim(-1, 1)
      # ax[1,2].set_ylim(-1, 1)
      
      # ax[0,0].set_ylabel("Amplitude")
      # ax[1,0].set_ylabel("Delta")
      # ax[1,0].set_xlabel("psi1")
      # ax[1,1].set_xlabel("psi2")
      # ax[1,2].set_xlabel("m")
         
      # ax[0,0].grid(alpha = .3)
      # ax[0,1].grid(alpha = .3)
      # ax[0,2].grid(alpha = .3)
      # ax[1,0].grid(alpha = .3)
      # ax[1,1].grid(alpha = .3)
      # ax[1,2].grid(alpha = .3)
      
      # plt.suptitle(f"{model_name}\nfft -> latitude mean -> ensemble mean")
      # plt.savefig(f"/media/volume/sdc/lenny_outputs/model_output_noise/singleSteps_4-8-24_2/{model_name}_numNoise_steps-{steps}_meanSpectrumPlot1.png")
      # plt.close()
   
    

