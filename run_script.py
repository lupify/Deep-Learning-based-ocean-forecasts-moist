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

################################################################
# configs
################################################################
# path_outputs = '/home/exouser/ocean_reanalysis_daily_other_baselines/outputs/'
## make sure that nimrod xl 1 is mounted correctly
moist_dir = '/media/volume/sdb'

moist_loc_151 = f"{moist_dir}/moist_5_daily/151/output.3d.nc"
moist_loc_153 = f"{moist_dir}/moist_5_daily/153/output.3d.nc"
moist_loc_155 = f"{moist_dir}/moist_5_daily/155/output.3d.nc"
moist_loc_156 = f"{moist_dir}/moist_5_daily/156/output.3d.nc"
moist_loc_157 = f"{moist_dir}/moist_5_daily/157/output.3d.nc"
moist_loc_158 = f"{moist_dir}/moist_5_daily/158/output.3d.nc"
moist_loc_159 = f"{moist_dir}/moist_5_daily/159/output.3d.nc"
moist_loc_162 = f"{moist_dir}/moist_5_daily/162/output.3d.nc"
moist_loc_163 = f"{moist_dir}/moist_5_daily/163/output.3d.nc"
moist_loc_164 = f"{moist_dir}/moist_5_daily/164/output.3d.nc"
moist_loc_701 = f"{moist_dir}/moist_5_daily/701/output.3d.nc"
moist_loc_702 = f"{moist_dir}/moist_5_daily/702/output.3d.nc"
moist_loc_703 = f"{moist_dir}/moist_5_daily/703/output.3d.nc"
moist_loc_704 = f"{moist_dir}/moist_5_daily/704/output.3d.nc"
moist_loc_705 = f"{moist_dir}/moist_5_daily/705/output.3d.nc"
moist_loc_706 = f"{moist_dir}/moist_5_daily/706/output.3d.nc"
moist_loc_707 = f"{moist_dir}/moist_5_daily/707/output.3d.nc"

moist_data_locs = {
                   151 : moist_loc_151, 
                   153 : moist_loc_153,  
                   155 : moist_loc_155,  
                   156 : moist_loc_156,  
                   157 : moist_loc_157,  
                   158 : moist_loc_158,  
                   159 : moist_loc_159,
                   162 : moist_loc_162,
                   163 : moist_loc_163,
                   164 : moist_loc_164,
                   701 : moist_loc_701,
                   702 : moist_loc_702,
                   703 : moist_loc_703,
                   704 : moist_loc_704,
                   705 : moist_loc_705,
                   706 : moist_loc_706,
                   707 : moist_loc_707,
                  }

save_dir = "/home/exouser/nimrodxl1_mymount/"
#save_fno_step_loc = f"/media/volume/sdb/lenny_outputs/save_fno_step_10-31-2023_meanNormalizedOnly.data"
output_dir = f"{save_dir}/lenny_outputs"
data_dir = f"{output_dir}/model_data"
data_loc = f"{data_dir}/save_fno_step_3-29-23_allNorm_noRampUp.pkl"
models_dir = f"{output_dir}/models/singleSteps_allLoss_2-5-24/"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

channels = {
            "psi1" : ["mean", "std"],
            "psi2" : ["mean", "std"],
            "m" : ["mean", "std"],
           }
channel_names = ["psi1","psi2", "m"]

# test load 
if False:
    moist_data_locs = {
                   151 : moist_loc_151}
    data_loc = f"{output_dir}/models/singleSteps_twoOnly_test.pkl"
    
if not os.path.exists(data_loc):
    
    ## can change this to create date with/without channels, or can be done at a later step
    

    moists_keep_fno, moists_keep_fno_timestamps, moists_info = datau.data_prep_save(data_loc,
                                                                                      moist_data_locs = moist_data_locs,
                                                                                      channels = channels,
                                                                                      rampuptstamp = 0,
                                                                                      save = True)
else:
    moists_keep_fno, moists_keep_fno_timestamps, moists_info =  datau.data_load(data_loc)

"""
step_methods -- 
   "directstep" : directstep,
   "RK4step" : RK4step,
   "PECstep" : PECstep, 
   "Eulerstep" : Eulerstep,
"""

## can remove the moisture underneath here
## removes moisture channel, changes channel_names that are kept
if False:
    for m in moists_keep_fno:
        moists_keep_fno[m] = moists_keep_fno[m][:,:,:,:2]
    
    channels.pop("m", None)
    models_dir = f"{output_dir}/models/singleSteps_dry_1-24-24/"
    channel_names = ["psi1","psi2"]
    print(r"running dry only")

data_prep = "singleStep"
# data_prep = "tsteps"

## not saving modifications anymore
#data_mod_loc = f"{output_dir}/model_data/fno_{data_prep}_11-23-23_allNorm.pkl"
data_mod_loc = f"na"

"""
below, we can choose single step (uses a single timestep for input to the fno), or multistep, which uses multiple time inputs
"""
if data_prep == "singleStep":
    data_prep_args = {"ts_in" : 1, "lead" : 0, "ts_out" : 1, "overlap" : True}
elif data_prep == "twoStep":
    data_prep_args = {"ts_in" : 1, "lead" : 0, "ts_out" : 1, "overlap" : True}
elif data_prep == "tsteps":
    data_prep_args = {"ts_in" : 4, "lead" : 0, "ts_out" : 1, "overlap" : True}

# iterations
train = [153, 155, 156, 157, 158, 159, 162, 163, 164, 701, 702, 703, 704, 705, 706, 707]
test = [151]

## available integration steps
## huber loss also implemented
integration_methods = \
  {
   "directstep" : directstep,
   "RK4step" : RK4step,
   "PECstep" : PECstep, 
   "Eulerstep" : Eulerstep,
  }

losses = \
  {
    "spectral_loss_channels_og" : spectral_loss_channels_og,
    "spectral_loss_channels_sqr" : spectral_loss_channels_sqr,
    "huber_loss" : huber_loss,
  }
  

clear_mem()

load_num = 4
mi = 0
train_lists = []
train_copy = np.copy(train)
np.random.shuffle(train_copy)
while mi < len(train_copy):
    train_lists.append(train_copy[mi:mi+load_num])
    mi+=load_num


"""
Running training for variations on step methods and lambdas given
"""

## only direct step works for multiple tsteps method at the moment
# step_methods = ['directstep']
# lambda_ffts = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]
    
# step_methods = ['directstep']
step_methods = ['directstep', "RK4step", "PECstep", "Eulerstep"]
# lambda_ffts = [.01]
lambda_ffts = [0.0, .05, .1, .15, .2, .4]
# lambda_ffts = [.0, .5]

param_vars = list(itertools.product(*[step_methods, lambda_ffts]))

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
autoregsteps=250*4
max_tstep_animation = 100*4

for step_method, lambda_fft in param_vars:
    
    lambda_fft_str = str(lambda_fft).replace('.','p')
    model_name = f"FNO2D_stepMethod-{step_method}_lambda-{lambda_fft_str}_dataPrep-{data_prep}"
    nn_dir = f"{models_dir}/{model_name}"
    nn_loc = f"{nn_dir}/model.pt"
    pred_plots_dir = f"{nn_dir}/pred_plots"
    
    # if os.path.exists(f"{pred_plots_dir}/pred-act.mp4"):
        # print(f"Final plot exists: {pred_plots_dir}/pred-act.mp4 continuing...")
        # continue
        
    if not os.path.exists(nn_loc):
        print(f"''{nn_loc}'' does not exist. Training model...")
    
        model_params = \
             {
              "model_name" : model_name,
              "data_loc" : data_loc,
              "data_prep" : data_prep,
              "data_prep_args" : data_prep_args,
              "data_mod_loc" : data_mod_loc,
              "model_dir" : nn_dir,
              "model_loc" : nn_loc,
              "step_method" : step_method,
              "num_epochs" : 20, 
              "lambda_fft" : lambda_fft,
              "wavenum_init" : 20, 
              "wavenum_init_ydir" : 20,
              "modes1" : 64, 
              "modes2" : 64, 
              "width" : 32, 
              "batch_size" : 40, 
              "learning_rate" : 0.001,
              "optimizer" : "AdamW",
              "lossFunction" : "spectral_loss_channels_sqr",
             }
        
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
        step_method = model_params["step_method"]
        optimizer_name = model_params["optimizer"]
        lossFunction = model_params["lossFunction"]
        
        pprint.pprint("model_params:")
        pprint.pprint(model_params)
        
        ## to get shape of input
        d = {t:moists_keep_fno[t] for t in [151]}
        inputs, targets = datau.load_data_fno(d, data_prep, data_prep_args)
        
        net = FNO2d(modes1, modes2, width, channels = inputs.shape[3], channelsout = targets.shape[3]).to("cuda")
        print(f"model parameter count: {count_params(net)}")

        del inputs
        del targets
        del d
        
        #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)
        #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)#, weight_decay=1e-4)
        #optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)
        optimizer = getattr(torch.optim, optimizer_name)(net.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        train_losses = np.empty([0,3])
        epoch_losses = np.empty([0,3])

        # for step in range(0, inputs.shape[0]):
        for epoch in range(0, num_epochs):  # loop over the dataset multiple times
            for traini, train_use in enumerate(train_lists,1):
                print(f"**epoch {epoch+1} : running on moist {train_use}, {traini}/{len(train_lists)}**")
                data = {t:moists_keep_fno[t] for t in train_use}
                inputs, targets = datau.load_data_fno(data, data_prep, data_prep_args)
            
                for step in range(0, inputs.shape[0], batch_size):
                    ## since number may not be multiple of batch_size
                    maxstep = np.min([step+batch_size, inputs.shape[0]])

                    input1 = inputs[np.arange(step,maxstep)]
                    target1 = targets[np.arange(step,maxstep)]

                    if step == 0 and epoch == 0:
                        print(f"Training input and target shape: input {input1.shape}, target {target1.shape}")

                    optimizer.zero_grad()
                    if data_prep == "twoStep":
                        output0 = integration_methods[step_method](net, input1)
                        output1 = integration_methods[step_method](net, output0)
                    elif data_prep == "singleStep":
                        output1 = integration_methods[step_method](net, input1)
                        
                    ## grid value size to put grid and spectrum on similar footing?
                                                                     
                    if lossFunction == "huber_loss":
                        loss = losses[lossFunction](output1,
                                                 target1,
                                                 lambda_fft)
                    else:
                        loss, loss_grid, loss_fft = losses[lossFunction](output1,
                                                                     target1,
                                                                     wavenum_init,
                                                                     wavenum_init_ydir,
                                                                     lambda_fft = lambda_fft,
                                                                     grid_valid_size = targets.shape[1]*targets.shape[2],
                                                                     channels = targets.shape[3])
                    train_losses = np.concatenate([train_losses,
                                                   np.array([[epoch, step, loss.item()]])],
                                                   axis = 0)

                    loss.backward()
                    optimizer.step()
                     
                    if step % 5000 == 0:
                        print(f"{epoch+1}, {step+1} : loss {loss.item()}, lambda_fft {lambda_fft}")
                        # print(f"    loss_grid {loss_grid.item()}")
                        # print(f"    loss_fft {loss_fft.item()}")
                
            print(f"**{epoch+1}, {step+1} (final of epoch) : loss {loss.item()}**")
            # print(f"    loss_grid {loss_grid.item()}")
            # print(f"    loss_fft {loss_fft.item()}")
            
            epoch_losses = np.concatenate([epoch_losses,
                                               np.array([[epoch, step, loss.item()]])],
                                               axis = 0)

        print('Finished Training')
        
        os.mkdir(nn_dir)
        
        with open(f"{nn_dir}/model_params.yml", 'w') as h:
            yaml.dump(model_params, h, default_flow_style=False)
        
        with open(f"{nn_dir}/epoch_losses.pkl", 'wb') as h:
            pickle.dump(epoch_losses, h)
        
        torch.save(net.state_dict(), nn_loc)
        print('FNO Model and Params Saved')
        
        ## plotting epoch losses
        clipLosses = epoch_losses[:,2]
        nSamples = np.arange(0, clipLosses.shape[0])
        plt.plot(nSamples, clipLosses)
        plt.grid(alpha = .5)
        plt.xlabel("epochs")
        plt.ylabel(r"loss")
        plt.title(f"{model_name}\nloss function: {lossFunction}, lambda_fft: {np.around(lambda_fft,4)}")
        plt.savefig(f"{nn_dir}/{model_name}_losses.png")
        plt.close()

    else:
        print(f"''{nn_loc}'' exist! Loading model...")
        with open(f"{nn_dir}/model_params.yml", 'r') as h:
            model_params = yaml.load(h, Loader = yaml.Loader)
            
        with open(f"{nn_dir}/epoch_losses.pkl", 'rb') as h:
            epoch_losses = pickle.load(h)
        
        data_loc = model_params["data_loc"]
        data_prep = model_params["data_prep"]
        data_prep_args = model_params["data_prep_args"]
        data_mod_loc = model_params["data_mod_loc"]
        nn_dir = model_params["model_dir"]
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
        
        ## to get shape of input
        d = {t:moists_keep_fno[t] for t in [151]}
        inputs, targets = datau.load_data_fno(d, data_prep, data_prep_args)
        
        net = FNO2d(modes1, modes2, width, channels = inputs.shape[3], channelsout = targets.shape[3]).to("cuda")
        net.load_state_dict(torch.load(nn_loc))
        net = net.eval()
        
        del inputs
        del targets
        del d
        
    #if not os.path.exists(f"{pred_plots_dir}/pred-act.mp4"):
    if 1:
        # actual = moists_keep_fno[test[0]][:autoregsteps+1]
        
        ts_start = data_prep_args["ts_in"]
        ts_start = 1000
        ## if the model only contains dry, and singleStep
        ## starting on the first valid step that can be made with autoregsteps, or another starting tstamp
        
        actual = moists_keep_fno[test[0]][ts_start:autoregsteps+ts_start+1]
        tstamp_start = moists_keep_fno_timestamps[test[0]][ts_start]
        ## singlestep to be saved to autoreg_pred, to compare to actual later on
        autoreg_pred = actual[[0]] ## unseen data
        #previnput = torch.from_numpy(fno_form(moists_keep_fno[151][0:ts_in]).shape).float().cuda() ## seen data, part of training
        
        ## doing it directly from the fno data, since we can just compare directly
        actual_fno_in, actual_fno_tar = datau.prep_in_tar_data_2(moists_keep_fno[test[0]][:autoregsteps+1], **data_prep_args)
        # actual_fno = moists_keep_fno[test[0]][:autoregsteps+1]
        previnput = actual_fno_in[[0]]
        
        ## single step
        print(f"Running autoregression {data_prep}...")
        if data_prep == "singleStep":
            
            ## if the model only contains dry, and singleStep
            ## starting on the first valid step that can be made with autoregsteps
            actual = moists_keep_fno[test[0]][0:autoregsteps+1]
            tstamp_start = moists_keep_fno_timestamps[test[0]][0]
            ## singlestep to be saved to autoreg_pred, to compare to actual later on
            autoreg_pred = actual[[0]] ## unseen data
            #previnput = torch.from_numpy(fno_form(moists_keep_fno[151][0:ts_in]).shape).float().cuda() ## seen data, part of training
            
            ## doing it directly from the fno data, since we can just compare directly
            #actual_fno_in, actual_fno_tar = prep_in_tar_data_2(moists_keep_fno[test[0]][:autoregsteps+1], **data_prep_args)
            actual_fno = moists_keep_fno[test[0]][:autoregsteps+1]
            previnput = actual_fno[[0]]
        
            for step in range(autoregsteps):
                # grid = net.get_grid(previnput.shape, previnput.device)
                # previnput = torch.cat((previnput, grid), dim=-1)
                output = integration_methods[step_method](net, torch.tensor(previnput).cuda()).cpu().detach().numpy()
                autoreg_pred = np.concatenate([autoreg_pred, output], axis = 0) 
                previnput = output
        
        ## tsteps
        elif data_prep == "tsteps":
            
            ts_start = data_prep_args["ts_in"]
            ## if the model only contains dry, and singleStep
            ## starting on the first valid step that can be made with autoregsteps
            actual = moists_keep_fno[test[0]][ts_start:autoregsteps+ts_start+1]
            tstamp_start = moists_keep_fno_timestamps[test[0]][ts_start]
            ## singlestep to be saved to autoreg_pred, to compare to actual later on
            autoreg_pred = actual[[0]] ## unseen data
            #previnput = torch.from_numpy(fno_form(moists_keep_fno[151][0:ts_in]).shape).float().cuda() ## seen data, part of training
            
            ## doing it directly from the fno data, since we can just compare directly
            actual_fno_in, actual_fno_tar = prep_in_tar_data_2(moists_keep_fno[test[0]][:autoregsteps+1], **data_prep_args)
            # actual_fno = moists_keep_fno[test[0]][:autoregsteps+1]
            previnput = actual_fno_in[[0]]
            
            for step in range(1, autoregsteps+1):
                output = integration_methods[step_method](net, torch.tensor(previnput).cuda()).cpu().detach().numpy()
                output_step = output[:,:,:,:3]
                autoreg_pred = np.concatenate([autoreg_pred, output_step], axis = 0)
                ## shifts, removes channels associated with previous start time, and adds last of new output
                previnput = np.concatenate([previnput[:,:,:,3:], output_step], axis = 3)
                if step % 100 == 0:
                    specloss = losses[lossFunction](torch.tensor(output).cuda(),
                                                       torch.tensor(actual[[step]]).cuda(),
                                                       wavenum_init,
                                                       wavenum_init_ydir,
                                                       lambda_fft = lambda_fft,
                                                       grid_valid_size = output.shape[1]*output.shape[2],
                                                       channels = output.shape[3])[0]
                    print(f"step {step}: loss: {specloss.item()}")
                    
        if not os.path.exists(pred_plots_dir):
            os.mkdir(pred_plots_dir)
        
        with open(f"{pred_plots_dir}/pred.pkl","wb") as h:
            pickle.dump(autoreg_pred, h)
            
        gs_dir = f"{pred_plots_dir}/grid_spectrum"
        if not os.path.exists(gs_dir):
            os.mkdir(gs_dir)
        
        # tsteps_pred = autoreg_pred.shape[0]
        steps_save = [0,1,2,5,20,50,200,500]
        for step in steps_save:
            plotting.plot_2d_grid_spectrum(autoreg_pred,
                                           channels = channel_names,
                                           frame=step,
                                           savename = f"pred_step-{step}",
                                           output_dir = gs_dir,
                                           title = f"{model_name} autoregressive predictions; moist {test[0]} init",
                                           begframe = tstamp_start)
                                           
            plotting.plot_2d_grid_spectrum(actual,
                                           channels = channel_names,
                                           frame=step,
                                           savename = f"actual_step-{step}",
                                           output_dir = gs_dir,
                                           title = f"Actual predictions moist {test[0]}",
                                           begframe = tstamp_start)
        
        plotting.plot_rmse(autoreg_pred, actual, channels = channel_names, loc = f"{pred_plots_dir}/mseVtime_full.png")
        plotting.plot_acc(autoreg_pred, actual, channels = channel_names, loc = f"{pred_plots_dir}/accVtime_full.png")
        plotting.plot_spectrums(autoreg_pred, actual, channels = channel_names, loc = f"{pred_plots_dir}/spectrum_graphs_full.png")
        plotting.plot_spectrums2(autoreg_pred, actual, channels = channel_names, loc = f"{pred_plots_dir}/spectrum_graphs2_full.png")
        
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
            
        for step in np.arange(0,max_tstep_animation,1):
            if step%(4*10) == 0:
                print(f"grid spectrum plot for step {step}")
            str_step = "0"*(6-len(str(step)))+str(step)
            plotting.plot_2d_grid_spectrum(autoreg_pred,
                                           channels = channel_names,
                                           frame=step,
                                           savename = f"pred_{str_step}",
                                           output_dir = gsp_dir,
                                           title = f"{model_name} autoregressive predictions; moist {test[0]} init",
                                           cmap = cm.viridis,
                                           begframe = tstamp_start)
                                           
        os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {gsp_dir}/pred_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {pred_plots_dir}/pred.mp4')
        
        ## plot saving for animation, predictions - actual
        gspa_dir = f"{pred_plots_dir}/pred-actual_pngs"
        if not os.path.exists(gspa_dir):
            os.mkdir(gspa_dir)
        
        ## plot saving for animation
        for step in np.arange(0,max_tstep_animation,1):
            if step%(4*10) == 0:
                print(f"grid spectrum plot for step {step}")
            str_step = "0"*(6-len(str(step)))+str(step)
            plotting.plot_2d_grid_spectrum(autoreg_pred-actual,
                                           channels = channel_names,
                                           frame=step,
                                           savename = f"pred-actual_{str_step}",
                                           output_dir = gspa_dir,
                                           title = f"{model_name} autoregressive predictions-actual; moist {test[0]} init",
                                           cmap = cm.bwr,
                                           begframe = tstamp_start)
                                           
        os.system(f'ffmpeg -y -r 20 -f image2 -s 1920x1080 -i {gspa_dir}/pred-actual_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {pred_plots_dir}/pred-act.mp4')
        
        ## should implement a spectrum loss method to approximate differece between predicted and actual mean (right now it does actual snapshot)
    
## for mp4 of the actual predictions
## for single steps
if 1:
    ## for actual prediction
    actual_dir = f"{models_dir}/actual_moist-151"

    if not os.path.exists(actual_dir):
        os.makedirs(actual_dir)

    actual = moists_keep_fno[test[0]][0:autoregsteps+1]
    with open(f"{actual_dir}/actual.pkl","wb") as h:
        pickle.dump(actual, h)
    
    gsa_dir = f"{actual_dir}/actual_pngs"
    if not os.path.exists(gsa_dir):
        os.mkdir(gsa_dir)
        
    ## plot saving for animation
    for step in np.arange(0,actual.shape[0],1):
        if step%(4*10) == 0:
            print(f"grid spectrum plot for step {step}")
        str_step = "0"*(6-len(str(step)))+str(step)
        plotting.plot_2d_grid_spectrum(actual,
                                       channels = channel_names,
                                       frame=step,
                                       savename = f"actual_{str_step}",
                                       output_dir = gsa_dir,
                                       title = f"actual target values",
                                       begframe = moists_keep_fno_timestamps[151][0])
                                           
    os.system(f'ffmpeg -y -r 30 -f image2 -s 1920x1080 -i {gsa_dir}/actual_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {actual_dir}/actual.mp4')



## code for creating comparisons between each of the different runs
if 1:
    model_list = os.listdir(models_dir)
    actual_loc = "/home/exouser/nimrodxl1_mymount/lenny_outputs/models/singleSteps_1-12-24/actual_moist-151/actual.pkl"

    def load_pkl(fileloc):
        with open(fileloc, "rb") as h:
            return pickle.load(h)

    actual = load_pkl(actual_loc)

    gen_autoreg_loc = lambda model: f"/home/exouser/nimrodxl1_mymount/lenny_outputs/models/singleSteps_1-12-24/{model}/pred_plots/pred.pkl"

    # ## comparison files 
    # compare_runs = ['FNO2D_stepMethod-Eulerstep_lambda-0p0_dataPrep-singleStep',
                     # 'FNO2D_stepMethod-Eulerstep_lambda-0p16_dataPrep-singleStep',
                     # 'FNO2D_stepMethod-PECstep_lambda-0p0_dataPrep-singleStep',
                     # 'FNO2D_stepMethod-PECstep_lambda-0p16_dataPrep-singleStep',
                     # 'FNO2D_stepMethod-RK4step_lambda-0p0_dataPrep-singleStep',
                     # 'FNO2D_stepMethod-RK4step_lambda-0p16_dataPrep-singleStep',
                     # 'FNO2D_stepMethod-directstep_lambda-0p0_dataPrep-singleStep',
                     # 'FNO2D_stepMethod-directstep_lambda-0p16_dataPrep-singleStep',]

    # compare_run_names = [r"Eulerstep $\lambda=0.0$",
                         # r"Eulerstep $\lambda=0.16$",
                         # r"PECstep $\lambda=0.0$",
                         # r"PECstep $\lambda=0.16$",
                         # r"RK4step $\lambda=0.0$",
                         # r"RK4step $\lambda=0.16$",
                         # r"directstep $\lambda=0.0$",
                         # r"directstep $\lambda=0.16$",]

    # compare_run_colors = ["red",
                          # "red",
                          # "purple",
                          # "purple",
                          # "blue",
                          # "blue",
                          # "green",
                          # "green"]

    # compare_run_ls = ["-",
                      # "--",
                      # "-",
                      # "--",
                      # "-",
                      # "--",
                      # "-",
                      # "--"]    
    
    ## comparison files 
    compare_runs = ['FNO2D_stepMethod-Eulerstep_lambda-0p0_dataPrep-singleStep',
                     'FNO2D_stepMethod-Eulerstep_lambda-0p16_dataPrep-singleStep',
                     'FNO2D_stepMethod-directstep_lambda-0p0_dataPrep-singleStep',
                     'FNO2D_stepMethod-directstep_lambda-0p16_dataPrep-singleStep',]

    compare_run_names = [r"Eulerstep $\lambda=0.0$",
                         r"Eulerstep $\lambda=0.16$",
                         r"directstep $\lambda=0.0$",
                         r"directstep $\lambda=0.16$",]

    compare_run_colors = ["red",
                          "red",
                          "green",
                          "green"]

    compare_run_ls = ["-",
                      "--",
                      "-",
                      "--"]
                      

    autoreg_preds = []
    ## direct predictions from autoregressive predictions, up to 1+1000 time steps
    for c in compare_runs: 
        c_pred_loc = gen_autoreg_loc(c)
        autoreg_preds.append(load_pkl(c_pred_loc))
        
    """
    ACC -- three panels (psi1, psi2, m) with and without spectral loss, with and without PEC.
    Same for RMSE.
    Same for Spectrum

    Panels.
    """

    ## ACC
    plot_shared_dir = "/home/exouser/nimrodxl1_mymount/lenny_outputs/models/singleSteps_1-12-24/shared_plots_pecVSdirect"

    if not os.path.exists(plot_shared_dir):
        os.mkdir(plot_shared_dir)
        
    short_tsteps = 14*4
    long_tsteps = 60*4
    labels_properties = {}
    for i, rn in enumerate(compare_run_names, 0):
        labels_properties[i] = {"label" : compare_run_names[i],
                               "color" : compare_run_colors[i],
                               "linestyle" : compare_run_ls[i]}
        

    plotting.plot_compare_accs(actual[:short_tsteps,...], 
                                  [a[:short_tsteps,...] for a in autoreg_preds],
                                  labels_properties = labels_properties,
                                  channels = channel_names,
                                  loc = f"{plot_shared_dir}/plot_accs_short.png")
                                  
    plotting.plot_compare_rmse(actual[:short_tsteps,...], 
                                  [a[:short_tsteps,...] for a in autoreg_preds],
                                  labels_properties = labels_properties,
                                  channels = channel_names,
                                  loc = f"{plot_shared_dir}/plot_rmse_short.png")

    plotting.plot_compare_accs(actual[:long_tsteps,...], 
                                  [a[:long_tsteps,...] for a in autoreg_preds],
                                  labels_properties = labels_properties,
                                  channels = channel_names,
                                  loc = f"{plot_shared_dir}/plot_accs_long.png")
                                  
    plotting.plot_compare_rmse(actual[:long_tsteps,...], 
                                  [a[:long_tsteps,...] for a in autoreg_preds],
                                  labels_properties = labels_properties,
                                  channels = channel_names,
                                  loc = f"{plot_shared_dir}/plot_rmse_long.png")

    plotting.plot_compare_accs(actual[:,...], 
                                  [a[:,...] for a in autoreg_preds],
                                  labels_properties = labels_properties,
                                  channels = channel_names,
                                  loc = f"{plot_shared_dir}/plot_accs_full.png")
                                  
    plotting.plot_compare_rmse(actual[:,...], 
                                  [a[:,...] for a in autoreg_preds],
                                  labels_properties = labels_properties,
                                  channels = channel_names,
                                  loc = f"{plot_shared_dir}/plot_rmse_full.png")
                                  
    
    plotting.plot_compare_spectrums2(actual,  
                                        autoreg_preds,
                                        labels_properties,
                                        channels = channel_names,
                                        tsteps = [4*7],
                                        kex = 1,
                                        cmap = cm.rainbow,
                                        loc = f"{plot_shared_dir}/plot_spec2_1week.png")
    
    plotting.plot_compare_spectrums2(actual,  
                                        autoreg_preds,
                                        labels_properties,
                                        channels = channel_names,
                                        tsteps = [4*7*4],
                                        kex = 1,
                                        cmap = cm.rainbow,
                                        loc = f"{plot_shared_dir}/plot_spec2_4week.png")