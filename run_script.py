import torch.nn.functional as F
from utilities3 import *
from timeit import default_timer

import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
from data_loader_SSH import load_test_data
# from data_loader_SSH_two_step import load_test_data
from data_loader_SSH import load_train_data
# from data_loader_SSH_two_step import load_train_data
from count_trainable_params import count_parameters
import hdf5storage
import gc

torch.manual_seed(0)
np.random.seed(0)

from fourier2D_two_step_moist import SpectralConv2d,\
                                     MLP, \
                                     FNO2d, \
                                     regular_loss, \
                                     ocean_loss, \
                                     estimate_cyclone, \
                                     estimate_anticyclone,\
                                     spectral_loss_channels,\
                                     RK4step,\
                                     Eulerstep,\
                                     PECstep,\
                                     directstep#,\
                                     #get_grid
                    
# from plotting import plot_2d_quantity, \
                     # animate_2d_quantity, \
                     # plot_singleSpectrum

import plotting
import matplotlib.pyplot as plt
import pickle
import os
from time import time
import yaml

################################################################
# configs
################################################################
# path_outputs = '/home/exouser/ocean_reanalysis_daily_other_baselines/outputs/'
volumename = '/media/volume/sdb'

moist_loc_151 = f"{volumename}/moist_5_daily/151/output.3d.nc"
moist_loc_153 = f"{volumename}/moist_5_daily/153/output.3d.nc"
moist_loc_155 = f"{volumename}/moist_5_daily/155/output.3d.nc"
moist_loc_156 = f"{volumename}/moist_5_daily/156/output.3d.nc"
moist_loc_157 = f"{volumename}/moist_5_daily/157/output.3d.nc"
moist_loc_158 = f"{volumename}/moist_5_daily/158/output.3d.nc"
moist_loc_159 = f"{volumename}/moist_5_daily/159/output.3d.nc"

moist_data_locs = {
                   151 : moist_loc_151, 
                   153 : moist_loc_153,  
                   155 : moist_loc_155,  
                   156 : moist_loc_156,  
                   157 : moist_loc_157,  
                   158 : moist_loc_158,  
                   159 : moist_loc_159,
                  }

#save_fno_step_loc = f"/media/volume/sdb/lenny_outputs/save_fno_step_10-31-2023_meanNormalizedOnly.data"
output_dir = f"/media/volume/sdb/moist_5_daily/lenny_outputs"
data_loc = f"{output_dir}/model_data/save_fno_step_11-08-23_allNorm.pkl"

if not os.path.exists(data_loc):
    print(f"Loading moist dataset, and saving to loc {data_loc}")
    print("loading moist datasets...")
    moists_full = {}
    
    for m in moist_data_locs:
        moists_full[m] = nc.Dataset(moist_data_locs[m])

    channels = {
                "psi1" : ["mean", "std"],
                "psi2" : ["mean", "std"],
                "m" : ["mean", "std"],
               }
    moists_keep = {}

    print(f"pulling {list(channels.keys())} and concatenating as numpy array")
    for moist in moists_full:
        moists_keep[moist] = []
        for ch in channels:
            moists_keep[moist].append([np.asarray(moists_full[moist][ch])])
        moists_keep[moist] = np.concatenate(moists_keep[moist], axis = 0)
        moists_keep[moist] = np.moveaxis(moists_keep[moist], 0, 3)

    moists_keep_raw = moists_keep.copy()
    ## if made error or something
    # moists_keep = moists_keep_raw.copy()

    ## computing normalized
    print("normalizing (much more efficient in numpy), and converting to torch tensors")
    print("Normalizations:")
    print(channels)

    moists_info = {}
    for moist in moists_full:
        moists_info[moist] = {}
        for i, ch in enumerate(list(channels.keys()), 0):

            if "std" in channels[ch]:
                std = np.std(moists_keep[moist][:,:,:,i])
            else:
                std = 1.0
            if "mean" in channels[ch]:
                mean = np.mean(moists_keep[moist][:,:,:,i])
            else:
                mean = 0.0

            moists_info[moist][ch] = {"std" : std, "mean" : mean, "index" : i}
            moists_keep[moist][:,:,:,i] = (moists_keep[moist][:,:,:,i] - mean)/std

    ## ramp up period, dimensions: feature, time, height (lattitude), width (longitude)
    rampuptstamp = 10000
    moists_keep_fno_timestamps = {}
    moists_keep_fno = {}

    for moist in moists_keep:
        moists_keep_fno_timestamps[moist] = np.arange(moists_keep[moist].shape[0])
        moists_keep_fno[moist] = moists_keep[moist][rampuptstamp:,:,:,:]
        moists_keep_fno_timestamps[moist] = moists_keep_fno_timestamps[moist][rampuptstamp:]
        moists_keep_fno[moist] = moists_keep_fno[moist]
        moists_keep_fno_timestamps[moist] = moists_keep_fno_timestamps[moist]

    with open(data_loc, "wb") as h:
        pickle.dump([moists_keep_fno,
                     moists_keep_fno_timestamps,
                     moists_info,
                    ], h)

else:
    print(f"Loading previously saved data at loc {data_loc}")
    with open(data_loc, "rb") as h:
        moists_keep_fno, moists_keep_fno_timestamps, moists_info = pickle.load(h)

train = [153, 155, 156, 157, 158, 159]
test = [151]

## available integration steps
integration_methods = \
  {
   "directstep" : directstep,
   "RK4step" : RK4step,
   "PECstep" : PECstep, 
   "Eulerstep" : Eulerstep,
  }
  
## number of timesteps function is evaluated at as input (output)

##
## load data at specific times, not all at once...for future...how can you shuffle data this way?
## maybe presave shuffled data, the load portions with each batch

## save animations of spectrum data over time
# animate_2d_grid_spectrum(moists_keep_fno[151],
                         # channels = ["psi1","psi2","moist"],
                         # start = None,
                         # stop = 100,
                         # step = 1,
                         # savename = None,
                         # begframe = 10000)

gsdir = f"{output_dir}/plots"

if not os.path.exists(gsdir):
    os.mkdir(gsdir)

if False:
    ## plotting grid spectrum for different timestamps
    for t in [*test, *train]:
        for chan in moists_info[t].keys():
            ind = moists_info[t][chan]["index"]
            cmin, cmax = moists_keep_fno[t][:,:,:,ind].min(), moists_keep_fno[t][:,:,:,ind].max()

        for frame in range(0,moists_keep_fno[t].shape[0],2000):
            title = [f"Channel Grid and Spectrum plots for moist {t}", moists_info[t]]
            # for k in moists_info[t].keys():
                # title.append(moists_info[t][k])
            title = "\n".join([str(s) for s in title])

            plotting.plot_2d_grid_spectrum(moists_keep_fno[t],
                                              channels = ["psi1","psi2","moist"],
                                              savename = f"moist-{t}_frame-{frame}",
                                              output_dir = gsdir,
                                              kex = 1,
                                              frame = frame,
                                              title = title,
                                              begframe = 10000)

## isn't removing from memory?
def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

## only works for one timestep
def concatenate_data_singleStep(data,
                     timestamps,
                     moistnames,
                     timesteps_eval = 1,
                     do_permute = True):

    inputs = np.zeros(shape=(0,
                    data[moistnames[0]].shape[1],
                    data[moistnames[0]].shape[2],
                    data[moistnames[0]].shape[3]),
                    dtype=np.float64)

    targets = inputs.copy()

    ind = 0
    for t in moistnames:
        train_tile = np.tile(t, [timestamps[t].shape[0]-1,1])
        inputs = np.concatenate([inputs, data[t][:-1,:,:,:]])
        targets = np.concatenate([targets, data[t][1:,:,:,:]])
        ind += data[t].shape[0] - 1

    print(f"inputs.shape : {inputs.shape}, targets.shape : {targets.shape}")
    assert inputs.shape == targets.shape

    if do_permute:
       indices = np.random.permutation(inputs.shape[0])
       inputs, targets = inputs[indices,:,:,:], targets[indices,:,:,:]
       #times_input, times_target = times_input[indices], times_target[indices]

    return torch.from_numpy(inputs).float().cuda(), torch.from_numpy(targets).float().cuda()#, times_input, times_target

## modify later when using multiple timestamps for prediction...done
def fno_form(datastep):
    ## rearranges timesteps into single row, with each of the function input/solutions put together
    s = datastep.shape
    # return datastep.permute(1,2,0,3).reshape((1, s[1],s[2],s[0]*s[3]))
    return np.transpose(datastep, (1,2,0,3)).reshape((1, s[1],s[2],s[0]*s[3]))

def prep_in_tar_data(data, ts_in = 10, lead = 0, ts_out = 1):

    data_input = None
    for i in range(0, data.shape[0]-(ts_in + lead + ts_out + 1), ts_in+lead):
        sin = i
        fin = i+ts_in
        star = fin+lead
        ftar = fin+lead+ts_out
        data_in = data[sin:fin,:,:,:]
        data_tar = data[star:ftar,:,:,:]

        ## turning multiple timestep outputs into single output
        data_in_fno = fno_form(data_in)
        data_tar_fno = fno_form(data_tar)

        if data_input is None:
            data_input = data_in_fno
            data_target = data_tar_fno
        else:
            data_input = np.concatenate([data_input, data_in_fno], axis = 0)
            data_target = np.concatenate([data_target, data_tar_fno], axis = 0)

    return data_input, data_target

def concatenate_data_tsteps(data_dict, do_permute = True, pitd_kwargs = {}):
    datas_input = None
    for m in data_dict.keys():
        print(f"concatenating: {m}")
        data_input, data_target = prep_in_tar_data(data_dict[m], **pitd_kwargs)
        if datas_input is None:
            datas_input = data_input
            datas_target = data_target
        else:
            datas_input = np.concatenate([datas_input, data_input], axis = 0)
            datas_target = np.concatenate([datas_target, data_target], axis = 0)

    datas_input = torch.from_numpy(datas_input).float().cuda()
    datas_target = torch.from_numpy(datas_target).float().cuda()

    if do_permute:
       indices = np.random.permutation(datas_input.shape[0])
       datas_input = datas_input[indices]
       datas_target = datas_target[indices]

    return datas_input, datas_target

## single timestep prediction


## for multiple timesteps function outputs as input, with dictionary form of each moist file
#inputs, targets = concatenate_data_tsteps({t:moists_keep_fno[t] for t in train})

## for dry psi channels only
# inputs = inputs[:,:,:,[0,1]]
# targets = targets[:,:,:,[0,1]]


## 10/16/23 notes
## plot spectrums over time on the normalized to see if there are inconsistencies...done. none i see
## check losses for bad samples
## try different normalizations (across all 3 channels)..couple tried, still experimenting
## check matrix dimension direction to make sure input and target dimensions match...seems ok
## normalize by the mean, but dont divide by standard deviation...done, loss blows up after certain number of epochs
## -> try a bunch of normalization techniques
## plot predictive snapshots, can add layers


## 11/6/23
## normalization variations:
##  normalize moisture only
##  remove moisture, psi1, psi2 only
##

## 11/8/23
## wasn't using optimizer.zero_grad()....:(
## variations:

"""
step methods -- 
   "directstep" : directstep,
   "RK4step" : RK4step,
   "PECstep" : PECstep, 
   "Eulerstep" : Eulerstep,
"""

data_prep = "singleStep"
data_mod_loc = f"{output_dir}/model_data/save_fno_step_11-18-23_allNorm_{data_prep}.pkl"

if not os.path.exists(data_mod_loc):
    
    print(f"Concatenating training data: {data_prep} processed.")
    if data_prep == "singleStep":
        inputs, targets = concatenate_data_singleStep(moists_keep_fno, moists_keep_fno_timestamps, train, do_permute = True)
    elif data_prep == "tsteps":
        inputs, targets = concatenate_data_tsteps({t:moists_keep_fno[t] for t in train})
        
    # print(f"Saving training data to {data_mod_loc}")
    # with open(data_mod_loc, "wb") as h:
        # pickle.dump([inputs, targets], h)
else:
    # with open(data_mod_loc, "rb") as h:
        # inputs, targets = pickle.load(h)
    pass
    
# iterations
step_methods = ['PECstep', 'directstep', 'RK4step',  'Eulerstep']
lambda_ffts = [0.0, .5, 1.0]

clear_mem()

for step in step_methods:
    for lambda_fft in lambda_ffts:
        # lambda_fft = 1.0
        # step = "directstep"
        lambda_fft_str = str(lambda_fft).replace('.','p')

        model_name = f"FNO2D_{step}_lambda-{lambda_fft_str}_1"
        nn_dir = f"{output_dir}/models/{model_name}"
        nn_loc = f"{nn_dir}/model.pt"
        
        model_params = \
             {
              "model_name" : model_name,
              "data_loc" : data_loc,
              "data_prep" : data_prep,
              "data_mod_loc" : data_mod_loc,
              "model_dir" : nn_dir,
              "model_loc" : nn_loc,
              "step" : step,
              "num_epochs" : 20, 
              "lambda_fft" : lambda_fft, 
              "wavenum_init" : 20, 
              "wavenum_init_ydir" : 20,
              "modes1" : 64, 
              "modes2" : 64, 
              "width" : 20, 
              "batch_size" : 40, 
              "learning_rate" : 0.001,
             }


        if not os.path.exists(nn_dir):
            ## changing each time in beefore calls

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
            step = model_params["step"]
            
            stepmethod = integration_methods[step]
            
            net = FNO2d(modes1, modes2, width, channels = inputs.shape[3], channelsout = targets.shape[3]).to("cuda")
            print(count_params(net))

            #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)
            #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)#, weight_decay=1e-4)
            optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)

            train_losses = np.empty([0,5])
            epoch_losses = np.empty([0,5])
            print(f"training samples: {inputs.shape[0]}")
            for epoch in range(0, num_epochs):  # loop over the dataset multiple times

                # for step in range(0, inputs.shape[0]):
                for step in range(0, inputs.shape[0], batch_size):
                    ## since number may not be multiple of batch_size
                    maxstep = np.min([step+batch_size, inputs.shape[0]])

                    input1 = inputs[np.arange(step,maxstep)]
                    target1 = targets[np.arange(step,maxstep)]

                    if step == 0 and epoch == 0:
                        print(f"Training input and target shape: input {input1.shape}, target {target1.shape}")

                    optimizer.zero_grad()
                    output1 = stepmethod(net, input1)

                    ## grid value size to put grid and spectrum on similar footing?
                    loss, loss_grid, loss_fft = spectral_loss_channels(output1,
                                                                       target1,
                                                                       wavenum_init,
                                                                       wavenum_init_ydir,
                                                                       lambda_fft = lambda_fft,
                                                                       grid_valid_size = targets.shape[1]*targets.shape[2],
                                                                       channels = targets.shape[3])
                    train_losses = np.concatenate([train_losses,
                                                   np.array([[epoch, step, loss.item(), loss_grid.item(), loss_fft.item()]])],
                                                   axis = 0)

                    loss.backward()
                    optimizer.step()
                    if step % 1000 == 0:
                        # print(f"{epoch+1}, {step+1} : loss {loss.item()}, loss_grid {loss_grid}, loss_fft {loss_fft}")
                        print(f"{epoch+1}, {step+1} : loss {loss.item()}\n  loss_grid {loss_grid.item()}\n  loss_fft {loss_fft.item()}")
                        print(f"    output: std: {output1.std()}, mean: {output1.mean()}")
                        print(f"    target: std: {target1.std()}, mean: {target1.mean()}")
                        #print(f"  midlatlon value comparison [0,64,64,:]:\n  output: {output1[0,64,64,:]}\n  target: {target1[0,64,64,:]}")

                epoch_losses = np.concatenate([epoch_losses,
                                                   np.array([[epoch, step, loss.item(), loss_grid.item(), loss_fft.item()]])],
                                                   axis = 0)

            print('Finished Training')
            
            os.mkdir(nn_dir)
            
            with open(f"{nn_dir}/model_params.yml", 'w') as h:
                yaml.dump(model_params, h, default_flow_style=False)
            
            with open(f"{nn_dir}/epoch_losses.pkl", 'wb') as h:
                pickle.dump(epoch_losses, h)
            
            torch.save(net.state_dict(), nn_loc)
            print('FNO Model and Params Saved')

        else:
            with open(f"{nn_dir}/model_params.yml", 'r') as h:
                model_params = yaml.load(h, Loader = yaml.Loader)
                
            with open(f"{nn_dir}/epoch_losses.pkl", 'rb') as h:
                epoch_losses = pickle.load(h)
            
            data_loc = model_params["data_loc"]
            data_prep = model_params["data_prep"]
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
            step = model_params["step"]
            
            stepmethod = integration_methods[step]
            
            net = FNO2d(modes1, modes2, width, channels = inputs.shape[3]).to("cuda")
            net.load_state_dict(torch.load(nn_loc))
            net = net.eval()
            

        ## plot losses over training
        ## for each individual batch
        # clipLosses = np.clip(train_losses[:,2],0,10000)
        # nSamples = np.arange(0, clipLosses.shape[0])
        # epoch = train_losses[:,0]
        # epochdiff = epoch[1:]-epoch[:-1]
        # plt.plot(nSamples, clipLosses)

        # for epochDiffLoc in np.where(epochdiff == 1)[0]:
            # plt.plot(np.tile(epochDiffLoc+1,2), [clipLosses.min(), clipLosses.max()], linestyle = "--", color = "red")

        # plt.grid(alpha = .5)
        # plt.show()

        ## for epochs
        # clipLosses = np.clip(epoch_losses[:,2],0,10000)
        clipLosses = epoch_losses[:,2]
        nSamples = np.arange(0, clipLosses.shape[0])
        plt.plot(nSamples, clipLosses)
        plt.grid(alpha = .5)
        #plt.yscale("log")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        # plt.title(nn_name)
        plt.savefig(f"{nn_dir}/{model_name}_losses.png")
        plt.close()

        ## autoregression
        ## 151 autorgression

        autoregsteps=1000
        actual = moists_keep_fno[test[0]][:autoregsteps+1]
        previnput = actual[[0]]
        autoreg_pred = previnput ## unseen data
        #previnput = torch.from_numpy(fno_form(moists_keep_fno[151][0:ts_in]).shape).float().cuda() ## seen data, part of training

        for step in range(autoregsteps):
            # grid = net.get_grid(previnput.shape, previnput.device)
            # previnput = torch.cat((previnput, grid), dim=-1)
            output = stepmethod(net, torch.tensor(previnput).cuda()).cpu().detach().numpy()
            autoreg_pred = np.concatenate([autoreg_pred, output], axis = 0)
            previnput = output

        pred_plots = f"{nn_dir}/pred_plots"
        if not os.path.exists(pred_plots):
            os.mkdir(pred_plots)

        tsteps_pred = 100*4

        # tsteps_pred = autoreg_pred.shape[0]
        for step in np.arange(0, tsteps_pred, 4*20):
            print(step)
            
            plotting.plot_2d_grid_spectrum(autoreg_pred,
                                           channels = ["psi1","psi2","moist"],
                                           frame=step,
                                           savename = f"{model_name}_step-{step}.png",
                                           output_dir = pred_plots,
                                           title = f"{model_name} autoregressive predictions; moist {test[0]} init",
                                           begframe = 10000)
                                           
            plotting.plot_2d_grid_spectrum(actual,
                                           channels = ["psi1","psi2","moist"],
                                           frame=step,
                                           savename = f"actual_step-{step}.png",
                                           output_dir = pred_plots,
                                           title = f"Actual predictions moist {test[0]}",
                                           begframe = 10000)


