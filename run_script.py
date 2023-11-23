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
import scipy.signal as signal

torch.manual_seed(0)
np.random.seed(0)

from fourier2D_two_step_moist import SpectralConv2d,\
                                     MLP, \
                                     FNO2d, \
                                     regular_loss, \
                                     ocean_loss, \
                                     estimate_cyclone, \
                                     estimate_anticyclone,\
                                     spectral_loss_channels_sqr,\
                                     spectral_loss_channels_og,\
                                     RK4step,\
                                     Eulerstep,\
                                     PECstep,\
                                     directstep#,\
                                     #get_grid
                                     # spectral_loss_channels,
                    
# from plotting import plot_2d_quantity, \
                     # animate_2d_quantity, \
                     # plot_singleSpectrum

import plotting
import matplotlib.pyplot as plt
import pickle
import os
from time import time
import yaml
import pprint

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
data_loc = f"{data_dir}/save_fno_step_11-21-23_allNorm.pkl"

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

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
    channels_list = ["psi1", "psi2", "m"]
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
    rampup_tstamp = 10000
    moists_keep_fno_timestamps = {}
    moists_keep_fno = {}

    for moist in moists_keep:
        moists_keep_fno_timestamps[moist] = np.arange(moists_keep[moist].shape[0])
        moists_keep_fno[moist] = moists_keep[moist][rampup_tstamp:,:,:,:]
        moists_keep_fno_timestamps[moist] = moists_keep_fno_timestamps[moist][rampup_tstamp:]
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
  
## instead of this, will try actually including convolution layers in the nn itself
# def laplacian(field, step = 1):
    # lapkernel = np.tile(np.array([[[0, 1, 0],
                        # [1, -4, 1],
                        # [0, 1, 0]]]), (field.shape[0],1,1))
    # filters.convolve(field,lapkernel,mode='wrap')/step**2
    # return field

# def laplacian_mod(data, channels = [0,1]):
    # datalap = np.empty(data[:,:,:,channels].shape)
    # for ch in channels:
        # datalap = np.concatenate([datalap, laplacian(datalap[...,ch])], axis = -1)
        # print(datalap.shape)
        # raise

## number of timesteps function is evaluated at as input (output)
##
## load data at specific times, not all at once...for future...how can you shuffle data this way?
## maybe presave shuffled data, the load portions with each batch
## isn't removing from memory?
def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()

## only works for one timestep
def concatenate_data_singleStep(data_dict,
                                 timesteps_eval = 1,
                                 do_permute = True):#timestamps,
    data_keys = list(data_dict.keys())
    inputs = np.zeros(shape=(0,
                    data_dict[data_keys[0]].shape[1],
                    data_dict[data_keys[0]].shape[2],
                    data_dict[data_keys[0]].shape[3]),
                    dtype=np.float64)

    targets = inputs.copy()

    ind = 0
    for t in data_keys:
        #train_tile = np.tile(t, [timestamps[t].shape[0]-1,1])
        inputs = np.concatenate([inputs, data_dict[t][:-1,:,:,:]])
        targets = np.concatenate([targets, data_dict[t][1:,:,:,:]])
        ind += data_dict[t].shape[0] - 1

    #print(f"inputs.shape : {inputs.shape}, targets.shape : {targets.shape}")
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

def prep_in_tar_data(data, ts_in = 5, lead = 0, ts_out = 5):
    data_input = None
    for i in range(0, data.shape[0]-(ts_in + lead + ts_out + 1), ts_in+lead):
        sin = i
        fin = i+ts_in
        star = fin+lead
        ftar = star+ts_out
        data_in = data[sin:fin,:,:,:]
        data_tar = data[star:ftar,:,:,:]

        ## turning multiple timestep outputs into single output
        data_in_fno = fno_form(data_in)
        data_tar_fno = fno_form(data_tar)
        #data_in_fno = data_in
        #data_tar_fno = data_tar

        if data_input is None:
            data_input = data_in_fno
            data_target = data_tar_fno
        else:
            data_input = np.concatenate([data_input, data_in_fno], axis = 0)
            data_target = np.concatenate([data_target, data_tar_fno], axis = 0)

    return data_input, data_target

def prep_in_tar_data_2(data, ts_in = 5, lead = 0, ts_out = 5, overlap = True):
    
    sin = 0
    fin = sin+ts_in
    star = fin+lead
    ftar = star+ts_out
    
    if overlap:
        skip = ts_in
    else:
        skip = ts_in+lead+ts_out
        
    envelope = ts_in+lead+ts_out
    
    data_in_indices = []
    for i in range(sin, data.shape[0]-envelope, skip):
        data_in_indices.append(np.arange(ts_in)+i)
    data_in_indices = np.array(data_in_indices).flatten()
    data_in = data[data_in_indices]
    
    data_tar_indices = []
    for i in range(star, data.shape[0]-ts_out, skip):
        data_tar_indices.append(np.arange(ts_out)+i)
    data_tar_indices = np.array(data_tar_indices).flatten()
    data_tar = data[data_tar_indices]
    
    s = data_in.shape
    data_input = np.transpose(data_in, (1,2,0,3)).reshape((s[1], s[2], s[0]//ts_in, s[3]*ts_in)).transpose((2,0,1,3))
    
    r = data_tar.shape
    data_target = np.transpose(data_tar, (1,2,0,3)).reshape((r[1], r[2], r[0]//ts_out, r[3]*ts_out)).transpose((2,0,1,3))
    
    return data_input, data_target

def concatenate_data_tsteps(data_dict, do_permute = True, pitd_kwargs = {}):
    datas_input = None
    print(pitd_kwargs)
    for m in data_dict.keys():
        print(f"concatenating: {m}")
        # data_input, data_target = prep_in_tar_data(data_dict[m], **pitd_kwargs)
        data_input, data_target = prep_in_tar_data_2(data_dict[m], **pitd_kwargs)
        if datas_input is None:
            datas_input = data_input
            datas_target = data_target
        else:
            datas_input = np.concatenate([datas_input, data_input], axis = 0)
            datas_target = np.concatenate([datas_target, data_target], axis = 0)
    
    if do_permute:
        print("premuting...")
        indices = np.random.permutation(datas_input.shape[0])
        datas_input = datas_input[indices]
        datas_target = datas_target[indices]
       
    datas_input = torch.from_numpy(datas_input).float().cuda()
    datas_target = torch.from_numpy(datas_target).float().cuda()
    
    return datas_input, datas_target

"""
step methods -- 
   "directstep" : directstep,
   "RK4step" : RK4step,
   "PECstep" : PECstep, 
   "Eulerstep" : Eulerstep,
"""

data_prep = "singleStep"
# data_prep = "tsteps"
data_mod_loc = f"{output_dir}/model_data/fno_{data_prep}_11-23-23_allNorm.pkl"

if data_prep == "singleStep":
    data_prep_args = {"ts_in" : 1, "lead" : 0, "ts_out" : 1, "overlap" : True}
elif data_prep == "tsteps":
    data_prep_args = {"ts_in" : 16, "lead" : 0, "ts_out" : 1, "overlap" : True}

def load_data_fno(data_dict, data_prep, data_prep_args, do_permute = True):
    print(f"Concatenating training data. data_prep: {data_prep}")
    if data_prep == "singleStep":
        data_prep_args = {"ts_in" : 1, "lead" : 0, "ts_out" : 1, "overlap" : True}
        inputs, targets = concatenate_data_singleStep(data_dict, do_permute = do_permute)
        # test_inputs, test_targets = concatenate_data_singleStep(moists_keep_fno, test, do_permute = False)
    elif data_prep == "tsteps":
        data_prep_args = {"ts_in" : 16, "lead" : 0, "ts_out" : 1, "overlap" : True}
        inputs, targets = concatenate_data_tsteps(data_dict, do_permute = True, pitd_kwargs = data_prep_args)
        # test_inputs, test_targets = concatenate_data_tsteps({t:moists_keep_fno[t] for t in test}, do_permute = False, pitd_kwargs = data_prep_args)
    print(f"inputs.shape : {inputs.shape}, targets.shape : {targets.shape}")
    return inputs, targets

# iterations
train = [153, 155, 156, 157, 158, 159, 162, 163, 164, 701, 702, 703, 704, 705, 706, 707]
test = [151]

## available integration steps
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
  }
  
## only direct step works for tstep method at the moment
# step_methods = ['directstep']
# lambda_ffts = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

step_methods = ['PECstep', 'directstep', 'RK4step',  'Eulerstep']
lambda_ffts = [0.0, .5, 1.0]

clear_mem()
models_dir = f"{output_dir}/models/singleSteps_11-23-23/"

load_num = 4
mi = 0
train_lists = []
train_copy = np.copy(train)
np.random.shuffle(train_copy)
while mi < len(train_copy):
    train_lists.append(train_copy[mi:mi+load_num])
    mi+=load_num

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    
for step_method in step_methods:
    for lambda_fft in lambda_ffts:
    
        lambda_fft_str = str(lambda_fft).replace('.','p')
        model_name = f"FNO2D_stepMethod-{step_method}_lambda-{lambda_fft_str}_dataPrep-{data_prep}"
        nn_dir = f"{models_dir}/{model_name}"
        nn_loc = f"{nn_dir}/model.pt"
        
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
                  "num_epochs" : 8, 
                  "lambda_fft" : lambda_fft, 
                  "wavenum_init" : 20, 
                  "wavenum_init_ydir" : 20,
                  "modes1" : 64, 
                  "modes2" : 64, 
                  "width" : 32, 
                  "batch_size" : 20, 
                  "learning_rate" : 0.001,
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
            lossFunction = model_params["lossFunction"]
            
            pprint.pprint("model_params:")
            pprint.pprint(model_params)
            
            ## to get shape of input
            d = {t:moists_keep_fno[t] for t in [151]}
            inputs, targets = load_data_fno(d, data_prep, data_prep_args)
            
            net = FNO2d(modes1, modes2, width, channels = inputs.shape[3], channelsout = targets.shape[3]).to("cuda")
            print(f"model parameter count: {count_params(net)}")

            del inputs
            del targets
            del d
            
            #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)
            #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)#, weight_decay=1e-4)
            optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)

            train_losses = np.empty([0,5])
            epoch_losses = np.empty([0,5])

            # for step in range(0, inputs.shape[0]):
            for epoch in range(0, num_epochs):  # loop over the dataset multiple times
                for traini, train_use in enumerate(train_lists,1):
                    print(f"**epoch {epoch+1} : running on moist {train_use}, {traini}/{len(train_lists)}**")
                    data = {t:moists_keep_fno[t] for t in train_use}
                    inputs, targets = load_data_fno(data, data_prep, data_prep_args)
                
                    for step in range(0, inputs.shape[0], batch_size):
                        ## since number may not be multiple of batch_size
                        maxstep = np.min([step+batch_size, inputs.shape[0]])

                        input1 = inputs[np.arange(step,maxstep)]
                        target1 = targets[np.arange(step,maxstep)]

                        if step == 0 and epoch == 0:
                            print(f"Training input and target shape: input {input1.shape}, target {target1.shape}")

                        optimizer.zero_grad()
                        output1 = integration_methods[step_method](net, input1)

                        ## grid value size to put grid and spectrum on similar footing?
                        loss, loss_grid, loss_fft = losses[lossFunction](output1,
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
                         
                        if step % 5000 == 0:
                            # print(f"{epoch+1}, {step+1} : loss {loss.item()}, loss_grid {loss_grid}, loss_fft {loss_fft}")
                            print(f"{epoch+1}, {step+1} : loss {loss.item()}, lambda_fft {lambda_fft}")
                            print(f"    loss_grid {loss_grid.item()}")
                            print(f"    loss_fft {loss_fft.item()}")
                            # print(f"    output: std: {output1.std()}, mean: {output1.mean()}")
                            # print(f"    target: std: {target1.std()}, mean: {target1.mean()}")
                            #print(f"  midlatlon value comparison [0,64,64,:]:\n  output: {output1[0,64,64,:]}\n  target: {target1[0,64,64,:]}")
                    
                print(f"**{epoch+1}, {step+1} (final of epoch) : loss {loss.item()}**")
                print(f"    loss_grid {loss_grid.item()}")
                print(f"    loss_fft {loss_fft.item()}")
                
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
            
            ## plotting epoch losses
            # clipLosses = np.clip(epoch_losses[:,2],0,10000)
            clipLosses = epoch_losses[:,2]
            nSamples = np.arange(0, clipLosses.shape[0])
            plt.plot(nSamples, clipLosses)
            plt.grid(alpha = .5)
            #plt.yscale("log")
            plt.xlabel("epochs")
            plt.ylabel("loss: {")
            # plt.title(nn_name)
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
            step_method = model_params["step_method"]
            lossFunction = model_params["lossFunction"]
            
            ## to get shape of input
            d = {t:moists_keep_fno[t] for t in [151]}
            inputs, targets = load_data_fno(d, data_prep, data_prep_args)
            
            net = FNO2d(modes1, modes2, width, channels = inputs.shape[3], channelsout = targets.shape[3]).to("cuda")
            net.load_state_dict(torch.load(nn_loc))
            net = net.eval()
            
            del inputs
            del targets
            del d
            
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

        ## autoregression
        ## 151 autorgression

        ## ashesh code, fixed 
        # nn_loc_asheshTrained = "/media/volume/sdb/moist_5_daily/lenny_outputs/models/ashesh_debug/FNO2D_ts_loss_psiOnlyDry_batch-20_samples-49494_epochs-100(partial)_kx-20.pt"
        # nn_dir = "/media/volume/sdb/moist_5_daily/lenny_outputs/models/ashesh_debug"
        # net = FNO2d_ashesh(64, 64, 20).to("cuda")
        # net.load_state_dict(torch.load(nn_loc_asheshTrained))
        # net = net.eval()
        # step_method = "Eulerstep"
        # model_name = "ashesh_debug"
        ## use this too with asheshes code
        # actual = actual[...,[0,1]]
        
        pred_plots_dir = f"{nn_dir}/pred_plots"
        
        if not os.path.exists(f"{pred_plots_dir}/spectrum_graphs.png"):
            autoregsteps=1000
            # actual = moists_keep_fno[test[0]][:autoregsteps+1]
            
            ts_start = data_prep_args["ts_in"]
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
                    # grid = net.get_grid(previnput.shape, previnput.device)
                    # previnput = torch.cat((previnput, grid), dim=-1)
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
                                                           grid_valid_size = targets.shape[1]*targets.shape[2],
                                                           channels = targets.shape[3])[0]
                        print(f"step {step}: loss: {specloss.item()}")
                        
            if not os.path.exists(pred_plots_dir):
                os.mkdir(pred_plots_dir)
            
            gs_dir = f"{pred_plots_dir}/grid_spectrum"
            if not os.path.exists(gs_dir):
                os.mkdir(gs_dir)
                
            # tsteps_pred = autoreg_pred.shape[0]
            steps_save = [0,1,2,5,20,50,200,500,1000]
            #steps_save = [0]
            for step in steps_save:
                plotting.plot_2d_grid_spectrum(autoreg_pred,
                                               channels = ["psi1","psi2","moist"],
                                               frame=step,
                                               savename = f"{model_name}_step-{step}",
                                               output_dir = gs_dir,
                                               title = f"{model_name} autoregressive predictions; moist {test[0]} init",
                                               begframe = tstamp_start)
                                               
                plotting.plot_2d_grid_spectrum(actual,
                                               channels = ["psi1","psi2","moist"],
                                               frame=step,
                                               savename = f"actual_step-{step}",
                                               output_dir = gs_dir,
                                               title = f"Actual predictions moist {test[0]}",
                                               begframe = tstamp_start)
            
            plotting.plot_squared_error(autoreg_pred, actual, loc = f"{pred_plots_dir}/mseVtime.png")
            plotting.plot_acc(autoreg_pred, actual, loc = f"{pred_plots_dir}/accVtime.png")
            plotting.plot_spectrums(autoreg_pred, actual, loc = f"{pred_plots_dir}/spectrum_graphs.png")
            
            # with open(f"{nn_dir}/predictions_{test[0]}.pkl","wb") as h:
                # pickle.dump(autoreg_pred, h)
            
            ## takes a ton of time?
            # stop_frame = 50
            # animate_2d_grid_spectrum(autoreg_pred,
                                 # channels = ["psi1","psi2","moist"],
                                 # start = None, 
                                 # stop = stop_frame, 
                                 # step = 1,
                                 # savename = f"autoreg_pred_{test[0]}_stop-{stop_frame}",
                                 # output_dir = pred_plots_dir,
                                 # title = model_name,
                                 # kex = 1,
                                 # begframe = 10000,
                                 # interval = 100)
                                 
            # animate_2d_grid_spectrum(actual,
                                 # channels = ["psi1","psi2","moist"],
                                 # start = None, 
                                 # stop = 100, 
                                 # step = 1,
                                 # savename = f"actual_{test[0]}",
                                 # output_dir = pred_plots_dir,
                                 # title = "actual target",
                                 # kex = 1,
                                 # begframe = 10000,
                                 # interval = 200)