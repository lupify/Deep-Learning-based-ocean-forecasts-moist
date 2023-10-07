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
                                     spectral_loss_single,\
                                     RK4step,\
                                     Eulerstep,\
                                     PECstep,\
                                     directstep#,\
                                     #get_grid

from plotting import plot_2d_quantity, \
                     animate_2d_quantity, \
                     plot_singleSpectrum

import matplotlib.pyplot as plt
import pickle
import os
from time import time
################################################################
# configs
################################################################
path_outputs = '/home/exouser/ocean_reanalysis_daily_other_baselines/outputs/'
save_fno_step_loc = "./outputs/save_fno_step_3-10-2023.data"

# FF=nc.Dataset('/home/exouser/mount/ocean_reanalysis_daily/EnKF_surface_2020_5dmean_gom.nc')
volumename = '/media/volume/sdb'

moist_loc_151 = f"{volumename}/moist_5_daily/151/output.3d.nc"
moist_loc_153 = f"{volumename}/moist_5_daily/153/output.3d.nc"
moist_loc_155 = f"{volumename}/moist_5_daily/155/output.3d.nc"
moist_loc_156 = f"{volumename}/moist_5_daily/156/output.3d.nc"
moist_loc_157 = f"{volumename}/moist_5_daily/157/output.3d.nc"
moist_loc_158 = f"{volumename}/moist_5_daily/158/output.3d.nc"
moist_loc_159 = f"{volumename}/moist_5_daily/159/output.3d.nc"
rampuptstamp = 10000

if not os.path.exists(save_fno_step_loc):
    print(f"Loading moist dataset, and saving to loc {save_fno_step_loc}")
    print("loading moist datasets...")
    moists_full = {151 : nc.Dataset(moist_loc_151),
                   153 : nc.Dataset(moist_loc_153),
                   155 : nc.Dataset(moist_loc_155),
                   156 : nc.Dataset(moist_loc_156),
                   157 : nc.Dataset(moist_loc_157),
                   158 : nc.Dataset(moist_loc_158),
                   159 : nc.Dataset(moist_loc_159),
                  }
                  
    features = ["psi1","psi2","m"]
    moists_keep = {}

    # for moist in moists_full:
        # moists_keep[moist] = {}
        # for f in features:
            # # moists_keep[moist][f] = torch.from_numpy(np.asarray(moists_full[moist][f])).float().cuda()
            # moists_keep[moist][f] = np.asarray(moists_full[moist][f])
         
    print(f"pulling {features} and concatenating as numpy array")
    for moist in moists_full:
        moists_keep[moist] = []
        for f in features:
            moists_keep[moist].append([np.asarray(moists_full[moist][f])])
        moists_keep[moist] = np.concatenate(moists_keep[moist], axis = 0)
        moists_keep[moist] = np.moveaxis(moists_keep[moist], 0, 3)

        
    moists_keep_raw = moists_keep.copy()
    ## if made error or something
    # moists_keep = moists_keep_raw.copy()

    ## computing normalized
    print("normalizing (much more efficient in numpy), and converting to torch tensors")
    moists_info = {}
    for moist in moists_full:
        moists_info[moist] = {}
        for i, feat in enumerate(features, 0):
            std = np.std(moists_keep[moist][:,:,:,i])
            mean = np.mean(moists_keep[moist][:,:,:,i])
            moists_info[moist][feat] = {"std" : std, "mean" : mean, "index" : i}
            moists_keep[moist][:,:,:,i] = (moists_keep[moist][:,:,:,i] - mean)/std
        #moists_keep[moist] = torch.from_numpy(moists_keep[moist]).float().cuda()
        
    ## ramp up period, dimensions: feature, time, height (lattitude), width (longitude)

    # moists_keep

    ## make timestep 40 timsetep_DNS
    ## files already loaded such that dt is 40 dns
    tfno = 1
    print(f"making appropriate time steps: t_FNO = {tfno} t_DNS")

    moists_keep_fno_timestamps = {}
    moists_keep_fno = {}
    for moist in moists_keep:
        moists_keep_fno_timestamps[moist] = np.arange(moists_keep[moist].shape[0])
        moists_keep_fno[moist] = moists_keep[moist][rampuptstamp:,:,:,:] 
        moists_keep_fno_timestamps[moist] = moists_keep_fno_timestamps[moist][rampuptstamp:]
        moists_keep_fno[moist] = moists_keep_fno[moist][::tfno,:,:,:]
        moists_keep_fno_timestamps[moist] = moists_keep_fno_timestamps[moist][::tfno]
        
    with open(save_fno_step_loc, "wb") as h: 
        pickle.dump([moists_keep_fno, 
                     moists_keep_fno_timestamps,
                     moists_info,
                    ], h)
                      
else:
    print(f"Loading previously saved data at loc {save_fno_step_loc}")
    with open(save_fno_step_loc, "rb") as h:
        moists_keep_fno, moists_keep_fno_timestamps, moists_info= pickle.load(h)

train = [153, 155, 156, 157, 158, 159]
test = [151]

## only works for one timestep
def concatenate_data(moists_keep_fno, 
                     timestamps,
                     moistnames, 
                     do_permute = True):
    
    input = np.zeros(shape=(0, 
                    moists_keep_fno[moistnames[0]].shape[1], 
                    moists_keep_fno[moistnames[0]].shape[2], 
                    moists_keep_fno[moistnames[0]].shape[3]), 
                    dtype=np.float64)
    
    #np.random.seed(0)
    target = input.copy()
        
    
    times_input = np.zeros(((moists_keep_fno[151].shape[0]-1)*len(moistnames),2))
    times_target = np.zeros(((moists_keep_fno[151].shape[0]-1)*len(moistnames),2))

    ind = 0
    for t in moistnames:
        # input = np.concatenate([input, moists_keep_fno[t][:-1,:,:,:].cpu()])
        # target = np.concatenate([target, moists_keep_fno[t][1:,:,:,:].cpu()])
        ti = timestamps[t][:-1].reshape(-1,1)
        tt = timestamps[t][1:].reshape(-1,1)
        train_tile = np.tile(t, [timestamps[t].shape[0]-1,1])
        
        times_input[ind:ind+(timestamps[t].shape[0]-1),:] = np.concatenate([train_tile, ti], axis = 1)
        times_target[ind:ind+(timestamps[t].shape[0]-1),:] = np.concatenate([train_tile, tt], axis = 1)
        
        input = np.concatenate([input, moists_keep_fno[t][:-1,:,:,:]])
        target = np.concatenate([target, moists_keep_fno[t][1:,:,:,:]])
        ind += moists_keep_fno[t].shape[0] - 1
        
    print(f"input.shape : {input.shape}, target.shape : {target.shape}")
    assert input.shape == target.shape
    
    if do_permute:
       indices = np.random.permutation(input.shape[0])
       input, target, times_input, times_target = \
           input[indices,:,:,:], target[indices,:,:,:], times_input[indices], times_target[indices]
       
    return torch.from_numpy(input).float().cuda(), torch.from_numpy(target).float().cuda(), times_input, times_target

## modify later when using multiple timestamps for prediction...done
def fno_form(datastep):
    # ## for the last column, we need the u(t, x, y),..., x, y (function solution, x coord, y coord)
    # datastepnp = datastep.cpu().numpy()
    # data_shape = datastepnp.shape
    # data_fno_tensor = np.zeros((1, data_shape[1],data_shape[2], data_shape[0]*data_shape[3]))
    
    
    # for x in range(data_shape[1]):
        # for y in range(data_shape[2]):
            # fnoform = np.empty(0)
            # for f in range(data_shape[3]):
                # for t in range(data_shape[0]):
                    # fnoform = np.concatenate([fnoform, [datastepnp[t, x, y, f]]])
        # data_fno_tensor[0,x,y,:] = fnoform
    
    # ## took grid method and put it here for convenience. makes more sense to do it outside of the fno nn code
    # #grid = get_grid(data_fno_tensor.shape, datastep.device)
    # #x = torch.from_numpy(data_fno_tensor).float().cuda()
    # # return torch.cat((x, grid), dim=-1)
    # return torch.from_numpy(data_fno_tensor).float().cuda()
    s = datastep.shape
    return datastep.permute(1,2,0,3).reshape((1, s[1],s[2],s[0]*s[3]))


inputs, targets, times_input, times_target = concatenate_data(moists_keep_fno, moists_keep_fno_timestamps, train, do_permute = True)

## isn't removing from memory?
if False:
    moists_keep_fno = None
    moists_keep_fno_timestamps = None
    del moists_keep_fno
    del moists_keep_fno_timestamps
    gc.collect()
    torch.cuda.empty_cache()

Nlat = 128
Nlon = 128
channels = 3
ts_train = 1 ## keep at one until modify previous code above

## nn parameters
# num_epochs = 100
num_epochs = 100

# lamda_reg =0.9
lambda_fft =0.0
wavenum_init=20
wavenum_init_ydir=20

#modes = 128
modes1 = 64
modes2 = 64
width = 20

batch_size = 1
learning_rate = 0.001

nn_loc = f"./outputs/FNO2D_test8_loss_batch{batch_size}_samples{inputs.shape[0]}_epochs{num_epochs}.pt"
torch.cuda.empty_cache()

if not os.path.exists(nn_loc):
    
    net = FNO2d(modes1, modes2, width, channels = ts_train*channels).to("cuda")
    print(count_params(net))

    #optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-4)
    #optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)#, weight_decay=1e-4)
    ## error here:
    ## "RuntimeError: The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 4"
    ## works w/o weight_decay for Adam optimizer...submit ticket google?
    ## fixed by using AdamW, which apparently is the correct implementation?
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_losses = np.empty([0,5])
    print(f"training samples: {inputs.shape[0]}")
    for epoch in range(0, num_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        
        # for step in range(0, inputs.shape[0]):
        for step in range(0, inputs.shape[0], batch_size):
            ## make separate method for this, and index w/o for loop
            # input1, target1 = inputs[np.arange(step, step+ts_train)], targets[np.arange(step+ts_train, step+2*ts_train)]
            # input1, target1 = inputs[[step]], targets[[step]]
            ## since number may not be multiple of batch_size
            maxstep = np.min([step+batch_size, inputs.shape[0]])
            
            input1 = inputs[np.arange(step,maxstep)]
            target1 = targets[np.arange(step,maxstep)]
            #input1, target1 = input[:,:,:,[0]], target[:,:,:,[0]] # psi1
            #input2, target2 = input[:,:,:,[1]], target[:,:,:,[1]] # psi2
            
            #input = input.permute()
            #target = input.permute()
            
            #t1 = time()
            # input1 = fno_form(input1)
            # target1 = fno_form(target1)
            #t2 = time()
            
            ## something is wrong with the net output prediction (only gives 1d channel, instead of 3)
            #output1 = PECstep(net, input1)
            #directoutput1 = net(input1)
            output1 = directstep(net, input1)
            # print(output1.shape)
            #print('shape of FNO2D output',output1.shape)
            #t3 = time()
            # check ocean grid number
            loss, loss_grid, loss_fft = spectral_loss_single(output1, 
                                                             target1, 
                                                             wavenum_init, 
                                                             wavenum_init_ydir, 
                                                             lambda_fft = lambda_fft, 
                                                             grid_valid_size = Nlat*Nlon)
            #t4 = time()
            train_losses = np.concatenate([train_losses, 
                                           np.array([[epoch, step, loss.item(), loss_grid.item(), loss_fft.item()]])], 
                                           axis = 0)
                                           
            loss.backward()
            #t5 = time()
            optimizer.step()
            #t6 = time()
            if step % 1000 == 0:
                print(f"inputs: {times_input[step]} + {maxstep-step-1} others")
                # print('[%d, %5d] loss: %.3f' %
                      # (epoch + 1, step + 1, loss))
                print(f"{epoch+1}, {step+1} : loss {loss}, loss_grid {loss_grid}, loss_fft {loss_fft}")
                print(f" step output: std: {output1.std()}, mean: {output1.mean()}")
                print(f" target: std {target1.std()}, mean: {target1.mean()}")
                print(f" midlatlon value comparison [0,64,64,:]:\n  output: {output1[0,64,64,:]}\n  target: {target1[0,64,64,:]}")
                #print(f" net output: std: {directoutput1.std()}, mean: {directoutput1.mean()}")

                #print(f"  times: fno_form {np.around(t2-t1,3)}, pecstep {np.around(t3 -t2,3)}, spectral_loss {np.around(t4 -t3,3)}, loss {np.around(t5 -t4,3)}, optimizer {np.around(t6 -t5,3)}")
                          
                # print('[%d, %5d] val_loss: %.3f' %
                       # (epoch + 1, step + 1, val_loss))
                running_loss = 0.0     
    print('Finished Training')
    torch.save(net.state_dict(), nn_loc)
    print('FNO Model Saved')
    
else:
    net = FNO2d(modes1, modes2, width, channels = ts_train*channels).to("cuda")
    net.load_state_dict(torch.load(nn_loc))
    net = net.eval()

## autoregression
autoregsteps=100
autoreg_pred = torch.from_numpy(np.empty([0,Nlat,Nlon,3])).float().cuda()
# previnput = torch.from_numpy(moists_keep_fno[151][[0]]).float().cuda() ## unseen data
previnput = torch.from_numpy(moists_keep_fno[153][[0]]).float().cuda() ## seen data, part of training

autoreg_pred = torch.concatenate([autoreg_pred, previnput], axis = 0)

for step in range(autoregsteps):
    # grid = net.get_grid(previnput.shape, previnput.device)
    # previnput = torch.cat((previnput, grid), dim=-1)
    previnput = PECstep(net, previnput)
    autoreg_pred = torch.concatenate([autoreg_pred, previnput], axis = 0)

animate_2d_quantity(autoreg_pred.cpu().detach().numpy()[:,:,:,0])

