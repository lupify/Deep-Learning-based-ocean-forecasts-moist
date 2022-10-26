import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
from saveNCfile import savenc


def load_test_data(FF,lead):

  ucur=  np.asarray(FF['SSU'])
  vcur = np.asarray(FF['SSV'])
  SSH = np.asarray(FF['SSH'])

##### convert Nans to zero ####
  ucur[np.isnan(ucur)]=0.0
  vcur[np.isnan(vcur)]=0.0
  SSH[np.isnan(SSH)]=0.0
################################
  ucur=ucur[700:,:,:]
  vcur=vcur[700:,:,:]
  SSH=SSH[700:,:,:]
  
  uv = np.zeros([np.size(ucur,0),3,np.size(ucur,1),np.size(ucur,2)])

  uv [:,0,:,:] = ucur
  uv [:,1,:,:] = vcur
  uv [:,2,:,:] = SSH

  uv_test_input = uv[0:np.size(ucur,0)-lead,:,:,:]
  uv_test_label = uv[lead:np.size(ucur,0),:,:,:]
 


## convert to torch tensor
  uv_test_input_torch = torch.from_numpy(uv_test_input).float()
  uv_test_label_torch = torch.from_numpy(uv_test_label).float()

  return uv_test_input_torch, uv_test_label_torch


def load_train_data(GG, lead,trainN):
  
     ucur=np.asarray(GG['SSU'])
#### remove NANS###########
     ocean_grid_size = np.count_nonzero(~np.isnan(ucur[0,:,:]))

     ucur[np.isnan(ucur)]=0
     ucur=ucur[0:trainN,:,:]

     vcur=np.asarray(GG['SSV'])
     vcur[np.isnan(vcur)]=0
     vcur=vcur[0:trainN,:,:]

     SSH=np.asarray(GG['SSH'])
     SSH[np.isnan(SSH)]=0
     SSH=SSH[0:trainN,:,:]

     uv = np.zeros([np.size(ucur,0),3,np.size(ucur,1),np.size(ucur,2)])
     uv[:,0,:,:] = ucur
     uv[:,1,:,:] = vcur
     uv[:,2,:,:] = SSH

     uv_train_input = uv[0:np.size(ucur,0)-lead,:,:,:]
     uv_train_label = uv[lead:np.size(ucur,0),:,:,:]

     uv_train_input_torch = torch.from_numpy(uv_train_input).float()
     uv_train_label_torch = torch.from_numpy(uv_train_label).float()

     return uv_train_input_torch, uv_train_label_torch, ocean_grid_size
