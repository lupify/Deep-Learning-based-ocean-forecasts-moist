import numpy as np
import torch
import netCDF4 as nc



def load_test_data(FF, lead, spinup):


    psi1 = np.asarray(FF['psi1'])
    psi2 = np.asarray(FF['psi2'])
    m = np.asarray(FF['m'])

    psi1 = psi1[spinup:,:,:]
    psi2 = psi2[spinup:,:,:]
    m = m[spinup:,:,:]


    uv = np.zeros([np.size(psi1, 0), 3, np.size(psi1, 1), np.size(psi1, 2)])
    uv[:, 0, :, :] = psi1
    uv[:, 1, :, :] = psi2
    uv[:, 2, :, :] = m

    uv_test_input = uv[0:np.size(psi1, 0) - lead, :, :, :]
    uv_test_label = uv[lead:np.size(psi1, 0), :, :,]

    # Convert to torch tensor
    uv_test_input_torch = torch.from_numpy(uv_test_input).float()
    uv_test_label_torch = torch.from_numpy(uv_test_label).float()

    return uv_test_input_torch, uv_test_label_torch

def load_train_data(GG, lead, spinup, trainN):
    psi1 = np.asarray(GG['psi1'])
    psi1 = psi1[spinup:spinup+trainN, :, :]

    psi2 = np.asarray(GG['psi2'])
    psi2 = psi2[spinup:spinup+trainN, :, :]

    m = np.asarray(GG['m'])
    m = m[spinup:spinup+trainN, :, :]

    uv = np.zeros([np.size(psi1, 0), 3, np.size(psi1, 1), np.size(psi1, 2)])

    uv[:, 0, :, :] = psi1
    uv[:, 1, :, :] = psi2
    uv[:, 2, :, :] = m

    uv_train_input = uv[0:np.size(psi1, 0) - lead, :, :, :]
    uv_train_label = uv[lead:np.size(psi1, 0), :, :, :]

    uv_train_input_torch = torch.from_numpy(uv_train_input).float()
    uv_train_label_torch = torch.from_numpy(uv_train_label).float()

    return uv_train_input_torch, uv_train_label_torch
