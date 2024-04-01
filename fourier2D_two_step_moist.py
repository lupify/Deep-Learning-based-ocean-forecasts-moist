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


torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, padding = 8, channels = 3, channelsout = 3):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic
        self.p = nn.Linear(channels+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, channelsout, self.width * 4) # output channel is 1: u(x, y)
        
## this is not working....bring back the grid addition in this step if cant fix...done
    def forward(self, x):
        # adds grid coordinates to the end of the input array for FNO prediction x,y coordinates
        grid = self.get_grid(x.shape, x.device)
        #grid = get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        #x = F.pad(x, [0,self.padding, 0, 0]) # pad the domain if input is non-periodic
        x = x.permute(0, 3, 1, 2)
        
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        #print(x.shape)
        x = self.q(x) # why does the resulting shape have the same?
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        #x = x[:, :-self.padding, :, :] # pad the domain if input is non-periodic
        return x

    def get_grid(self, shape, device):
        # normalized from 0 to 1?
        ## this is a pretty strange way to implement for grid free prediction
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class FNO2d_mod1(nn.Module):
    def __init__(self, modes1, modes2, width, padding = 8, channels = 3, channelsout = 3):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 8 # pad the domain if input is non-periodic
        self.c = nn.Conv2d(channels+2, channels+2, kernel_size = Union[int, Tuple[int, int, int]],)
        self.p = nn.Linear(channels+2, self.width) # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, channelsout, self.width * 4) # output channel is 1: u(x, y)
        
## this is not working....bring back the grid addition in this step if cant fix...done
    def forward(self, x):
        # adds grid coordinates to the end of the input array for FNO prediction x,y coordinates
        grid = self.get_grid(x.shape, x.device)
        #grid = get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.c(x)
        x = self.p(x)
        #x = F.pad(x, [0,self.padding, 0, 0]) # pad the domain if input is non-periodic
        x = x.permute(0, 3, 1, 2)

        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2
        #print(x.shape)
        x = self.q(x) # why does the resulting shape have the same?
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        #x = x[:, :-self.padding, :, :] # pad the domain if input is non-periodic
        return x

    def get_grid(self, shape, device):
        # normalized from 0 to 1?
        ## this is a pretty strange way to implement for grid free prediction
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

def regular_loss(output, target):

 loss = torch.mean((output-target)**2)
 return loss

def ocean_loss(output, target, ocean_grid):

 loss = (torch.sum((output-target)**2))/ocean_grid
 return loss

def estimate_cyclone(output,target,mask):

    target_reshape = target[:,:,:,0]*mask
    output_reshape = output[:,:,:,0]*mask
#    index= (target_reshape >=0.17).nonzero(as_tuple = False)
    index= (target_reshape).ge(0.17)
    output_LC = torch.masked_select(output_reshape,index)
    target_LC = torch.masked_select(target_reshape,index)

    loss = torch.mean((output_LC - target_LC)**2)

    return loss
    
def estimate_anticyclone (output,target,mask):

    target_reshape = target[:,:,:,0]*mask
    output_reshape = output[:,:,:,0]*mask
#    index= (target_reshape >=0.17).nonzero(as_tuple = False)
    index= (target_reshape).le(-0.1)
    output_LC = torch.masked_select(output_reshape,index)
    target_LC = torch.masked_select(target_reshape,index)


#    output_LC = torch.cat([output_reshape[xs,ys,zs].unsqueeze(0) for xs,ys,zs in zip(index[:,0],index[:,1],index[:,2])])
#    target_LC = torch.cat([target_reshape[xs,ys,zs].unsqueeze(0) for xs,ys,zs in zip(index[:,0],index[:,1],index[:,2])])

#    output_LC = torch.index_select(output_reshape,0,index)
#    target_LC = torch.index_select(target_reshape,0,index)

    loss = torch.mean((output_LC - target_LC)**2)

    return loss

def spectral_loss_channels_og(output, 
                           target, 
                           wavenum_init_lon, 
                           wavenum_init_lat, 
                           lambda_fft,
                           grid_valid_size,
                           lat_lon_bal = .5,
                           channels = 3):

    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    
    ## loss from grid space
    # loss_grid = torch.sum((output-target)**2)/(grid_valid_size*channels)
    loss_grid = torch.mean((output-target)**2)
    # loss1 = torch.abs((output-tnparget))/ocean_grid
 
    run_loss_run = torch.zeros(1).float().cuda()
    
    for c in range(channels):
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lat
        out_fft_lat = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,c],dim=1)),dim=2)
        target_fft_lat = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,c],dim=1)),dim=2)
        loss_fft_lat = torch.mean(torch.abs(out_fft_lat[:,wavenum_init_lat:]-target_fft_lat[:,wavenum_init_lat:]))
        ## lon
        out_fft_lon = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,c],dim=2)),dim=1)
        target_fft_lon = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,c],dim=2)),dim=1)
        loss_fft_lon = torch.mean(torch.abs(out_fft_lon[:,wavenum_init_lon:]-target_fft_lon[:,wavenum_init_lon:]))
        
        run_loss_run += (1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat
    
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/(channels*fft_loss_scale)
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    return loss, loss_grid, loss_fft

def spectral_loss_channels_sqr(output, 
                               target, 
                               wavenum_init_lon, 
                               wavenum_init_lat, 
                               lambda_fft,
                               grid_valid_size,
                               lat_lon_bal = .5,
                               channels = 3,
                               fft_loss_scale = 1./110.):
        
    """
    Grid and spectral losses, both with mse
    """
    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    
    ## loss from grid space
    # loss_grid = torch.sum((output-target)**2)/(grid_valid_size*channels)
    loss_grid = torch.mean((output-target)**2)
    # loss1 = torch.abs((output-tnparget))/ocean_grid

    run_loss_run = torch.zeros(1).float().cuda()
    
    for c in range(channels):
        ## it makes sense for me to take fft differences before...if you take mean, you lose more important differences at the equator?
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lat
        out_fft_lat = torch.abs(torch.fft.rfft(output[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
        target_fft_lat = torch.abs(torch.fft.rfft(target[:,:,:,c],dim=1))[:,wavenum_init_lon:,:]
        loss_fft_lat = torch.mean((out_fft_lat - target_fft_lat)**2)
        ## lon
        out_fft_lon = torch.abs(torch.fft.rfft(output[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        target_fft_lon = torch.abs(torch.fft.rfft(target[:,:,:,c],dim=2))[:,:,wavenum_init_lon:]
        loss_fft_lon = torch.mean((out_fft_lon - target_fft_lon)**2)

        run_loss_run += (1-lat_lon_bal)*loss_fft_lon + lat_lon_bal*loss_fft_lat
    
    ## fft_loss_scale included, so the lambda_fft is more intuitive (lambda_fft = .5 --> around half weighted shared between grid and fft loss)
    loss_fft = run_loss_run/(channels)*fft_loss_scale
    loss_fft_weighted = lambda_fft*loss_fft
    loss_grid_weighted = ((1-lambda_fft))*loss_grid
    loss = loss_grid_weighted + loss_fft_weighted
    
    return loss, loss_grid, loss_fft

def huber_loss(output,
               target,
               delta = .2):
    
    loss_grid = torch.mean(torch.abs(output-target))
    
    if loss_grid > delta:
        loss = .5*loss_grid**2
    else:
        loss = delta*(loss_grid - .5*delta)
    
    return loss
               
## these only work for single steps...
## think about inclusion of hyper viscocity in output predictions???

def RK4step(net, input_batch):
    output_1 = net(input_batch.cuda())
    output_2 = net(input_batch.cuda()+0.5*output_1)
    output_3 = net(input_batch.cuda()+0.5*output_2)
    output_4 = net(input_batch.cuda()+output_3)

    return input_batch.cuda() + (output_1+2*output_2+2*output_3+output_4)/6
 
def Eulerstep(net, input_batch, delta_t = 1.0):
    output_1 = net(input_batch.cuda())
    return input_batch.cuda() + delta_t*(output_1)
 
def PECstep(net, input_batch, delta_t = 1.0):
    output_net = net(input_batch.cuda())
    #assert output_net.shape == input_batch.shape 
    output_1 = delta_t*output_net + input_batch.cuda() ## delta_t variances, to force jacobian of output to have smaller eigenvalues -> store matrix for single timestep, the can do eigenvalue decomp on bigger cluster.
    ## torch will return tensor rank 6 -> reshape to rank 2 (128x128x3)x(128x128x3)
    #print(net(input_batch.cuda()).shape, input_batch.cuda().shape)
    return input_batch.cuda() + delta_t*0.5*(net(input_batch.cuda())+net(output_1))

def directstep(net, input_batch):
    output_1 = net(input_batch.cuda())
    return output_1        

## instead of this, will try actually including convolution layers in the nn itself?
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