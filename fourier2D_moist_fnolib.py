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
    def __init__(self, modes1, modes2, width, padding = 8, channels = 3):
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
        self.q = MLP(self.width, channels, self.width * 4) # output channel is 1: u(x, y)
        
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

        
        x = self.q(x)
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

# def get_grid(shape, device):
    # # normalized from 0 to 1?
    # batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    # gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    # gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    # gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    # gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    # return torch.cat((gridx, gridy), dim=-1).to(device)

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

# old
# def spectral_loss(output, output2, target, target2, wavenum_init,wavenum_init_ydir,lamda_reg,ocean_grid):

    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    # # loss1 = torch.abs((output-target))/ocean_grid

    # out_fft = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,0],dim=2)),dim=1)
    # target_fft = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,0],dim=2)),dim=1)

    # out_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,0],dim=1)),dim=2)
    # target_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,0],dim=1)),dim=2)

    # out2_fft = torch.mean(torch.abs(torch.fft.rfft(output2[:,:,:,0],dim=2)),dim=1)
    # target2_fft = torch.mean(torch.abs(torch.fft.rfft(target2[:,:,:,0],dim=2)),dim=1)

    # out2_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(output2[:,:,:,0],dim=1)),dim=2)
    # target2_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(target2[:,:,:,0],dim=1)),dim=2)


    # # loss2 = torch.mean(torch.abs(out_fft[:,0:wavenum_init]-target_fft[:,0:wavenum_init]))
    # # loss2_ydir = torch.mean(torch.abs(out_fft_ydir[:,0:wavenum_init_ydir]-target_fft_ydir[:,0:wavenum_init_ydir]))

    # loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:]-target_fft[:,wavenum_init:]))
    # loss2_ydir = torch.mean(torch.abs(out_fft_ydir[:,wavenum_init_ydir:]-target_fft_ydir[:,wavenum_init_ydir:]))

    # # LC_loss_cyclone = estimate_cyclone (output,target,mask_rho_torch)
    # # LC_loss_anticyclone = estimate_anticyclone (output,target,mask_rho_torch)

    # loss2_next_time = torch.mean(torch.abs(out2_fft[:,wavenum_init:]-target2_fft[:,wavenum_init:]))
    # loss2_ydir_next_time = torch.mean(torch.abs(out2_fft_ydir[:,wavenum_init_ydir:]-target2_fft_ydir[:,wavenum_init_ydir:]))


    # # loss = (1-lamda_reg)*loss1 + 0.33*lamda_reg*loss2 + 0.33*lamda_reg*loss2_ydir + 0.33*LC_loss
    # loss = ((1-lamda_reg)*loss1 + 0.25*(lamda_reg)*loss2 + 0.25*(lamda_reg)*loss2_ydir + 0.25*(lamda_reg)*loss2_next_time+0.25*(lamda_reg)*loss2_ydir_next_time)

    # return loss

# def spectral_loss(output, output2, target, target2, wavenum_init,wavenum_init_ydir,lamda_reg,ocean_grid):

    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    # # loss1 = torch.abs((output-target))/ocean_grid

    # out_fft = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,0],dim=2)),dim=1)
    # target_fft = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,0],dim=2)),dim=1)

    # out_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,0],dim=1)),dim=2)
    # target_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,0],dim=1)),dim=2)

    # out2_fft = torch.mean(torch.abs(torch.fft.rfft(output2[:,:,:,0],dim=2)),dim=1)
    # target2_fft = torch.mean(torch.abs(torch.fft.rfft(target2[:,:,:,0],dim=2)),dim=1)

    # out2_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(output2[:,:,:,0],dim=1)),dim=2)
    # target2_fft_ydir = torch.mean(torch.abs(torch.fft.rfft(target2[:,:,:,0],dim=1)),dim=2)


    # # loss2 = torch.mean(torch.abs(out_fft[:,0:wavenum_init]-target_fft[:,0:wavenum_init]))
    # # loss2_ydir = torch.mean(torch.abs(out_fft_ydir[:,0:wavenum_init_ydir]-target_fft_ydir[:,0:wavenum_init_ydir]))

    # loss2 = torch.mean(torch.abs(out_fft[:,wavenum_init:]-target_fft[:,wavenum_init:]))
    # loss2_ydir = torch.mean(torch.abs(out_fft_ydir[:,wavenum_init_ydir:]-target_fft_ydir[:,wavenum_init_ydir:]))

    # # LC_loss_cyclone = estimate_cyclone (output,target,mask_rho_torch)
    # # LC_loss_anticyclone = estimate_anticyclone (output,target,mask_rho_torch)

    # loss2_next_time = torch.mean(torch.abs(out2_fft[:,wavenum_init:]-target2_fft[:,wavenum_init:]))
    # loss2_ydir_next_time = torch.mean(torch.abs(out2_fft_ydir[:,wavenum_init_ydir:]-target2_fft_ydir[:,wavenum_init_ydir:]))


    # # loss = (1-lamda_reg)*loss1 + 0.33*lamda_reg*loss2 + 0.33*lamda_reg*loss2_ydir + 0.33*LC_loss
    # loss = ((1-lamda_reg)*loss1 + 0.25*(lamda_reg)*loss2 + 0.25*(lamda_reg)*loss2_ydir + 0.25*(lamda_reg)*loss2_next_time+0.25*(lamda_reg)*loss2_ydir_next_time)

    # return loss

## removed spectrum loss
def spectral_loss_single(output, 
                         target, 
                         wavenum_init_lon, 
                         wavenum_init_lat, 
                         lambda_fft,
                         grid_valid_size,
                         channels = 3):

    # loss1 = torch.sum((output-target)**2)/ocean_grid + torch.sum((output2-target2)**2)/ocean_grid
    
    ## loss from grid space
    loss_grid = torch.sum((output-target)**2)/(grid_valid_size*channels)
    # loss1 = torch.abs((output-tnparget))/ocean_grid

    run_loss_run = torch.empty(1)
    
    for c in range(channels):
        ## losses from fft, both lat (index 1) and lon (index 2) directions
        ## lat
        out_fft_lat = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,c],dim=1)),dim=2)
        target_fft_lat = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,c],dim=1)),dim=2)
        ## lon
        out_fft_lon = torch.mean(torch.abs(torch.fft.rfft(output[:,:,:,c],dim=2)),dim=1)
        target_fft_lon = torch.mean(torch.abs(torch.fft.rfft(target[:,:,:,c],dim=2)),dim=1)
        
        loss_fft_lat = torch.mean(torch.abs(out_fft_lat[:,wavenum_init_lon:]-target_fft_lat[:,wavenum_init_lon:]))
        loss_fft_lon = torch.mean(torch.abs(out_fft_lon[:,wavenum_init_lat:]-target_fft_lon[:,wavenum_init_lat:]))

        run_loss_run = torch.sum(loss_fft_lat+loss_fft_lon)
        
    loss_fft = lambda_fft*.5*torch.mean(run_loss_run)
    loss_grid = ((1-lambda_fft))*loss_grid
    loss = loss_grid + loss_fft
    
    return loss, loss_grid, loss_fft

def RK4step(net,input_batch):
    output_1 = net(input_batch.cuda())
    output_2= net(input_batch.cuda()+0.5*output_1)
    output_3 = net(input_batch.cuda()+0.5*output_2)
    output_4 = net(input_batch.cuda()+output_3)

    return input_batch.cuda() + (output_1+2*output_2+2*output_3+output_4)/6
    
delta_t = 1.0

def Eulerstep(net,input_batch):
    output_1 = net(input_batch.cuda())
    return input_batch.cuda() + delta_t*(output_1)


def PECstep(net,input_batch):
    output_net = net(input_batch.cuda())
    assert output_net.shape == input_batch.shape 
    output_1 = output_net + input_batch.cuda()
    #print(net(input_batch.cuda()).shape, input_batch.cuda().shape)
    return input_batch.cuda() + delta_t*0.5*(net(input_batch.cuda())+net(output_1))

def directstep(net,input_batch):
    output_1 = net(input_batch.cuda())
    return output_1

def main():
    ################################################################
    # configs
    ################################################################
    path_outputs = '/home/exouser/ocean_reanalysis_daily_other_baselines/outputs/'

    # FF=nc.Dataset('/home/exouser/mount/ocean_reanalysis_daily/EnKF_surface_2020_5dmean_gom.nc')
    volumename = '/home/exouser/moist'

    moist_loc_151 = f"{volumename}/moist_5_daily/151/output.3d.nc"
    moist_loc_153 = f"{volumename}/moist_5_daily/153/output.3d.nc"
    moist_loc_155 = f"{volumename}/moist_5_daily/155/output.3d.nc"
    moist_loc_156 = f"{volumename}/moist_5_daily/156/output.3d.nc"
    moist_loc_157 = f"{volumename}/moist_5_daily/157/output.3d.nc"
    moist_loc_158 = f"{volumename}/moist_5_daily/158/output.3d.nc"
    moist_loc_159 = f"{volumename}/moist_5_daily/159/output.3d.nc"

    print("loading moist datasets...")
    moists_full = {"151" : nc.Dataset(moist_loc_151),
                   "153" : nc.Dataset(moist_loc_153),
                   "155" : nc.Dataset(moist_loc_155),
                   "156" : nc.Dataset(moist_loc_156),
                   "157" : nc.Dataset(moist_loc_157),
                   "158" : nc.Dataset(moist_loc_158),
                   "159" : nc.Dataset(moist_loc_159),
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

    ## computing normalized
    print("normalizing (much more efficient in numpy), and converting to torch tensors")
    moists_info = {}
    for moist in moists_full:
        moists_info[moist] = {}
        for i, feat in enumerate(features, 0):
            std = np.std(moists_keep[moist][:,:,:,i])
            mean = np.mean(moists_keep[moist][:,:,:,i])
            moists_info[moist][feat] = {"std" : std, "mean" : mean}
            moists_keep[moist][:,:,:,i] = (moists_keep[moist][:,:,:,i] - mean)/std
        moists_keep[moist] = torch.from_numpy(moists_keep[moist]).float().cuda()
        
    ## ramp up period, dimensions: feature, time, height (lattitude), width (longitude)

    train = ["153", "155", "156", "157", "158", "159"]
    test = ["151"]

    rampuptstamp = 10000
    lead = 6000

    ## previous code 
    if False:
        mask_rho = np.asarray(FF['mask_rho'])
        mask_rho_torch= torch.from_numpy(mask_rho).float().cuda()
        lead = 5
        delta_t = 1.0
        psi_test_input_Tr_torch, psi_test_label_Tr_torch,_  = load_test_data(FF,lead)


        M_test_level1=torch.mean((psi_test_input_Tr_torch.flatten()))
        STD_test_level1=torch.std((psi_test_input_Tr_torch.flatten()))


        psi_test_input_Tr_torch_norm_level1 = ((psi_test_input_Tr_torch[:,0,None,:,:]-M_test_level1)/STD_test_level1)
        psi_test_label_Tr_torch_norm_level1 = ((psi_test_label_Tr_torch[:,0,None,:,:]-M_test_level1)/STD_test_level1)

        print('mean value',M_test_level1)
        print('std value',STD_test_level1)








    ## previous code

    psi_test_label_Tr = psi_test_label_Tr_torch.detach().cpu().numpy()
    Nlat = np.size(psi_test_label_Tr,2)
    Nlon = np.size(psi_test_label_Tr,3)

    batch_size = 10 
    num_epochs = 100
    num_samples = 2

    lamda_reg =0.9
    wavenum_init=50
    wavenum_init_ydir=50



    modes = 128
    width = 20

    batch_size = 5
    learning_rate = 0.001

    ################################################################
    # training and evaluation
    ################################################################
    net = FNO2d(modes, modes, width).cuda()
    print(count_params(net))
    net.load_state_dict(torch.load('BNN_FNO2D_LC_loss_Eulerstep_SSH_ocean_spectral_loss_modes_128_wavenum50lead5.pt'))
    net.eval()

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(0, num_epochs):  # loop over the dataset multiple times

     running_loss = 0.0


     for k in range(1993,2020):
      print('File index',k)

      ## normalization, and loading data
      G = nc.Dataset('/home/exouser/mount/ocean_reanalysis_daily/EnKF_surface_'+str(k)+'_5dmean_gom.nc')
      trainN=350
      psi_train_input_Tr_torch, psi_train_label_Tr_torch, psi_train_label2_Tr_torch, ocean_grid  = load_train_data(G,lead,trainN)

      M_train_level1=torch.mean((psi_train_input_Tr_torch.flatten()))
      STD_train_level1=torch.std((psi_train_input_Tr_torch.flatten()))




      psi_train_input_Tr_torch_norm_level1 = ((psi_train_input_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1)
      psi_train_label_Tr_torch_norm_level1 = ((psi_train_label_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1)
      psi_train_label2_Tr_torch_norm_level1 = ((psi_train_label2_Tr_torch[:,0,None,:,:]-M_train_level1)/STD_train_level1)


      for step in range(0,trainN-2*batch_size,batch_size):
            # get the inputs; data is a list of [inputs, labels]
            indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
            input_batch, label_batch, label2_batch = psi_train_input_Tr_torch_norm_level1[indices,:,:,:], psi_train_label_Tr_torch_norm_level1[indices,:,:,:],psi_train_label2_Tr_torch_norm_level1[indices,:,:,:]

            input_batch = input_batch.permute(0,2,3,1)
            label_batch = label_batch.permute(0,2,3,1)
            label2_batch = label2_batch.permute(0,2,3,1)

            print('shape of input', input_batch.shape)
            print('shape of label1', label_batch.shape)
            print('shape of label2', label2_batch.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
    #        output,_,_,_,_,_,_ = net(input_batch.cuda())
            output = PECstep(net,input_batch.cuda())
            output2 = PECstep(net,output.cuda())
            print('shape of FNO2D output',output.shape)
            print('shape of FNO2D output2',output2.shape)
            #        loss = regular_loss(output, label_batch_crop.cuda())
            # print statistics
            loss = spectral_loss(output, output2, label_batch.cuda(),label2_batch.cuda(),wavenum_init,wavenum_init_ydir,lamda_reg,(torch.tensor(ocean_grid)).cuda())
    #        loss = ocean_loss(output, label_batch_crop.cuda(),(torch.tensor(ocean_grid)).cuda())
            loss.backward()
            optimizer.step()
    #        output_val = Eulerstep(net,(psi_test_input_Tr_torch_norm_level1[0:num_samples,:,0:Nlat,0:Nlon].reshape([num_samples,1,Nlat,Nlon])).permute(0,2,3,1))
    #        output_val2 = Eulerstep(net,output_val)
    #        val_loss = spectral_loss(output_val, output_val2,(psi_test_label_Tr_torch_norm_level1[0:num_samples,:,0:Nlat,0:Nlon].reshape([num_samples,1,Nlat,Nlon])).permute(0,2,3,1).cuda(),wavenum_init,wavenum_init_ydir,lamda_reg,(torch.tensor(ocean_grid)).cuda())
    #        val_loss = ocean_loss(output_val, psi_test_label_Tr_torch[0:num_samples,:,0:Nlat-2,0:Nlon-1].reshape([num_samples,1,Nlat-2,Nlon-1]).cuda(),(torch.tensor(ocean_grid)).cuda())
            if step % 100 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, loss))
    #            print('[%d, %5d] val_loss: %.3f' %
    #                  (epoch + 1, step + 1, val_loss))
                running_loss = 0.0
                
    print('Finished Training')

    torch.save(net.state_dict(), './BNN_FNO2D_two_step__loss_PECstep_SSH_ocean_spectral_loss_modes_'+str(modes)+'_wavenum'+str(wavenum_init)+'delta_t_'+str(delta_t)+'lead'+str(lead)+'.pt')

    print('BNN Model Saved')


    ############# Auto-regressive prediction #####################

    psi_test_label_Tr_torch_denorm = psi_test_label_Tr_torch_norm_level1*STD_test_level1+M_test_level1
    psi_test_label_Tr = psi_test_label_Tr_torch_denorm.detach().cpu().numpy()

    M=100
    autoreg_pred = np.zeros([M,1,Nlat,Nlon])

    for k in range(0,M):

      if (k==0):

        out = (PECstep(net,(psi_test_input_Tr_torch_norm_level1[k,:,0:Nlat,0:Nlon].reshape([1,1,Nlat,Nlon])).permute(0,2,3,1).cuda()))
        autoreg_pred[k,:,:,:] = (out.permute(0,3,1,2)).detach().cpu().numpy()

      else:

        out = (PECstep(net,(torch.from_numpy(autoreg_pred[k-1,:,0:Nlat,0:Nlon].reshape([1,1,Nlat,Nlon])).float()).permute(0,2,3,1).cuda()))
        autoreg_pred[k,:,:,:] = (out.permute(0,3,1,2)).detach().cpu().numpy()



    M_test_level1 = M_test_level1.detach().cpu().numpy()
    STD_test_level1 = STD_test_level1.detach().cpu().numpy()

    matfiledata = {}
    matfiledata[u'prediction'] = autoreg_pred*STD_test_level1+M_test_level1
    matfiledata[u'Truth'] = psi_test_label_Tr
    hdf5storage.write(matfiledata, '.', path_outputs+'predicted_FNO_2D_two_step_loss_PECstep_SSH_level_ocean_spectral_loss_5day_modes_'+str(modes)+'train_wavenumber'+str(wavenum_init)+'delta_t_'+str(delta_t)+'lead'+str(lead)+'lambda_'+str(lamda_reg)+'.mat', matlab_compatible=True)

    print('Saved Predictions')

if __name__ == "__main__":
    #main()
    pass