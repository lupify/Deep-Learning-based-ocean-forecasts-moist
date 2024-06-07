import pprint
import netCDF4 as nc
## testing the spectral_loss_channels...want (initially) for the fft loss to be same order as grid loss with noise

targett = torch.tensor(moists_keep_fno[151][[0]])
outputt = torch.tensor(targett)
outputt_noise = outputt + torch.tensor(np.random.normal(scale = .001, size = outputt.shape))

# targett = torch.tensor(np.random.normal(scale = 1, size = outputt.shape))
# outputt = torch.tensor(targett)

## computing noisy dataset, for scaling factor on spectrum loss channels for fft norm
## could potentiall do for each channel? seems like moist has a little more loss typically via fft
scales = 10.**(-np.arange(0,10))
dict_losses = {}
ratios = []
for scale in scales:
    losses = np.empty((1,3))
    for i in range(100):
        outputt_noise = outputt + torch.tensor(np.random.normal(scale = scale, size = outputt.shape))
        outputt_fft_noise = torch.fft.rfft(outputt_noise[...,0], axis = 2)
        loss, loss_grid, loss_fft = spectral_loss_channels_sqr(outputt_noise[...,[0,1]], targett[...,[0,1]], 0, 0, .5, 128*128, 2, fft_loss_scale = 1)
        # loss, loss_grid, loss_fft = spectral_loss_channels(outputt_noise, targett, 20, 20, .5, 128*128, 3, 1)
        losses = np.concatenate([losses, np.array([[loss.item(), loss_grid.item(), loss_fft.item()]])], axis = 0)
    
    losses_means = np.mean(losses, axis = 0)
    dict_losses[scale] = {"loss_grid" : losses_means[1], "loss_fft" : losses_means[2], "loss_ratio" : losses_means[1]/losses_means[2]}
    ratios.append(losses_means[1]/losses_means[2])
    
pprint.pprint(dict_losses)
print(f"ave ratio: {np.mean(ratios)}")
print(f"std ratio: {np.std(ratios)}")

## ratio between grid and squared fft loss is 32, when data is just noise...
## from moist data 151, the ratio average is around 

## random from norm: 
"""
{1e-09: {'loss_fft': 6.241789987502076e-09,
         'loss_grid': 9.607427620256111e-11,
         'loss_ratio': 0.01539210329007071},
 1e-08: {'loss_fft': 6.111792808307744e-07,
         'loss_grid': 9.62191948900629e-09,
         'loss_ratio': 0.015743203002443473},
 1e-07: {'loss_fft': 6.304207821895113e-07,
         'loss_grid': 9.70350179645213e-09,
         'loss_ratio': 0.01539210329131433},
 1e-06: {'loss_fft': 6.17291067094903e-05,
         'loss_grid': 9.71813858394007e-07,
         'loss_ratio': 0.015743203007416908},
 1e-05: {'loss_fft': 6.367243350063068e-05,
         'loss_grid': 9.80052680592293e-07,
         'loss_ratio': 0.015392103406611362},
 0.0001: {'loss_fft': 0.006234633242619345,
          'loss_grid': 9.815309976692686e-05,
          'loss_ratio': 0.01574320348083378},
 0.001: {'loss_fft': 0.006430262008703227,
         'loss_grid': 9.897532991837869e-05,
         'loss_ratio': 0.015392114626809547},
 0.01: {'loss_fft': 0.6296325918759675,
        'loss_grid': 0.00991246355839664,
        'loss_ratio': 0.01574325040713476},
 0.1: {'loss_fft': 0.6429228470858662,
       'loss_grid': 0.009896584008263652,
       'loss_ratio': 0.015393112957676402},
 1.0: {'loss_fft': 62.93830087150042,
       'loss_grid': 0.9911551435449638,
       'loss_ratio': 0.015748044192813226}}
ave ratio: 0.01556824416631245

"""

## moist 151
"""
{1e-09: {'loss_fft': 9.915203857640922e-09,
         'loss_grid': 9.604825680330662e-11,
         'loss_ratio': 0.009686967427229373},
 1e-08: {'loss_fft': 1.1807217351067426e-06,
         'loss_grid': 9.60672172327404e-09,
         'loss_ratio': 0.008136313102092212},
 1e-07: {'loss_fft': 1.0013708565533808e-06,
         'loss_grid': 9.700873837133372e-09,
         'loss_ratio': 0.00968759353604799},
 1e-06: {'loss_fft': 0.0001192528298830051,
         'loss_grid': 9.7027888405268e-07,
         'loss_ratio': 0.008136317477787217},
 1e-05: {'loss_fft': 0.00010113832629927321,
         'loss_grid': 9.79787257567125e-07,
         'loss_ratio': 0.009687596121255625},
 0.0001: {'loss_fft': 0.01204452921271116,
          'loss_grid': 9.799806720432532e-05,
          'loss_ratio': 0.00813631363033296},
 0.001: {'loss_fft': 0.010214315526843827,
         'loss_grid': 9.894851245337055e-05,
         'loss_ratio': 0.009687238679217221},
 0.01: {'loss_fft': 1.2164321503290823,
        'loss_grid': 0.009896805396639386,
        'loss_ratio': 0.00813592882592095},
 0.1: {'loss_fft': 1.025105300517053,
       'loss_grid': 0.009893820471050613,
       'loss_ratio': 0.009651516255023038},
 1.0: {'loss_fft': 122.11158095256023,
       'loss_grid': 0.9895744266989138,
       'loss_ratio': 0.008103854024159746}}
ave ratio: 0.008904963907906632
"""

targett_fft = torch.fft.rfft(targett[...,0], axis = 2) 
output_fft_noise = torch.fft.rfft(output_fft_noise[...,0], axis = 2)

plt.imshow(torch.abs(outputt_fft_noise)[0,:,20:])
plt.show()

plt.imshow(torch.abs(targett_fft)[0,:,20:])
plt.show()


plt.imshow(torch.abs(outputt_noise-targett)[0,:,20:])
plt.show()



## plotting zonal mean
moist_dir = '/media/volume/sdb'
moist_loc_151 = f"{moist_dir}/moist_5_daily/151/output.3d.nc"
d = nc.Dataset(moist_loc_151)

# moist_loc_151_2d = f"{moist_dir}/moist_5_daily/151/output.2d.nc"
# d_2d = nc.Dataset(moist_loc_151_2d)

f, axs = plt.subplots(2,3, figsize = (10,6))

for ich, ch in enumerate(["psi1", "psi2", "m"],0):
    chvals= np.asarray(d[ch])[10000:]
    chvals_time_lat_mean = chvals.mean(axis = 0).mean(axis = 1)
    axs[0,ich].imshow(chvals.mean(axis = 0))
    axs[0,ich].set_title(ch)
    axs[1,ich].plot(np.arange(chvals_time_lat_mean.shape[0]), chvals_time_lat_mean)
    
axs[1,0].set_ylabel("zone")
axs[1,0].set_xlabel("zonal mean")

plt.show()



## testing the ramp up period
plotting.plot_2d_grid_spectrum(moists_keep_fno[151],
                                       channels = channel_names,
                                       frame=10000,
                                       savename = f"151_test",
                                       output_dir = "./outputs",
                                       title = f"actual target values",
                                       begframe = -1)
                                       
                                       
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


#####
## testing energy-time plots

channels = {
            "psi1" : ["mean", "std"],
            "psi2" : ["mean", "std"],
            "m" : ["mean", "std"],
           }
channel_names = ["psi1","psi2", "m"]

qgm_sim_dir = "/media/volume/sdc/qgm_sim/"
qgm_names_noNoise_loc = os.path.join(qgm_sim_dir, 'load_resDir-151_noise-none_seed-0', "output.3d.nc")

data = nc.Dataset(qgm_names_noNoise_loc)
actual = np.array(np.stack([data.variables["psi1"][:], data.variables["psi2"][:], data.variables["m"][:]],axis=3)).astype(float)

## no mean, need raw values
# for ich, ch in enumerate(channel_names, 0):
    # if "std" in channels[ch]:
        # std = np.std(actual[:,:,:,ich])
    # else:
        # std = 1.0
    # if "mean" in channels[ch]:
        # mean = np.mean(actual[:,:,:,ich])
    # else:
        # mean = 0.0

    # actual[:,:,:,ich] = (actual[:,:,:,ich] - mean)/std


## energy at timesteps, longitude mean
loc = "/home/exouser/lenny_scripts/Deep-Learning-based-ocean-forecasts-moist/outputs/actual151_energy_timeSteps" 
fig, ax = plt.subplots(1, actual.shape[2], dpi = 200, figsize = (12,4))
tsteps = [1,28,64]
dt = .25

for ich, ch in enumerate(channel_names[:2],0):
  # first lat, second long
  [dpsidt, dpsidy, dpsidx] = np.gradient(actual[:,:,:,ich])
  uactual, vactual = -dpsidy, dpsidx
  
  totE_actual = np.mean(uactual**2+vactual**2, axis = 2)
  totE_actual = np.mean(totE_actual, axis = 0)
  ax[ich].plot(np.arange(128)[4:-4], totE_actual[4:-4], color = "black", linestyle = "--", label = f"actual (long mean {ch}, t averaged)", zorder = 20)


ax[len(channel_names[:2])-1].legend(bbox_to_anchor=(1.1, 1.05))
for ich, ch in enumerate(channel_names[:2],0):
    ax[ich].set_title(ch)
    
ax[0].set_xlabel(r"Latitude")
ax[0].set_ylabel(r"Total KE at Latitude")
ax[0].grid(), ax[1].grid()

plt.plot()
plt.savefig(fname=loc, bbox_inches='tight')
plt.close()

## energy over timesteps of actual
loc = "/home/exouser/lenny_scripts/Deep-Learning-based-ocean-forecasts-moist/outputs/actual151_energyVStime" 
fig, ax = plt.subplots(1, actual.shape[2], dpi = 200, figsize = (12,4))
tsteps = [1,28,64]
dt = .25

for ich, ch in enumerate(channel_names[:2],0):
  # first lat, second long
  [dpsidt, dpsidy, dpsidx] = np.gradient(actual[:,:,:,ich])
  uactual, vactual = -dpsidy, dpsidx
  
  totE_actual_time = uactual**2+vactual**2
  totE_actual_time = np.mean(np.mean(totE_actual_time, axis = 1), axis = 1)
  ax[ich].plot(np.arange(totE_actual_time.shape[0])/dt, totE_actual_time, color = "black", linestyle = "--", label = f"actual {ch} (mean all) ", zorder = 20)


ax[len(channel_names[:2])-1].legend(bbox_to_anchor=(1.1, 1.05))
for ich, ch in enumerate(channel_names[:2],0):
    ax[ich].set_title(ch)
    
ax[0].set_xlabel(r"Time")
ax[0].set_ylabel(r"Total KE at timestep")
ax[0].grid(), ax[1].grid()

plt.plot()
plt.savefig(fname=loc, bbox_inches='tight')
plt.close()