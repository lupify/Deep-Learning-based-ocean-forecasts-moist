import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
from scipy import fft
import os
import multiprocessing as mp
from multiprocessing import Pool
import matplotlib.cm as cm

def plot_2d_quantity(data, 
                     day = 10, 
                     day_dtdns = 40*5, 
                     loc = None):
    plt.imshow(data)
    plt.colorbar()
    #plt.show()
    
    if loc is not None:
        plt.savefig(loc)
    
    plt.close()

def animate_2d_quantity(data,
                        start = None, 
                        stop = None, 
                        step = None):
    fig, ax = plt.subplots()
    
    if start is None:
        start = 0
    if stop is None:
        stop = data.shape[0]
    if step is None:
        step = 1
        
    stime = ctime = time.time()
    
    frame = start
    artists = []
    while frame < stop:
        if (time.time() - ctime) > 5:
            print(f"{time.time() - stime},  frame: {frame}")
            ctime = time.time()
            
        container = ax.imshow(data[frame,:,:])
        #ax.set_title(f"day: {np.around(frame/4, 2)}")
        artists.append([container])
        frame += step
    
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=100)
    timenow = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    ani.save(filename=f"./outputs/test_{timenow}.mp4", writer="ffmpeg")
    
def animate_2d_grid_spectrum(data,
                             channels = ["psi1","psi2","moist"],
                             start = None, 
                             stop = 100, 
                             step = 1,
                             savename = None,
                             output_dir = "./outputs",
                             title = "",
                             kex = 1,
                             begframe = 10000,
                             interval = 200):
    

    if savename is None:
        timenow = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        loc = f"{output_dir}/gridSpectrum_{timenow}.mp4"
    else:
        loc = f"{output_dir}/{savename}.mp4"
        
    assert not os.path.exists(loc)
    
    fig, ax = plt.subplots(2, data.shape[3], dpi = 100)

    if start is None:
        start = 0
    if stop is None:
        stop = data.shape[0]
        
    stime = ctime = time.time()
    frame = start
    artists = []
    print(f"Writing grid spectrum graphs to {loc}")
    
    while frame < stop:
        if (time.time() - ctime) > 5:
            print(f"time: {np.around(time.time() - stime,2)},  frame: {frame}")
            ctime = time.time()
            
        container = []  
        
        for i in range(data.shape[3]):
            # set title to channel name
            text = ax[0,i].set_title(channels[i])
            # excluding first mode
            spectrum = np.abs(fft.rfft(data[frame,:,:,i], axis = 1)[:,kex:64])
            spectrum_mean = np.mean(spectrum, axis = 0)
            # grid space 
            gridplot = ax[0,i].imshow(data[frame,:,:,i])
            container.append(gridplot)
            
            divider = make_axes_locatable(ax[0,i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(gridplot, cax=cax, orientation='vertical')
            
            # spectrum
            container.append(ax[1,i].plot(spectrum_mean, color = "blue")[0])
            container.append(text)
        
        daytitle = plt.text(0.5,1.15,f"{title}\nframe: {frame}, day: {np.around((frame+begframe)/4, 2)}", ha="center",va="bottom", transform=ax[0,int(data.shape[3]/2)].transAxes, fontsize="small", color = "black")    
        
        container.append(daytitle)   
        artists.append(container)
        frame += step
        
    print(f"final - time: {np.around(time.time() - stime,2)},  frame: {frame}")
    
    writervideo = animation.FFMpegWriter(fps=8) 
    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=interval)
    ani.save(filename=loc, writer=writervideo)

def plot_2d_grid_spectrum(data, 
                             channels = ["psi1","psi2","moist"],
                             savename = None,
                             output_dir = "./outputs",
                             kex = 1,
                             frame = 0,
                             vmin = -2,
                             vmax = 2,
                             title = "Channel Grid and Spectrum Plots",
                             begframe = 10000):
    

    if savename is None:
        timenow = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        loc = f"{output_dir}/gridSpectrum_frame-{frame}_time-{timenow}.png"
    else:
        loc = f"{output_dir}/{savename}.png"
        
    fig, ax = plt.subplots(2, data.shape[3], dpi = 200, figsize = (10,6))

    #print(f"Plotting grid spectrum graphs to {loc}")

    for i in range(data.shape[3]):
        # set title to channel name
        # dmin, dmax = data[frame,:,:,i].min(), data[frame,:,:,i].max()
        # text = ax[0,i].set_title(f"{channels[i]}\nmin = {np.around(dmin,3)}, max = {np.around(dmax)}")
        text = ax[0,i].set_title(f"{channels[i]}")
        # excluding first mode
        # grid space 
        ax[0,i].get_xaxis().set_visible(False)
        ax[0,i].get_yaxis().set_visible(False)
        gridplot = ax[0,i].imshow(data[frame,:,:,i], vmin = vmin, vmax = vmax)
        
        divider = make_axes_locatable(ax[0,i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(gridplot, cax=cax, orientation='vertical')
        
        # spectrum
        spectrum = np.abs(fft.rfft(data[frame,:,:,i], axis = 1)[:,kex:64])
        spectrum_mean = np.mean(spectrum, axis = 0)
        ax[1,i].plot(spectrum_mean, color = "blue")[0]
        ax[1,i].grid(alpha = .5)
    
    daytitle = plt.text(0.5,1.15,f"{title}\nframe: {frame}, day: {np.around((frame+begframe)/4, 2)}", ha="center",va="bottom", transform=ax[0,int(data.shape[3]/2)].transAxes, fontsize="small", color = "black")    
    #plt.tight_layout()
    plt.plot()
    plt.savefig(fname=loc)
    plt.close()

def plot_squared_error(data1, 
                        data2, 
                        channels = ["psi1","psi2","moist"],
                        loc = "./plot_squared_error.png"):
    assert data1.shape == data2.shape
    fig, ax = plt.subplots(1, data1.shape[3], dpi = 200, figsize = (11,4))
    
    for ich, ch in enumerate(channels,0):
        sqerrors = []
        tsteps = np.arange(data1.shape[0])
        for tstep in tsteps:
            sqerrors.append(np.mean((data1[tstep,:,:,ich]-data2[tstep,:,:,ich])**2))
        ax[ich].plot(tsteps, sqerrors)
        ax[ich].set_title(f"{ch} MSE")
        ax[ich].set_xlabel(f"time steps")
        if ich == 0:
            ax[ich].set_ylabel(f"MSE")
        ax[ich].grid(alpha = .8)
    
    plt.plot()
    plt.savefig(fname=loc)
    plt.close()

def plot_acc(data1, 
                        data2, 
                        channels = ["psi1","psi2","moist"],
                        loc = "./plot_acc.png"):
    assert data1.shape == data2.shape
    fig, ax = plt.subplots(1, data1.shape[3], dpi = 200, figsize = (11,4))
    
    for ich, ch in enumerate(channels,0):
        accs = []
        tsteps = np.arange(data1.shape[0])
        #d1c = data1[:,:,:,ich].mean(axis = 0)
        d2c = data2[:,:,:,ich].mean(axis = 0)
        
        for tstep in tsteps:
            d1t = data1[tstep,:,:,ich]
            d2t = data2[tstep,:,:,ich]
            num = np.sum((d1t - d2c)*(d2t - d2c))
            den = np.sqrt(np.sum((d1t - d2c)**2))*np.sqrt(np.sum((d2t - d2c)**2))
            accs.append(num/den)
            
        ax[ich].plot(tsteps, accs)
        ax[ich].set_title(f"{ch} ACC")
        ax[ich].set_xlabel(f"time steps")
        if ich == 0:
            ax[ich].set_ylabel(f"ACC")
        ax[ich].grid(alpha = .8)
    
    plt.plot()
    plt.savefig(fname=loc)
    plt.close()

def plot_spectrums(data1, 
                    data2, 
                    channels = ["psi1","psi2","moist"],
                    data_names = ["autoreg_pred", "actual"],
                    tsteps = [0,5,20,100,500],
                    kex = 1,
                    cmap = cm.rainbow,
                    loc = "./plot_acc.png"):
    assert data1.shape == data2.shape
    fig, ax = plt.subplots(2, data1.shape[3], dpi = 200, figsize = (12,8))
    ks = np.arange(kex,64)
    
    for idata, data in enumerate([data1, data2], 0):
        for ich, ch in enumerate(channels,0):
            for itstep, tstep in enumerate(tsteps,0):
                spectrum = np.abs(fft.rfft(data[tstep,:,:,ich], axis = 1)[:,ks])
                spectrum_mean = np.mean(spectrum, axis = 0)
                ax[idata,ich].plot(ks, spectrum_mean, color = cmap(itstep/(len(tsteps)-1)), label = f"tstep = {tstep}")
                ax[idata,ich].grid(alpha = .8)
    
    ax[0,0].legend()
    for ich, ch in enumerate(channels,0):
        ax[0, ich].set_title(ch)
        ax[1, ich].set_xlabel(r"Wavenumber, $k$")
    for idata, data in enumerate(data_names,0):
        ax[idata, 0].set_ylabel(r"$E$")
    
    #ax[0, 1].set_title(data_names[0])
    #ax[1, 1].set_title(data_names[1])
    
    plt.plot()
    plt.savefig(fname=loc)
    plt.close() 

def plot_singleSpectrum(data, 
                        label = "default",
                        starttimestep = 0,
                        stoptimestep = None,
                        latrange = [None, None],
                        xlabel = r"$k_x$",
                        ylabel = ""):
    """
    Date inputted is raw data grid feature data, across lattitude (horizontally)
    """
    
    # a = np.abs(np.real(fft.fft(moists_keep[moist][2,10000:,64-10:64+10,:], axis = 2)[:,:,1:64]))
    a = np.abs(fft.rfft(data[starttimestep:stoptimestep,latrange[0] : latrange[1], :], axis = 2)[:,:,1:64])
    b = np.mean(a, axis = 1)
    c = np.mean(b, axis = 0)
    
    plt.plot(c, label = label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)  
    print(c.shape)
    plt.grid(alpha = .5)
    
def plot_loss(losses, ylim = 5):
    plt.plot(losses)
    plt.grid()
    plt.ylim(0, ylim)
    plt.savefig("./outputs/test.png")
    plt.close()