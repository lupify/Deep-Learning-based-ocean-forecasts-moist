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
                            actual_spectrum = None,
                             channels = ["psi1","psi2","moist"],
                             savename = None,
                             output_dir = "./outputs",
                             kex = 1,
                             frame = 0,
                             vmin = -2,
                             vmax = 2,
                             title = "Channel Grid and Spectrum Plots",
                             cmap = cm.viridis,
                             begframe = 10000,
                             showfig = False):
    
    if savename is None:
        timenow = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        loc = f"{output_dir}/gridSpectrum_frame-{frame}_time-{timenow}.png"
    else:
        loc = f"{output_dir}/{savename}.png"
        
    fig, ax = plt.subplots(2, data.shape[3], dpi = 200, figsize = (10,6))

    #print(f"Plotting grid spectrum graphs to {loc}")
    
    # axes limits on spectrum plots
    lims = [2.5, 20, 2]
    
    for i in range(data.shape[3]):
        # set title to channel name
        # dmin, dmax = data[frame,:,:,i].min(), data[frame,:,:,i].max()
        # text = ax[0,i].set_title(f"{channels[i]}\nmin = {np.around(dmin,3)}, max = {np.around(dmax)}")
        text = ax[0,i].set_title(f"{channels[i]}")
        # excluding first mode
        # grid space 
        ax[0,i].get_xaxis().set_visible(False)
        ax[0,i].get_yaxis().set_visible(False)
        gridplot = ax[0,i].imshow(data[frame,:,:,i], vmin = vmin, vmax = vmax, cmap = cmap)
        
        divider = make_axes_locatable(ax[0,i])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(gridplot, cax=cax, orientation='vertical')
        
        # spectrum
        spectrum = np.abs(fft.rfft(data[frame,:,:,i], axis = 1)[:,kex:64])
        spectrum_mean = np.mean(spectrum, axis = 0)
        ax[1,i].plot(spectrum_mean, color = "blue")[0]
        ax[1,i].grid(alpha = .5)
        ax[1,i].set_ylim(0, lims[i])
        
        if actual_spectrum is not None:
            # spectrum = np.abs(fft.rfft(actual[:,:,:,i], axis = 2)[:,:,kex:64,:], axis = 1)
            # spectrum_mean = np.mean(spectrum, axis = 0)
            # spectrum_mean = np.abs(fft.rfft(actual[:,:,:,i], axis = 2)[:,:,kex:64]).mean(axis = 1).mean(axis = 0)
            spectrum_mean = actual_spectrum[...,i]
            ax[1,i].plot(spectrum_mean, color = "black", linestyle = "--", alpha = .7)[0] 
        
    ax[0,0].set_xlabel("lon")
    ax[0,0].set_ylabel("lat")
    ax[1,0].set_xlabel(r"wavenumber, $k$")
    ax[1,0].set_ylabel(r"Amplitude")
    
    daytitle = plt.text(0.5,1.15,f"{title}\nframe: {frame}\nday: {frame/4.:.2f}", ha="center",va="bottom", transform=ax[0,int(data.shape[3]/2)].transAxes, fontsize="small", color = "black")    
    #plt.tight_layout()
    if showfig:
        plt.plot()
    else:
        plt.savefig(fname=loc)
    plt.close()

def plot_rmse(data1, 
                       data2, 
                       channels = ["psi1","psi2","moist"],
                       loc = "./plot_rmse.png"):
    assert data1.shape == data2.shape
    fig, ax = plt.subplots(1, data1.shape[3], dpi = 200, figsize = (11,4))
    
    for ich, ch in enumerate(channels,0):
        sqerrors = []
        tsteps = np.arange(data1.shape[0])
        for tstep in tsteps:
            sqerrors.append(np.sqrt(np.mean((data1[tstep,:,:,ich]-data2[tstep,:,:,ich])**2)))
            
        ax[ich].plot(tsteps/4., sqerrors)
        ax[ich].set_title(f"{ch} RMSE")
        ax[ich].set_xlabel(f"days")
        if ich == 0:
            ax[ich].set_ylabel(f"RMSE")
        ax[ich].grid(alpha = .8)
    
    plt.plot()
    plt.savefig(fname=loc, bbox_inches='tight')
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
        ## time mean computed
        d2c = data2[:,:,:,ich].mean(axis = 0)
        
        for tstep in tsteps:
            d1t = data1[tstep,:,:,ich]
            d2t = data2[tstep,:,:,ich]
            num = np.sum((d1t - d2c)*(d2t - d2c))
            den = np.sqrt(np.sum((d1t - d2c)**2))*np.sqrt(np.sum((d2t - d2c)**2))
            accs.append(num/den)
            
        ax[ich].plot(tsteps/4., accs)
        ax[ich].set_title(f"{ch} ACC")
        ax[ich].set_xlabel(f"days")
        if ich == 0:
            ax[ich].set_ylabel(f"ACC")
        ax[ich].grid(alpha = .8)
    
    plt.plot()
    plt.savefig(fname=loc, bbox_inches='tight')
    plt.close()

def plot_spectrums(data1, 
                    data2, 
                    channels = ["psi1","psi2","moist"],
                    data_names = ["autoreg_pred", "actual"],
                    tsteps = [0,5,20,100,500],
                    kex = 1,
                    cmap = cm.rainbow,
                    loc = "./plot_spec1.png"):
                    
    assert data1.shape == data2.shape
    fig, ax = plt.subplots(2, data1.shape[3], dpi = 200, figsize = (12,8))
    ks = np.arange(kex,64)
    
    for idata, data in enumerate([data1, data2], 0):
        for ich, ch in enumerate(channels,0):
            for itstep, tstep in enumerate(tsteps,0):
                spectrum = np.abs(fft.rfft(data[tstep,:,:,ich], axis = 1)[:,ks])
                spectrum_mean = np.mean(spectrum, axis = 0)
                ax[idata,ich].plot(ks, spectrum_mean, color = cmap(itstep/(len(tsteps)-1)), label = f"day = {np.around(tstep/4.,2)}")
                ax[idata,ich].grid(alpha = .8)
                ax[idata,ich].set_yscale("log")
    
    ax[0,0].legend()
    for ich, ch in enumerate(channels,0):
        ax[0, ich].set_title(ch)
        ax[1, ich].set_xlabel(r"Wavenumber, $k$")
    for idata, data in enumerate(data_names,0):
        ax[idata, 0].set_ylabel(r"Amplitude")
    
    #ax[0, 1].set_title(data_names[0])
    #ax[1, 1].set_title(data_names[1])
    
    plt.plot()
    plt.savefig(fname=loc, bbox_inches='tight')
    plt.close()

def plot_spectrums2(data1, 
                        data2, 
                        channels = ["psi1","psi2","moist"],
                        data_names = ["autoreg_pred", "actual"],
                        tsteps = [0,5,20,100,500],
                        kex = 1,
                        cmap = cm.rainbow,
                        loc = "./plot_spec2.png"):
    assert data1.shape == data2.shape
    fig, ax = plt.subplots(1, data1.shape[3], dpi = 200, figsize = (12,4))
    ks = np.arange(kex,64)
    
    for ich, ch in enumerate(channels,0):
        for itstep, tstep in enumerate(tsteps,0):
            spectrum = np.abs(fft.rfft(data1[tstep,:,:,ich], axis = 1)[:,ks])
            spectrum_mean = np.mean(spectrum, axis = 0)
            ax[ich].plot(ks, spectrum_mean, color = cmap(itstep/(len(tsteps)-1)), label = f"day = {np.around(tstep/4.,2)}")
            ax[ich].grid(alpha = .8)
            ax[ich].set_yscale("log")
    
        spectrum = np.mean(np.abs(fft.rfft(data2[:,:,:,ich], axis = 2)[:,:,ks]), axis = 0)
        spectrum_mean = np.mean(spectrum, axis = 0)
        ax[ich].plot(ks, spectrum_mean, color = "black", linestyle = "--", label = f"actual (mean)", zorder = 20)

    
    ax[0].legend()
    for ich, ch in enumerate(channels,0):
        ax[ich].set_title(ch)
        
    ax[0].set_xlabel(r"Wavenumber, $k$")
    ax[0].set_ylabel(r"Amplitude")
    
    #ax[0, 1].set_title(data_names[0])
    #ax[1, 1].set_title(data_names[1])
    
    plt.plot()
    plt.savefig(fname=loc, bbox_inches='tight')
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
    

def plot_compare_accs(actual, 
                      preds,
                      labels_properties,
                      channels = ["psi1","psi2","moist"],
                      loc = "./plot_acc.png"):
                      
    fig, ax = plt.subplots(1, actual.shape[3], dpi = 200, figsize = (11,4))
    
    
    for ich, ch in enumerate(channels,0):
        for ipr, pred in enumerate(preds, 0):
            accs = []
            tsteps = np.arange(actual.shape[0])
            #d1c = data1[:,:,:,ich].mean(axis = 0)
            ## time mean computed
            d2c = actual[:,:,:,ich].mean(axis = 0)
            
            for tstep in tsteps:
                d1t = pred[tstep,:,:,ich]
                d2t = actual[tstep,:,:,ich]
                num = np.sum((d1t - d2c)*(d2t - d2c))
                den = np.sqrt(np.sum((d1t - d2c)**2))*np.sqrt(np.sum((d2t - d2c)**2))
                accs.append(num/den)
                
            ax[ich].plot(tsteps/4., 
                         accs, 
                         label = labels_properties[ipr]["label"], 
                         color = labels_properties[ipr]["color"], 
                         linestyle = labels_properties[ipr]["linestyle"])
                         
            ax[ich].set_title(f"{ch} ACC")
            ax[ich].set_xlabel(f"days")
            
            if ich == 0:
                ax[ich].set_ylabel(f"ACC")
            ax[ich].grid(alpha = .8)
            
        if ich == (len(channels)-1):
            ax[ich].legend(bbox_to_anchor=(1.1, 1.05))
    
    plt.plot()
    plt.savefig(fname=loc, bbox_inches='tight')
    plt.close()
    

def plot_compare_rmse(actual,
                      preds, 
                      labels_properties,
                      channels = ["psi1","psi2","moist"],
                      loc = "./plot_rmse.png"):
                      
    fig, ax = plt.subplots(1, actual.shape[3], dpi = 200, figsize = (11,4))
    
    
    for ich, ch in enumerate(channels,0):
        for ipr, pred in enumerate(preds, 0):
            sqerrors = []
            tsteps = np.arange(actual.shape[0])
            #d1c = data1[:,:,:,ich].mean(axis = 0)
            ## time mean computed
            d2c = pred[:,:,:,ich].mean(axis = 0)
            
            for tstep in tsteps:
                d1t = pred[tstep,:,:,ich]
                d2t = actual[tstep,:,:,ich]
                sqerrors.append(np.sqrt(np.mean((d1t-d2t)**2)))
                
            ax[ich].plot(tsteps/4., 
                         sqerrors, 
                         label = labels_properties[ipr]["label"], 
                         color = labels_properties[ipr]["color"], 
                         linestyle = labels_properties[ipr]["linestyle"])
                         
            ax[ich].set_title(f"{ch} RMSE")
            ax[ich].set_xlabel(f"days")
            
            if ich == 0:
                ax[ich].set_ylabel(f"RMSE")
            ax[ich].grid(alpha = .8)
            
        if ich == (len(channels)-1):
            ax[ich].legend(bbox_to_anchor=(1.1, 1.05))
    
    plt.plot()
    plt.savefig(fname=loc, bbox_inches='tight')
    plt.close()
    
def plot_compare_spectrums2(actual,  
                            preds,
                            labels_properties,
                            channels = ["psi1","psi2","moist"],
                            tsteps = [28],
                            kex = 1,
                            cmap = cm.rainbow,
                            loc = "./plot_spec2.png"):
                            
    fig, ax = plt.subplots(1, actual.shape[3], dpi = 200, figsize = (12,4))
    ks = np.arange(kex,64)
    
    for ich, ch in enumerate(channels,0):
        for ipr, pred in enumerate(preds, 0):
            for itstep, tstep in enumerate(tsteps,0):
                spectrum = np.abs(fft.rfft(pred[tstep,:,:,ich], axis = 1)[:,ks])
                spectrum_mean = np.mean(spectrum, axis = 0)
                ax[ich].plot(ks, 
                             spectrum_mean, 
                             color = labels_properties[ipr]['color'], 
                             label = f"{labels_properties[ipr]['label']}, day = {np.around(tstep/4.,2)}",
                             linestyle = labels_properties[ipr]['linestyle'])
                ax[ich].grid(alpha = .8)
                ax[ich].set_yscale("log")
        
        spectrum = np.mean(np.abs(fft.rfft(actual[:,:,:,ich], axis = 2)[:,:,ks]), axis = 0)
        spectrum_mean = np.mean(spectrum, axis = 0)
        ax[ich].plot(ks, spectrum_mean, color = "black", linestyle = "--", label = f"actual (mean)", zorder = 20)

    
    ax[len(channels)-1].legend(bbox_to_anchor=(1.1, 1.05))
    for ich, ch in enumerate(channels,0):
        ax[ich].set_title(ch)
        
    ax[0].set_xlabel(r"Wavenumber, $k$")
    ax[0].set_ylabel(r"Amplitude")
    
    plt.plot()
    plt.savefig(fname=loc, bbox_inches='tight')
    plt.close()

def plot_compare_energy_tsteps(actual,  
                            preds,
                            labels_properties,
                            channels = ["psi1","psi2"],
                            tsteps = [1, 28],
                            kex = 1,
                            cmap = cm.rainbow,
                            dt = .25,
                            loc = "./plot_spec2.png"):
                            
    fig, ax = plt.subplots(1, actual.shape[3], dpi = 200, figsize = (12,4))
    
    for ich, ch in enumerate(channels,0):
        for ipr, pred in enumerate(preds, 0):
            for itstep, tstep in enumerate(tsteps,0):
                
                upred = -(pred[tstep,1:,:,ich] - pred[tstep,:-1,:,ich])/dt
                vpred = (pred[tstep,:,1:,ich] - pred[tstep,:,:-1,ich])/dt
                
                # latitude mean
                totE_pred = np.sum(upred**2+vpred**2, axis = 1)/127
        
        uactual = -(np.mean(actual[tstep,1:,:,ich], axis = 0) - np.mean(actual[0,:-1,:,ich], axis = 0))/dt
        vactual = (np.mean(actual[tstep,:,1:,ich], axis = 0) - np.mean(actual[tstep,:,:-1,ich], axis = 0))/dt
        totE_actual = np.sum(uactual**2+vactual**2, axis = 1)/127
        
        ax[ich].plot(ks, spectrum_mean, color = "black", linestyle = "--", label = f"actual (mean)", zorder = 20)

    
    ax[len(channels)-1].legend(bbox_to_anchor=(1.1, 1.05))
    for ich, ch in enumerate(channels,0):
        ax[ich].set_title(ch)
        
    ax[0].set_xlabel(r"Lattitude")
    ax[0].set_ylabel(r"Energy")
    
    plt.plot()
    plt.savefig(fname=loc, bbox_inches='tight')
    plt.close()