import matplotlib.pyplot as plt
import numpy as np
import datetime
import matplotlib.animation as animation
import time
from scipy import fft

def plot_2d_quantity(data, 
                     day = 10, 
                     day_dtdns = 40*5, 
                     loc = None):
    plt.imshow(data)
    plt.colorbar()
    plt.show()
    
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