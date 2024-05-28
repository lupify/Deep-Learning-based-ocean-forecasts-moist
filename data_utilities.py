import netCDF4 as nc
import pickle
import numpy as np
import torch

def data_prep_save(data_loc,
                   moist_data_locs,
                   channels = {
                               "psi1" : ["mean", "std"],
                               "psi2" : ["mean", "std"],
                               "m" : ["mean", "std"],
                              },
                   rampuptstamp = 4000,
                   endstamp = 18000,
                   save = True):
                  
    print(f"Loading moist dataset")
    print(f"...save: {save}, saving loc: {data_loc}")
    print("loading moist datasets...")
    
    moists_full = {}
    for m in moist_data_locs.keys():
        try:
            print(f"{m} : {moist_data_locs[m]}")
            moists_full[m] = nc.Dataset(moist_data_locs[m])
        except:
            print(f"...skipping. Error loading nc")
            continue
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
                std = np.std(moists_keep[moist][rampuptstamp:endstamp,:,:,i])
            else:
                std = 1.0
            if "mean" in channels[ch]:
                mean = np.mean(moists_keep[moist][rampuptstamp:endstamp,:,:,i])
            else:
                mean = 0.0

            moists_info[moist][ch] = {"std" : std, "mean" : mean, "index" : i}
            moists_keep[moist][:,:,:,i] = (moists_keep[moist][:,:,:,i] - mean)/std

    ## ramp up period, dimensions: feature, time, height (latitude), width (longitude)
    moists_keep_fno_timestamps = {}
    moists_keep_fno = {}

    for moist in moists_keep:
        moists_keep_fno_timestamps[moist] = np.arange(moists_keep[moist].shape[0])
        moists_keep_fno[moist] = moists_keep[moist][rampuptstamp:endstamp,:,:,:]
        moists_keep_fno_timestamps[moist] = moists_keep_fno_timestamps[moist][rampuptstamp:endstamp]
        moists_keep_fno[moist] = moists_keep_fno[moist]
        moists_keep_fno_timestamps[moist] = moists_keep_fno_timestamps[moist]
    
    if save:
        with open(data_loc, "wb") as h:
            pickle.dump([moists_keep_fno,
                         moists_keep_fno_timestamps,
                         moists_info,
                        ], 
                        h)

    return moists_keep_fno, moists_keep_fno_timestamps, moists_info

def data_load(data_loc):
    print(f"Loading previously saved data at loc {data_loc}")
    with open(data_loc, "rb") as h:
        moists_keep_fno, moists_keep_fno_timestamps, moists_info = pickle.load(h)
    
    return moists_keep_fno, moists_keep_fno_timestamps, moists_info

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

def prep_in_tar_data_2(data, ts_in = 8, lead = 0, ts_out = 1, overlap = True):
    
    """
    Much more efficient for tsteps concatenation
    """
    
    sin = 0
    fin = sin+ts_in
    star = fin+lead
    ftar = star+ts_out
    
    ## doesn't include overlapping data in inputs, even if overlap is True
    if overlap is True:
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
        print(f"concatenating: {m}...")
        data_input, data_target = prep_in_tar_data_2(data_dict[m], **pitd_kwargs)
        if datas_input is None:
            datas_input = data_input
            datas_target = data_target
        else:
            datas_input = np.concatenate([datas_input, data_input], axis = 0)
            datas_target = np.concatenate([datas_target, data_target], axis = 0)
    
    if do_permute:
        print("permuting...")
        indices = np.random.permutation(datas_input.shape[0])
        datas_input = datas_input[indices]
        datas_target = datas_target[indices]
       
    datas_input = torch.from_numpy(datas_input).float().cuda()
    datas_target = torch.from_numpy(datas_target).float().cuda()
    
    return datas_input, datas_target

def load_data_fno(data_dict, data_prep, data_prep_args, do_permute = True):
    print(f"Concatenating training data. data_prep: {data_prep}")
    if data_prep == "singleStep":
        inputs, targets = concatenate_data_singleStep(data_dict, do_permute = do_permute)
    elif data_prep == "tsteps":
        inputs, targets = concatenate_data_tsteps(data_dict, do_permute = do_permute, pitd_kwargs = data_prep_args)
    print(f"inputs.shape : {inputs.shape}, targets.shape : {targets.shape}")
    return inputs, targets

def load_data_fno(data_dict, data_prep, data_prep_args, do_permute = True):
    print(f"Concatenating training data. data_prep: {data_prep}")
    if data_prep == "singleStep":
        inputs, targets = concatenate_data_singleStep(data_dict, do_permute = do_permute)
    elif data_prep == "twoStep":
        inputs, targets = concatenate_data_singleStep(data_dict, do_permute = do_permute)
        ## two step predictions
        inputs = inputs[:-1]
        targets = targets[1:]
    elif data_prep == "tsteps":
        inputs, targets = concatenate_data_tsteps(data_dict, do_permute = do_permute, pitd_kwargs = data_prep_args)
    print(f"inputs.shape : {inputs.shape}, targets.shape : {targets.shape}")
    return inputs, targets 