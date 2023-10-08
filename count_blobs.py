import os
import numpy as np
import pandas as pd
import cc3d
import datetime
import pickle

from filehandling import read_nifti, write_nifti

def load_cached_brain(settings, brain):
    """ 
    Check if we already performed a costly connected component analysis
    If so, return the name of the file (contains the number of connected components)
    Else, return False
    """
    path_in = settings["postprocessing"]["output_location"]
    result = False
    for item in [x for x in os.listdir(path_in) if ".npy" in x]:
        if brain in item:
            result = os.path.join(path_in, item)  
    return result

def load_cached_stats(settings, brain):
    """ 
    Check if we already performed a costly connected component statistics analysis
    If so, return the name of the file 
    Else, return False
    """
    path_in = settings["postprocessing"]["output_location"]
    result = False
    for item in [x for x in os.listdir(path_in) if ".pickle" in x]:
        if brain in item:
            result = os.path.join(path_in, item)  
    return result

def count_blobs(settings, path_in, brain_i, brain, stack_shape, min_size=-1, max_size=-1):
    path_out    = settings["postprocessing"]["output_location"]
    
    if not os.path.exists(path_out):
        os.mkdir(path_out)

    len_b = len(os.listdir(path_in))
    start = datetime.datetime.now()
    print(f"{start} Start {brain} - {brain_i}/{len_b}")
    brain_path = os.path.join(path_in, brain, "binary_segmentations", "binaries.npy")
    x = dataset_on_disk = np.memmap(brain_path,dtype=np.uint8,mode='r+',shape=stack_shape[1:])
    x = x[0,:,:,:]
    mid = datetime.datetime.now()
    print(f"{mid} Reading took {mid - start}")
    #x = np.swapaxes(x, 0, -1)
    mid2 = datetime.datetime.now()
    #print(f"{mid2} Swapping axes took {mid2 - mid} : {x.shape}")
    #temporarily store cc3d output
    temp_cc3d_output_path =  os.path.join(path_out,brain+"temp_cc3d_store.npy")

    if not load_cached_brain(settings, brain):
        print("No cached brain found, performing cc3d...")
        labels, N = cc3d.connected_components(x, return_N=True,out_file = temp_cc3d_output_path)
         #write_nifti(os.path.join(path_out, f"{brain}-{N}-cc3d.nii.gz"), labels)
        np.save(os.path.join(path_out, f"{brain}-{N}-cc3d.npy"), labels)
        os.remove(temp_cc3d_output_path) 
    else:
        path_cached_brain = load_cached_brain(settings, brain)
        N = int(path_cached_brain.split("/")[-1].split("-")[1])
        print(f"Cached brain found at {path_cached_brain} with {N} components, loading...")
        #labels = read_nifti(path_cached_brain)
        labels = np.load(path_cached_brain)
        
    # if not os.path.exists(os.path.join(path_out, brain + "-cc3d.nii.gz")):
    # labels, N = cc3d.connected_components(x, return_N=True)
    mid3 = datetime.datetime.now()
    print(f"{mid3} cc3d+writing/loading took {mid3 - mid2} : {N}")

    if not load_cached_stats(settings, brain):
        print("No stats found, performing cc3d.statistics...")
        stats = cc3d.statistics(labels)
        path_stats = os.path.join(path_out, f"{brain}-stats.pickle")
        with open(path_stats, "wb") as file:
            pickle.dump(stats, file, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        path_stats = load_cached_stats(settings, brain)
        print(f"Found stats at {path_stats}")
        with open(path_stats, "rb") as file:
            stats = pickle.load(file)

    mid4 = datetime.datetime.now()
    print(f"{mid4} stats took {mid4-mid3}")

    columns = ["Blob", "Coords", "Size"]
    df = pd.DataFrame(columns = columns)

    for i in range(1, N):
        if min_size == -1:
            if max_size == -1:
                centroids_list = [stats["centroids"][i].tolist()]
                df_l = pd.DataFrame({"Blob":i,\
                        "Coords":centroids_list,\
                        "Size":stats["voxel_counts"][i]})
                df = pd.concat([df, df_l])
            else:
                if stats["voxel_counts"] <= max_size:
                    centroids_list = [stats["centroids"][i].tolist()]
                    df_l = pd.DataFrame({"Blob":i,\
                            "Coords":centroids_list,\
                            "Size":stats["voxel_counts"][i]})
                    df = pd.concat([df, df_l])
        else:
            if stats["voxel_counts"] >= min_size:
                centroids_list = [stats["centroids"][i].tolist()]
                df_l = pd.DataFrame({"Blob":i,\
                        "Coords":centroids_list,\
                        "Size":stats["voxel_counts"][i]})
                df = pd.concat([df, df_l])


    output_name = f"{x.shape}_{brain.replace('.nii.gz','')}.csv"
    df.to_csv(path_out + output_name)
    end = datetime.datetime.now()
    end_delta = end- start
    remaining_time = (len_b - brain_i) * end_delta
    print(f"{end} {brain} {brain_i} / {len_b} Done; Took {end_delta}, ETA {remaining_time}")

