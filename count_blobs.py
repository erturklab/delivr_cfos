import os
import numpy as np
import pandas as pd
import cc3d
import datetime

from filehandling import read_nifti

def count_blobs(settings):
    path_in     = settings["postprocessing"]["input_location"]
    path_out    = settings["postprocessing"]["output_location"]
    len_b = len(os.listdir(path_in))
    for brain_i, brain in enumerate(os.listdir(path_in)):
        start = datetime.datetime.now()
        print(f"{start} Start {brain} - {brain_i}/{len_b}")
        x = read_nifti(path_in + brain)
        mid = datetime.datetime.now()
        print(f"{mid} Reading took {mid - start}")
        x = np.swapaxes(x, 0, -1)
        mid2 = datetime.datetime.now()
        print(f"{mid2} Swapping axes took {mid2 - mid} : {x.shape}")

        labels, N = cc3d.connected_components(x, return_N=True)
        mid3 = datetime.datetime.now()
        print(f"{mid3} cc3d took {mid3 - mid2} : {N}")

        stats = cc3d.statistics(labels)
        mid4 = datetime.datetime.now()
        print(f"{mid4} stats took {mid4-mid3}")

        columns = ["Blob", "Coords", "Size"]
        df = pd.DataFrame(columns = columns)

        for i in range(1, N):
            # start = datetime.datetime.now()
            df_l = {"Blob":i,\
                    "Coords":stats["centroids"][i],\
                    "Size":stats["voxel_counts"][i]}
            df = df.append(df_l, ignore_index=True)
            # delta = datetime.datetime.now()
            # rem = (N - i) * delta
            # print(f"{i} / {N}, ETA {rem}", end="\r", flush=True)

        df.to_csv(path_out + str(x.shape) + "_" + brain.replace(".nii.gz",".csv"))
        end = datetime.datetime.now()
        end_delta = end- start
        remaining_time = (len_b - brain_i) * end_delta
        print(f"{end} {brain} {brain_i} / {len_b} Done; Took {end_delta}, ETA {remaining_time}")

