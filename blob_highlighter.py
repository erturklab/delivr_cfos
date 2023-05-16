import os
import datetime
import cc3d
import pickle
import cv2
import numpy as np
import pandas as pd
import shutil
from filehandling import read_nifti, write_nifti
from inference.inference import create_empty_memmap
from skimage.morphology import binary_dilation
from skimage.draw import ellipsoid
from skimage import io

def fast_dilate(img, kernel_size):                                                                                 
     kernel = np.ones((kernel_size,kernel_size), np.uint8)                                                          
     for x in range(img.shape[2]):                                                                                  
         #img_x = img[:,:,x]
         img[:,:,x] = cv2.dilate(img[:,:,x], kernel, iterations=1)                                                       
     for y in range(img.shape[1]):                                                                                  
         #img_y = img[:, y, :]                                                                                     
         img[:,y,:] = cv2.dilate(img[:,y,:], kernel, iterations=1)                                                       
     return img  

def check_brain(path, brain):
    res = -1
    for item in os.listdir(path):
        if brain in item and "cc3d" in item:
            res = item
    return res

def color_sphere(brain, components, stats, item, color):
    assert(brain.shape == components.shape)
    z_0, z_1 = int(stats["bounding_boxes"][item][0].start), int(stats["bounding_boxes"][item][0].stop)
    y_0, y_1 = int(stats["bounding_boxes"][item][1].start), int(stats["bounding_boxes"][item][1].stop)
    x_0, x_1 = int(stats["bounding_boxes"][item][2].start), int(stats["bounding_boxes"][item][2].stop)

    size = 8
    #generate ellipsoid. Note that skimage.draw.ellipsoid assumes xyz, so it must be transposed here 
    el = ellipsoid(size, size, size/(6 / 1.63)).astype(np.uint8)
    el = np.swapaxes (el, 0,-1)

    if el.shape[2] > x_1 - x_0:
        x_1 = x_0 + el.shape[2]
    if el.shape[1] > y_1 - y_0:
        y_1 = y_0 + el.shape[1]
    if el.shape[0] > z_1 - z_0:
        z_1 = z_0 + el.shape[0]

    sub_p = components[z_0:z_1, y_0:y_1, x_0:x_1]

    el *= color
    #TODO Messy: will not be centered
    sub_p[0:el.shape[0],0:el.shape[1],0:el.shape[2]] = el

    brain[z_0:z_1, y_0:y_1, x_0:x_1] = sub_p

    return brain

def color_bb(brain, components, stats, item, color):
    assert(brain.shape == components.shape)
    z_0, z_1 = int(stats["bounding_boxes"][item][0].start), int(stats["bounding_boxes"][item][0].stop)
    y_0, y_1 = int(stats["bounding_boxes"][item][1].start), int(stats["bounding_boxes"][item][1].stop)
    x_0, x_1 = int(stats["bounding_boxes"][item][2].start), int(stats["bounding_boxes"][item][2].stop)
    try:
        sub_p = components[z_0:z_1, y_0:y_1, x_0:x_1]
        sub_p[np.where(sub_p  == item)] = color
    except TypeError as te:
        print(f"{te} where")
        exit()

    try:
        brain[z_0:z_1, y_0:y_1, x_0:x_1] = sub_p
    except TypeError as te:
        print(f"{te} brain")
        exit()

    return brain

def color_bb_rgb(brain, components, stats, item, color):
    try:
        z_0, z_1 = 1 + int(stats["bounding_boxes"][item][0].start), int(stats["bounding_boxes"][item][0].stop)
        y_0, y_1 = 1 + int(stats["bounding_boxes"][item][1].start), int(stats["bounding_boxes"][item][1].stop)
        x_0, x_1 = 1 + int(stats["bounding_boxes"][item][2].start), int(stats["bounding_boxes"][item][2].stop)
        sub_p = components[z_0:z_1, y_0:y_1, x_0:x_1]
        sub_p = np.stack((sub_p, sub_p, sub_p),axis=3)
        #sub_p = np.swapaxes(sub_p, 0, -1)
        #sub_p = np.swapaxes(sub_p, 0, 1)
        #sub_p = np.swapaxes(sub_p, 1, 2)
        coords = tuple(zip(*np.where(sub_p == item)))
        for coord in coords:
            if coord[-1] == 0:
                sub_p[coord] = color[0]
            elif coord[-1] == 1:
                sub_p[coord] = color[1]
            elif coord[-1] == 2:
                sub_p[coord] = color[2]
        brain[z_0:z_1, y_0:y_1, x_0:x_1,:] = sub_p
    except IndexError as ie:
        print(f"{ie} Brain {brain.shape} Components {components.shape} subp {sub_p.shape} {color}")
    except ValueError as ve:
        print(f"{ve} Brain {brain.shape} Components {components.shape} subp {sub_p.shape} {color}")

    return brain

def get_color_set(df):
    """Get df, select # unique RGB triplets, return list of triplets to color according to list
    """
    return sorted(list(df["color-hex-triplet"].unique()))

def blob_highlighter(settings, brain_item,stack_shape):
    """Color blobs by their corresponding atlas region
    """

    path_binary         = settings["visualization"]["input_prediction_location"]
    if settings["visualization"]["input_size_location"] == "":
        settings["visualization"]["input_size_location"] = settings["postprocessing"]["output_location"]
    path_size_csv       = settings["visualization"]["input_size_location"]

    path_cell_csv       = settings["visualization"]["input_csv_location"]

    path_out            = settings["visualization"]["output_location"]
    path_cache          = settings["visualization"]["cache_location"]

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    if not os.path.exists(path_cache):
        os.mkdir(path_cache)

    
    RGB = False

    draw_ellipsoid = True

    #TODO check if it goes through all brains??
    brain = brain_item[0]
    highlight_area = brain_item[1]
    dilation_kernel = 4
    if highlight_area != "":
        print(f"{datetime.datetime.now()} Highlighting {highlight_area} in {brain}")
    else:
        print(f"{datetime.datetime.now()} Highlighting everything in {brain}")

    path_brain_binary       = path_binary   + [x for x in os.listdir(path_binary) if brain in x][0] + "/binary_segmentations/binaries.npy"
    path_brain_size_csv     = path_size_csv + [x for x in os.listdir(path_size_csv) if brain in x][0]
    path_brain_cell_csv     = path_cell_csv + [x for x in os.listdir(path_cell_csv) if brain in x and "cells" in x][0]
    print(path_brain_cell_csv)

    # Load csvs
    print(f"{datetime.datetime.now()} : Loading csv")
    size_csv    = pd.read_csv(path_brain_size_csv,index_col="Blob")
    cell_csv    = pd.read_csv(path_brain_cell_csv,index_col=0)

    # Load binarized outputs
    print(f"{datetime.datetime.now()} : Loading brain")
    #brain_nifti         = read_nifti(path_brain_binary)
    #brain_nifti         = np.swapaxes(brain_nifti, 0, -1)
    #load npy
    brain_nifti = dataset_on_disk = np.memmap(path_brain_binary,dtype=np.uint8,mode='r+',shape=stack_shape[1:])
    brain_nifti = brain_nifti[0,:,:,:]


    # CC3D
    print(f"{datetime.datetime.now()} : CC3D")
    brain_exists = check_brain(path_cache, brain)
    if brain_exists != -1:
        print(f"{datetime.datetime.now()} : CC3D exists, reading...")
        #labels, N = read_nifti(f"{path_cache}{brain_exists}"), int(brain_exists.split("_")[-2])
        #labels, N = np.load(f"{path_cache}{brain_exists}"), int(brain_exists.split("_")[-2])
        labels, N = np.memmap(f"{path_cache}{brain_exists}",dtype=np.uint32,shape=stack_shape[2:]),int(brain_exists.split("_")[-2])
    else:
        print(f"{datetime.datetime.now()} : Calculating CC3D")
        #labels = create_empty_memmap(path_cache + f"{brain}_{N}_cc3d.npy", brain_nifti.shape,dtype=np.uint8,return_torch=False)
        labels, N   = cc3d.connected_components(brain_nifti, return_N=True)
        print(f"{datetime.datetime.now()} : Writing cc3d nifti")
        #write_nifti(path_cache + f"{brain}_{N}_cc3d.nii", labels)
        #np.save(path_cache + f"{brain}_{N}_cc3d.npy", labels.astype(np.uint8))
        #debug
        #print("labels dtype: ", labels.dtype)
        #print("max labels value: ", labels.max())
        np.save(path_cache + f"{brain}_{N}_cc3d.npy", labels)
        

    # Make np zero same size as brain
    print(f"{datetime.datetime.now()} : Creating empty volumes")
    if RGB:
        #cell_nifti      = np.zeros_like(labels, dtype = np.uint8)
        #cell_nifti      = np.stack((cell_nifti, cell_nifti, cell_nifti))
        #cell_nifti      = np.swapaxes(cell_nifti, 0, -1)
        #cell_nifti      = np.swapaxes(cell_nifti, 0, 1)
        #cell_nifti      = np.swapaxes(cell_nifti, 1, 2)
        cell_nifti_path  = path_out + f"{brain}_RGB.npy"
        cell_nifti       = create_empty_memmap(cell_nifti_path, shape=(*brain_nifti.shape,3),dtype=np.uint8,return_torch=False)
    else:
        #cell_nifti      = np.zeros_like(labels, dtype = np.uint8)
        cell_nifti_path  = path_out + f"{brain}_visualization.npy"
        cell_nifti       = create_empty_memmap(cell_nifti_path, shape=brain_nifti.shape,dtype=np.uint8,return_torch=False)
    #debug
    print("cell_nifti shape: ", cell_nifti.shape)

    if not os.path.exists(f"{path_cache}{brain}_statistics.pickledump"):
        print(f"{datetime.datetime.now()} : Calculating statistics")
        stats = cc3d.statistics(labels)
        with open(f"{path_cache}{brain}_statistics.pickledump", "xb") as file:
            pickle.dump(stats, file)
    else:
        print(f"{datetime.datetime.now()} : Reading statistics")
        with open(f"{path_cache}{brain}_statistics.pickledump", "rb") as file:
            stats = pickle.load(file)

    key_error_count = 0
    print(f"{datetime.datetime.now()} : Coloring")
    color_set = get_color_set(cell_csv)
    print(f"Len colorset {len(color_set)}")
    with open(f"{path_cache}{brain}_color_set.txt", "w") as file:
        file.write(str(color_set))

    #debug
    #color_cell_list = []
    print(f"Cell nifti {cell_nifti.shape}")
    print(f"labels {labels.shape}")
    print(cell_csv)
    for label in range(1,N):
        start = datetime.datetime.now()
        if label > 0 and label in cell_csv.index:
            try:
                size    = size_csv.loc[label]["Size"]
                location= size_csv.loc[label]["Coords"]
                if RGB:
                    color_cell = [-1, -1, -1]
                    color_cell[0] = int(cell_csv.loc[label]["red"])
                    color_cell[1] = int(cell_csv.loc[label]["green"])
                    color_cell[2] = int(cell_csv.loc[label]["blue"])
                    cell_nifti = color_bb_rgb(cell_nifti, labels, stats, label, color_cell)
                else:
                    color_cell = color_set.index(cell_csv.loc[label]["color-hex-triplet"])  
                    #debug
                    #color_cell_list.append(color_cell)
                    if highlight_area != "":
                        if highlight_area == cell_csv.loc[label]["color-hex-triplet"]:
                            if draw_ellipsoid:
                                cell_nifti = color_sphere(cell_nifti, labels, stats, label, color_cell)
                            else:
                                cell_nifti = color_bb(cell_nifti, labels, stats, label, color_cell)
                    else:
                        if draw_ellipsoid:
                            cell_nifti = color_sphere(cell_nifti, labels, stats, label, color_cell)
                        else:
                            cell_nifti = color_bb(cell_nifti, labels, stats, label, color_cell)
            # except KeyError as ke:
                # if "key_error_count" in locals():
                # key_error_count += 1
                # if label == 1:
                    # print(f"No first label found - Big oopsie! Loc vs iLoc?\nKeyError {ke}")
                    # exit()
                # print(f"\nKey Error {ke} Count {key_error_count}",end="\r",flush=True)
            except TypeError as te:
                print(f"\nType Error {te} {label}")
            except ValueError as ve:
                print(f"\nValue Error {ve} {label}")
            except IndexError as ie:
                print(f"\nIndex Error {ie} {label}")
        #debug
        #with open(f"{path_cache}{brain}_color_set_list.txt", "w") as file:
        #   file.write(str(color_cell_list))

        if label % 50 == 0:
            d = datetime.datetime.now()
            print(f"{label} / {N} | KeyErrors: {key_error_count} ETA {((N - label) * (d-start).total_seconds())/60:.2f} Minutes remaining",end="\r",flush=True)
    #cell_nifti = np.swapaxes(cell_nifti, 0, 2)

    print(f"{datetime.datetime.now()} : Writing cell nifti {np.amin(cell_nifti)} {np.amax(cell_nifti)}")
    output_name = f"{path_out}{brain}"
    if RGB:
        output_name += "_rgb"
    else:
        if dilation_kernel > 0:
            cell_nifti = fast_dilate(cell_nifti, dilation_kernel)
            output_name += "_dilated"
        output_name += "_singlecolor"
        if highlight_area != "":
            output_name += f"_{highlight_area}"

    if draw_ellipsoid:
        output_name += "_ellipsoid"
    output_name += ".npy"
    #write_nifti(output_name, cell_nifti)

    os.makedirs(os.path.join(path_out, brain),exist_ok=True)
    #output tiff stack
    for z in range(cell_nifti.shape[0]):
        if RGB: 
            z_plane = cell_nifti[z,:,:,:]
        else:
            z_plane = cell_nifti[z,:,:]
        #debug
        print("z: ",z," z_plane shape: ",z_plane.shape)
        io.imsave(output_name+"_Z"+str(z).zfill(4)+".tif", z_plane, compression='lzw', check_contrast=False)
     
    #close the file and rename 
    cell_nifti.flush()
    del cell_nifti
    #rename the file 
    os.rename(cell_nifti_path, output_name)
    