import os
import datetime
import cc3d
import pickle
import cv2
import numpy as np
import pandas as pd
import shutil
import os
import tifffile
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
    brain = brain_item[0] # subfolder in "/data/output/02_blob_detection/output/brain_id_subfolder"
    highlight_area = brain_item[1] #not currently implemented 
    #dilation_kernel = 4
    if highlight_area != "":
        print(f"{datetime.datetime.now()} Highlighting {highlight_area} in {brain}")
    else:
        print(f"{datetime.datetime.now()} Highlighting everything in {brain}")
    
    #define paths 
    path_binary         = settings["visualization"]["input_prediction_location"] #"/data/output/02_blob_detection/output/"
    path_cell_csv       = settings["visualization"]["input_csv_location"] #"/data/output/04_atlas_alignment/collection/", should be "/data/output/05_region_assignment/",
    path_out            = settings["visualization"]["output_location"] #"/data/output/06_visualization/output/"
    path_cache          = settings["visualization"]["cache_location"] #"/data/output/06_visualization/cache/"
    
    #add the brain name to the     
    path_out = os.path.join(path_out,brain,brain+"_rgb_tiffs")
    path_cache = os.path.join(path_cache,brain)
    
    try:
        os.makedirs(path_out,exist_ok=True)
        os.makedirs(path_cache,exist_ok=True)
    except:
        pass

    path_brain_binary       = path_binary   + [x for x in os.listdir(path_binary) if brain in x][0] + "/binary_segmentations/binaries.npy"
    path_brain_cell_csv     = path_cell_csv + [x for x in os.listdir(path_cell_csv) if "cells_" + brain in x and ".csv" in x][0]
    print(path_brain_cell_csv)

    # Load cells, filter out cells annotated as background:
    print(f"{datetime.datetime.now()} : Loading csv")
    cell_csv    = pd.read_csv(path_brain_cell_csv,index_col=0)
    cell_csv = cell_csv.loc[cell_csv["acronym"] != "bgr"]

    # Load binarized outputs
    print(f"{datetime.datetime.now()} : Loading brain")
    bin_img = np.memmap(path_brain_binary,dtype=np.uint8,mode='r+',shape=stack_shape[1:])
    bin_img = bin_img[0,:,:,:]
    
    #calculate cc3d statistics (no caching needed because thanks to no_slice_conversion it's fast)
    print(f"{datetime.datetime.now()} : calculating connected-component analysis")
    labels, N = cc3d.connected_components(bin_img, return_N=True)
    stats = cc3d.statistics(labels,no_slice_conversion=True)
    
    #create temporary red, green, and blue arrays 
    print(f"{datetime.datetime.now()} : coloring blobs")
    R_img = np.lib.format.open_memmap(os.path.join(path_cache,"R_img.npy"),mode='w+',dtype=np.uint8,shape=stack_shape[2:])
    G_img = np.lib.format.open_memmap(os.path.join(path_cache,"G_img.npy"),mode='w+',dtype=np.uint8,shape=stack_shape[2:])
    B_img = np.lib.format.open_memmap(os.path.join(path_cache,"B_img.npy"),mode='w+',dtype=np.uint8,shape=stack_shape[2:])

    #iterate cells, color one by one 
    for cc_id in cell_csv['connected_component_id']:
        #extract cell (returns empty df if not present) 
        current_cell = cell_csv.loc[cell_csv['connected_component_id'] == cc_id]
        #extract bounding box coordinates 
        bb = stats['bounding_boxes'][cc_id]
        #color the positive values inside the BB. 
        #Note this is optimized for small, round-ish blobs. Long, diagonal blobs (i.e. blood vessels) might accidentally re-color other blobs close by. 
        R_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = bin_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] * current_cell['red'].to_numpy()
        G_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = bin_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] * current_cell['green'].to_numpy()
        B_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = bin_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] * current_cell['blue'].to_numpy()

    #output RGB images 
    print(f"{datetime.datetime.now()} : Generating tiffs")
    for z in range(bin_img.shape[0]):
        zplane_r = R_img[z,:,:]
        zplane_g = G_img[z,:,:]
        zplane_b = B_img[z,:,:]
        zplane_rgb = np.stack((zplane_r,zplane_g,zplane_b))
        tifffile.imwrite(os.path.join(path_out,"rgb_z"+str(z).zfill(4)+".tif"),zplane_rgb,compression='lzw')
        
    #cleanup
    print(f"{datetime.datetime.now()} : Cleanup")
    shutil.rmtree(path_cache)