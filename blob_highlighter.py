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
from blob_depthmap import depth_map_blobs

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
    path_out_rgb = os.path.join(path_out,brain,brain+"_rgb_tiffs")
    path_cache = os.path.join(path_cache,brain)
    
    try:
        os.makedirs(path_out_rgb,exist_ok=True)
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
    
    #default: make RGB area color-coded output
    if settings["visualization"]["region_id_rgb"]: 
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
        print(f"{datetime.datetime.now()} : Generating RGB tiffs")
        for z in range(bin_img.shape[0]):
            zplane_r = R_img[z,:,:]
            zplane_g = G_img[z,:,:]
            zplane_b = B_img[z,:,:]
            zplane_rgb = np.stack((zplane_r,zplane_g,zplane_b))
            tifffile.imwrite(os.path.join(path_out_rgb,"rgb_z"+str(z).zfill(4)+".tif"),zplane_rgb,compression='lzw')

    #optionally, also save with region_id as gray values 
    print(f"{datetime.datetime.now()} : Generating region_id gray-value tiffs")
    if settings["visualization"]["region_id_grayvalues"]: 
        #create another output folder 
        path_out_region_id = os.path.join(path_out,brain,brain+"_region_id_tiffs")
        try:
            os.makedirs(path_out_region_id,exist_ok=True)
        except:
            pass 
        #create temporary 16-bit gray value npy file in cache dir 
        region_id_img = np.lib.format.open_memmap(os.path.join(path_cache,"region_id_img.npy"),mode='w+',dtype=np.uint16,shape=stack_shape[2:])
        #iterate through the list and color-code 
        for cc_id in cell_csv['connected_component_id']:
            #extract cell (returns empty df if not present) 
            current_cell = cell_csv.loc[cell_csv['connected_component_id'] == cc_id]
            #extract bounding box coordinates 
            bb = stats['bounding_boxes'][cc_id]
            #color the positive values inside the BB. 
            #Note this is optimized for small, round-ish blobs. Long, diagonal blobs (i.e. blood vessels) might accidentally re-color other blobs close by. 
            region_id_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = bin_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]].astype(np.uint16) * current_cell['graph_order'].to_numpy()
        for z in range(bin_img.shape[0]):
            zplane_region = region_id_img[z,:,:]
            tifffile.imwrite(os.path.join(path_out_region_id,"region_id_"+str(z).zfill(4)+".tif"),zplane_region,compression='lzw')

    #optionally, only map the blobs over their distance from the sample's outside. Useful if there is no atlas at hand (e.g. other organs). 
    if settings["visualization"]["no_atlas_depthmap"]: 
        blob_depthmap(settings,brain,stack_shape)
        
    #cleanup
    print(f"{datetime.datetime.now()} : Cleanup")
    shutil.rmtree(path_cache)