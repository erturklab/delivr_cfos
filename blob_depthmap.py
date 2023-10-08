#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 09:24:13 2023

@author: wirrbel
"""

import numpy as np
import os
import tifffile
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.ndimage import distance_transform_edt
import glob
import shutil
import cc3d
import datetime


def calculate_mask_distance(root_dir,spacing=(1,1,1),collection_dir=None,intensity_max=None,prefix=None):
    #determine sample name (root dir name)
    sample_name = os.path.split(root_dir)[-1]
    
    #determine input/output dir 
    input_dir = os.path.join(root_dir,"C01")
    output_dir = os.path.join(collection_dir,prefix+sample_name+"_C01_output")
    
    #load mask image 
    masked_stack = tifffile.imread(sorted(glob.glob(input_dir + "/*.tif")))
    
    #prepare output dir (overwriting previous input)
    try:
        shutil.rmtree(output_dir)
    except:
        pass
    os.makedirs(output_dir,exist_ok=True)

    #save to temporary npy array
    #np.save(os.path.join(output_dir,"distances.npy"), np.zeros(shape=masked_stack.shape,dtype=np.float64))
    #distances = np.memmap(os.path.join(output_dir,"distances.npy"),mode= 'r+',dtype=np.float64,shape=masked_stack.shape)
    
    #compute distance transform
    distances = distance_transform_edt(masked_stack,sampling=spacing)
    
    #write out as tiff 
    #tifffile.imwrite(os.path.join(output_dir,sample_name+"_mask_heatmap.tif"),distances.astype(np.uint16))
    
    #read stack if already present
    #distances = tifffile.imread(os.path.join(output_dir,sample_name+"mask_heatmap.tif"))
    
    #transform to 1D arrays + bring into pandas dataframe 
    depth_values = np.ravel(distances)
    pixel_values = np.ravel(masked_stack)
    combined_values = np.asarray((depth_values,pixel_values))
    combined_values = np.swapaxes(combined_values,0,1)
    combined_data = pd.DataFrame(data=depth_values,columns=["depth"])
    combined_data["intensity"] = pixel_values
    
    #strip out background
    combined_data = combined_data.loc[combined_data["depth"] > 0]
    
    #slice into even distance bins 
    bin_range = range(0,int(combined_data['depth'].max()))
    combined_data['intensity_bins'] = pd.cut(combined_data['depth'],bins=bin_range)
    
    #measure median in each bin 
    depth_intensity_profile = combined_data['intensity'].groupby(combined_data['intensity_bins']).median()
    
    #plot and save 
    plt.clf()
    plt.plot(bin_range[:-1],depth_intensity_profile.values)
    plt.title('depth profile')
    plt.ylabel("median intensity (a.u.)")
    plt.xlabel("depth (µm)")
    if intensity_max is not None:
        plt.xmax(intensity_max)
    plt.savefig(os.path.join(output_dir,"depthmap_01"+".svg"))
    
    #optional: save to collection folder
    if collection_dir is not None:
            plt.savefig(os.path.join(collection_dir,prefix+sample_name+"_depthmap_01"+".svg"))
            combined_data.to_csv(os.path.join(collection_dir,prefix+sample_name+"_combined_data.csv"))

    
    #cleanup temp file
    del distances
    try:
        os.remove(os.path.join(output_dir,"distances.npy"))
        os.remove(os.path.join(output_dir,"*.tif"))
    except:
        pass




def depth_map_blobs(settings, brain,stack_shape):
    #define input folder for cc3d 
    
    #define paths 
    path_binary         = settings["visualization"]["input_prediction_location"] #"/data/output/02_blob_detection/output/"
    path_out            = settings["visualization"]["output_location"] #"/data/output/06_visualization/output/"
    path_cache          = settings["visualization"]["cache_location"] #"/data/output/06_visualization/cache/"
    
    #add the brain name to the     
    path_out_depthmap = os.path.join(path_out,brain,brain+"_depthmap_tiffs")
    path_cache = os.path.join(path_cache,brain)
    
    try:
        os.makedirs(path_out_depthmap,exist_ok=True)
        os.makedirs(path_cache,exist_ok=True)
    except:
        pass

    path_brain_binary       = path_binary   + [x for x in os.listdir(path_binary) if brain in x][0] + "/binary_segmentations/binaries.npy"
    
    #do cc3d 
    # Load binarized outputs
    print(f"{datetime.datetime.now()} : Loading brain")
    bin_img = np.memmap(path_brain_binary,dtype=np.uint8,mode='r+',shape=stack_shape[1:])
    bin_img = bin_img[0,:,:,:]
    
    #calculate cc3d statistics (no caching needed because thanks to no_slice_conversion it's fast)
    print(f"{datetime.datetime.now()} : calculating connected-component analysis")
    labels, N = cc3d.connected_components(bin_img, return_N=True)
    stats = cc3d.statistics(labels,no_slice_conversion=True)        

    #calculate depth map from downsampled masked stack
    print(f"{datetime.datetime.now()} : calculating euclidean distance transform")
    mask_detection_folder = settings["mask_detection"]["output_location"] # e.g. "/data/output/01_mask_detection/output/"
    downsampled_masked_stack_path = os.path.join(mask_detection_folder,brain,"downsampled_masked_stack.tif") 

    #get measurements from settings 
    original_dims_x = settings["mask_detection"]["downsample_steps"]["original_um_x"]
    original_dims_y = settings["mask_detection"]["downsample_steps"]["original_um_y"]
    original_dims_z = settings["mask_detection"]["downsample_steps"]["original_um_z"]
    downsample_dims_x = settings["mask_detection"]["downsample_steps"]["downsample_um_x"]
    downsample_dims_y = settings["mask_detection"]["downsample_steps"]["downsample_um_y"]
    downsample_dims_z = settings["mask_detection"]["downsample_steps"]["downsample_um_z"]
    
    #load masked stack
    masked_stack = tifffile.imread(downsampled_masked_stack_path)
    
    #pad with zeroes in all directions to avoid EDT errors at edges 
    stack_padded = np.pad(masked_stack,((1,1,),(1,1,),(1,1,)))
    
    #calculate distance from the edges 
    distances = distance_transform_edt(stack_padded,sampling=(downsample_dims_z,downsample_dims_y,downsample_dims_x))

    #remove padding
    distances = distances[1:-1,1:-1,1:-1]
    distances = distances.astype(np.uint16)
    
    print(f"{datetime.datetime.now()} : resampling cc3d centroids")
    #resample cc3d points 
    coordinates = stats["centroids"].copy()
    coordinates[:,0] = coordinates[:,0]/(downsample_dims_z/original_dims_z)
    coordinates[:,1] = coordinates[:,1]/(downsample_dims_y/original_dims_y)
    coordinates[:,2] = coordinates[:,2]/(downsample_dims_x/original_dims_x)
    coordinates = coordinates.astype(int)
    

    print(f"{datetime.datetime.now()} : generating depth-coded blob map")
    #create temporary 16-bit gray value npy file in cache dir 
    depthmap_img = np.lib.format.open_memmap(os.path.join(path_cache,"path_out_depthmap.npy"),mode='w+',dtype=np.uint16,shape=stack_shape[2:])
    #iterate through the list and color-code 
    for cc_id in range(N):
        #extract cell (returns empty df if not present) 
        current_cell = coordinates[cc_id]
        current_cell_depth = distances[current_cell[0],current_cell[1],current_cell[2]]
        #extract bounding box coordinates 
        bb = stats['bounding_boxes'][cc_id]
        #color the positive values inside the BB. 
        #Note this is optimized for small, round-ish blobs. Long, diagonal blobs (i.e. blood vessels) might accidentally re-color other blobs close by. 
        depthmap_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]] = bin_img[bb[0]:bb[1],bb[2]:bb[3],bb[4]:bb[5]].astype(np.uint16) * current_cell_depth
        
    print(f"{datetime.datetime.now()} : exporting depth-coded tiffs")
    #export as tiff 
    for z in range(bin_img.shape[0]):
        zplane_depthmap = depthmap_img[z,:,:]
        tifffile.imwrite(os.path.join(path_out_depthmap,"depthmap_"+str(z).zfill(4)+".tif"),zplane_depthmap,compression='lzw')

    #cleanup
    print(f"{datetime.datetime.now()} : Cleanup")
    try:
        shutil.rmtree(path_cache)
    except:
        pass

    
