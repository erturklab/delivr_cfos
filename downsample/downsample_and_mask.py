#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:20:47 2022

@author: wirrbel
"""

import os
import glob
from skimage import io, transform
from scipy.ndimage import zoom
import numpy as np
import multiprocessing as mp 
import tempfile
import cv2
import datetime

def get_real_size(raw_folder):
    z = len([i for i in os.listdir(raw_folder) if ".tif" in i])
    img = cv2.imread(raw_folder + "/" + os.listdir(raw_folder)[0])
    x = img.shape[0]
    y = img.shape[1]
    return (x, y, z)

def downsample_zplanes(raw_location,raw_image_list,x_ratio,y_ratio,z_ratio,temp_dir,z_planes):
    #define empty chunk list 
    z_chunk = []
    #generate list of all planes in the chunk 
    for plane in range(z_planes[0],z_planes[1]):
        z_chunk.append(os.path.join(raw_location,raw_image_list[plane]))
            
    #generate collection, then read into memory
    z_chunk_imgs = io.imread_collection(z_chunk)
    z_chunk_imgs = io.concatenate_images(z_chunk_imgs)

    #downsample to the right dimensions 
    z_chunk_downsampled = transform.downscale_local_mean(z_chunk_imgs,(z_ratio,y_ratio,x_ratio)).astype('uint16')
    
    #save to temporary directory 
    io.imsave(os.path.join(temp_dir.name,'downsampled_um_z' + str(z_planes[0]).zfill(4) + '-' + str(z_planes[1]).zfill(4) + '.tif'),z_chunk_downsampled,compress=True)    

def ilastik_ventricles(results_folder,downsampled_name,ilastik_path,ventricle_masking_ilastik_project):
    #runs Ilastik as command line tool 
    
    #assemble command 
    cmd = str(os.path.join(ilastik_path,'run_ilastik.sh')
              + ' --headless '
              + ' --project=' + ventricle_masking_ilastik_project + ' '
              + os.path.join(results_folder,downsampled_name + '.tif')
              )
    
    #run command
    res = os.system(cmd)
    
    #reassemble images
    mask_dir = os.path.join(results_folder,downsampled_name + ".tif")
    mask = io.imread(mask_dir) 
    
    #cast to 8-bit (should be superfluous)
    mask = mask.astype('uint8')
    
    #Ilastik exports probabilities mapped between 0 and 255.
    #thresholding at 128: everything smaller = 0, everything larger = 1
    mask[mask < 128] = 0
    mask[mask >= 128] = 1
    
    #save in main folder 
    io.imsave(os.path.join(results_folder,downsampled_name + '_mask.tif'),mask,compress=True)
    
    return mask 

def collect_measurements(original_um_x,original_um_y,original_um_z,downsampled_um_x,downsampled_um_y,downsampled_um_z,raw_location,raw_image_list,mask):
    yx = io.imread(os.path.join(raw_location,raw_image_list[0]))

    #create dictionary for original dimensions 
    original_dims = {'original_um_x' : original_um_x,
                    'original_um_y' : original_um_y,
                    'original_um_z' : original_um_z,
                    'original_px_x' : yx[1],
                    'original_px_x' : yx[0],
                    'original_px_x' : len(raw_image_list)
                    }
    
    #create dict for downsampled dimensions 
    downsampled_dims = {'downsampled_um_x' : downsampled_um_x,
                        'downsampled_um_y' : downsampled_um_y,
                        'downsampled_um_z' : downsampled_um_z,
                        'downsampled_px_x' : mask.shape[2],
                        'downsampled_px_y' : mask.shape[1],
                        'downsampled_px_z' : mask.shape[0]
                        }
    
    return original_dims, downsampled_dims
    
def upsample(image : np.array,
        original_um_x : float,
        original_um_y : float,
        original_um_z : float,
        downsample_um_x : float,
        downsample_um_y : float,
        downsample_um_z : float) -> np.array:
    """
    Upsamples image based on original values
    Args:
        image (np.array) : Downsampled image
        original_um_x (float) : Original x step
        original_um_y (float) : Original y step
        original_um_z (float) : Original z step
        downsample_um_x (float) : Downsampled x step 
        downsample_um_y (float) : Downsampled y step
        downsample_um_z (float) : Downsampled z step
    Returns:
        mask_us (np.array) : Upscaled image
    """
    x_ratio = downsample_um_x/original_um_x
    y_ratio = downsample_um_y/original_um_y
    z_ratio = downsample_um_z/original_um_z
    mask_us = zoom(image,(z_ratio, y_ratio, x_ratio),output='uint8')
    return mask_us

    
# if __name__ == '__main__':
def downsample_mask(settings, brain):

    #stitched images are here 
    raw_location = os.path.join(settings["raw_location"], brain)
    #generate a sorted list of all images 
    raw_image_list = sorted(glob.glob(raw_location+'/*.tif'))

    original_um_x = settings["mask_detection"]["downsample_steps"]["original_um_x"]
    original_um_y = settings["mask_detection"]["downsample_steps"]["original_um_y"]
    original_um_z = settings["mask_detection"]["downsample_steps"]["original_um_z"]

    downsampled_um_x = settings["mask_detection"]["downsample_steps"]["downsample_um_x"]
    downsampled_um_y = settings["mask_detection"]["downsample_steps"]["downsample_um_y"]
    downsampled_um_z = settings["mask_detection"]["downsample_steps"]["downsample_um_z"]

    ilastik_path = settings["mask_detection"]["ilastik_location"]

    ventricle_masking_ilastik_project = settings["mask_detection"]["ilastik_model"]

    #calculate the ratios 
    x_ratio = round(downsampled_um_x/original_um_x)
    y_ratio = round(downsampled_um_y/original_um_y)
    z_ratio = round(downsampled_um_z/original_um_z)
    
    #split the image list into this many chunks along z
    z_series = np.arange(0,len(raw_image_list),z_ratio)
    
    #create a new results folder
    results_folder = settings["mask_detection"]["output_location"]
    if results_folder == "":
        parent_dir,raw_folder = os.path.split(os.path.abspath(raw_location))
        results_folder = os.path.join(parent_dir,raw_folder+'_results')

    #try to create
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
        
    #create a temporary storage directory for downsampling 
    temp_dir = tempfile.TemporaryDirectory() 

    #open multiprocessing pool 
    pool = mp.Pool(processes = 6)

    #run resampling in parallel 
    for z_planes in zip(z_series,z_series[1:]):
        pool.apply_async(downsample_zplanes, (raw_location,raw_image_list,x_ratio,y_ratio,z_ratio,temp_dir,z_planes))
    
    ##close pool 
    pool.close()
    pool.join()
    # for z_planes in zip(z_series, z_series[1:]):
    #     downsample_zplanes(raw_location, raw_image_list, x_ratio, y_ratio, z_ratio, temp_dir, z_planes)
    
    #list images 
    downsampled_list = sorted([os.path.join(temp_dir.name, file) for file in os.listdir(temp_dir.name)])
    
    #debug
    print(downsampled_list)
    
    #reassemble images to one big stack 
    downsampled_stack = io.imread_collection(downsampled_list)
    try:
        downsampled_stack = io.concatenate_images(downsampled_stack)
    except ValueError as ve:
        if len(downsampled_list) > 0:
            print(f"{ve}: {downsampled_list[0]} {downsampled_list[-1]}")
            exit()
        else:
            print("Downsampled list empty!")
    
    #save as new stack in results_folder
    downsampled_name = 'stack_resampled'
    io.imsave(os.path.join(results_folder,downsampled_name + '.tif'),downsampled_stack,compress=True)
    
    #cleanup 
    temp_dir.cleanup()
    
    #run Ilastik 
    print(results_folder)
    print(downsampled_name)
    print(ilastik_path)
    print(ventricle_masking_ilastik_project)
    start = datetime.datetime.now()
    downsampled_mask = ilastik_ventricles(results_folder,downsampled_name,ilastik_path,ventricle_masking_ilastik_project)
    delta = datetime.datetime.now() -start
    print(f"Ilastik: {delta}")
    
    #define dimension dictionaries 
    original_dims, downsampled_dims = collect_measurements(original_um_x,original_um_y,original_um_z,downsampled_um_x,downsampled_um_y,downsampled_um_z,raw_location,raw_image_list,downsampled_mask)
    ##calculate the ratios
    #x_ratio = downsampled_um_x/original_um_x
    #y_ratio = downsampled_um_y/original_um_y
    #z_ratio = downsampled_um_z/original_um_z
    # zoom(downsampled_mask,(z_ratio, y_ratio, x_ratio),output='uint8')
    # mask_us = upsample(downsampled_mask, 
    #                     original_um_x, 
    #                     original_um_y, 
    #                     original_um_z,
    #                     downsampled_um_x,
    #                     downsampled_um_y,
    #                     downsampled_um_z)

    raw_shape = get_real_size(raw_location)
    print(f"Before upsampling: {downsampled_mask.shape}")
    print(f"Raw shape {raw_shape}")

    z_ratio = raw_shape[2] / downsampled_mask.shape[0] 
    y_ratio = raw_shape[0] / downsampled_mask.shape[1] 
    x_ratio = raw_shape[1] / downsampled_mask.shape[2] 

    start = datetime.datetime.now()
    mask_us = zoom(downsampled_mask,(z_ratio, y_ratio, x_ratio),output='uint8')
    delta = datetime.datetime.now() - start
    print(f"Zoom: {delta}")
    
    mask_us = np.swapaxes(mask_us, 0, 2)
    
    print(f"Final shape {mask_us.shape}")

    if mask_us.shape != raw_shape:
        mask_us = np.squeeze(mask_us)
        mask_us = np.swapaxes(mask_us, 0, 1)
    print(f"Saving final masked data {mask_us.shape}\n")
    start = datetime.datetime.now()
    for i, item in enumerate(sorted([x for x in os.listdir(raw_location) if ".tif" in x])):
        img = cv2.imread(raw_location + "/" + item, -1)
        print(f"{item} {i} / {mask_us.shape[2]} MASK {mask_us.shape} RAW {img.shape}")
        img *= mask_us[:,:,i]
        io.imsave(os.path.join(results_folder, item), img)
    delta = datetime.datetime.now() - start
    print(f"Masking: {delta}")


    
