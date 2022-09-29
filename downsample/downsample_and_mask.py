#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:20:47 2022

@author: wirrbel
"""

import os
import glob
import numpy as np
import nibabel as nib
import multiprocessing as mp 
import tempfile
import cv2
import datetime
from skimage import io, transform
from scipy.ndimage import zoom
from subprocess import Popen
from collections import deque


def write_nifti(path,volume):
    """
    write_nifti(path,volume)
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. Taken from Olivers filehandling class
    """
    if(path.find(".nii.gz")==-1):
        path = path + ".nii.gz"
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume, 0, 1), affine=affmat)
    nib.save(NiftiObject, os.path.normpath(path))

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
    io.imsave(os.path.join(temp_dir.name,'downsampled_um_z' + str(z_planes[0]).zfill(4) + '-' + str(z_planes[1]).zfill(4) + '.tif'),z_chunk_downsampled,compress=True,check_contrast=False)    

def save_vaa3d(teraconverter_path, item_path, results_path):
    cmd = str(f"{teraconverter_path}teraconverter --sfmt=\"TIFF (3D)\" \
            --dfmt=\"Vaa3D raw (tiled, 3D)\" \
            -s=\"{item_path}\" \
            -d=\"{results_path}\"")
    Popen(cmd, shell=True).wait()

def ilastik_ventricles(results_folder,downsampled_name,ilastik_path,ventricle_masking_ilastik_project):
    #runs Ilastik as command line tool 
    
    #assemble command 
    cmd = str(os.path.join(ilastik_path,'run_ilastik.sh')
              + ' --headless '
              + ' --project=' + ventricle_masking_ilastik_project + ' '
              + os.path.join(results_folder,downsampled_name + '.tif')
              )
    
    #run command
    # res = os.system(cmd)
    res = Popen(cmd, shell=True).wait()
    
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

def setup_folders(settings):
    # Location of the raw tiff files
    raw_location = os.path.join(settings["raw_location"], brain)

    # General output folder
    results_folder = settings["mask_detection"]["output_location"]

    #  If there's no specified output folder, we make our own
    if results_folder == "":
        parent_dir,raw_folder = os.path.split(os.path.abspath(raw_location))
        results_folder = os.path.join(parent_dir,raw_folder+'mask_detection_results')

    # Try to create the general output folder
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    #create a temporary storage directory for downsampling 
    temp_dir = tempfile.TemporaryDirectory() 
    

    # Folder for the masked data as a tiff stack
    downsampled_masked_name     = 'stack_masked_downsampled'
    downsampled_masked_path     = os.path.join(results_folder, downsampled_masked_name)
    if not os.path.exists(downsampled_masked_path):
        os.mkdir(downsampled_masked_path)

    # Downsampled stack
    downsampled_name = 'stack_resampled'
    downsampled_masked_name     = 'stack_downsampled'
    downsampled_masked_path     = os.path.join(results_folder, downsampled_masked_name)
    downsampled_masked_vaa3d    = os.path.join(results_folder, "stack_downsampled_vaa3d")

    if not os.path.exists(downsampled_masked_vaa3d):
        os.mkdir(downsampled_masked_vaa3d)

    if not os.path.exists(os.path.join(results_folder, "masked_tiffs")):
        os.mkdir(os.path.join(results_folder, "masked_tiffs"))

    if not os.path.exists(os.path.join(results_folder, "masked_niftis")):
        os.mkdir(os.path.join(results_folder, "masked_niftis"))

    
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
    
    #XXX XXX XXX XXX XXX XXX XXX XXX
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
    #XXX XXX XXX XXX XXX XXX XXX XXX

    #open multiprocessing pool 
    pool = mp.Pool(processes = 6)

    #run resampling in parallel 
    for z_planes in zip(z_series,z_series[1:]):
        pool.apply_async(downsample_zplanes, (raw_location,raw_image_list,x_ratio,y_ratio,z_ratio,temp_dir,z_planes))
    
    ##close pool 
    pool.close()
    pool.join()
    # Debug
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
    io.imsave(os.path.join(results_folder,downsampled_name + '.tif'),downsampled_stack,compress=True,check_contrast=False)

    #TODO save downsampled stack
    downsampled_name = 'stack_resampled'
    downsampled_masked_name     = 'stack_downsampled'
    downsampled_masked_path     = os.path.join(results_folder, downsampled_masked_name)
    downsampled_masked_vaa3d    = os.path.join(results_folder, "stack_downsampled_vaa3d")
    if not os.path.exists(downsampled_masked_vaa3d):
        os.mkdir(downsampled_masked_vaa3d)
    teraconverter_path = settings["mask_detection"]["teraconverter_location"]
    save_vaa3d(teraconverter_path, os.path.join(results_folder,downsampled_name + '.tif'), downsampled_masked_vaa3d)
    
    #cleanup 
    temp_dir.cleanup()
    
    #run Ilastik 
    print(results_folder)
    print(downsampled_name)
    print(ilastik_path)
    print(ventricle_masking_ilastik_project)
    start = datetime.datetime.now()
    downsampled_mask = ilastik_ventricles(results_folder,downsampled_name,ilastik_path,ventricle_masking_ilastik_project)
    print(f"Downsampled mask: {np.min(downsampled_mask)} {np.max(downsampled_mask)} {downsampled_mask.dtype}")
    downsampled_mask[downsampled_mask < 0.5] = 0
    downsampled_mask[downsampled_mask >= 0.5] = 1
    delta = datetime.datetime.now() -start
    print(f"Ilastik: {delta}")
    


    #make downsampled masked stack
    downsampled_masked_stack    = downsampled_mask * downsampled_stack
    downsampled_masked_name     = 'stack_masked_downsampled'
    downsampled_masked_path     = os.path.join(results_folder, downsampled_masked_name)
    downsampled_masked_vaa3d    = os.path.join(results_folder, "stack_masked_downsampled_vaa3d")
    print(f"downsampled stack {downsampled_stack.shape}")
    print(f"downsampled mask {downsampled_mask.shape}")
    print(f"downsampled masked stack {downsampled_masked_stack.shape}")
    start = datetime.datetime.now()
    if not os.path.exists(downsampled_masked_path):
        os.mkdir(downsampled_masked_path)
    if not os.path.exists(downsampled_masked_vaa3d):
        os.mkdir(downsampled_masked_vaa3d)
    print(downsampled_masked_path)
    # for downsampled_masked_slice in range(downsampled_masked_stack.shape[0]):
        # io.imsave(downsampled_masked_path + f"/{downsampled_masked_slice}.tif", downsampled_masked_stack[downsampled_masked_slice,:,:], compress=True)
    io.imsave(downsampled_masked_path + "/downsampled_masked_stack.tif", downsampled_masked_stack, compress=True,check_contrast=False)

    save_vaa3d(teraconverter_path, downsampled_masked_path + "/downsampled_masked_stack.tif", downsampled_masked_vaa3d)
    delta = datetime.datetime.now() -start
    print(f"Saving downsampled mask as tiff and vaa3d: {delta}")


    #define dimension dictionaries 
    original_dims, downsampled_dims = collect_measurements(original_um_x,original_um_y,original_um_z,downsampled_um_x,downsampled_um_y,downsampled_um_z,raw_location,raw_image_list,downsampled_mask)

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

    #TODO Mask * raw, save as tiff
    #TODO Convert tiff stack as .v3draw
    #TODO Copy .v3draw from subfolders
    #TODO Setup folders
    
    mask_us = np.swapaxes(mask_us, 0, 2)
    
    print(f"Final shape {mask_us.shape}")

    if mask_us.shape != raw_shape:
        mask_us = np.squeeze(mask_us)
        mask_us = np.swapaxes(mask_us, 0, 1)
    print(f"Saving final masked data {mask_us.shape}\n")
    start = datetime.datetime.now()

    # Save all tiffs neatly in a folder
    if not os.path.exists(os.path.join(results_folder, "masked_tiffs")):
        os.mkdir(os.path.join(results_folder, "masked_tiffs"))
    
    masked_nii = deque()
    for i, item in enumerate(sorted([x for x in os.listdir(raw_location) if ".tif" in x])):
        img = cv2.imread(raw_location + "/" + item, -1)
        print(f"{item} {i} / {mask_us.shape[2]} MASK {mask_us.shape} RAW {img.shape}")
        # Mask raw data with upscaled mask
        img *= mask_us[:,:,i]
        # Add to final Nifti output
        masked_nii.append(img)
        # Save masked raw data
        io.imsave(os.path.join(results_folder, "masked_tiffs", item), img, check_contrast=False)
    delta = datetime.datetime.now() - start
    print(f"Masking: {delta}")


    print("Saving final data as Nifti")
    start = datetime.datetime.now()
    # Save all tiffs neatly in a folder
    if not os.path.exists(os.path.join(results_folder, "masked_niftis")):
        os.mkdir(os.path.join(results_folder, "masked_niftis"))

    masked_nii = np.array(masked_nii)

    masked_nii = np.swapaxes(masked_nii, 0, -1)
    masked_nii = np.swapaxes(masked_nii, 0, 1)
    write_nifti(os.path.join(results_folder, "masked_niftis") + "/masked_nifti.nii.gz", masked_nii)
    delta = datetime.datetime.now() - start
    print(f"Nifti: {delta}")
    
