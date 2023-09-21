#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:20:47 2022

@author: wirrbel
"""

import os
import glob
import shutil
import numpy as np
import nibabel as nib
import multiprocessing as mp 
import tempfile
import cv2
import datetime
from skimage import io, transform
from skimage.util import img_as_ubyte
from scipy.ndimage import zoom
from subprocess import Popen
from collections import deque
from subprocess import DEVNULL



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
    y = img.shape[0]
    x = img.shape[1]
    return (z, y, x)

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
    io.imsave(os.path.join(temp_dir,'downsampled_um_z' + str(z_planes[0]).zfill(4) + '-' + str(z_planes[1]).zfill(4) + '.tif'),z_chunk_downsampled,compression='lzw',check_contrast=False)    

def save_vaa3d(teraconverter_path, item_path, results_path):
    cmd = str(f"{teraconverter_path}teraconverter --sfmt=\"TIFF (3D)\" \
            --dfmt=\"Vaa3D raw (tiled, 3D)\" \
            -s=\"{item_path}\" \
            -d=\"{results_path}\"")
    Popen(cmd, shell=True).wait()
    print(results_path)
    '''
    vaa3d_output = os.path.join(results_path, [x for x in os.listdir(results_path) if "RES" in x][0])  # RES (XxYxZ) folder
    vaa3d_output = os.path.join(vaa3d_output, os.listdir(vaa3d_output)[0])
    vaa3d_output = os.path.join(vaa3d_output, os.listdir(vaa3d_output)[0])
    vaa3d_output = os.path.join(vaa3d_output, os.listdir(vaa3d_output)[0])
    # vaa3d_output = os.path.join(vaa3d_output, os.listdir(os.listdir(os.listdir(vaa3d_output)[0])[0])[0])    #00000_0000 -> 000000 -> .raw
    '''
    #walk through Terastitcher output and find .raw file 
    for folder, subfolders, files in os.walk(results_path):
        for f in files:
            if f.endswith("raw"):
                 vaa3d_output = os.path.join(folder,f)
                 break

    old_path = results_path
    if results_path[-1] == "/":
        results_path = results_path[:-1]
    results_path += ".v3draw"
    print(f"Moving from {vaa3d_output}\n{results_path}")
    shutil.move(vaa3d_output, results_path)
    shutil.rmtree(old_path)

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
    res = Popen(cmd, shell=True,stdout=DEVNULL,stderr=DEVNULL).wait()
    
    #assemble the Ilastik output to a stack 
    ilastik_output_list = sorted(glob.glob(os.path.join(results_folder,'ventricles_zplanes','*.tif')))
    mask = io.concatenate_images(io.imread_collection(ilastik_output_list))
    
    #cast to 8-bit (should be superfluous)
    #mask = mask.astype('uint8')
    
    #Ilastik exports probabilities mapped between 0 and 255.
    #thresholding at 128: everything smaller = 0, everything larger = 1
    #mask[mask < 128] = 0
    #mask[mask >= 128] = 1
    
    #save in main folder 
    io.imsave(os.path.join(results_folder,downsampled_name + '_mask.tif'),mask,compression='lzw')
    print(f"{np.min(mask)} {np.mean(mask)} {np.max(mask)}")
    
    return mask 

def collect_measurements(original_um_x,original_um_y,original_um_z,downsampled_um_x,downsampled_um_y,downsampled_um_z,raw_location,raw_image_list,mask):
    yx = io.imread(os.path.join(raw_location,raw_image_list[0]))

    #create dictionary for original dimensions 
    original_dims = {'original_um_x' : original_um_x,
                    'original_um_y' : original_um_y,
                    'original_um_z' : original_um_z,
                    'original_px_y' : yx[1],
                    'original_px_x' : yx[0],
                    'original_px_z' : len(raw_image_list)
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

def histogram_equalization_8b(stack):
    #equalizes the histogram of a given image stack (as np array) to exclude top/bottom 1% 
    #and stretch the contrast to cover the remaining bits 
    
    #determine top/bottom 1% cutoffs 
    minval = round(np.percentile(stack.ravel(),1))
    maxval = round(np.percentile(stack.ravel(),99))
    
    #mask out everything above / below cutoffs
    stack[stack <= minval] = minval
    stack[stack >= maxval] = maxval
    
    #equalize the histogram for the remainder
    stack_equalized = (((stack - minval) / (maxval-minval))*65534).astype('uint16')
    
    #downsample to 8-bit
    stack_equalized_8bit = img_as_ubyte(stack_equalized)
    
    return stack_equalized_8bit

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
    results_folder = os.path.join(settings["mask_detection"]["output_location"], brain)
    if results_folder == "":
        parent_dir,raw_folder = os.path.split(os.path.abspath(raw_location))
        results_folder = os.path.join(parent_dir,raw_folder+'_results')

    #try to create
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    #create a temporary storage directory for downsampling 
    #temp_dir = tempfile.TemporaryDirectory() 
    temp_dir = os.path.join(results_folder,"temp_downsampling_cache")
    os.makedirs(temp_dir, exist_ok=True)
    #XXX XXX XXX XXX XXX XXX XXX XXX

    #open multiprocessing pool 
    pool = mp.Pool(processes = int(os.cpu_count()/2))

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
    downsampled_list = sorted([os.path.join(temp_dir, file) for file in os.listdir(temp_dir)])
    
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

    #skimage.io generates a 4-dimensional stack, but we need 3 dims 
    downsampled_stack = np.squeeze(downsampled_stack)

    io.imsave(os.path.join(results_folder,downsampled_name + '.tif'),downsampled_stack,compression='lzw',check_contrast=False)
    downsampled_stack_8bit = histogram_equalization_8b(downsampled_stack)
    io.imsave(os.path.join(results_folder,downsampled_name + '_8bit.tif'),downsampled_stack_8bit,compression='lzw',check_contrast=False)



    #save downsampled stack
    #Note that Teraconverter requires at least 250 px in all dimensions, so skip if too small 
    #TODO: Pad so that Teraconverter still works (required for atlas alignment)
    #if min(downsampled_stack.shape) > 250:
    downsampled_name = 'stack_resampled'
    downsampled_masked_name     = 'stack_downsampled'
    downsampled_masked_path     = os.path.join(results_folder, downsampled_masked_name)
    downsampled_masked_vaa3d    = os.path.join(results_folder, "stack_downsampled")
    if not os.path.exists(downsampled_masked_vaa3d):
        os.mkdir(downsampled_masked_vaa3d)
    teraconverter_path = settings["mask_detection"]["teraconverter_location"]
    save_vaa3d(teraconverter_path, os.path.join(results_folder,downsampled_name + '_8bit.tif'), downsampled_masked_vaa3d)
    
    #cleanup 
    #temp_dir.cleanup()
    shutil.rmtree(temp_dir)
    
    #run Ilastik 
    print(results_folder)
    print(downsampled_name)
    print(ilastik_path)
    print(ventricle_masking_ilastik_project)
    start = datetime.datetime.now()
    downsampled_mask = ilastik_ventricles(results_folder,downsampled_name,ilastik_path,ventricle_masking_ilastik_project)
    print(f"Downsampled mask: {np.min(downsampled_mask)} {np.max(downsampled_mask)} {downsampled_mask.dtype}")
    # Important: Saved masks have propabilities 0 - 255 instead of 0 - 1
    downsampled_mask[downsampled_mask < 125] = 0
    downsampled_mask[downsampled_mask >= 125] = 1
    delta = datetime.datetime.now() -start
    print(f"Ilastik: {delta}")
    


    #make downsampled masked stack
    print(f"Downsampled stack {np.min(downsampled_stack)} {np.max(downsampled_stack)} {downsampled_stack.dtype}")
    downsampled_masked_stack    = downsampled_mask * downsampled_stack
    print(f"Masked downsampled stack {np.min(downsampled_masked_stack)} {np.max(downsampled_masked_stack)} {downsampled_masked_stack.dtype}")
    downsampled_masked_name     = 'stack_masked_downsampled'
    downsampled_masked_path     = os.path.join(results_folder, downsampled_masked_name)
    downsampled_masked_vaa3d    = os.path.join(results_folder, "stack_masked_downsampled")
    print(f"downsampled stack {downsampled_stack.shape}")
    print(f"downsampled mask {downsampled_mask.shape}")
    print(f"downsampled masked stack {downsampled_masked_stack.shape}")
    start = datetime.datetime.now()
    if not os.path.exists(downsampled_masked_path):
        os.makedirs(downsampled_masked_path)
    if not os.path.exists(downsampled_masked_vaa3d):
        os.makedirs(downsampled_masked_vaa3d)
    print(downsampled_masked_path)
    # for downsampled_masked_slice in range(downsampled_masked_stack.shape[0]):
        # io.imsave(downsampled_masked_path + f"/{downsampled_masked_slice}.tif", downsampled_masked_stack[downsampled_masked_slice,:,:], compression='lzw')
    io.imsave(os.path.join(results_folder,"downsampled_masked_stack.tif"), downsampled_masked_stack, compression='lzw',check_contrast=False)
    downsampled_masked_stack_8bit = histogram_equalization_8b(downsampled_masked_stack)
    io.imsave(os.path.join(results_folder,"downsampled_masked_stack_8bit.tif"), downsampled_masked_stack_8bit, compression='lzw',check_contrast=False)

    io.imsave(downsampled_masked_path + "/downsampled_masked_stack_8bit.tif", downsampled_masked_stack_8bit, compression='lzw',check_contrast=False)

    #again, Teraconverter requires stacks larger than 250 px in all directions
    #if min(downsampled_stack.shape) > 250:
    save_vaa3d(teraconverter_path, downsampled_masked_path + "/downsampled_masked_stack_8bit.tif", downsampled_masked_vaa3d)
    delta = datetime.datetime.now() -start
    print(f"Saving downsampled mask as tiff and vaa3d: {delta}")


    #define dimension dictionaries 
    original_dims, downsampled_dims = collect_measurements(original_um_x,original_um_y,original_um_z,downsampled_um_x,downsampled_um_y,downsampled_um_z,raw_location,raw_image_list,downsampled_mask)
    
    #get the shape of the raw image stack
    raw_shape = get_real_size(raw_location)
    print(f"Before upsampling: {downsampled_mask.shape}")
    print(f"Raw shape {raw_shape}")
    
    #calculate ratios for upscaling (this replaces the previously used ratios for downsampling)
    z_ratio = raw_shape[0] / downsampled_mask.shape[0] 
    y_ratio = raw_shape[1] / downsampled_mask.shape[1] 
    x_ratio = raw_shape[2] / downsampled_mask.shape[2] 

    #upscale the mask (this may take quite a bit, order=2 should be a bit faster than default order=3)
    start = datetime.datetime.now()
    #np.save(os.path.join(results_folder, "mask_us.npy"), np.zeros(shape=raw_shape,dtype=np.uint8))
    #mask_us = np.memmap(os.path.join(results_folder, "mask_us.npy"), mode='r+', dtype=np.uint8, shape=raw_shape)
    
    #directly create a npy file on disk
    mask_us = np.lib.format.open_memmap(os.path.join(results_folder, "mask_us.npy"), mode='w+', dtype=np.uint8, shape=raw_shape)

    #upscale the mask. Scipy.ndimage.zoom is single-threaded and likely to take a while.  
    mask_us = zoom(downsampled_mask,(z_ratio, y_ratio, x_ratio),output=mask_us,order=2, prefilter=False)   
    
    delta = datetime.datetime.now() - start
    print(f"Zoom: {delta}")

    #mask_us = np.swapaxes(mask_us, 0, 2)
    
    print(f"Final shape {mask_us.shape}")

    #fallback for when dimensions don't add up 
    if mask_us.shape != raw_shape:
        print("Warning: Dimensions were wrong, swapped mask_us axes 0&1")
        mask_us = np.squeeze(mask_us)
        mask_us = np.swapaxes(mask_us, 0, 1)
    print(f"Saving final masked data {mask_us.shape}\n")
    start = datetime.datetime.now()

    # Save all tiffs neatly in a folder
    if not os.path.exists(os.path.join(results_folder, "masked_tiffs")):
        os.mkdir(os.path.join(results_folder, "masked_tiffs"))

    # Save all npys neatly in a folder
    if not os.path.exists(os.path.join(results_folder, "masked_niftis")):
        os.mkdir(os.path.join(results_folder, "masked_niftis"))
    
    '''
    masked_nii = deque()
    for i, item in enumerate(sorted([x for x in os.listdir(raw_location) if ".tif" in x])):
        img = cv2.imread(raw_location + "/" + item, -1)
        print(f"{item} {i} / {mask_us.shape[0]} MASK {mask_us.shape} RAW {img.shape}")
        # Mask raw data with upscaled mask
        img *= mask_us[i,:,:]
        # Add to final Nifti output
        masked_nii.append(img)
        # Save masked raw data
        io.imsave(os.path.join(results_folder, "masked_tiffs", item), img, compression='lzw', check_contrast=False)
    delta = datetime.datetime.now() - start
    print(f"Masking: {delta}")
    '''
    #reserve memmap on disk, the 2 extra dimensions and dtype=float16 are to prepare for the tensor conversion 
    #masked_nii = np.zeros(shape=(1,1,*raw_shape), dtype=np.uint16)
    #np.save(os.path.join(results_folder, "masked_niftis", "masked_nifti.npy"), masked_nii)
    #masked_nii = np.memmap(os.path.join(results_folder, "masked_niftis", "masked_nifti.npy"),mode='r+',dtype=np.uint16,shape=(1,1,*raw_shape))

    #load the crop_size from the settings (default (64,64,32))
    crop_size_0 = settings["blob_detection"]["window_dimensions"]["window_dim_0"]
    crop_size_1 = settings["blob_detection"]["window_dimensions"]["window_dim_1"]
    crop_size_2 = settings["blob_detection"]["window_dimensions"]["window_dim_2"]
    #assemble crop_size
    crop_size = (crop_size_0,crop_size_1,crop_size_2)

    #pre-compute the size of the array so that it doesn't need to be padded in RAM during inference 
    raw_shape_pad = list(raw_shape)
    for idx, dim in enumerate(raw_shape_pad):
        raw_shape_pad[idx] = int(np.ceil(dim/crop_size[idx])*crop_size[idx])

    #create a memmapped npy array on disk. This method should be RAM-friendlier than having to create the array in RAM, then saving to disk
    masked_nii = np.lib.format.open_memmap(os.path.join(results_folder, "masked_niftis", "masked_nifti.npy"), mode='w+', dtype=np.uint16,shape=(1,1,*raw_shape_pad))

    for i, item in enumerate(sorted([x for x in os.listdir(raw_location) if ".tif" in x])):
        img = cv2.imread(raw_location + "/" + item, -1)
        print(f"{item} {i} / {mask_us.shape[0]} MASK {mask_us.shape} RAW {img.shape}")
        # Mask raw data with upscaled mask
        img *= mask_us[i,:,:]
        # Add to final Npy output
        masked_nii[0,0,i,0:raw_shape[1],0:raw_shape[2]] = np.expand_dims(np.expand_dims(img, axis=0),axis=0).astype(np.uint16)
        # Save masked raw data
        io.imsave(os.path.join(results_folder, "masked_tiffs", item), img, compression='lzw', check_contrast=False)
    delta = datetime.datetime.now() - start
    print(f"Masking: {delta}")
    
    '''
    #optional: also save upscaled mask 
    mask_output_folder = os.path.join(results_folder,'upscaled_mask')
    #try to create
    if not os.path.exists(mask_output_folder):
        os.mkdir(mask_output_folder)
    #save every z-plane of the upscaled mask as tif
    for z in range(mask_us.shape[0]):
        z_plane = mask_us[z,:,:]    
        io.imsave(os.path.join(mask_output_folder,'mask_upscaled_z_'+str(z).zfill(4)+'.tif'),z_plane,compression='lzw',check_contrast=False)
    '''
    '''
    #delete mask_us from memory to free up RAM
    del mask_us

    #save the nifti in a folder
    
    print("Saving final data as Nifti")
    start = datetime.datetime.now()
    if not os.path.exists(os.path.join(results_folder, "masked_niftis")):
        os.mkdir(os.path.join(results_folder, "masked_niftis"))

    masked_nii = np.array(masked_nii)

    masked_nii = np.swapaxes(masked_nii, 0, -1)
    masked_nii = np.swapaxes(masked_nii, 0, 1)
    write_nifti(os.path.join(results_folder, "masked_niftis", "masked_nifti.nii.gz"), masked_nii)
    delta = datetime.datetime.now() - start
    print(f"Nifti: {delta}")
    '''

    #remove mask_us from the disk to save space (the rest will be kept if the setting "SAVE_MASK_OUTPUT":true is set in config.json)
    os.remove(os.path.join(results_folder, "mask_us.npy"))

    #return name of the mouse brain and its original shape 
    return brain,masked_nii.shape
