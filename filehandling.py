import os
import cv2 
import numpy as np
import nibabel as nib

def write_nifti(path,volume):
    '''
    writeNifti(path,volume)
    
    Takes a Numpy volume, converts it to the Nifti1 file format, and saves it to file under
    the specified path. 
    '''
    #path = path.replace("/", "\\")
    if(path.find('.nii')==-1):
        path = path + '.nii.gz'
    # Save volume with adjusted orientation
    # --> Swap X and Y axis to go from (y,x,z) to (x,y,z)
    # --> Show in RAI orientation (x: right-to-left, y: anterior-to-posterior, z: inferior-to-superior)
    affmat = np.eye(4)
    affmat[0,0] = affmat[1,1] = -1
    NiftiObject = nib.Nifti1Image(np.swapaxes(volume,0,1), affine=affmat)
    nib.save(NiftiObject,os.path.normpath(path))

def read_nifti(path):
    '''
    volume = readNifti(path)
    
    Reads in the NiftiObject saved under path and returns a Numpy volume.
    '''
    if(path.find('.nii')==-1):
        path = path + '.nii'
    NiftiObject = nib.load(path)
    # Load volume and adjust orientation from (x,y,z) to (y,x,z)
    volume = np.swapaxes(NiftiObject.dataobj,0,1)
    return volume
