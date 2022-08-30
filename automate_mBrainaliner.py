#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:28:02 2022

@author: wirrbel
"""

import pandas as pd
import os
import glob
import re
import tifffile 
import shutil
import multiprocessing as mp
import numpy as np

def atlas_align(source_file, output_dir,mouse_name,mBrainAligner_location): 
    #run mBrainAligner global + local registration. Lightly adapted from the fLSM example windows batch file. 
    
    #first (global) alignment 
    cmd_global = str (mBrainAligner_location + "/binary/win64_bin/global_registration.exe " + 
    " -f " + mBrainAligner_location + "/examples/target/CCF_u8_xpad.v3draw" +
    " -c " + mBrainAligner_location + "/examples/target/CCF_mask.v3draw "+
    " -m " + str(source_file) +
    " -p r+f+n " +
    " -o " + str(output_dir)+ 
    " -d 70")
    
    print ('running global alignment for mouse ' + str(mouse_name))
    #run the command via command line 
    res_global = os.system(cmd_global)
    
    #second (local) alignment 
    cmd_local = str (mBrainAligner_location + "/binary/win64_bin/local_registration.exe " + 
    " -p " + mBrainAligner_location + "/examples/config/LSFM_config.txt " + 
    " -s " + str(output_dir) + "/global.v3draw " +
    " -l " + mBrainAligner_location + "/examples/target/target_landmarks/low_landmarks.marker " + 
    " -g " + mBrainAligner_location + "/examples/target " + 
    " -o " + str(output_dir + "/"))
    
    print ('running local alignment for mouse ' + str(mouse_name))
    #run the command via command line 
    res_local = os.system(cmd_local)

def rewrite_swc(entry, output_dir,XYZ=False):
    #define source file 
    #Note this needs to be the ONLY file ending in binary.scv! 
    csv_path = glob.glob(entry+"/*binary.csv")[0]

    #read file 
    file = pd.read_csv(csv_path)

    #strip duplicate "Blob" column 
    file = file.drop(["Blob"],axis=1)
    
    #replace double spaces 
    file["Coords"] =  file["Coords"].str.replace("\s{2,}"," ")

    #strip first "[ ", then redo for only "["
    file["Coords"] =  file["Coords"].str.replace(re.escape("[ "),"")
    file["Coords"] =  file["Coords"].str.replace(re.escape("["),"")

    #strip trailing ' ]' and  ']'
    file["Coords"] =  file["Coords"].str.replace(re.escape(" ]"),"")
    file["Coords"] =  file["Coords"].str.replace(re.escape("]"),"")

    #split Coords into three columns 
    split = file["Coords"].str.split(" ",n=2,expand=True)

    #reassign to file 
    if not XYZ:
        file["z"] = split[0]
        file["x"] = split[1]
        file["y"] = split[2]
    elif XYZ: #specifically for 3411 + 3407
        file["x"] = split[0]
        file["y"] = split[1]
        file["z"] = split[2]
        
    file.drop("Coords",axis=1,inplace=True)
    file.drop("Unnamed: 0",axis=1,inplace=True)

    #round to 3 post-comma digits
    file["x"] = file["x"].astype('float').round(3)
    file["y"] = file["y"].astype('float').round(3)
    file["z"] = file["z"].astype('float').round(3)

    #debug: plot histogram of size values 
    #ax = file["Size"].plot.hist(bins=1000,log=True) #with log scale
    #ax = file["Size"].plot.hist(bins=1000,log=False) #on linear scale 

    #Other option (default as of v06): do not filter here, and filter after the atlas transform instead 
    file_filtered = file.copy()
    
    ###assemble df with SWC-conform files

    #insert structure columns with value =1 (should be soma according to this definition)
    file_filtered.insert(0,"Structure", 1)

    #reorder columns so that size is at end again 
    file_filtered = file_filtered[["Structure","x","y","z","Size"]]

    #add "Parent" column pointing to -1 (=no parent in swc notation)
    file_filtered.insert(5,"Parent", -1)

    #save actual file as swc
    target_swc_file_name = os.path.join(output_dir,os.path.split(csv_path)[1]+".swc")
    #remove whitespaces as those will trip up the program 
    target_swc_file_name = target_swc_file_name.replace(' ','')
    
    with open(target_swc_file_name, 'w') as swc_target:
        swc_target.write('##n type x y z radius parent\n')
        file_filtered.to_csv(swc_target, header=False, sep=' ')
        swc_target.close()
        
    print('successfully wrote ' + str(target_swc_file_name))
    
    return target_swc_file_name

def split_parameters (file_path): 
    #retrieve file name
    filename = os.path.split(file_path)[1]
    
    #extract everything between brackets as single string 
    #note that this is optimized to work with Rami's naming scheme for the csv / swc files 
    parameters = re.findall('\(([^)]+)',filename)
    
    #split the single string into a list of three parameters, then return 
    parameters_list = str(parameters[0]).replace(" ","").split(sep=',')

    #cast as int
    parameters_list = list(map(int, parameters_list))
    
    return parameters_list

def reattach_size_and_copy(entry,swc_local,mouse_name,output_dir,aligned_results_folder):
    #### re-attach size column from the original csv 
    #read tranformed swc
    registered_cells = pd.read_csv(swc_local, sep = ' ',skiprows=3,names=['n','type', 'x','y','z','radius','parent'])
    #read the original csv 
    csv_path = glob.glob(entry+"/*binary.csv")[0]
    original_csv = pd.read_csv(csv_path)
    #merge to new df
    merged = registered_cells.copy()
    merged['Size'] = original_csv['Size']
    merged = merged.drop(['radius','parent'],axis=1)
    
    #define output file name 
    output_file_name = mouse_name+'local_registered_with_original_size.csv'
    #write to csv 
    merged.to_csv(os.path.join(output_dir,output_file_name),sep=' ',index=False)
    #then, write again to the coillections folder (instead of copying )
    merged.to_csv(os.path.join(aligned_results_folder,output_file_name),sep=' ',index=False)
    
    #copy the local_registered_data to the collections folder 
    #shutil.copyfile(swc_resampled, os.path.join(aligned_results_folder,str(mouse_name)+"_resampled.swc"))
    #shutil.copyfile(swc_global, os.path.join(aligned_results_folder,str(mouse_name)+"_global_data.swc"))
    #shutil.copyfile(swc_local, os.path.join(aligned_results_folder,str(mouse_name)+"_local_registered_data.swc"))


def register_swc_to_atlas (entry,swc_file,mouse_name,output_dir,aligned_results_folder,XYZ=False,mBrainAligner_location): 
    #run mBrainaligner s swc registration 
    #create a command that looks like this: 
    '''
    swc_registration.exe ^
    -C ./test\autofluo_resampled.tif_RPM_tar.marker ^
    -M ./test\autofluo_resampled.tif_RPM_sub.marker ^
    -o ./test\test_3407.swc ^
    -T ./test\ori_local_registered_tar.marker ^
    -S ./test\ori_local_registered_sub.marker ^
    -d ./test\autofluo_resampled.tif_FFD_grid.swc ^
    -X 5616 -Y 6656 -Z 737 ^
    -x 364 -y 432 -z 176 ^
    -a 324 -b 200 -c 268 -p 20 ^
    -r ./result\resampled.swc ^
    -f ./result\global_data.swc ^
    -s ./result\local_registered_data.swc
    '''
    #determine the shape of the downsampled image 
    #Note this ONLY works if the v3draw file does contain the complete name of the resampled tif stack (incl '.tif') and it is in the same directory! 
    
    #determine source v3draw file (again)
    source_file = glob.glob(entry+"/*.v3draw")[0]
    #then, split off .v3draw to get resampled tiff stack name 
    resampled_tif_stack_name = os.path.splitext(os.path.split(source_file)[1])[0]
    #remove a '_processed' from the name, if present 
    if resampled_tif_stack_name.endswith('_processed'):
        resampled_tif_stack_name = resampled_tif_stack_name[:-10]
    #laod tiff stack, then determine size 
    resampled_tif_stack = tifffile.imread(os.path.join(entry,resampled_tif_stack_name))
    z,y,x = resampled_tif_stack.shape

    #determine non-downsampled (original) stack dimensions from swc file name 
    #nicer parsing function 
    if not XYZ:
        #usually the parameters are noted in (Z, X, Y)
        Z, X, Y = split_parameters(swc_file)
    elif XYZ:
        #however in some cases the parameters are (X, Y, Z)
        X, Y, Z = split_parameters(swc_file)
    
    #define path names for the resulting swc files 
    swc_resampled = os.path.join(output_dir,str(mouse_name)+"_resampled.swc")
    swc_global = os.path.join(output_dir,str(mouse_name)+"_global_data.swc")
    swc_local = os.path.join(output_dir,str(mouse_name)+"_local_registered_data.swc")
    
    
    cmd_swc =  str (mBrainAligner_location + "/examples/swc_registration/swc_registration.exe " + 
        " -C " + glob.glob(output_dir+"/*RPM_tar.marker")[0] +
        " -M " + glob.glob(output_dir+"/*RPM_sub.marker")[0] +
        " -o " + swc_file + 
        " -T " + glob.glob(output_dir+"/ori_local_registered_tar.marker")[0] +
        " -S " + glob.glob(output_dir+"/ori_local_registered_sub.marker")[0] +
        " -d " + glob.glob(output_dir+"/*FFD_grid.swc")[0] +
        " -X " + str(X) +
        " -Y " + str(Y) + 
        " -Z " + str(Z) +
        " -x " + str(x) +
        " -y " + str(y) +
        " -z " + str(z) +
        " -a 324 -b 200 -c 268 -p 20" +
        " -r " + swc_resampled +
        " -f " + swc_global +
        " -s " + swc_local
        )
    
    #run the registration 
    print("registering swc for " + mouse_name)
    res = os.system(cmd_swc)
    
    #debug
    print('swc_local: ', swc_local)
    print('swc_file: ',swc_file)
    print('mouse_name: ',mouse_name)
    print('output_dir: ',output_dir)
    print('aligned_results_folder: ',aligned_results_folder)
    
    #re-attach original size column and copy to new 
    reattach_size_and_copy(entry,swc_local,mouse_name, output_dir, aligned_results_folder)
    
    
 

def run_mbrainaligner_and_swc_reg(entry, xyz=False, latest_output=None,aligned_results_folder,mBrainAligner_location):  
    ''' 
    # TODO: automate v3d generation, under linux use the following commandline code https://www.nitrc.org/forum/message.php?msg_id=18446 
    # (though the dll now is in plugins/data_IO/convert_file_format/convert_file_format.dll)
    
    Inputs are organized as follows:
    entry = path to the v3d file 
    xyz = whether the connected_component analysis data is XYZ (True) or ZXY (False). Default False. 
    latest_output = if there are previously computed alignment files, you can put the path here. Default None. 
    aligned_results_folder = a collection folder where all results are collected (they will be locally saved next to the original data as well)
    mBrainAligner_location = the location of the mBrainAligner files, i.e. D:/MoritzNegwer/Documents/mBrainAligner
     
    '''
    #if latest is not given, fill in something 
    if latest_output is None:
        latest_output = '2022-04-19_mBrainAligner_'
    
    #define source file. This WILL fail if there is more than one v3draw file in the folder! 
    source_file = glob.glob(entry+"/*.v3draw")[0]
    print (source_file)
    
    #extract mouse name from entry directory 
    mouse_name = os.path.split(entry)[1][:9]
    
    #define output dir and try to ceeate it. If it already exists, skip creation (and subsequently overwrite contents)
    output_dir = os.path.join(entry,latest_output + mouse_name)
    try: 
        os.mkdir(output_dir)
    except:
        pass    

    #run mBrainAligner-to-atlas registration
    atlas_align(source_file,output_dir,mouse_name)
    
    #rewrite swc from Rami's blob swc
    swc_file = rewrite_swc(entry, output_dir,XYZ=xyz)
    
    #align swc to atlas
    register_swc_to_atlas(entry, swc_file, mouse_name, output_dir, aligned_results_folder,XYZ=xyz)
    

