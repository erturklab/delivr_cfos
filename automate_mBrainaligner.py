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

import pandas as pd
import os
import glob
import re
import tifffile 
import shutil
import multiprocessing as mp
import numpy as np

def atlas_align(mBrainAligner_location, source_file, output_dir,mouse_name): 
    #run mBrainAligner global + local registration. Lightly adapted from the fLSM example windows batch file. 
    
    print(f"Source File:\n{source_file}")
    print(f"Output Dir:\n{output_dir}")
    #first (global) alignment 
    cmd_global = str (f" {mBrainAligner_location}binary/linux_bin/global_registration " + 
    f" -f {mBrainAligner_location}examples/target/50um/ " +
    f" -m \"{source_file}\"" +
    " -p r+f+n " +
    f" -o \"{output_dir}\"/"+ 
    " -d 70 " +
    " -l 30+30+30 " +
    " -u 0" )
    
    
    #print ('running global alignment for mouse ' + str(mouse_name))
    print("global alignment command: ", cmd_global)
    #run the command via command line 
    res_global = os.system(cmd_global)
    
    #TODO Include config
    #second (local) alignment 
    cmd_local = str(f" {mBrainAligner_location}binary/linux_bin/local_registration " + 
    f" -p {mBrainAligner_location}examples/config/LSFM_config_DeliVR.txt " + 
    f" -s {output_dir}/global.v3draw " +
    f" -l {mBrainAligner_location}examples/target/50um/target_landmarks/low_landmarks.marker " + 
    f" -g {mBrainAligner_location}examples/target/50um/ " + 
    f" -o {output_dir}/"+
    " -u 0")
    
    #print ('running local alignment for mouse ' + str(mouse_name))
    print("local alignment command: ", cmd_local)
    #run the command via command line 
    res_local = os.system(cmd_local)


def rewrite_swc(csv_path, output_dir,XYZ=False):
    #define source file 
    #Note this needs to be the ONLY file ending in binary.scv! 
    #TODO remove?
    # csv_path = glob.glob(entry+"/*binary.csv")[0]

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

    file["z"] = file["z"].str.replace(re.escape(","), "")
    file["x"] = file["x"].str.replace(re.escape(","), "")
    file["y"] = file["y"].str.replace(re.escape(","), "")
    # print(split)
    # exit()
        
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

def reattach_size_and_copy(csv_path,swc_local,mouse_name,output_dir,aligned_results_folder):
    #### re-attach size column from the original csv 
    #read tranformed swc
    registered_cells = pd.read_csv(swc_local, sep = ' ',skiprows=3,names=['n','type', 'x','y','z','radius','parent'])
    #read the original csv 
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


def register_swc_to_atlas (csv_path,mBrainAligner_location, swc_file, source_file, tiff_path, mouse_name,output_dir,aligned_results_folder,XYZ=False): 
    #run mBrainaligner s swc registration 
    #create a command that looks like this: 
    '''
    ../binary/linux_bin/local_registration  
    -p config/LSFM_config_original.txt 
    -s result/LSFM/global.v3draw 
    -l target/50um/target_landmarks/low_landmarks.marker  
    -g target/50um/ 
    -o result/LSFM/ 
    -u 0
    '''
    #determine the shape of the downsampled image 
    
    #laod tiff stack, then determine size 
    resampled_tif_stack = tifffile.imread(tiff_path)
    downsampled_z,downsampled_y,downsampled_x = resampled_tif_stack.shape

    #determine non-downsampled (original) stack dimensions from swc file name 
    #nicer parsing function 
    if not XYZ:
        #usually the parameters are noted in (Z, X, Y)
        original_z, original_x, original_y = split_parameters(swc_file)
    elif XYZ:
        #however in some cases the parameters are (X, Y, Z)
        original_x, original_y, original_z = split_parameters(swc_file)
    
    #calculate the downsampling factor for each dimension
    ds_factor_x = original_x/downsampled_x
    ds_factor_y = original_y/downsampled_y
    ds_factor_z = original_z/downsampled_z
    
    #define path names for the resulting swc files 
    swc_resampled = os.path.join(output_dir,str(mouse_name)+"_resampled.swc")
    swc_global = os.path.join(output_dir,str(mouse_name)+"_global_data.swc")
    swc_ffd = os.path.join(output_dir,str(mouse_name)+"_FFD_data.swc")
    swc_local = os.path.join(output_dir,str(mouse_name)+"_local_registered_data.swc")
    
    
    cmd_swc =  str (f" {mBrainAligner_location}examples/swc_registration/binary/linux_bin/swc_registration " + 
        " -C " + glob.glob(output_dir+"/*RPM_tar.marker")[0] +
        " -M " + glob.glob(output_dir+"/*RPM_sub.marker")[0] +
        " -o " + "\"" + swc_file + "\"" +
        " -T " + glob.glob(output_dir+"/local_registered_tar.marker")[0] +
        " -S " + glob.glob(output_dir+"/local_registered_sub.marker")[0] +
        " -d " + glob.glob(output_dir+"/*FFD_grid.swc")[0] +
        " -x " + str(ds_factor_x) +
        " -y " + str(ds_factor_y) +
        " -z " + str(ds_factor_z) +
        " -a 228 -b 264 -c 160 " +
        " -r " + swc_resampled +
        " -g " + swc_global +
        " -f " + swc_ffd + 
        " -s " + swc_local
        )
    
    #run the registration 
    print("registering swc for " + mouse_name)
    print("running swc registration: " + cmd_swc)
    res = os.system(cmd_swc)
    
    #debug
    #print('swc_local: ', swc_local)
    #print('swc_file: ',swc_file)
    #print('mouse_name: ',mouse_name)
    #print('output_dir: ',output_dir)
    #print('aligned_results_folder: ',aligned_results_folder)
    
    #re-attach original size column and copy to new 
    reattach_size_and_copy(csv_path,swc_local,mouse_name, output_dir, aligned_results_folder)
    
def run_mbrainaligner_and_swc_reg(entry, settings, xyz=False, latest_output=None,aligned_results_folder=-1,mBrainAligner_location=-1):  
    ''' 
    # TODO: automate v3d generation, under linux use the following commandline code https://www.nitrc.org/forum/message.php?msg_id=18446 
    # (though the dll now is in plugins/data_IO/convert_file_format/convert_file_format.dll)
    
    Inputs are organized as follows:
    entry = path to the v3d file 
    settings = configuration file
    xyz = whether the connected_component analysis data is XYZ (True) or ZXY (False). Default False. 
    latest_output = if there are previously computed alignment files, you can put the path here. Default None. 
    aligned_results_folder = a collection folder where all results are collected (they will be locally saved next to the original data as well)
    mBrainAligner_location = the location of the mBrainAligner files, i.e. D:/MoritzNegwer/Documents/mBrainAligner
     
    '''
    entry_folder = entry.split("/")[-1].replace(".csv","")
    brain = ("_").join(entry_folder.split("_")[1:])
    orientation = entry_folder.split("_")[0]
    #TODO Find the v3draw
    v3draw_path = os.path.join(settings["mask_detection"]["output_location"], brain,"stack_downsampled.v3draw")
    tiff_path   = os.path.join(settings["mask_detection"]["output_location"], brain,"stack_resampled.tif")
    csv_path    = entry#os.path.join(settings["postprocessing"]["output_location"], brain) 

    #if latest is not given, fill in something 
    if latest_output is None:
        latest_output = '2022-04-19_mBrainAligner_'
    
    #define source file. 
    source_file = v3draw_path 
    print(f"Brain {brain}")
    print(f"Source file {source_file}")
    print(f"CSV path {csv_path}")
    
    #extract mouse name from entry directory 
    #TODO Define mouse name directly in input
    mouse_name = brain#os.path.split(entry)[1][:9]
    
    #define output dir and try to ceeate it. If it already exists, skip creation (and subsequently overwrite contents)
    #TODO Define output dir directly in input
    output_dir = settings["atlas_alignment"]["output_location"]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, mouse_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # output_dir = os.path.join(aligned_results_folder,mouse_name)
    # try: 
    #     os.mkdir(output_dir)
    # except:
    #     pass    
    print(f"latest_output {latest_output}")
    print(f"Output dir {output_dir}")

    mBrainAligner_location = settings["atlas_alignment"]["mBrainAligner_location"]
    #run mBrainAligner-to-atlas registration
    atlas_align(mBrainAligner_location, source_file,output_dir,mouse_name)
    
    #rewrite swc from Rami's blob swc
    swc_file = rewrite_swc(csv_path, output_dir,XYZ=xyz)
    
    #align swc to atlas
    register_swc_to_atlas(mBrainAligner_location, csv_path, swc_file, source_file, tiff_path, mouse_name, output_dir, aligned_results_folder,XYZ=xyz)
    

