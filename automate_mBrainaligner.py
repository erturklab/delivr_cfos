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
import more_itertools
import subprocess
import datetime

def atlas_align(mBrainAligner_location, source_file, output_dir,mouse_name, settings): 
    #run mBrainAligner global + local registration. Lightly adapted from the fLSM example windows batch file. 
    
    print(f"Source File:\n{source_file}")
    print(f"Output Dir:\n{output_dir}")
    source_folder = os.path.split(source_file)[0]
    #first (global) alignment 
    
    if(settings["atlas_alignment"]["landmarks_hemisphere"] == True):
        #use the landmarks to do the initial affine transformation. Recommended for hemispheres but requires manual landmark generation
        cmd_global = str (f" {mBrainAligner_location}binary/linux_bin/global_registration " + 
        f" -f {mBrainAligner_location}examples/target/50um/ " +
        f" -m \"{source_file}\"" +
        " -p a " +
        f" -o \"{output_dir}/\""+ 
        " -d 20 " +
        " -l 0+0+0 " +
        " -u 0" +
        f" -t {source_folder}/atlas_landmarks.marker" +
        f" -s {source_folder}/brain_landmarks.marker")
    else
        #use mBrainaligner's built-in registration tools. Recommended for whole-brain image stacks 
        cmd_global = str (f" {mBrainAligner_location}binary/linux_bin/global_registration " + 
        f" -f {mBrainAligner_location}examples/target/50um/ " +
        f" -m \"{source_file}\"" +
        " -p r+f+n " +
        f" -o \"{output_dir}/\""+ 
        " -d 20 " +
        " -l 0+0+0 " +
        " -u 0" )    
    
    #print ('running global alignment for mouse ' + str(mouse_name))
    print("global alignment command: ", cmd_global)
    #run the command via command line 
    #res_global = os.system(cmd_global)
    subprocess.run(cmd_global,shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
    #TODO Include config
    #second (local) alignment 
    cmd_local = str(f" {mBrainAligner_location}binary/linux_bin/local_registration " + 
    f" -p {mBrainAligner_location}examples/config/LSFM_half_config.txt " + 
    f" -s {output_dir}/global.v3draw " +
    f" -l {mBrainAligner_location}examples/target/50um/target_landmarks/low_landmarks.marker" + 
    f" -g {mBrainAligner_location}examples/target/50um/ " + 
    f" -o \"{output_dir}/\""+
    " -u 0")
    
    #print ('running local alignment for mouse ' + str(mouse_name))
    print("local alignment command: ", cmd_local)
    #run the command via command line 
    #res_local = os.system(cmd_local)
    subprocess.run(cmd_local,shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def rewrite_swc(csv_path, output_dir,XYZ=False,parallel_processing=False):
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
        file["y"] = split[1]
        file["x"] = split[2]
    elif XYZ: #specifically for 3411 + 3407
        file["x"] = split[0]
        file["y"] = split[1]
        file["z"] = split[2]

    file["z"] = file["z"].str.replace(re.escape(","), "")
    file["y"] = file["y"].str.replace(re.escape(","), "")
    file["x"] = file["x"].str.replace(re.escape(","), "")
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
    
    
    #populate list of output files
    target_swc_file_name_list = []
    
    #If creating several swc sub-files for parallel processing
    if parallel_processing:
        #Step 1: Determine number of CPUs (-1 to be safe) and split the cell list into approx. even chunks
        n_chunks = os.cpu_count()-1
        chunk_length = round(np.ceil(len(file_filtered)/n_chunks))
        
        #step 2: Split list into chunks and save as separate swc files 
        for chunk in more_itertools.sliced(file_filtered,chunk_length):
            #determine the row number of the first to help with reassembly)
            first_cell_number = chunk.iloc[0].name
            #make it a zero-padded string 
            first_cell_number = str(first_cell_number).zfill(7)
            #save chunk as file 
            target_swc_file_name = os.path.join(output_dir,os.path.split(csv_path)[1]+"chunk_"+first_cell_number+".swc")
            #remove whitespaces as those will trip up the program 
            target_swc_file_name = target_swc_file_name.replace(' ','')
            #remove brackets as those will trip up the program 
            target_swc_file_name = target_swc_file_name.replace('(','').replace(')','')
            
            #DEBUG: only take first 10 cells in each chunk
            #chunk = chunk[0:10]
            
            with open(target_swc_file_name, 'w') as swc_target:
                swc_target.write('##n type x y z radius parent\n')
                chunk.to_csv(swc_target, header=False, sep=' ')
                swc_target.close()
                
            print('successfully wrote ' + str(target_swc_file_name))
            #add to file list 
            target_swc_file_name_list += [target_swc_file_name]
            
        return target_swc_file_name_list
    
    else:
        #save actual file as swc
        target_swc_file_name = os.path.join(output_dir,os.path.split(csv_path)[1]+".swc")
        #remove whitespaces as those will trip up the program 
        target_swc_file_name = target_swc_file_name.replace(' ','')
        #remove brackets as those will trip up the program 
        target_swc_file_name = target_swc_file_name.replace('(','').replace(')','')
        
        
        with open(target_swc_file_name, 'w') as swc_target:
            swc_target.write('##n type x y z radius parent\n')
            file_filtered.to_csv(swc_target, header=False, sep=' ')
            swc_target.close()
            
        print('successfully wrote ' + str(target_swc_file_name))
        
        return [target_swc_file_name]

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

def reassemble_swcs (output_dir):
    #This function reassembles a single csv from the computed chunks. 
    aligned_chunks = sorted(glob.glob(output_dir+"/chunk*_local_registered_data.swc"))
    
    merged_swc = pd.DataFrame()
    
    #read all and reattach 
    for chunk in aligned_chunks:
        chunk_pd = pd.read_csv(chunk, sep = ' ',skiprows=3,names=['n','type', 'x','y','z','radius','parent'],index_col='n')
        
        merged_swc = pd.concat([merged_swc,chunk_pd],axis=0)
        
    #write to collected swc file: 
    target_swc_file_name = os.path.join(output_dir,'complete_local_registered_data.swc')
    
    with open(target_swc_file_name, 'w') as swc_target:
        swc_target.write('##n type x y z radius parent\n')
        merged_swc.to_csv(swc_target, header=False, sep=' ')
        swc_target.close()
        
    return target_swc_file_name

def reattach_size_and_copy(csv_path,swc_local,mouse_name,output_dir,aligned_results_folder):
    #### re-attach size column from the original csv 
    #read tranformed swc
    registered_cells = pd.read_csv(swc_local, sep = ' ',skiprows=1,names=['n','type', 'x','y','z','radius','parent'])
    #read the original csv 
    original_csv = pd.read_csv(csv_path)
    #merge to new df
    merged = registered_cells.copy()
    merged['Size'] = original_csv['Size']
    merged = merged.drop(['radius','parent'],axis=1)

    #define output file name 
    output_file_name = mouse_name+'_local_registered_with_original_size.csv'
    #write to csv 
    merged.to_csv(os.path.join(output_dir,output_file_name),sep=' ',index=False)
    #then, write again to the coillections folder (instead of copying )
    merged.to_csv(os.path.join(aligned_results_folder,output_file_name),sep=' ',index=False)
    
    #copy the local_registered_data to the collections folder 
    #shutil.copyfile(swc_resampled, os.path.join(aligned_results_folder,str(mouse_name)+"_resampled.swc"))
    #shutil.copyfile(swc_global, os.path.join(aligned_results_folder,str(mouse_name)+"_global_data.swc"))
    #shutil.copyfile(swc_local, os.path.join(aligned_results_folder,str(mouse_name)+"_local_registered_data.swc"))


def compute_sampling_factors(mBrainAligner_location, swc_file , tiff_path, XYZ=False,parallel_processing=False):
    #determine the shape of the downsampled image 
    
    #load tiff stack, then determine size 
    #tifffile _multifile=False means the metadata is ignored, and tifffile does not load the entire stack when accessing the first image of an ome.tif (that has infos about the stack in its metadata).
    #specifically added to deal with odd ome.tifs produced by some versions of Miltenyi/Lavision Ultramicroscope's Imspector software. 
    resampled_tif_stack = tifffile.imread(tiff_path,_multifile=False) 
    downsampled_z,downsampled_y,downsampled_x = resampled_tif_stack.shape

    #determine non-downsampled (original) stack dimensions from swc file name 
    #nicer parsing function 
    if not XYZ:
        #usually the parameters are noted in (Z, Y, X)
        original_z, original_y, original_x = split_parameters(swc_file)
    elif XYZ:
        #however in some cases the parameters are (X, Y, Z)
        original_x, original_y, original_z = split_parameters(swc_file)
    
    #calculate the downsampling factor for each dimension
    ds_factor_x = original_x/downsampled_x
    ds_factor_y = original_y/downsampled_y
    ds_factor_z = original_z/downsampled_z
    
    return ds_factor_x,ds_factor_y,ds_factor_z

def execute_swc_commandline(command):
    #wrapper to multithread the swc execution without displaying shell output in spyder 
    #debug:
    print("running command: ",command)
    subprocess.run(command,shell=True,stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    
def register_swc_to_atlas (mBrainAligner_location, target_swc_file_name_list, original_swc_file, tiff_path, mouse_name,output_dir,aligned_results_folder,XYZ=False,parallel_processing=False): 
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
    #compute downsampling factors 
    ds_factor_x,ds_factor_y,ds_factor_z = compute_sampling_factors(mBrainAligner_location, original_swc_file, tiff_path,XYZ=XYZ)


    if parallel_processing:
        #Determine number of CPUs (-1 to be safe)
        n_chunks = os.cpu_count()-1
        
        #start multiprocessing pool
        pool = mp.Pool(processes=n_chunks)
        
        #assemble command list
        cmd_list = []
        
        for target_swc in target_swc_file_name_list:
            source_folder = os.path.join("/data/output/01_mask_detection/output/",mouse_name)

            #remove .swc file ending from target_swc 
            target_swc_name = target_swc[-17:-4]
            
            #define path names for the resulting swc files 
            swc_resampled = os.path.join(output_dir,target_swc_name+"_resampled.swc")
            swc_global = os.path.join(output_dir,target_swc_name+"_global_data.swc")
            swc_ffd = os.path.join(output_dir,target_swc_name+"_FFD_data.swc")
            swc_local = os.path.join(output_dir,target_swc_name+"_local_registered_data.swc")
                
            #define path names for the resulting swc files 
            #swc_resampled = os.path.join(output_dir,str(mouse_name)+"_resampled.swc")
            #swc_global = os.path.join(output_dir,str(mouse_name)+"_global_data.swc")
            #swc_ffd = os.path.join(output_dir,str(mouse_name)+"_FFD_data.swc")
            #swc_local = os.path.join(output_dir,str(mouse_name)+"_local_registered_data.swc")
            
            cmd_swc =  str (f" {mBrainAligner_location}examples/swc_registration/binary/linux_bin/swc_registration " + 
                f" -C {source_folder}/atlas_landmarks.marker" +
                f" -M {source_folder}/brain_landmarks.marker"+
                " -o " + "\"" + target_swc + "\"" +
                " -T " + glob.glob(output_dir+"/local_registered_tar.marker")[0] +
                " -S " + glob.glob(output_dir+"/local_registered_sub.marker")[0] +
                " -x " + str(ds_factor_x) +
                " -y " + str(ds_factor_y) +
                " -z " + str(ds_factor_z) +
                " -a 264 -b 160 -c 228 " +
                " -r " + swc_resampled +
                " -g " + swc_global +
                " -f " + swc_ffd + 
                " -s " + swc_local  
                )
            
            #run the registration 
            #print("registering swc for " + mouse_name)
            print("running swc registration: " + cmd_swc)
            #res = os.system(cmd_swc)
                
            #add to list 
            cmd_list += [cmd_swc]
            
            #debug
            #print('swc_local: ', swc_local)
            #print('swc_file: ',swc_file)
            #print('mouse_name: ',mouse_name)
            #print('output_dir: ',output_dir)
            #print('aligned_results_folder: ',aligned_results_folder)
        
        #run all swc_alingment commands simultaneously
        pool.map(execute_swc_commandline,cmd_list)
        
        
        #return empty, the separate swc_local files will be reassembled at the end 
        return None
        
 
    else:   
            #if single-threaded, run everything only once. 
            #automatically the target_swc is the first (and hopefully only) file in the list
            target_swc = target_swc_file_name_list[0]
            #remove .swc file ending from target_swc 
            target_swc = target_swc[-17:-4]
            
            #define path names for the resulting swc files 
            swc_resampled = str(target_swc+"_resampled.swc")
            swc_global = str(target_swc+"_global_data.swc")
            swc_ffd = str(target_swc+"_FFD_data.swc")
            swc_local = str(target_swc+"_local_registered_data.swc")
                
            #define path names for the resulting swc files 
            #swc_resampled = os.path.join(output_dir,str(mouse_name)+"_resampled.swc")
            #swc_global = os.path.join(output_dir,str(mouse_name)+"_global_data.swc")
            #swc_ffd = os.path.join(output_dir,str(mouse_name)+"_FFD_data.swc")
            #swc_local = os.path.join(output_dir,str(mouse_name)+"_local_registered_data.swc")
            
            cmd_swc =  str (f" {mBrainAligner_location}examples/swc_registration/binary/linux_bin/swc_registration " + 
                " -C " + glob.glob(output_dir+"/*RPM_tar.marker")[0] +
                " -M " + glob.glob(output_dir+"/*RPM_sub.marker")[0] +
                " -o " + "\"" + target_swc + "\"" +
                " -T " + glob.glob(output_dir+"/local_registered_tar.marker")[0] +
                " -S " + glob.glob(output_dir+"/local_registered_sub.marker")[0] +
                " -d " + glob.glob(output_dir+"/*FFD_grid.swc")[0] +
                " -x " + str(ds_factor_x) +
                " -y " + str(ds_factor_y) +
                " -z " + str(ds_factor_z) +
                " -a 264 -b 160 -c 228 " +
                " -r " + swc_resampled +
                " -g " + swc_global +
                " -f " + swc_ffd + 
                " -s " + swc_local
                )
            
            #run the registration 
            print("registering swc for " + mouse_name)
            print("running swc registration: " + cmd_swc)
            res = os.system(cmd_swc)
            
            #return swc_local, as all cells are already contained no extra reassembly is needed
            return swc_local
    
def run_mbrainaligner_and_swc_reg(entry, settings, xyz=False, latest_output=None,aligned_results_folder=-1,mBrainAligner_location=-1,parallel_processing=False):  
    ''' 
    Inputs are organized as follows:
    entry = path to the v3d file 
    settings = configuration file
    xyz = whether the connected_component analysis data is XYZ (True) or ZXY (False). Default False. 
    latest_output = if there are previously computed alignment files, you can put the path here. Default None. 
    aligned_results_folder = a collection folder where all results are collected (they will be locally saved next to the original data as well)
    mBrainAligner_location = the location of the mBrainAligner files, i.e. D:/MoritzNegwer/Documents/mBrainAligner
     
    '''
    print(f"{datetime.datetime.now()} : Setting up atlas alignment parameters")
    entry_folder = entry.split("/")[-1].replace(".csv","")
    brain = ("_").join(entry_folder.split("_")[1:])
    orientation = entry_folder.split("_")[0]
    #TODO Find the v3draw
    v3draw_path = os.path.join(settings["mask_detection"]["output_location"], brain,"stack_downsampled.v3draw")
    tiff_path   = os.path.join(settings["mask_detection"]["output_location"], brain,"stack_resampled.tif")
    csv_path    = entry#os.path.join(settings["postprocessing"]["output_location"], brain) 

    #min_size = settings["postprocessing"]["min_size"]
    #max_size = settings["postprocessing"]["max_size"]

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
    mouse_name = brain #os.path.split(entry)[1][:9]
    
    #define output dir and try to ceeate it. If it already exists, skip creation (and subsequently overwrite contents)
    output_dir = os.path.join(settings["atlas_alignment"]["output_location"], mouse_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(aligned_results_folder):
        os.makedirs(aligned_results_folder)
    
    print(f"latest_output {latest_output}")
    print(f"Output dir {output_dir}")

    mBrainAligner_location = settings["atlas_alignment"]["mBrainAligner_location"]
    
    #run mBrainAligner-to-atlas registration
    print(f"{datetime.datetime.now()} : Registering brain to atlas using mBrainaligner")
    atlas_align(mBrainAligner_location, source_file,output_dir,mouse_name,settings)
    
    print(f"{datetime.datetime.now()} : Mapping cell coordinates into atlas space")
    #rewrite swc from Rami's blob swc
    target_swc_file_name_list = rewrite_swc(csv_path, output_dir,XYZ=xyz,parallel_processing=parallel_processing)
    
    #align swc to atlas
    local_swc = register_swc_to_atlas   (mBrainAligner_location, target_swc_file_name_list, csv_path, tiff_path, mouse_name,output_dir,aligned_results_folder,XYZ=xyz,parallel_processing=parallel_processing)
    
    
    if parallel_processing != False:
        local_swc = reassemble_swcs(output_dir) 
    
    #re-attach original size column and copy to new 
    reattach_size_and_copy(csv_path,local_swc,mouse_name, output_dir, aligned_results_folder)

    print(f"{datetime.datetime.now()} : Atlas registration finished")
    #return mouse_name so it can be added to the mouse_name list 
    return mouse_name