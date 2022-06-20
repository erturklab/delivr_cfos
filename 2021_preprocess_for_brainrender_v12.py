#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:56:37 2020

@author: wirrbel
"""

# We begin by adding the current path to sys.path to make sure that the imports work correctly
import sys
sys.path.append('/home/wirrbel/brainrender_update_2022-01/')
import os
import pandas as pd
import numpy as np
from shutil import copyfile
import in_place

from vedo import embedWindow  # for more explanations about these two lines checks the notebooks workflow example
embedWindow(None)

# Import variables
from brainrender import * # <- these can be changed to personalize the look of your renders

# Import brainrender classes and useful functions
from brainrender import Scene
from brainrender.actors import Points, PointsDensity, Volume
from brainrender.video import VideoMaker

import myterial
from vedo.colors import getColor,getColorName

#hardcode paths 
# check for yourself by running "whereis transformix" in the command line 
transformix_bin = '/usr/bin/transformix'

def copy_and_optimize_transformParameters (new_folder,transform): 
    #takes the list of TransformParametersFile, copies them to the new folder and adapts their pointers 
    
    #copy over transform files 
    for transforms in transform:
        #
        file_to_copy = os.path.join(new_folder,os.path.split(transforms)[1])
        copyfile (transforms, file_to_copy)
        
        
        # adapt the InitialTransform parameter so that it points to the local file 
        # search and replace paths to InitialTransforms 
        with in_place.InPlace(file_to_copy) as file:
            for line in file:
                if ('Initial' in line) and ('NoInitialTransform' not in line):
                    #if 'NoInitialTransform' not in line:
                        #replace area between first quote and last slash with directory path
                        first_quote = line.find(' \"')
                        last_slash = line.rfind('/') 
                        line = line[:first_quote] + ' \"' + new_folder + line[last_slash:]
                file.write(line)
        file.close()
    

def parseElastixOutputPoints(filename, indices = True):
    #from Clearmap 1 Elastix.py
    """Parses the output points from the output file of transformix
    
    Arguments:
        filename (str): file name of the transformix output file
        indices (bool): if True return pixel indices otherwise float coordinates
        
    Returns:
        points (array): the transformed coordinates     
    """
    
    with open(filename) as f:
        lines = f.readlines()
        f.close();
    
    length = len(lines);
    
    if length == 0:
        return numpy.zeros((0,3));
    
    points = np.zeros((length, 3));
    k = 0;
    for line in lines:
        ls = line.split();
        if indices:
            for i in range(0,3):
                points[k,i] = float(ls[i+22]);
        else:
            for i in range(0,3):
                points[k,i] = float(ls[i+30]);
        
        k += 1;
    
    return points;

def transform_points (cellsfile, transform):
        #make temporary text file 
        cells_folder, file_name = os.path.split(cellsfile)
        #define filename with .txt extension
        txt_name = file_name[:-4] + ".txt"
        
        #make new subfolder 
        new_folder = os.path.join(cells_folder,'Aligned_CCF3')
        
        #if it doesn't exist yet, make new_folder 
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
        
        #save np file as txt 
        points = np.load(cellsfile)
        filename = os.path.join(new_folder,txt_name)
        
        with open(filename, 'w') as pointfile:
            #if indices:
            #    pointfile.write('index\n')
            #else:
            pointfile.write('point\n')
        
            pointfile.write(str(points.shape[0]) + '\n');
            np.savetxt(pointfile, points, delimiter = ' ', newline = '\n', fmt = '%.5e')
            pointfile.close();
            #np.savetxt(os.path.join(new_folder,txt_name), np.load(cellsfile), delimiter = ' ', newline = '\n', fmt = '%.5e')
        
        #copy transforms over and insert proper paths 
        copy_and_optimize_transformParameters (new_folder,transform)
        
        
        ##==== Step 1: intermediate alignment step to result.tif (translated atlas) === 
        
        #define copied transform 
        copied_transform = os.path.join(new_folder,os.path.split(transform[0])[1])
        
        #create transformix command
        intermediate_align = (transformix_bin
                     + ' -def ' + os.path.join(new_folder,txt_name) 
                     + ' -tp ' + copied_transform
                     + ' -out ' + new_folder)
        
        # run command 
        res = os.system(intermediate_align);
        
        ###=== Step 2: Re-save as new points file
        #load as np array 
        points_intermediate_align = parseElastixOutputPoints(os.path.join(new_folder,'outputpoints.txt'),indices=True)
        
        #write to intermediate points file 
        filename = os.path.join(new_folder,'transformed_points_intermediate.txt')
                
        with open(filename, 'w') as pointfile:
            pointfile.write('point\n')
            pointfile.write(str(points_intermediate_align.shape[0]) + '\n');
            np.savetxt(pointfile, points_intermediate_align, delimiter = ' ', newline = '\n', fmt = '%.5e')
            pointfile.close();
        
        ###=== Step 3: final (inverse) alignment 
        final_align = (transformix_bin
                       + ' -def ' + os.path.join(new_folder,'transformed_points_intermediate.txt')
                       + ' -tp ' +  os.path.join(new_folder,'TransformParameters.1.txt')
                       + ' -out ' + new_folder)
        
        res = os.system(final_align)
        
        ###=== Step 4: Load finally aligned cells 
        points_finally_aligned = parseElastixOutputPoints(os.path.join(new_folder,'outputpoints.txt'),indices=True)


        return points_finally_aligned 

def read_aligned_points(cellsfile):
    #subset of transform_points if you are really sure the transform has happened already
    cells_folder, file_name = os.path.split(cellsfile)
    
    #make new subfolder 
    new_folder = os.path.join(cells_folder,'Aligned_CCF3')
    
    points_finally_aligned = parseElastixOutputPoints(os.path.join(new_folder,'outputpoints.txt'),indices=True)

    return points_finally_aligned 

def render_screenshot (screenshots_folder, cells, output_name ,cells_color,region_to_extract,camera=None,density=None):
    #try to make save folder 
    try:
        os.mkdir(screenshots_folder)
    except:
        pass
        
    # Create a scene
    scene = Scene (title=None,screenshots_folder=screenshots_folder,inset=None)
    #scene1.add_cells_from_file('/home/wirrbel/2020-03-02_Lynn_PV_P14/2020-03-02_Lynn_PV_P14_m2/cells_transformed_CCF.csv', color="red", radius=10,alpha=0.5)
    
    #for P56 alignment 
    #cellsfile = "/media/wirrbel/MN_2/Lynn_PV_P50/2020-02-19_Lynn_PV_P50_m3/cells_transformed_to_Atlas_aligned.npy"
    #cells = np.load(cellsfile)
    cells = cells
    
    #define region name to extract 
    #region_to_extract = 'HIP'
    #region_to_extract = region_to_extract
    '''
    #extract name of parent folder from cellsfile (assumed to be brain name)
    #for single cFos brains
    brain_name = str(region_to_extract) + "_" + output_name
    
    #for averages
    #brain_name = str(region_to_extract) + "_" + os.path.basename(os.path.split(cells_path)[0]) 
     
    ##subset for region of interest 
    #region to subset (added, but not displayed)
    #region_subset = scene.add_brain_region(region_to_extract, alpha=0.1,color=myterial.grey)
    region_subset = scene.add_brain_region(region_to_extract, alpha=0.0)
    
    #subset for region 
    cells = region_subset.mesh.insidePoints(cells).points()
    '''
    #region_to_extract can be a single region name, or a list 
    #if it is a single name, then add to the scene and subset any cells for it 
    if not isinstance(region_to_extract,list):
        #define name 
        brain_name = "video_" + str(region_to_extract) + "_" + output_name 
    
        #region to subset (added, but not displayed)
        region_subset = scene.add_brain_region(region_to_extract, alpha=0.2)
        #subset for region 
        cells = region_subset.mesh.insidePoints(cells).points()
        #add points to scene 
        scene.add(Points(cells,colors=cells_color,alpha=0.2,res=5,radius=15))
    else:
        #when trying to use a list of regions, go through them one by one and subset cells each time 
        #define name 
        brain_name = "video_" + "_" + output_name 
        for idx, region in enumerate(region_to_extract):
            region_subset = scene.add_brain_region(region, alpha=0.0)
            region_color_hex = str('#'+cells_color[idx]) #this assumes that cell_color is a list with the same length as region_to_extract, containing hex values 
            region_color_rgb = getColor(rgb=region_color_hex)
            
            #debug
            #print('region_color_hex = ',region_color_hex)
            #print('region_color_rgb = ',region_color_rgb)
            
            cells_sub = cells.copy()
            try:
                cells_sub = region_subset.mesh.insidePoints(cells).points()
                scene.add(Points(cells_sub,colors=getColorName(region_color_hex) ,alpha=0.4,res=5,radius=15))
                print('added ',region)
            except:
                print('region skipped:',region)
                pass #ignore errors
            #define name 
        
    

    brain_name = "cells_" + brain_name

    
    '''
    #slice one hemisphere 
    plane = scene.atlas.get_plane(plane="sagittal", norm=[0, 0, -1])
    #set sagittal cutting plane ever so slightly off the center 
    #to prevent a render error with the "cartoon" render setting (see settings.py)
    plane.mesh.SetPosition(7829.739797878185, 4296.026746612369, -5700.497969379508)
    scene.slice(plane,close_actors=False)
    '''
    '''
    if density is not None:
        #Changed points.py to be able to adapt color map, standard = "Dark2" from vedo: https://vedo.embl.es/autodocs/_modules/vedo/colors.html 
        scene.add(PointsDensity(cells,dims=(100,100,100),colormap="twilight",radius=750))
        brain_name = "density_" + brain_name
    else:
        #add points to scene 
        scene.add(Points(cells,colors=cells_color,alpha=0.4,res=5,radius=15))
        brain_name = "cells_" + brain_name
    '''

    #scene.render()
    
    #if region has a special camera angle attached to it, use this one. Othewise, use default (see brainrender settings)
    if camera is not None:
        scene.render(camera=camera, interactive=False)
        scene.screenshot(name=brain_name)
        scene.close()
    else: 
        scene.render(camera=camera, interactive=True)
        #scene.screenshot(name=brain_name)
        #scene.close()
    '''
    #debug
    scene.render()
    #periodically report camerra parameters
    import sched
    #scheduler = sched.scheduler(time.time,time.sleep)
    def periodically_report_cam_params (sc):
        try: 
            print(get_camera_params(scene=scene))
        except: 
            pass
        #scheduler.enter(5,1,periodically_report_cam_params, (sc))
        
    #scheduler.enter(5,1,periodically_report_cam_params, (sc))
    #scheduler.run()
    
    #debug purposes only (see below for test SST)
    #return plane
    '''
    
    
def mbrainaligner_atlas_to_ccf(cells):
    '''
    #swap axes 
    cells[["z","x","y"]] = cells[["x","y","z"]]
    
    #flip x 
    cells['x'] = 528-cells['x']
    cells['y'] = 320-cells['y']
    '''
    #remove padding (empirically determined by checking a tiff extract of their CCF_u8_xpad vs standard CCF3)
    cells["x"] = cells["x"]-12
    cells["y"] = cells["y"]-20
    cells["z"] = cells["z"]-20
    
    
    cells["x"] = cells["x"]*50
    cells["y"] = cells["y"]*50
    cells["z"] = cells["z"]*50
    
    #transfer to numpy array for brainrender
    cells_np = cells.to_numpy()
    

    return cells_np

def render_videos (screenshots_folder, cells, output_name ,cells_color,region_to_extract,camera=None,density=None):
    
    # Create a scene
    scene = Scene(inset=None)
    #scene.add_brain_region("TH")
    
    #region_to_extract can be a single region name, or a list 
    #if it is a single name, then add to the scene and subset any cells for it 
    if not isinstance(region_to_extract,list):
        #define name 
        brain_name = "video_" + str(region_to_extract) + "_" + output_name 
    
        #region to subset (added, but not displayed)
        region_subset = scene.add_brain_region(region_to_extract, alpha=0.2)
        #subset for region 
        cells = region_subset.mesh.insidePoints(cells).points()
        #add points to scene 
        scene.add(Points(cells,colors=cells_color,alpha=0.2,res=5,radius=15))
    else:
        #when trying to use a list of regions, go through them one by one and subset cells each time 
        #define name 
        brain_name = "video_" + "_" + output_name 
        for region in region_to_extract:
            region_subset = scene.add_brain_region(region, alpha=0.2)
            region_color = region_subset.mesh.property.GetAmbientColor()
            cells_sub = cells.copy()
            cells_sub = region_subset.mesh.insidePoints(cells).points()
            scene.add(Points(cells_sub,colors=region_color,alpha=0.4,res=5,radius=3))
            #define name 
        
    

    brain_name = "cells_" + brain_name
    
    # Create an instance of video maker
    vm = VideoMaker(scene, screenshots_folder, brain_name ,size="3840x3840")
    
    # make a video with the custom make frame function
    # this just rotates the scene
    #vm.make_video(elevation=2, duration=2, fps=15)
    vm.make_video(azimuth=-2,elevation=0, duration=30, fps=15)
    
    scene.close()
###

    

target_regions = [#"grey", #All brain
                  #"HIP", #Hippocampal region
                  #"Isocortex", #Cortex proper
                  #"CNU", #Striatum + GP
                  #"TH", #Thalamus +
                  #"CTXsp", #cortical subplate 
                  #"BLA", #BAsolateral Amygdala
                  #"MB", #Superior / inferior colliculus
                  #"SSp-bfd", #S1 Barrel Field
                  #"SS", #S1 
                  #"AUD", #A1
                  #"VIS", #V1
                  #"HY", #Hypothalamus
                  #"MTN", #midline nuclei of the thalamus
                  'PL', #prelimbic
                  'LHA', #Lateral Hypothalamus
                  ]

#list of (uncorrected t-test) significant cortical regions (NC26 vs C26)
#ctx_list = ["BLAp","RSPv6a","PL5","SSp-n6a","VISp6a","VISal6a","ORBm5","VISl6a","ORBvl6a","VISpm2/3","VISam4","ORBvl2/3","VISam6a","VISpm5","VISpm6a","AUDp1","RSPv5","AUDv1","SSp-ll5","ORBvl1","ORBvl5","VISam5","ECT6a","TEa6a","PL2/3","APr","VISam2/3","SSp-tr6a","ILA5","ACAd5","RSPd6a","SSs6a","SSp-n5","MOs6a","VISpl6a","VISpor6a","VISli6a","VISpm4","SSp-bfd6a","ACAd6a","VISa6a","ACAv5"]
#list of (uncorrected t-test) significant subcortical regions (NC26 vs C26)
#midbrain_list = ["AVP","LHA","PMv","IPN","IF","SF","MGd","SMT","SLD","GRN","SPA","SPVC","ICB","DN","TRN"]


#nc26 vs c26 (uncorrected t-test)
nc26_vs_c26 = ['RSPv6a','VISpm4','ACAd5','VISp5','PL5','AIv5','VISpm1','ORBvl5','IPN','ECT6b','FN','VISpm2/3',
               'RSPv5','VISam5','VISal5','RSPagl6a','PL6a','VISl5','VISa6a','RSPd6a','RSPd5','VISpm5','LHA',
               'PPT','PL2/3','BLAp','SSp-n5','SSp-tr5','VISam1','ORBm5','SSp-ll5','ORBvl6a','SSp-ul5','ORBvl2/3',
               'SSp-n6a','TU','VISa5','PRNr','SSp-un5','SSp-bfd5','ORBl5','VISrl6a','MOs5','PMd','PST','AUDv1',
               'ILA5','AUDv5','VISpm6a','PRC','VISpl5','SSp-un4','SSp-ll4','ECT6a','RSPagl5','VISli5','SCsg',
               'ILA6a','ACAd2/3','SCop','VISam4','VISp4','VISC6a','RCH','VISrl5','ORBl2/3','AIp1','SSp-ul4',
               'IAD','VISam2/3','LSr','AUDv4','TEa6a']

#c26 vs PBS 
c26_vs_pbs = ['VISa6a','SF','RCH','PD','SBPV','ICB','IPRL','RPO','AUDpo6a']

#nc26 vs PBS 
nc26_vs_pbs = ['TEa1','VISpl1','VPLpc','RSPv5','VISal1','AUDv1','SI','VPL','VPMpc','MOs2/3','SSp-ll5','MOp2/3',
               'RH','VISrl4','VISp5','SSp-m2/3','PGRNd','VISrl5','VISal5','VISp1','AUDp1','SSp-m4','VISrl2/3',
               'SSp-ul1','ACB','ORBvl6a','AUDpo1','SAG','SSp-ll4','VISp4','RSPd5','RSPd6a','PN','SSp-ul4',
               'SSp-bfd1','SSp-ul5','RSPagl6a','SSp-un4','VPM','VISpl5','VISl5','VISpm1','VISpm2/3','SCop',
               'MOp1','SSp-ll2/3']

'''
#from color-collapsed table:
nc26_vs_c26 = ['PL','RSP','ILA','VIS','LZ','ORB','SS','ACA','P-sat','EP','FN','MEZ','LSX','MBmot']

c26_vs_pbs = ['MEZ']

nc26_vs_pbs = ['PALv','MO','DORsm','FN','RSP','STRv','VIS','VS']
'''

techpaper_cam_01 = {
    "pos": (2093, 2345, -49727),
    "viewup": (0, -1, 0),
    "clippingRange": (33881, 52334),
    "focalPoint": (6888, 3571, -5717),
    "distance": 44288,
    }

cFosCamera_01 = {
    "pos": (-10104, -18549, 28684),
    "viewup": (0, -1, 0),
    "clippingRange": (25755, 66938),
    "focalPoint": (6888, 3571, -5717),
    "distance": 44288,
    }

cFosCamera_02 = {
     "pos": (-23429, -13179, 21883),
     "viewup": (0, -1, 0),
     "clippingRange": (23916, 68797),
     "focalPoint": (6888, 3571, -5717),
     "distance": 44288,
     }

cFos_Fig4_camera_01 =     {
     "pos": (-23001, -17333, 19405),
     "viewup": (0, -1, 0),
     "clippingRange": (25524, 67824),
     "focalPoint": (6888, 3571, -5717),
     "distance": 44288,
     "name":"cFos_Fig4_camera"
    }

cFos_sagittal =    {
     "pos": (8525, 2656, -49965),
     "viewup": (0, -1, 0),
     "clippingRange": (32907, 58823),
     "focalPoint": (6888, 3571, -5717),
     "distance": 44288,
     "name":"cFos_sagittal"
	}

cFos_coronal =    {
     "pos": (-37318, 916, -6157),
     "viewup": (0, -1, 0),
     "clippingRange": (29896, 61881),
     "focalPoint": (6888, 3571, -5717),
     "distance": 44288,
     "name":"cFos_coronal"
	}

cFos_top =     {
     "pos": (2613, -40510, -5917),
     "viewup": (-1, 0, 0),
     "clippingRange": (35416, 56124),
     "focalPoint": (6888, 3571, -5717),
     "distance": 44288,
     "name":"cFos_top"
     }

'''
cameras = { 
    "SSp-bfd" : {'pos': (6975, -881, -15113), # for S1 barrel field
     'viewup': (0, -1, 0),
     'clippingRange': (28, 27714),
     'focalPoint': (6691, 3915, -4824),
     'distance': 11356,},
    "SS" : { 'pos': (5996, 896, -17963), # for whole S1 
     'viewup': (0, -1, 0),
     'clippingRange': (722, 27588),
     'focalPoint': (5833, 3050, -6815),
     'distance': 11356,},
    "AUD" : { 'pos': (10341, 292, -20580), # for Auditory cortex
     'viewup': (0, -1, 0),
     'clippingRange': (821, 35103),
     'focalPoint': (6631, 4279, -4871),
     'distance': 16626, }, 
    "VIS" : {'pos': (14442, -3841, -16523), # for visual cortex
     'viewup': (-1, -1, 0),
     'clippingRange': (39, 38560),
     'focalPoint': (8618, 2032, -8742),
     'distance': 11356}
    }


target_regions = ["grey"]
'''



##DEBUG## 
'''
target_mice = [["/media/wirrbel/Moritz_Negwer/SST_Maren/2019-04-23_SST_P25_Sample1_57645_Het/cells_transformed_to_Atlas_aligned.npy", "green"], #Het
               #["/media/wirrbel/Moritz_Negwer/SST_Maren/2019-04-24_SST_P25_Sample2_57646_WT/cells_transformed_to_Atlas_aligned.npy", "green"], #WT
               #["/media/wirrbel/Moritz_Negwer/SST_Maren/2019-04-25_SST_P25_Sample3_57647_WT/cells_transformed_to_Atlas_aligned.npy", "green"], #WT
               #["/media/wirrbel/Moritz_Negwer/SST_Maren/2019-04-25_SST_P25_Sample4_57648_Het/cells_transformed_to_Atlas_aligned.npy", "green"], #Het
               #["/media/wirrbel/Moritz_Negwer/SST_Maren/2019-04-26_SST_P25_Sample5_57654_WT/cells_transformed_to_Atlas_aligned.npy", "green"], #WT
               #["/media/wirrbel/Moritz_Negwer/SST_Maren/2019-04-26_SST_P25_Sample6_57657_Het/cells_transformed_to_Atlas_aligned.npy", "green"], #Het
               ]

target_folder = "/media/wirrbel/MN_2/2022-01-16_brainrender_tech_paper/SST_Maren/"

target_transformation = ["/home/wirrbel/2021-03-29_brainrender_2_preprocessing/elastix_files/Par_rotate90degaroundX_CCF3_nrrd_directions.txt",
                         "/home/wirrbel/2021-03-29_brainrender_2_preprocessing/Kim_ref_P21_v2_brain_CCF3/TransformParameters.0.txt", 
                         "/home/wirrbel/2021-03-29_brainrender_2_preprocessing/Kim_ref_P21_v2_brain_CCF3/TransformParameters.1.txt"]

target_regions = ["grey"]

#### main function call 
for mouse in target_mice:
    #transform to CCF3 
    #cells = transform_points(mouse[0],target_transformation)
    cells = read_aligned_points(mouse[0])
    for region in target_regions:
        #render_screenshot (target_folder, cells, mouse[0], mouse[1], region, camera=cameras.get(region))
        plane = render_screenshot (target_folder, cells, mouse[0], mouse[1], region, camera=techpaper_cam_01)
'''
'''
target_folder = '/home/wirrbel/2022-02-22_cFos_mBrainaligner_visualization/2022-05-27_for_fig_4c/'
#try to create target folder, skip if it already exists
try: 
    os.mkdir(target_folder)
except:
    pass    

c26_cellsfiles = ["/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/c26_3403_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/c26_3406_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/c26_3407_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/c26_3408_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/c26_3409_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/c26_3411_local_registered_with_original_size.csv"
                    ]

nc26_cellsfiles = ["/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/nc26_3412local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/nc26_3413local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/Nc26_3416local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/nc26_3417local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/Nc26_3422local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/nc26_3423local_registered_with_original_size.csv"
                    ]

PBS_cellsfiles = ["/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/PBS_3400_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/PBS_3401_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/PBS_3402_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/PBS_3418_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/PBS_3419_local_registered_with_original_size.csv",
                    "/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/PBS_3420_local_registered_with_original_size.csv"
                    ]

#define empty cells array 
cells_c26 = np.empty((1,3))

#define empty cells array 
cells_nc26 = np.empty((1,3))

#define empty cells array 
cells_PBS = np.empty((1,3))


for cellsfile in c26_cellsfiles:
    #load xyz 
    #cellsfile = "/home/wirrbel/2022-02-22_cFos_mBrainaligner_visualization/v05_3407_working/local_registered_data.swc"
    cells_all = pd.read_csv(cellsfile,sep = ' ',usecols=['x','y','z','Size'])

    #define too-small segments 
    #cells_toosmall = cells_all.query("Size < 9")
    #cells_toosmall.drop('Size',axis=1,inplace=True)
    
    #define too-large segments 
    #cells_toolarge = cells_all.query("Size > 200")
    #cells_toolarge .drop('Size',axis=1,inplace=True)
    
    #define valid cells (mean + 3x SD = 104. manual min/max = 9/200)
    valid_cells = cells_all.query("Size < 104")
    valid_cells.drop('Size',axis=1,inplace=True)

    #remove padding (empirically determined by checking a tiff extract of their CCF_u8_xpad vs standard CCF3)
    cells_np = mbrainaligner_atlas_to_ccf(valid_cells)
    
    #add to cells_all 
    cells_c26 = np.concatenate((cells_np,cells_c26),axis=0)
    
    
    output_name =  os.path.split(cellsfile)[1][:9] 
    
    
    #render for c26
    #for region in target_regions:
    #    render_screenshot (target_folder, cells_np, output_name, 'purple', region, camera=cFosCamera_01,density=True)
    #    render_screenshot (target_folder, cells_np, output_name, 'purple', region, camera=cFosCamera_01)


for cellsfile in nc26_cellsfiles:
    #load xyz 
    #cellsfile = "/home/wirrbel/2022-02-22_cFos_mBrainaligner_visualization/v05_3407_working/local_registered_data.swc"
    cells_all = pd.read_csv(cellsfile,sep = ' ',usecols=['x','y','z','Size'])

    #define too-small segments 
    #cells_toosmall = cells_all.query("Size < 9")
    #cells_toosmall.drop('Size',axis=1,inplace=True)
    
    #define too-large segments 
    #cells_toolarge = cells_all.query("Size > 200")
    #cells_toolarge .drop('Size',axis=1,inplace=True)
    
    #define valid cells (mean + 3x SD = 104. manual min/max = 9/200)
    valid_cells = cells_all.query("Size < 104")
    valid_cells.drop('Size',axis=1,inplace=True)

    #remove padding (empirically determined by checking a tiff extract of their CCF_u8_xpad vs standard CCF3)
    cells_np = mbrainaligner_atlas_to_ccf(valid_cells)
    
    #add to cells_all 
    cells_nc26 = np.concatenate((cells_np,cells_nc26),axis=0)
    
    
    output_name =  os.path.split(cellsfile)[1][:9] 
    
    
    #for region in target_regions:
    #    render_screenshot (target_folder, cells_np, output_name, 'teal', region, camera=cFosCamera_01,density=True)
    #    render_screenshot (target_folder, cells_np, output_name, 'teal', region, camera=cFosCamera_01)
    
    
for cellsfile in PBS_cellsfiles:
    #load xyz 
    #cellsfile = "/home/wirrbel/2022-02-22_cFos_mBrainaligner_visualization/v05_3407_working/local_registered_data.swc"
    cells_all = pd.read_csv(cellsfile,sep = ' ',usecols=['x','y','z','Size'])

    #define too-small segments 
    #cells_toosmall = cells_all.query("Size < 9")
    #cells_toosmall.drop('Size',axis=1,inplace=True)
    
    #define too-large segments 
    #cells_toolarge = cells_all.query("Size > 200")
    #cells_toolarge .drop('Size',axis=1,inplace=True)
    
    #define valid cells (mean + 3x SD = 104. manual min/max = 9/200)
    valid_cells = cells_all.query("Size < 104")
    valid_cells.drop('Size',axis=1,inplace=True)

    #remove padding (empirically determined by checking a tiff extract of their CCF_u8_xpad vs standard CCF3)
    cells_np = mbrainaligner_atlas_to_ccf(valid_cells)
    
    #add to cells_all 
    cells_PBS = np.concatenate((cells_np,cells_PBS),axis=0)
    
    output_name =  os.path.split(cellsfile)[1][:9] 
    
    
    #for region in target_regions:
    #    render_screenshot (target_folder, cells_np, output_name, 'grey', region, camera=cFosCamera_01,density=True)
    #    render_screenshot (target_folder, cells_np, output_name, 'grey', region, camera=cFosCamera_01)

'''
'''

#debug
cellsfile = '/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/c26_3403_local_registered_with_original_size.csv'

cells_all = pd.read_csv(cellsfile,sep = ' ',usecols=['x','y','z','Size'])

#define too-small segments 
cells_toosmall = cells_all.query("Size < 9")
cells_toosmall.drop('Size',axis=1,inplace=True)

#define too-large segments 
cells_toolarge = cells_all.query("Size > 200")
cells_toolarge .drop('Size',axis=1,inplace=True)

valid_cells = cells_all.query("Size > 9 and Size < 200")
valid_cells.drop('Size',axis=1,inplace=True)

#convert to mBrainaligner coordinates
cells = mbrainaligner_atlas_to_ccf(valid_cells)
#render_screenshot(target_folder, cells, "debug", 'black', ["TH","Isocortex","VISp"], camera=cFosCamera_01)

'''

'''
#names 
c26_namefolder = "c26_avg_density"
nc26_namefolder = "nc26_avg_density"
PBS_namefolder = "pbs_avg_density"


#averages
#render for c26

for region in target_regions:
    #render_screenshot (target_folder, cells_c26, c26_namefolder, 'purple', '', camera=cFosCamera_02,density=True)
    render_screenshot (target_folder, [], c26_namefolder, 'purple', region, camera=cFosCamera_02,density=None)
    #render_screenshot (target_folder, cells_c26, c26_namefolder, 'purple', region, camera=cFosCamera_02)

#render for nc26
for region in target_regions:
    #render_screenshot (target_folder, cells_nc26, nc26_namefolder, 'teal', '', camera=cFosCamera_02,density=True)
    render_screenshot (target_folder, [], nc26_namefolder, 'teal', region, camera=cFosCamera_02,density=None)
    #render_screenshot (target_folder, cells_nc26, nc26_namefolder, 'teal', region, camera=cFosCamera_02)


#render for bc26
for region in target_regions:
    #render_screenshot (target_folder, cells_PBS, PBS_namefolder, 'grey', '', camera=cFosCamera_02,density=True)
    render_screenshot (target_folder, [], PBS_namefolder, 'grey', region, camera=cFosCamera_02,density=None)
    #render_screenshot (target_folder, cells_PBS, PBS_namefolder, 'grey', region, camera=cFosCamera_02)
'''
'''
### === Video === 
#render for c26
for region in target_regions:
    render_videos(target_folder, cells_c26, c26_namefolder, 'purple', region, camera=cFosCamera_01)

#render for c26
for region in target_regions:
    render_videos(target_folder, cells_nc26, nc26_namefolder, 'teal', region, camera=cFosCamera_01)

#render for c26
for region in target_regions:
    render_videos(target_folder, cells_PBS, PBS_namefolder, 'grey', region, camera=cFosCamera_01)

### === Video with multiple areas 
#c26 vs nc26
render_videos(target_folder, cells_c26, "c26vsnc26_c26_cells", 'grey', nc26_vs_c26, camera=cFosCamera_01)
render_videos (target_folder, cells_nc26, "c26vsnc26_nc26_cells", 'grey', nc26_vs_c26, camera=cFosCamera_01)

#c26 vs PBS 
render_videos(target_folder, cells_c26, "c26vspbs_c26_cells", 'grey', c26_vs_pbs, camera=cFosCamera_01)
render_videos(target_folder, cells_PBS, "c26vspbs_pbs_cells", 'grey', c26_vs_pbs, camera=cFosCamera_01)

#nc26 vs PBS 
render_videos(target_folder, cells_nc26, "nc26vspbs_nc26_cells", 'grey', nc26_vs_pbs, camera=cFosCamera_01)
render_videos(target_folder, cells_PBS, "nc26vspbs_pbs_cells", 'grey', nc26_vs_pbs, camera=cFosCamera_01)
'''

#### === for fig 4c, example brain 3400 === 

#file with segments 
cellsfile = '/home/wirrbel/2021-10-17_cFos_realignments/mBrainAligner_swc_results_collection_v06/PBS_3400_local_registered_with_original_size.csv'

#images go here 
target_folder = '/home/wirrbel/2022-02-22_cFos_mBrainaligner_visualization/2022-05-27_for_fig_4c/'
#try to create target folder, skip if it already exists
try: 
    os.mkdir(target_folder)
except:
    pass     

#load areas of interest (i.e. all color groups)
area_list = pd.read_csv('/home/wirrbel/2022-02-22_cFos_mBrainaligner_visualization/color_collapsed_area_list_for_fig_4c_v01.csv')

#load xyz 
#cellsfile = "/home/wirrbel/2022-02-22_cFos_mBrainaligner_visualization/v05_3407_working/local_registered_data.swc"
cells_all = pd.read_csv(cellsfile,sep = ' ',usecols=['x','y','z','Size'])

#define valid cells (mean + 3x SD = 104. manual min/max = 9/200)
valid_cells = cells_all.query("Size < 104")
valid_cells.drop('Size',axis=1,inplace=True)

#remove padding (empirically determined by checking a tiff extract of their CCF_u8_xpad vs standard CCF3)
cells_np = mbrainaligner_atlas_to_ccf(valid_cells)

output_name =  os.path.split(cellsfile)[1][:9] 

#interactive render 
#render_screenshot(target_folder, cells_np, output_name, area_list['ColorGroup'].to_list(), area_list['GroupAcronym'].to_list(), camera=None)

#render several views at once 
camera_list = [cFos_coronal,cFos_sagittal,cFos_top,cFos_Fig4_camera_01]

for camera in camera_list:  
    #add a name for the camera 
    try:
        #if the camera has a manually added name key, use that
        cam_name = str(camera.get('name'))
    except:
        #if not, take the XYZ position arguments 
        cam_name = str(camera.get('pos'))
    
    render_screenshot(target_folder, cells_np, output_name+'_'+cam_name, area_list['ColorGroup'].to_list(), area_list['GroupAcronym'].to_list(), camera=camera)
