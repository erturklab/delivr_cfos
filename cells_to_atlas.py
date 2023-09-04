#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:19:45 2022

@author: wirrbel
"""
import numpy as np
import pandas as pd
import csv
import nibabel as nib
import pandas as pd
import datetime
from xml.etree import ElementTree as ET
import io
from scipy import stats
from scipy.ndimage import gaussian_filter 
import warnings
import os
import tifffile
from PIL import Image
import pickle
import glob


def parseOntologyXML(ontologyInput=None):
    
    if ontologyInput == None: raise Exception("An Allen CCF ontology XML file must be provided.")
    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " Parsing ontology XML" )  
        
    outputFileColumns = ['id','name', 'acronym', 'red', 'green', 'blue', 'graph_order', 'parent_id', 'parent_acronym', 'color-hex-triplet','structure-level']
    
    with io.open(ontologyInput, 'r', encoding='utf-8-sig') as f:
        contents = f.read()
        root = ET.fromstring(contents)
    
    ontologyParsed = []
    ontologyParsed.append(tuple(outputFileColumns))
    
    #print(','.join(outputFileColumns))
        
    row=[0,'background','bgr',0,0,0,0,'None','None','000000',0]
    ontologyParsed.append(tuple(row))
    
    for atlasStructure in root.iter('structure'):
        
        #identify root structure and print its acronym
        structures = root.iter('structure')
        for tmp in structures:
            RandAttribute = tmp.findall('id')[0].text
            if RandAttribute == atlasStructure.find('parent-structure-id').text:
                ci_name = tmp.findall('acronym')[0].text               
                
                
        if atlasStructure.find('id-original') == None:
            structureId = atlasStructure.find('id').text
        else:
            structureId = atlasStructure.find('id-original').text
        #structureId = atlasStructure.find('id').text
            
        if int(structureId) == 997:
            ci_name='"root"'
            print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " parent of ID 997 is always mapped to root" )  
            
        if int(structureId) == 312782566:
            structureId=312782560
            print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " mapping ID 312782566 --> 312782560 (only the latter exists is in the annotation NRRD)" )  
            
        if int(structureId) == 614454277:
            structureId=614454272
            print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + " mapping ID 614454277 --> 614454272 (only the latter exists is in the annotation NRRD)" )
        
        
        row=[int(structureId) ,
              atlasStructure.find('name').text , 
              atlasStructure.find('acronym').text.replace('"','') , 
              tuple(int(atlasStructure.find('color-hex-triplet').text[i:i+2], 16) for i in (0, 2, 4))[0], 
              tuple(int(atlasStructure.find('color-hex-triplet').text[i:i+2], 16) for i in (0, 2, 4))[1],
              tuple(int(atlasStructure.find('color-hex-triplet').text[i:i+2], 16) for i in (0, 2, 4))[2],
              atlasStructure.find('graph-order').text,
              atlasStructure.find('parent-structure-id').text,             
              ci_name,             
              atlasStructure.find('color-hex-triplet').text,
              int(atlasStructure.find('st-level').text)]
        ontologyParsed.append(tuple(row))
        
    ontologyDF=pd.DataFrame.from_records(ontologyParsed[1:], columns=outputFileColumns)
    print(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ' Parsing finished, found ' + str(ontologyDF['id'].unique().shape[0]) + ' unique IDs' )
    
    return ontologyDF

def collapseToColorGroup(ElementsList, ontologyDF, excludeRegions=None):
       
    tmp = pd.DataFrame(ElementsList)  
    dfColNames=["ColorGroup", "GroupName", "GroupAcronym", "GroupedAcronyms", "BlobCount"]
    groupTemplate=pd.DataFrame(columns=dfColNames)

    groupTemplate['ColorGroup'] = ontologyDF['color-hex-triplet'].unique().tolist()
    
    for i, row in groupTemplate.iterrows():
        acronymList = ontologyDF[ontologyDF['color-hex-triplet'] == groupTemplate.at[i,'ColorGroup']]['acronym'].tolist()
        
        groupTemplate.at[i,'GroupedAcronyms']        = ', '.join(acronymList)
        groupTemplate.at[i,'GroupName']              = ontologyDF[ontologyDF['acronym']==acronymList[0]]['name'].to_string(index=False).strip()
        groupTemplate.at[i,'GroupAcronym']           = acronymList[0]
        
        groupTemplate.at[i,'BlobCount']              = tmp[tmp["color-hex-triplet"]==groupTemplate.at[i,'ColorGroup']]["number"].sum()
        
    if not excludeRegions == None:
        groupTemplate  = groupTemplate[~groupTemplate['ColorGroup'].isin(excludeRegions)]
        
    return groupTemplate  

def mbrainaligner_atlas_to_ccf(cells,LabelImage):
    
    #Flip X in mBrainAligner-Atlas space 
    cells['x'] = 264-cells['x']
    #Flip Y in mBrainAligner-Atlas space
    cells['y'] = 160 - cells['y']
    
    #swap X and Y in mBrainaligner-atlas space
    cells = cells.rename(columns={'x' : 'y', 'y' : 'x', 'z' : 'z'})

    #subtract padding from mBrainAligner atlas values 
    cells["x"] = cells["x"]
    cells["y"] = cells["y"]
    cells["z"] = cells["z"]
    
    #multiply by 2 to get to CCF3 dimensions of 25µm/voxel
    cells[["x","y","z"]] = cells[["x","y","z"]]*2

    #correct for the original Blob id from the connected component analysis starting at 1, not 0
    cells['connected_component_id'] += 1
    
    #round to int 
    cells = cells.round().astype(int)

    orig_cells_number = len(cells)
    
    #cut off anything that is out of bounds 
    cells = cells.drop(cells[cells.x >= LabelImage.shape[2]].index)
    cells = cells.drop(cells[cells.y >= LabelImage.shape[1]].index)
    cells = cells.drop(cells[cells.z >= LabelImage.shape[0]].index)
    cells = cells.drop(cells[cells.x < 0 ].index)
    cells = cells.drop(cells[cells.y < 0 ].index)
    cells = cells.drop(cells[cells.z < 0 ].index)
    
    cells = cells.reset_index (drop=True)
    print("discarded out of bounds cells: ", orig_cells_number - len(cells))
    
    return cells

def create_region_table(cells, ontology_df):
    ##count the cells per region (i.e. occurrence of each value) 
    #uniquetable = ontology_df.set_index('graph_order').iloc[cells['graph_order'].unique()]
    uniquetable = ontology_df.set_index('graph_order')
    uniquetable['number'] = cells['graph_order'].value_counts()
    #reset index for normal range 
    uniquetable = uniquetable.reset_index()
    # change order so that number is place #2 
    uniquetable = uniquetable.reindex(columns = ['id', 'number', 'name', 'acronym', 'red', 'green', 'blue', 'graph_order', 'parent_id', 'parent_acronym', 'color-hex-triplet','structure-level'])
    
    ##sort by graph order for easy comparability between mice 
    #first cast graph_order as int 
    uniquetable['graph_order'] = uniquetable['graph_order'].astype(int)
    #then sort by graph order 
    uniquetable = uniquetable.sort_values(by=['graph_order'])
    
    #fill NaN values with 0
    uniquetable['number'] = uniquetable['number'].fillna(0)
   
    return uniquetable 

def create_heatmap (cells, LabelImage):
    #create empty numpy array with same shape as LabelImage (i.e. the reference atlas)
    heatmap = np.zeros(shape=LabelImage.shape)
    
    #filter out cells with background (add other similar terms to remove e.g. fiber tracts)
    #cells = cells[cells.acronym != 'bgr']
    
    #determine the counts per coordinate
    counts = cells[['x','y','z']].value_counts()
    #reset index so tha the xyz coordinates become ordinary columns. Now column [0] contains the counts 
    counts = counts.reset_index()
    
    #add counts as value to the heatmap array 
    heatmap[counts['z'].astype(int),counts['y'].astype(int),counts['x'].astype(int)] = counts[0].astype(int)
    
    #cast as int 
    heatmap = heatmap.astype(int)
    
    #3D blurring. Assumes there are <7 cells / voxel (or it will exceed the 65500 uint16 maximum). 
    # 5 sigma works for CCF3 atlas-sized stacks with 456 x 528 x 320 
    #for documentation see https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    #blurred_heatmap = gaussian_filter(heatmap*10000, sigma=5) #for visualization 
    
    #for statistics
    blurred_heatmap = gaussian_filter(heatmap.astype('float32'), sigma=2.25) #for visualization 
    
    return blurred_heatmap #change this to heatmap for the unblurred, per-pixel plot     
    
def cells_to_atlas(cells,LabelImage,ontology_df):
   
    #find values for all cells 
    RegionID = LabelImage[cells['z'].to_list(),cells['y'].to_list(),cells['x'].to_list()]
    
    #find all values at once 
    #note that we need to add +1 everywhere except for 0 (=background) 
    #this is because the graph_order contains 2 graph_order=0, one for background and one for root, 
    #and we need to account for the offset. 
    
    #add +1 everywhere except 0 
    RegionID[RegionID!=0] += 1 
    #generate the region ID table for each cell 
    allvals = ontology_df.iloc[RegionID]
    #reset index 
    allvals = allvals.reset_index(drop=False)
        
    #add to cells list 
    cells = cells.merge(allvals,left_index=True,right_index=True)
    #cells[allvals.columns] = allvals
    #cells = pd.concat([cells,allvals],axis=1)
    
    #debug 
    return cells,allvals
    #return cells

def add_to_collection(collection_table, uniquetable, mouse_name):
    #set id column as index 
    uniquetable = uniquetable.set_index('id')
    
    #reindex according to the complete table (collection_table, same as ontology_df)
    uniquetable = uniquetable.reindex(index=collection_table.id)
    
    #merge into collection_table 
    #collection_table = collection_table.merge(uniquetable['number'].rename(mouse_name),left_index=True,right_index=True,how='left')
    collection_table[mouse_name] = uniquetable.reset_index()['number']
    
    return collection_table
    
#=== Main function body === 
#if __name__ == '__main__': 
def map_cells_to_atlas(OntologyFilePath,CCF3_filepath,source_folder, mouse_name_list,target_folder,hookoverall,hookfactor):
    #create a heatmap collection 
    heatmap_collection = {}

    #load the ontology df 
    ontology_df = parseOntologyXML(OntologyFilePath)
    
    #load region atlas 
    #LabelImage = np.asarray(nib.load(LabelFilePath).dataobj)
    LabelImage = tifffile.imread(CCF3_filepath)
    
    #try to create target dir 
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    #define an empty collections table 
    collection_region_table = ontology_df.copy()
    collection_region_table['graph_order'] = collection_region_table['graph_order'].astype(int)
    
    #create an empty collection df table 
    collection_collapsed_table = collapseToColorGroup(pd.DataFrame(columns=['id','number']+ontology_df.columns[2:].tolist()),ontology_df)
    
    print(f"\nMouse_name_list:\n{mouse_name_list}\n")

    #iterate through files and save as excel 
    for mouse_i, mouse_name in enumerate(mouse_name_list):
        
        #Hook for interaction with FIJI plugin
        print(f"HOOK:{hookoverall}:{hookfactor}:{mouse_i}:{len(mouse_name_list)}")
        #determine cellsfile 
        cellsfile = glob.glob(os.path.join(source_folder,mouse_name+"*"))
        cellsfile = [x for x in cellsfile if mouse_name in x and ".csv" in x][0]
        
        #debug
        print(f"\nMouse name:\n{mouse_name}")
        
        #load xyz coords
        print(f"Cellsfile:\n{cellsfile}\n")
        cells = pd.read_csv(cellsfile,sep=' ',usecols=["n","x","y","z"])
        
        #optional: rename "n" to "connected_component_id" for connected component analysis
        cells = cells.rename(columns={"n" : "connected_component_id"})
        
        #remove padding (empirically determined by checking a tiff extract of their CCF_u8_xpad vs standard CCF3)
        cells = mbrainaligner_atlas_to_ccf(cells,LabelImage)
        
        #check Region ID per cell and add write to csv
        cells,allvals = cells_to_atlas(cells,LabelImage,ontology_df)
        #then, save as csv 
        cells.to_csv(os.path.join(target_folder,"cells_" + mouse_name + ".csv"))
        
        #create region table, write to csv, and add to collection table 
        uniquetable = create_region_table(cells,ontology_df)
        uniquetable.to_csv(os.path.join(target_folder,"cells_overview_" + mouse_name + ".csv"))
        collection_region_table = add_to_collection(collection_region_table, uniquetable, mouse_name)
        
        #create collapsed table and write to csv
        color_region_table = collapseToColorGroup(uniquetable, ontology_df)
        color_region_table.to_csv(os.path.join(target_folder,"region_collapsed_" + mouse_name + ".csv"))
        collection_collapsed_table = collection_collapsed_table.merge(color_region_table['BlobCount'].rename(mouse_name),left_index=True,right_index=True,how='left')
        
        #create heatmap and write to tif
        heatmap = create_heatmap(cells,LabelImage)
        tifffile.imwrite(os.path.join(target_folder,"heatmap_" + mouse_name + ".tif"),heatmap.astype("float"),compression='lzw')
        #add to heatmap collection
        heatmap_collection[mouse_name] = heatmap
    
    
    
    ##at the end, save the collection table 
    #replace all empty cells with 0 (after all, we checked and could not identify any cells there)
    collection_region_table = collection_region_table.fillna(0)
    #write to excel table 
    collection_region_table.to_excel(os.path.join(target_folder,"region_overview.xlsx"))
    
    ##at the end, save collection_collapsed table 
    collection_collapsed_table = collection_collapsed_table.fillna(0)
    #write to excel 
    collection_collapsed_table.to_excel(os.path.join(target_folder,"region_collapsed_overview.xlsx"))
        
    #save heatmap collection 
    pickle.dump(heatmap_collection,open(os.path.join(target_folder,'heatmap_collection.pickledump'),'wb'))
    ####load again with the following command. note the 'rb'at the end. 
    ###blub = pickle.load(open('/path/to/heatmap_collection.pickledump','rb'))
    
    '''
    #define lists 
    mouse_list = list(heatmap_collection.keys())
    c26_list = mouse_list[0:6]
    nc26_list = mouse_list[6:12]
    pbs_list = mouse_list[12:]
    group_list = [c26_list,nc26_list,pbs_list]
    
    for group in group_list:
        #create empty heatmap
        heatmap_avg = np.zeros(shape=LabelImage.shape)
        for mouse in group:
            heatmap_avg = np.add(heatmap_avg,heatmap_collection[mouse])
        #divide by number of mice to normalize 
        heatmap_avg = heatmap_avg/len(group)
        heatmap_avg = heatmap_avg.astype('float32')
        #save heatmap 
        tifffile.imwrite(os.path.join(target_folder,"heatmap_average_" + group[0] + ".tif"),heatmap_avg.astype("float"),compression='lzw')
    '''
