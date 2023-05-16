import os
import json
import glob
import argparse
import numpy as np
from path import Path
from downsample.downsample_and_mask import downsample_mask, get_real_size
from inference import inference 
from count_blobs import count_blobs
from automate_mBrainaligner import run_mbrainaligner_and_swc_reg
from cells_to_atlas import map_cells_to_atlas
from blob_highlighter import blob_highlighter
from shutil import rmtree

#TODO Check if output data exists -> skip 

def setup_folders(settings):
    if not os.path.exists(settings["output_location"]):
        os.makedirs(settings["output_location"])
    work_packages = ["mask_detection", "blob_detection", "postprocessing", "atlas_alignment", "region_assignment", "visualization"]
    for work_package in work_packages:
        output_folder = Path(settings[work_package]["output_location"])
        parent_folder = output_folder.parent.parent
        if not os.path.exists(parent_folder):
            print(f"Making {parent_folder}")
            os.makedirs(parent_folder)
        if not os.path.exists(output_folder):
            print(f"Making {output_folder}")
            os.makedirs(output_folder)
        if work_package == "atlas_alignment":
            collections_folder = Path(settings[work_package]["collection_folder"])
            print(f"Making {collections_folder}")
            if not os.path.exists(collections_folder):
                os.makedirs(collections_folder)
        
def setup_config(settings):
    if not settings["FLAGS"]["ABSPATHS"]:
        work_packages = ["mask_detection", "blob_detection", "postprocessing", "atlas_alignment", "region_assignment", "visualization"]
        output_path = settings["output_location"]
        for work_package in work_packages:
            for key in settings[work_package].keys():
                if "input" in key or "output" in key or "collection" in key:
                    settings[work_package][key] = os.path.join(output_path, settings[work_package][key])
        print("Transformed paths in config file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DELIVR inference pipeline")
    parser.add_argument("config", metavar="config", type=str, nargs="*", default="config.json", help="Path for the config file; default is in the same folder as the __main__.py file (./config.json)")

    args = parser.parse_args()

    config_location = args.config

    if type(config_location) == type([]):
        config_location = config_location[0]

    # Load settings
    settings = {}
    with open(config_location,"r") as file:
        print(f"Loading {config_location}")
        settings = json.loads(file.read())

    # Make paths absolute (if they werent already)
    setup_config(settings)

    # Setup the file structure
    setup_folders(settings)

    #TODO Get overall "Hookfactor"
    #TODO Get "action" hooks
    hookfactor = 0
    hookoverall = 0
    if settings["FLAGS"]["MASK_DOWNSAMPLE"]: hookfactor += 1
    if settings["FLAGS"]["BLOB_DETECTION"]: hookfactor += 1
    if settings["FLAGS"]["POSTPROCESSING"]: hookfactor += 1
    if settings["FLAGS"]["ATLAS_ALIGNMENT"]: hookfactor += 1
    if settings["FLAGS"]["REGION_ASSIGNMENT"]: hookfactor += 1
    if settings["FLAGS"]["VISUALIZATION"]: hookfactor += 1
    print(f"HOOK:OVERALL:{hookfactor}")

    # Downsample
    # Downsample, filter out ventricles, upsample and mask the raw images
    # Multiple intermediate steps are saved for further steps down the pipeline
    mouse_list = []
    if settings["FLAGS"]["MASK_DOWNSAMPLE"]:
        print("Masking")
        brain_list = os.listdir(settings["raw_location"])
        hookoverall += 1
        for brain_i, brain in enumerate(brain_list):
            #Hook for communication with Fiji plugin
            print(f"HOOK:{hookoverall}:{hookfactor}:{brain_i}:{len(brain_list)}")
            #downsample and mask image stack
            if not os.path.exists(os.path.join(settings["mask_detection"]["output_location"], brain,"masked_niftis")):
                mouse_id, stack_shape = downsample_mask(settings, brain)
                mouse_list.append({"mouse_id":mouse_id,"stack_shape":stack_shape})
            else:
                print(f"{brain} exists, skipping...")

    # Infer
    # Run inference of the trained deep learning network on the 
    # masked brain images
    if settings["FLAGS"]["BLOB_DETECTION"]:
        print("Blob detection")
        batch_path = ""

        if settings["blob_detection"]["input_location"] == "":
            batch = Path(settings["mask_detection"]["output_location"])
        else:
            batch = Path(settings["blob_detection"]["input_location"])

        print(f"Blob detection in {batch}")
        mice = sorted(batch.dirs())
        print(f"Blob detection for {mice}")
        hookoverall += 1

        for mouse_i, mouse in enumerate(mice):
            #Hook for communicating with Fiji plugin
            print(f"HOOK:{hookoverall}:{hookfactor}:{mouse_i}:{len(mice)}")
            # 
            binary_path = settings["blob_detection"]["output_location"] + "/" + mouse.name + "/binary_segmentations/masked_nifti.npy"
            tta         = settings["FLAGS"]["TEST_TIME_AUGMENTATION"]
            try:
                stack_shape  = mouse_list['mouse_id'==mouse]['stack_shape']
            except:
                stack_shape = get_real_size(os.path.join(settings["raw_location"], mouse.name))
                stack_shape = (1,1,*stack_shape)
            if not os.path.exists(binary_path):
                print(f"Detecting in {mouse}")
                mouse_name = mouse.name
                mouse = os.path.join(mouse, "masked_niftis")
                slices = mouse.files("*.npy") 
                inference.run_inference(niftis           = slices,\
                                        output_folder    = settings["blob_detection"]["output_location"],\
                                        stack_shape      = stack_shape, \
                                        model_weights    = settings["blob_detection"]["model_location"], \
                                        tta              = tta, \
                                        comment          = mouse_name)
            else:
                print(f"{mouse} already processed, skipping...")

    # Post-processing
    # Counts individual blobs, filters by size and saves each blob, 
    # its size and its location (x/y/z) in a csv file for each brain
    if settings["FLAGS"]["POSTPROCESSING"]:
        print("Postprocessing")
        path_in     = settings["postprocessing"]["input_location"]
        path_out    = settings["postprocessing"]["output_location"]
        hookoverall += 1

        min_size = settings["postprocessing"]["min_size"]
        max_size = settings["postprocessing"]["max_size"]
        
        if not os.path.exists(path_out):
            os.mkdir(path_out)

        len_b = len(os.listdir(path_in))
        for brain_i, brain in enumerate(sorted(os.listdir(path_in))):
            #Hook for communicating with Fiji plugin
            print(f"HOOK:{hookoverall}:{hookfactor}:{brain_i}:{len(os.listdir(path_in))}")
            #get shape of the image stack
            try:
                stack_shape  = mouse_list['mouse_id'==brain]['stack_shape']
            except:
                stack_shape = get_real_size(os.path.join(settings["raw_location"], brain))
                stack_shape = (1,1,*stack_shape)
            count_blobs(settings, path_in, brain_i, brain, stack_shape, min_size, max_size)

    # Atlas alignment
    if settings["FLAGS"]["ATLAS_ALIGNMENT"]:
        print("Atlas alignment")
        postprocessed_files = Path(settings["postprocessing"]["output_location"]).files("*.csv")

        mouse_name_list = []
        hookoverall += 1
        for brain_i, blobcoordinates in enumerate(postprocessed_files):
            print(f"HOOK:{hookoverall}:{hookfactor}:{brain_i}:{len(postprocessed_files)}")
            mouse_name = run_mbrainaligner_and_swc_reg(entry                     = blobcoordinates,\
                                          settings                  = settings,\
                                          xyz                       = False,\
                                          latest_output             = None,\
                                          aligned_results_folder    = settings["atlas_alignment"]["collection_folder"],\
                                          mBrainAligner_location    = settings["atlas_alignment"]["mBrainAligner_location"],\
                                          parallel_processing       = settings["atlas_alignment"]["parallel_processing"])
             #add to mouse name list 
            mouse_name_list.append(mouse_name)

    # Region assignment 
    #TODO Folder erstellen
    if settings["FLAGS"]["REGION_ASSIGNMENT"]:
        print("Region assignment")
        #TODO HOOKS
        hookoverall += 1
        map_cells_to_atlas(OntologyFilePath = settings["region_assignment"]["CCF3_ontology"],\
                       CCF3_filepath    = settings["region_assignment"]["CCF3_atlasfile"],\
                       source_folder  = settings["atlas_alignment"]["collection_folder"],\
                       mouse_name_list  = mouse_name_list,\
                       target_folder    = settings["region_assignment"]["output_location"],
                       hookoverall      = hookoverall,
                       hookfactor       = hookfactor)

    # Visualization
    if settings["FLAGS"]["VISUALIZATION"]:
        print("Visualization")
        region = "" #TODO: User should be able to choose
        mouse_name_list = list(sorted(os.listdir(settings["visualization"]["input_prediction_location"])))
        brain_list = list(zip(mouse_name_list,[region]))
        hookoverall += 1
        for brain_i, brain_item in enumerate(brain_list):
            print(brain_item)
            #stack_shape = get_real_size(os.path.join(settings["raw_location"], brain_item[0]))
            stack_shape = (715, 6656, 5616)
            stack_shape = (1,1,*stack_shape)
            print(f"HOOK:{hookoverall}:{hookfactor}:{brain_i}:{len(brain_list)}")
            blob_highlighter(settings, brain_item,stack_shape)

    # Cleanup
    if settings["FLAGS"]["MASK_DOWNSAMPLE"] and not settings["FLAGS"]["SAVE_MASK_OUTPUT"]:
        print("Removing masking output...")
        rmtree(settings["mask_detection"]["output_location"])
        print("Done.")
    if settings["FLAGS"]["BLOB_DETECTION"] and not settings["FLAGS"]["SAVE_NETWORK_OUTPUT"]:
        print("Removing network output...")
        rmtree(settings["blob_detection"]["output_location"])
        print("Done.")
    if settings["FLAGS"]["POSTPROCESSING"] and not settings["FLAGS"]["SAVE_POSTPROCESSING_OUTPUT"]:
        print("Removing postprocessing output...")
        rmtree(settings["postprocessing"]["output_location"])
        print("Done.")
    if settings["FLAGS"]["ATLAS_ALIGNMENT"] and not settings["FLAGS"]["SAVE_ATLAS_OUTPUT"]:
        print("Removing atlas alignment output...")
        rmtree(settings["atlas_alignment"]["output_location"])
        print("Done.")
    print("DELIVR Done.")

