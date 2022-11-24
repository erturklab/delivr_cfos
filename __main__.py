import os
import json
from path import Path
from downsample.downsample_and_mask import downsample_mask
from inference import inference 
from count_blobs import count_blobs
from automate_mBrainaligner import run_mbrainaligner_and_swc_reg
from cells_to_atlas import map_cells_to_atlas
from blob_highlighter import blob_highlighter


def setup_subfolders(dict_entry):
    for item in dict_entry.values():
        if isinstance(item, dict):
            setup_subfolders(item)
        elif isinstance(item, str):
            if "/data/" in item and item[-1] == "/":
                item = item.replace("/data/","./data/")
                if not os.path.exists(item):
                    os.mkdir(item)

def setup_folders(settings):
    if not os.path.exists("./data/"):
        os.mkdir("./data/")
    setup_subfolders(settings)

# Load settings
settings = {}
with open("config_test.json","r") as file:
    settings = json.loads(file.read())

# # Setup the file structure
setup_folders(settings)

# # Downsample
# # Downsample, filter out ventricles, upsample and mask the raw images
# # Multiple intermediate steps are saved for further steps down the pipeline
for brain in os.listdir(settings["raw_location"]):
    downsample_mask(settings, brain)

# Infer
# Run inference of the trained deep learning network on the 
# masked brain images
batch_path = ""

if settings["blob_detection"]["input_location"] == "":
    batch = Path(settings["mask_detection"]["output_location"])
else:
    batch = Path(settings["blob_detection"]["input_location"])

print(f"Blob detection in {batch}")
mice = batch.dirs()
print(f"Blob detection for {mice}")

for mouse in mice:
    print(f"Detecting in {mouse}")
    mouse_name = mouse.name
    mouse = os.path.join(mouse, "masked_niftis")
    slices = mouse.files("*.nii.gz")
    inference.run_inference(niftis          = slices,\
                            output_folder   = settings["blob_detection"]["output_location"],\
                            model_weights   = settings["blob_detection"]["model_location"], \
                            tta             = False,
                            comment         = mouse_name)

# Post-processing
# Counts individual blobs, filters by size and saves each blob, 
# its size and its location (x/y/z) in a csv file for each brain
count_blobs(settings)

# Atlas alignment
postprocessed_files = Path(settings["postprocessing"]["output_location"]).files("*.csv")
for blobcoordinates in postprocessed_files:
    run_mbrainaligner_and_swc_reg(entry                     = blobcoordinates,\
                                  settings                  = settings,\
                                  xyz                       = False,\
                                  latest_output             = None,\
                                  aligned_results_folder    = settings["atlas_alignment"]["collection_folder"],\
                                  mBrainAligner_location    = settings["atlas_alignment"]["mBrainAligner_location"])


# Region assignment 
map_cells_to_atlas(OntologyFilePath = settings["region_assignment"]["CCF3_ontology"],\
                   CCF3_filepath    = settings["region_assignment"]["CCF3_atlasfile"],\
                   cell_file_list   = settings["region_assignment"]["input_location"].files("*local_registered_with_original_size.csv"),\
                   target_folder    = settings["region_assignment"]["output_location"])

# Visualization
blob_highlighter(settings)


