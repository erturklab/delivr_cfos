import os
import json
from path import Path
from downsample.downsample_and_mask import downsample_mask
from inference import inference 
from count_blobs import count_blobs
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
with open("config.json","r") as file:
    settings = json.loads(file.read())

# Setup the file structure
setup_folders(settings)

# Downsample
for brain in os.listdir(settings["raw_location"]):
    downsample.downsample_mask(settings, brain)

# Infer
batch_path = ""

if settings["blob_detection"]["input_location"] == "":
    batch = Path(settings["mask_detection"]["mask_output_location"])
else:
    Path(settings["blob_detection"]["input_location"])

mice = batch.dirs()

for mouse in mice:
    slices = mouse.files("*.nii.gz")
    inference.run_inference(niftis          = slices,\
                            output_folder   = settings["blob_detection"]["output_location"],\
                            model_weights   = settings["blob_detection"]["model_location"], \
                            comment         = mouse.name)

# Post-processing
count_blobs(settings)

# Atlas alignment
#TODO @Moritz add code snippet here

# Visualization
blob_highlighter(settings)


