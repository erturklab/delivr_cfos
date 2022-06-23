import os
import json
from path import Path
from downsample_and_mask.downsample_and_mask import downsample_mask
from inference import inference 
from count_blobs import count_blobs
from blob_highlighter import blob_highlighter

# Load settings
settings = {}
with open("config.json","r") as file:
    settings = json.loads(file.read())

# Downsample
for brain in os.listdir(settings["mask_detection"]["input_location"]):
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


