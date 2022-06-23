# DELiVR: A VR enabled, deep learning based cFos inference pipeline
## Introduction
## Requirements
### Python requirements
Python 3.8 on Linux for downsampling, masking, upsampling 

Python 3.8 on Linux for inference 

Python 3.8 on Windows for Atlas Alignment 

Python 3.8 on Linux for Visualization

### ILastik requirements
We recommend the latest beta version from the official website: https://www.ilastik.org/download.html. We used 1.4.0b26. 

### File requirements
We assume that the files are a series of 16-bit TIFF files, one per z-plane. Our model is optimized for data acquired with approx. 3.25 x 3.25 x 6 µm (x/y/z/) voxel size. 

## Setup and Execution
1. Clone this repository using `git clone https://github.com/erturklab/deliver_cfos.git`
2. Install the requirements using `pip install -r requirements.txt` (pip) `conda install --file requirements.txt` (anaconda)
3. Set the location of your raw data and intermediate folders in `config.json`
5. Run `python __main__.py` in the terminal

### Config.json parameters
- `raw_location` : Location of the raw image files of the brain as individual tiffs
- `mask_detection` `ilastik_location`  : Location of the ilastik executable file 
- `mask_detection` `ilastik_model`  : Location of the ilastik model 
- `mask_detection` `output_location`  : Location where the masked data will be saved to 
- `mask_detection` `downsample_steps`  : Original resoultion in um as well as downsample factor
- `blob_detection` `input_location` : Location of the masked data from previous step 
- `blob_detection` `model_location` : Location of the trained U-Net model
- `blob_detection` `output_location` : Location where the infered data will be saved to 
- `postprocessing` `input_location` : Location of the infered data 
- `postprocessing` `output_location` : Location where the csvs containing the size and location of cells will be saved to 
- `atlas_alignment` `input_location` : Location of the csvs containing the size and location of cells 
- `atlas_alignment` `output_location` : Location where the csvs containing the atlas region of each cells will be saved to
- `visualization` `input_csv_location` : Location of the csvs containing the atlas region of each cell 
- `visualization` `input_size_location` : Location of the csvs containing the size and location of cells 
- `visualization` `input_prediction_location` : Location of the infered data 
- `visualization` `cache_location` : Location for cached data 
- `visualization` `output_location` : Location of the final results, volumes where each cell is colored according to their atlas region
- `FLAGS` : currently inactive 

## Pipeline overview
### Downsampling, Masking, Upsampling
Pseudocode:
```python
Artifact Masking
load stack as stack
stack = downsample(stack)
foreach image in stack:
    image = randomForest_model(image)
    upscale(image)
save(stack)
```

### Deep Learning inference
Pseudocode: 
```python
load stack as stack
load deepLearning_model as model
sliding_window_inferrer = Sliding_Window_Inferrer(model)
result = sliding_window_inferrer(stack)
save(result)
```

### Atlas alignment

(this requires to manually convert the downsampled stack from tiff to Vaa3d's V3DRAW format. For this, use the most recent version of Vaa3d https://github.com/Vaa3D/release/releases/ )

This step also requires mBrainAligner, https://github.com/Vaa3D/vaa3d_tools/tree/master/hackathon/mBrainAligner 
The pre-compiled swc registration only runs on Windows. 

Pseudocode:
```python
load downsampled_stack.V3DRAW as stack
preprocessed_cell_coordinates = preprocess_connected_component_output (cc_output)
transfomation_file = mBrainAligner_create_alignment (stack)
mBrainAligner_swc_transformation(preprocessed_cell_coordinates, transformation_file)
```

### Visualization
This step requires BrainRender https://github.com/brainglobe/brainrender 

Pseudocode: 
```python
load mBrainAligner_transformed_file as coords
filter coords (size max=104)
BrainRender.render(coords)
```
