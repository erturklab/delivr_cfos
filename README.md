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

## Setup and Execution
1. Clone this repository using `git clone https://github.com/erturklab/deliver_cfos.git`
2. Install the requirements using `pip install -r ./inference/inference_requirements.txt` (pip) `conda install --file ./inference/inference_requirements.txt` (anaconda)
3. Set the location of your raw data in `config.json` under `raw_location`
4. Set the mask detection parameters accordingly
5. Run `python __main__.py` in the terminal

