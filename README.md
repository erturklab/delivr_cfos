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
We assume that the files are a series of 16-bit TIFF files, one per z-plane. Our model is optimized for data acquired with approx. 1.62 x 1.62 x 6 µm (x/y/z/) voxel size. 

## Setup and Execution 
1. Clone this repository using `git clone https://github.com/erturklab/deliver_cfos.git`
2. Install the requirements using `pip install -r requirements.txt` (pip) `conda install --file requirements.txt` (anaconda)
3. Set the location of your raw data and ilastik installation path in `config.json`
4. Download and install mBrainAligner https://github.com/Vaa3D/vaa3d_tools/tree/master/hackathon/mBrainAligner (Note that the pre-compiled swc registration only runs on Windows). Set the path in the config.json under `mBrainAligner location`
5. Download and install Ilastik (latest beta) https://www.ilastik.org/download.html. Set the path in the config.json under `ilastik_location`. 
6. Download the Allen Brain Atlas CCF3 version we used, and place it in the `/models/` folder. Use the following dropbox link: https://www.dropbox.com/s/inxmi2qv3sgzz50/CCF3_P56_annotation.tif?dl=0 
7. (Optional: Download the test raw data from this dropbox link (50 Gb total) and place the tiff files into `/data/raw/`: https://www.dropbox.com/sh/k3y7h0yovrsoz01/AABVodOqGKMdswRbV6DGhdUBa?dl=0 )
8. (If required, download the intermediate results and move them to into `/data/`)
9. Run `python __main__.py` in the terminal

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
- `atlas_alignment` `mBrainAligner_location` : Location of the mBrainAligner folder
- `atlas_alignment` `collection_folder` : Location where the atlas-aligned cell coordinates will be saved to 
- `atlas_alignment` `output_location` : Location where the csvs containing the atlas region of each cells will be saved to
- `region_assignment` `input_location` : Location where the atlas-aligned cell coordinates will be loaded from 
- `region_assignment` `CCF3_atlasfile` : Location of the CCF3 atlas file used to assign regions to each cell 
- `region_assignment` `CCF3_ontology` : Location of the ontology file providing the region names, IDs, and hierarchical structure 
- `region_assignment` `output_location` : Location of the csvs containing the atlas region of each cell will be saved to  
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

### Region Assignment 
DELiVR uses the Allen Brain Atlas CCF3 adult mouse brain atlas, as provided by the Scalable Brain Atlas https://scalablebrainatlas.incf.org/mouse/ABA_v3 
This step loads the atlas-aligned cell coordinates, assigns a region (and parent region, atlas color, etc) to each of them. It also produces a per-area summary table (cells/ region) and a 3D heatmap in atlas space. 

Pseudocode:
```python
for brain in brains:
    cell_region_table = assign_regions_to_each_cell(brain)
    region_overview_table = summarize_cell_counts_per_region(cell_region_table)
    create_indivisual_brain_heatmap(brain)
    
summarize_overview_table(all_overview_tables)
summarize_heatmaps(all_heatmaps)
```


### Visualization
This step requires BrainRender https://github.com/brainglobe/brainrender 

Pseudocode: 
```python
load mBrainAligner_transformed_file as coords
filter coords (size max=104)
BrainRender.render(coords)
```
