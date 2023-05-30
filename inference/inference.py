# basics
import os
import numpy as np
import nibabel as nib
from path import Path
from tqdm import tqdm
from skimage.util import view_as_windows
# import shutil
import time
#import psutil

# dl
import torch

import monai
from monai.networks.nets import BasicUNet
from .sliding_window_inferer import SlidingWindowInferer

from monai.transforms import RandGaussianNoise

def update_idx (old_idx,new_idx,total_size):
    #update both old_idx and new_idx, taking into account which dimensions could change
    for i in range(len(old_idx)):
        if new_idx[i] < total_size[i]:
            new_idx[i] += old_idx[i]
        #in case one dimension is completely filled, make sure that old_idx starts from beginning
        if old_idx[i] == total_size[i]:
            old_idx[i] = 0
    return old_idx, new_idx

def create_nifti_seg(
    threshold,
    model_output,
    output_file,
    network_output_file,
):
    
    #save activated network output as npy, then re-read
    np.save(network_output_file,np.zeros(shape=model_output.shape[1:], dtype=np.float32))
    activated_outputs = np.memmap(network_output_file,mode='w+',dtype=np.float32,shape=model_output.shape[1:])
    
    #save binarized network output as npy, then re-read
    np.save(output_file,np.zeros(shape=model_output.shape[1:], dtype=np.uint8))
    binarized_outputs = np.memmap(output_file,mode='w+',dtype=np.uint8,shape=model_output.shape[1:])
    
    #construct iterator over output_image array. Tweak buffer size for larger blocks
    img_iterator = np.lib.Arrayterator(model_output[0,0,:,:,:], 1000**3)
    
    #construct indices for placing the resulting values back
    idx = [0,0,0]
    old_idx = idx
    
    #iterate through model_output
    for subarr in img_iterator:
        #generate indices for placing the output in the correct array locations
        idx = list(subarr.shape)
        old_idx, idx = update_idx(old_idx,idx,list(img_iterator.shape))
        #apply sigmoid function to subarray
        subarr = torch.as_tensor(subarr,dtype=torch.float)
        sigmoid = (subarr[:, :, :].sigmoid()).detach().cpu().numpy()  
        #export to activated_outputs array
        activated_outputs[0,old_idx[0]:idx[0],old_idx[1]:idx[1],old_idx[2]:idx[2]] = sigmoid
        #threshold and output export to binarized_outputs array
        thresholded_sigmoid = sigmoid >= threshold
        binarized_outputs[0,old_idx[0]:idx[0],old_idx[1]:idx[1],old_idx[2]:idx[2]] = thresholded_sigmoid.astype(np.uint8)
        #update index 
        old_idx = idx
    
    #ensure exported arrays are written to disk 
    activated_outputs.flush()
    binarized_outputs.flush()


def create_empty_memmap (file_location, shape,dtype=np.uint16,return_torch=True):
    #creates a zeroed-out npy file with shape=shape on the disk, returns as torch tensor view. dtype is float16. 
    #first make sure previous temp files are erased 
    try:                
        os.remove (file_location)
    except:
        pass
    #save empty np array on disk 
    np.save(file_location,np.zeros(shape=shape, dtype=dtype))
    #now re-read from disk as memmap (saving RAM)
    empty_memmap = np.memmap(file_location,mode='w+',dtype=dtype,shape=shape)
    if return_torch:
        empty_memmap = torch.as_tensor(empty_memmap,dtype=torch.float16)
    return empty_memmap


# GO
def run_inference(
    niftis,
    output_folder,
    stack_shape,
    comment="none",
    model_weights="weights/inference_weights.tar",
    tta=False,
    threshold=0.5,
    cuda_devices="0,1",
    crop_size=(64,64, 32),
    workers=0,
    sw_batch_size=30, 
    overlap=0.5,
    verbosity=True,
):
    """
    call this function to run the sliding window inference.

    Parameters:
    niftis: list of nifti files to infer
    comment: string to comment
    model_weights: Path to the model weights
    tta: whether to run test time augmentations
    threshold: threshold for binarization of the network outputs. Greater than <theshold> equals foreground
    cuda_devices: which cuda devices should be used for the inference.
    crop_size: crop size for the inference
    workers: how many workers should the data loader use
    sw_batch_size: batch size for the sliding window inference
    overlap: overlap used in the sliding window inference

    see the above function definition for meaningful defaults.
    """
    # ~~<< S E T T I N G S >>~~
    # torch.multiprocessing.set_sharing_strategy("file_system")

    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    multi_gpu = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # clean memory
    torch.cuda.empty_cache()
    
    #smartly set sw_batch_size to occupy available VRAM 
    #available_devices = torch.cuda.device_count()
    #available_mem = available_devices*torch.cuda.mem_get_info()[0]
    available_mem = np.sum([torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())])
    available_mem = int(available_mem*0.95) #safety margin

    #convert to MB 
    available_mem = available_mem/(1024**2)
    #230 MB is empirically determined mem requirement for sw_crop (64,64,32)
    empirical_sw_batch_size = int(round(available_mem / 230))
    sw_batch_size = empirical_sw_batch_size

    # T R A N S F O R M S
    # datasets
    dataset_on_disk = np.memmap(niftis[0],dtype=np.uint16,mode='r+',shape=stack_shape)
    dataset = dataset_on_disk
    #dataset = torch.as_tensor(dataset_on_disk)

    # ~~<< M O D E L >>~~
    model = BasicUNet(
        dimensions=3,
        in_channels=1,
        out_channels=1,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
        act="mish",
    )

    model_weights = Path(os.path.abspath(model_weights))
    checkpoint = torch.load(model_weights, map_location="cpu")

    # inferer
    patch_size = crop_size

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        sw_device= torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        device=torch.device('cpu'),
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )

    # send model to device // very important for optimizer to work on CUDA
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # load
    model.load_state_dict(checkpoint["model_state"])

    ### DATA PREP ###
    #create output folder if not already present:
    #try to create output folder in case it's not there yet     
    os.makedirs(os.path.join(output_folder, comment), exist_ok=True)

    #create empty output tensors (memmapped npy underneath) 
    output_image = create_empty_memmap (file_location = os.path.join(output_folder, comment,"inference_output.npy"), shape = dataset_on_disk.shape,dtype=np.float16)
    count_map = create_empty_memmap (file_location = os.path.join(output_folder, comment ,"count_map.npy"), shape = dataset_on_disk.shape,dtype=np.float16)
    
    #if already part-way done, load results from disk:
    #output_image = torch.as_tensor(np.memmap(os.path.join(output_folder,comment ,"inference_output.npy"), mode = 'r+', dtype = np.float16, shape = dataset_on_disk.shape),dtype=torch.float16)
    #count_map = torch.as_tensor(np.memmap(os.path.join(output_folder,comment ,"count_map.npy"), mode = 'r+', dtype = np.float16, shape = dataset_on_disk.shape),dtype=torch.float16)

    # epoch stuff
    time_date = time.strftime("%Y-%m-%d_%H-%M-%S")
    print("start:", time_date)

    testing_session_path = Path(
        os.path.abspath(output_folder + "/" + comment)
    )

    netouts_path = testing_session_path + "/network_outputs/"
    os.makedirs(netouts_path, exist_ok=True)

    binaries_path = testing_session_path + "/binary_segmentations/"
    os.makedirs(binaries_path, exist_ok=True)

    # limit batch length?!
    batchLength = 0

    # eval
    with torch.no_grad():
        model.eval()
        #run sliding window inference 
        
        output_image = inferer(dataset, model, output_image = output_image, count_map = count_map)
        #print("first inference finished")           
         

        # test time augmentations
        if tta == True:
            for _ in range(4):
                #re-run inferer while creating noised data on the fly
                output_image = inferer(dataset, model, output_image = output_image, count_map = count_map, tta = True)
                
                #flip Z 
                output_image = inferer(dataset, model, output_image = output_image, count_map = count_map, tta = True, flip_dim = 2)
                
                #create an empty count map again
                output_image = inferer(dataset, model, output_image = output_image, count_map = count_map, tta = True, flip_dim = 3)
                
    #average output of all runs (even if just 1, count_map contains all data)
    #do it in memory if enough memory is present 
    #output_image = output_image / count_map    
    
    #do averaging block-wise as arrays from disk - runs with RAM < 2x input image as well 
    #construct iterator over output_image array. Tweak buffer size for larger blocks
    ouput_iterator = np.lib.Arrayterator(output_image[0,0,:,:,:], 1000**3)
    #construct indices for placing the resulting values back
    idx = [0,0,0]
    old_idx = idx
    #iterate through output_image
    for subarr in ouput_iterator:
        #generate indices for placing the output in the correct array locations
        idx = list(subarr.shape)
        old_idx, idx = update_idx(old_idx,idx,list(ouput_iterator.shape))
        #divide output image by count_map at same location 
        subarr = subarr / count_map[0,0,old_idx[0]:idx[0],old_idx[1]:idx[1],old_idx[2]:idx[2]]
        #place back in output_image
        output_image[0,0,old_idx[0]:idx[0],old_idx[1]:idx[1],old_idx[2]:idx[2]] = subarr
        #update index 
        old_idx = idx
    #ensure updated arrays are written to disk 
    output_image.flush()

    #delete the count_map (not required in any case, the rest is kept if the flag "SAVE_NETWORK_OUTPUT":true is set in config.json
    os.remove(os.path.join(output_folder, comment ,"count_map.npy"))
    # generate segmentation nifti
    output_file         = os.path.join(binaries_path,"binaries.npy")
    network_output_file = os.path.join(binaries_path,"network_output.npy")

    create_nifti_seg(
        threshold=threshold,
        model_output=output_image,
        output_file=output_file,
        network_output_file=network_output_file,
    )

    print("end:", time.strftime("%Y-%m-%d_%H-%M-%S"))
    return testing_session_path
