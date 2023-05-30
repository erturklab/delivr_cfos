# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option
from monai.inferers.inferer import Inferer
from monai.transforms import RandGaussianNoise







__all__ = ["sliding_window_inference"]


def sliding_window_inference(
    inputs: np.array,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    SIGMOID : bool = False,
    output_image: torch.tensor = None,
    count_map: torch.tensor = None,
    tta: bool = None,
    flip_dim: int = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window inference on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """


    #to inference.py

    num_spatial_dims = len(inputs.shape) - 2

    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size = list(inputs.shape[2:])
    batch_size = inputs.shape[0]
    
    
    roi_size = fall_back_tuple(roi_size, image_size)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    #print(f"Padding {inputs.shape} with {pad_size}, converting to float...")
    # inputs = inputs.float()
    # inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)
    np_pad = []
    for i, j in enumerate(inputs.shape):
         np_pad.append((pad_size[i], pad_size[i]))
    
    np_pad = tuple(np_pad)
    
    #this loads the entire input into RAM, though not necessary in most cases    
    #if max(max(np_pad)) > 0:
    #    inputs = np.pad(inputs, pad_width=np_pad, mode="reflect")
    #    print(f"After padding {np_pad} : {inputs.shape}")
    
    
    #moved to inference, still needed here
    #print("Padding done, converting down...")

    #with (64,64,32) roi_size this works out to (32,32,16)
    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows
    
    # Create window-level importance map
    importance_map = compute_importance_map(get_valid_patch_size(image_size, roi_size), mode='constant', sigma_scale=sigma_scale)

    # Perform predictions
    print("Inferring...")
    
    
    #create empty output tensors 
    #output_image, count_map = torch.tensor(0.0, dtype=torch.float16, device=device), torch.tensor(0.0, dtype=torch.float16, device=device)
    
    #Set initialized to False to trigger initialization on first run 
    _initialized = True
    
    #calculate the amount of slices in this batch (for "inferring... x / 142" progress report)
    slice_l = len(list(range(0, total_slices, sw_batch_size)))
    

    #split total slices into number of sw_batches (slice_i) and number of slices within the total_slices (slice_g), 
    #then run inference on each of those slices 
    for slice_i, slice_g in enumerate(range(0, total_slices, sw_batch_size)):
        print(f"{slice_i}/{slice_l}",end="\r",flush=True)

        #create the range of slice numbers within current sw_batch
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))

        #pick out the (indices of) the slices from among the window list 
        #unravel_slice = [
        #    [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
        #    for idx in slice_range
        #]

        unravel_slice = []
        for idx in slice_range:
            slice_block = [[list([each_slice.start , each_slice.stop]) for each_slice in slices[idx % num_win]]]
            unravel_slice += slice_block
        

        #load the data as subset from the main input dataset (that should be a tensor by now) and concatenate all sw_batch slices into one tensor
        #load data 
        data_to_load = []
        for win_slice in unravel_slice:
            #try to load selectively from disk
            single_slice_load = inputs[:,:,win_slice[0][0]:win_slice[0][1],win_slice[1][0]:win_slice[1][1],win_slice[2][0]:win_slice[2][1]].copy()
            #cast as signed 32-bit integer for conversion to 32-bit float
            single_slice_load = single_slice_load.astype(np.int32)
            #cast to tensor 
            single_slice_load = torch.as_tensor(single_slice_load,dtype=torch.int32)
            #add extra dimensions at beginning
            single_slice_load = single_slice_load[None,:,:,:]
            #single_slice_load = single_slice_load[None,:,:,:,:]
            data_to_load += single_slice_load
        
        #concatenate all win_slices to a single batch (influenced by sw_batch_size, set according to on the graphics card VRAM)
        window_data = torch.cat(data_to_load,dim=0)

        #skip computing if this tile is background (as filtered and set to 0 by the mask_detection step)
        if window_data.max() == 0.0:
            seg_prob = window_data

        #if this tile contains data, process it: 
        else: 
            #cast as 32-bit float and send to graphics card 
            window_data = window_data.type(torch.float32)     
            window_data = window_data.cuda()
            window_data.to(sw_device)

            #add noise if running with test-time augmentation
            if tta == True:
                #window_data = RandGaussianNoise(prob=1.0, std=0.001)(window_data)
                window_data[0,0,:,:,:] = window_data[0,0,:,:,:] + (0.001**0.5)*torch.randn(size=window_data[0,0,:,:,:].shape,out=window_data[0,0,:,:,:],dtype=torch.float32,device=sw_device)

            # if the data needs to be flipped, do this here (flip_dim: 2 = z, 3 = y, 4 = x) 
            if flip_dim is not None:
                window_data = torch.flip(window_data,dims=[flip_dim])

            #run the actual prediction 
            seg_prob = predictor(window_data, *args, **kwargs).to(device)  # batched patch segmentation
            
            if SIGMOID:
                seg_prob = torch.sigmoid(seg_prob)

            # flip data back if previously flipped 
            if flip_dim is not None:
                seg_prob = torch.flip(seg_prob,dims=[flip_dim])
            
            #send to cpu
            #seg_prob.to(device)

            #cast the results back to float16 
            seg_prob = seg_prob.to(torch.float16)
        
            # store the result in the proper location of the full output. Apply weights from importance map. (skip this if it is all background) 
            for idx, original_idx in zip(slice_range, unravel_slice):
                #print("original_idx: ",original_idx)
                #print("idx: ", idx)
                #print("unravel_slice: ",unravel_slice)            
                #print("output_image[original_idx]: ",output_image[original_idx])
                #print("importance_map shape: ",importance_map.shape)
                #print("seg_prob shape: ",seg_prob.shape)
                #print("current seg_prob shape: ",seg_prob[idx - slice_g].shape)
                output_image[:,:,original_idx[0][0]:original_idx[0][1],original_idx[1][0]:original_idx[1][1],original_idx[2][0]:original_idx[2][1]] += importance_map * seg_prob[idx - slice_g]
                count_map[:,:,original_idx[0][0]:original_idx[0][1],original_idx[1][0]:original_idx[1][1],original_idx[2][0]:original_idx[2][1]] += importance_map # directly replaces the values 

    # account for any overlapping sections (this should happen in inference.py at the end of all inferences)
    #output_image = output_image / count_map 
    

    #generate slice placement list (I think)
    
    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image.shape):
        final_slicing.insert(0, slice(None))
    return output_image[final_slicing]
    

def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)
    
class SlidingWindowInferer(Inferer):
    """
    Sliding window method for model inference,
    with `sw_batch_size` windows for every model.forward().
    Usage example can be found in the :py:class:`monai.inferers.Inferer` base class.

    Args:
    roi_size: the window size to execute SlidingWindow evaluation.
        If it has non-positive components, the corresponding `inputs` size will be used.
        if the components of the `roi_size` are non-positive values, the transform will use the
        corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
        to `(32, 64)` if the second spatial dimension size of img is `64`.
    sw_batch_size: the batch size to run window slices.
    overlap: Amount of overlap between scans.
    mode: {``"constant"``, ``"gaussian"``}
        How to blend output of overlapping windows. Defaults to ``"constant"``.

        - ``"constant``": gives equal weight to all predictions.
        - ``"gaussian``": gives less weight to predictions on edges of windows.

    sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
        Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
        When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
        spatial dimensions.
    padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
        Padding mode when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
        See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
    cval: fill value for 'constant' padding mode. Default: 0
    sw_device: device for the window data.
        By default the device (and accordingly the memory) of the `inputs` is used.
        Normally `sw_device` should be consistent with the device where `predictor` is defined.
    device: device for the stitched output prediction.
        By default the device (and accordingly the memory) of the `inputs` is used. If for example
        set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
        `inputs` and `roi_size`. Output is on the `device`.

    Note:
        ``sw_batch_size`` denotes the max number of windows per network inference iteration,
        not the batch size of inputs.

    """

    def __init__(
        self,
        roi_size: Union[Sequence[int], int],
        sw_batch_size: int = 1,
        overlap: float = 0.25,
        mode: Union[BlendMode, str] = BlendMode.CONSTANT,
        sigma_scale: Union[Sequence[float], float] = 0.125,
        padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
        cval: float = 0.0,
        sw_device: Union[torch.device, str, None] = None,
        device: Union[torch.device, str, None] = None,
    ) -> None:
        Inferer.__init__(self)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode: BlendMode = BlendMode(mode)
        self.sigma_scale = sigma_scale
        self.padding_mode = padding_mode
        self.cval = cval
        self.sw_device = sw_device
        self.device = device

    def __call__(
        self, inputs: torch.Tensor, network: Callable[..., torch.Tensor], *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """

        Args:
            inputs: model input data for inference.
            network: target model to execute inference.
                supports callables such as ``lambda x: my_torch_model(x, additional_config)``
            args: optional args to be passed to ``network``.
            kwargs: optional keyword args to be passed to ``network``.

        """
        return sliding_window_inference(
            inputs,
            self.roi_size,
            self.sw_batch_size,
            network,
            self.overlap,
            self.mode,
            self.sigma_scale,
            self.padding_mode,
            self.cval,
            self.sw_device,
            self.device,
            *args,
            **kwargs,
        )
