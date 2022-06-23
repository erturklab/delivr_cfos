# basics
import os
import numpy as np
import nibabel as nib
from path import Path
from tqdm import tqdm
# import shutil
import time

# dl
import torch
from torch.utils.data import DataLoader

import monai
from monai.networks.nets import BasicUNet
from monai.data import list_data_collate
from monai.inferers import SlidingWindowInferer

from monai.transforms import RandGaussianNoised
from monai.transforms import (
    Compose,
    LoadImageD,
    AddChanneld,
    Lambdad,
    ToTensord,
)


def create_nifti_seg(
    threshold,
    onehot_model_outputs_CHWD,
    output_file,
    network_output_file,
):

    # generate segmentation nifti
    activated_outputs = (
        (onehot_model_outputs_CHWD[0][:, :, :].sigmoid()).detach().cpu().numpy()
    )

    binarized_outputs = activated_outputs >= threshold

    binarized_outputs = binarized_outputs.astype(np.uint8)

    segmentation_image = nib.Nifti1Image(binarized_outputs, np.eye(4))
    nib.save(segmentation_image, output_file)

    network_output_image = nib.Nifti1Image(activated_outputs, np.eye(4))

    nib.save(network_output_image, network_output_file)


# GO
def run_inference(
    niftis,
    output_folder,
    comment="none",
    model_weights="weights/inference_weights.tar",
    tta=True,
    threshold=0.5,
    cuda_devices="0,1",
    crop_size=(96, 96, 64),
    workers=0,
    sw_batch_size=42,
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

    # clean memory
    torch.cuda.empty_cache()

    # T R A N S F O R M S
    inference_transforms = Compose(
        [
            # << PREPROCESSING transforms >>
            LoadImageD(keys="images"),
            AddChanneld(keys="images"),
            Lambdad(["images", "label"], np.nan_to_num),
            ToTensord(keys="images"),
        ]
    )
    # D A T A L O A D E R
    dicts = list()

    for nifti in niftis:
        print("nifti:", nifti)
        nifti = Path(os.path.abspath(nifti))
        images = [nifti]

        the_dict = {
            "exam": nifti.name,
            "micro": nifti,
            "images": images,
        }

        dicts.append(the_dict)

    # datasets
    inf_ds = monai.data.Dataset(data=dicts, transform=inference_transforms)

    # dataloaders
    data_loader = DataLoader(
        inf_ds,
        batch_size=1,
        num_workers=workers,
        collate_fn=list_data_collate,
        shuffle=False,
    )

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
        sw_device="cuda",
        device="cpu",
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )

    # send model to device // very important for optimizer to work on CUDA
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # load
    model.load_state_dict(checkpoint)

    # epoch stuff
    session_name = checkpoint["session_name"]

    time_date = time.strftime("%Y-%m-%d_%H-%M-%S")
    print("start:", time_date)

    best_epoch = checkpoint["best_epoch"]
    best_accuracy = checkpoint["best_accuracy"]
    print(
        f"testing {session_name} at best_epoch: {best_epoch} with best_accuracy: {best_accuracy} on validation set"
    )

    testing_session_path = Path(
        os.path.abspath(
            output_folder + "/" + time_date + "_" + session_name + "_" + comment
        )
    )

    # meta_path = testing_session_path + "/meta"
    # os.makedirs(meta_path, exist_ok=True)

    netouts_path = testing_session_path + "/network_outputs/"
    os.makedirs(netouts_path, exist_ok=True)

    binaries_path = testing_session_path + "/binary_segmentations/"
    os.makedirs(binaries_path, exist_ok=True)

    # model_weights = Path(model_weights)
    # shutil.copyfile(model_weights, meta_path + "/" + model_weights.name)
    # checkpoint["model_state"] = model.module.state_dict()
    # torch.save(checkpoint, resumeCheckpointFile.name)

    # limit batch length?!
    batchLength = 0

    # eval
    with torch.no_grad():
        model.eval()
        # loop through batches
        for counter, data in enumerate(tqdm(data_loader, 0)):
            if batchLength != 0:
                if counter == batchLength:
                    break

            # get the inputs and labels
            # print(data)
            # inputs = data["images"].float()
            inputs = data["images"]

            outputs = inferer(inputs, model)

            # test time augmentations
            if tta == True:
                n = 1.0
                for _ in range(4):
                    # test time augmentations
                    _img = RandGaussianNoised(keys="images", prob=1.0, std=0.001)(data)[
                        "images"
                    ]

                    output = inferer(_img, model)
                    outputs = outputs + output
                    n = n + 1.0
                    for dims in [[2], [3]]:
                        flip_pred = inferer(torch.flip(_img, dims=dims), model)

                        output = torch.flip(flip_pred, dims=dims)
                        outputs = outputs + output
                        n = n + 1.0
                outputs = outputs / n

            print("inputs shape:", inputs.shape)
            print("outputs:", outputs.shape)
            print("data length:", len(data))
            print("outputs shape 0:", outputs.shape[0])

            # loop through elements in batch
            for element in range(outputs.shape[0]):
                # generate segmentation nifti
                output_file = (
                    binaries_path + str(data["exam"][element][:-7]) + ".nii.gz"
                )
                network_output_file = (
                    netouts_path + str(data["exam"][element][:-7]) + "_out.nii.gz"
                )

                onehot_model_output = outputs[element]
                create_nifti_seg(
                    threshold=threshold,
                    onehot_model_outputs_CHWD=onehot_model_output,
                    output_file=output_file,
                    network_output_file=network_output_file,
                )

                print("the time:", time.strftime("%Y-%m-%d_%H-%M-%S"))

    print("end:", time.strftime("%Y-%m-%d_%H-%M-%S"))
    return testing_session_path
