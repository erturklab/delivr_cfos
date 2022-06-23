import os
import datetime
import cc3d
import pickle
import cv2
import numpy as np
import pandas as pd
from util_pkg.filehandling import read_nifti, write_nifti
from skimage.morphology import binary_dilation
from skimage.draw import ellipsoid

def fast_dilate(img, kernel_size):
    kernel = np.ones((kernel_size,kernel_size), np.uint8)
    for x in range(img.shape[0]):
        img[x,:,:] = cv2.dilate(img[x,:,:], kernel, iterations=1)
    for y in range(img.shape[1]):
        img[:,y,:] = cv2.dilate(img[:,y,:], kernel, iterations=1)
    # for z in range(img.shape[-1]):
    #     img[:,:,z] = cv2.dilate(img[:,:,z], kernel, iterations=1)
    # img = binary_dilation(img)
    return img

def check_brain(path, brain):
    res = -1
    for item in os.listdir(path):
        if brain in item and "cc3d" in item:
            res = item
    return res

def color_sphere(brain, components, stats, item, color):
    assert(brain.shape == components.shape)
    x_0, x_1 = int(stats["bounding_boxes"][item][0].start), int(stats["bounding_boxes"][item][0].stop)
    y_0, y_1 = int(stats["bounding_boxes"][item][1].start), int(stats["bounding_boxes"][item][1].stop)
    z_0, z_1 = int(stats["bounding_boxes"][item][2].start), int(stats["bounding_boxes"][item][2].stop)

    size = 8
    el = ellipsoid(size/ (6 / 1.63), size, size ).astype(np.uint8)


    if el.shape[0] > x_1 - x_0:
        x_1 = x_0 + el.shape[0]
    if el.shape[1] > y_1 - y_0:
        y_1 = y_0 + el.shape[1]
    if el.shape[2] > z_1 - z_0:
        z_1 = z_0 + el.shape[2]

    sub_p = components[x_0:x_1, y_0:y_1, z_0:z_1]

    el *= color
    #TODO Messy: will not be centered
    sub_p[0:el.shape[0],0:el.shape[1],0:el.shape[2]] = el

    brain[x_0:x_1, y_0:y_1, z_0:z_1] = sub_p

    return brain

def color_bb(brain, components, stats, item, color):
    assert(brain.shape == components.shape)
    x_0, x_1 = int(stats["bounding_boxes"][item][0].start), int(stats["bounding_boxes"][item][0].stop)
    y_0, y_1 = int(stats["bounding_boxes"][item][1].start), int(stats["bounding_boxes"][item][1].stop)
    z_0, z_1 = int(stats["bounding_boxes"][item][2].start), int(stats["bounding_boxes"][item][2].stop)
    try:
        sub_p = components[x_0:x_1, y_0:y_1, z_0:z_1]
        sub_p[np.where(sub_p  == item)] = color
    except TypeError as te:
        print(f"{te} where")
        exit()

    try:
        brain[x_0:x_1, y_0:y_1, z_0:z_1] = sub_p
    except TypeError as te:
        print(f"{te} brain")
        exit()

    return brain

def color_bb_rgb(brain, components, stats, item, color):
    try:
        x_0, x_1 = 1 + int(stats["bounding_boxes"][item][0].start), int(stats["bounding_boxes"][item][0].stop)
        y_0, y_1 = 1 + int(stats["bounding_boxes"][item][1].start), int(stats["bounding_boxes"][item][1].stop)
        z_0, z_1 = 1 + int(stats["bounding_boxes"][item][2].start), int(stats["bounding_boxes"][item][2].stop)
        sub_p = components[x_0:x_1, y_0:y_1, z_0:z_1]
        sub_p = np.stack((sub_p, sub_p, sub_p))
        sub_p = np.swapaxes(sub_p, 0, -1)
        sub_p = np.swapaxes(sub_p, 0, 1)
        sub_p = np.swapaxes(sub_p, 1, 2)
        coords = tuple(zip(*np.where(sub_p == item)))
        for coord in coords:
            if coord[-1] == 0:
                sub_p[coord] = color[0]
            elif coord[-1] == 1:
                sub_p[coord] = color[1]
            elif coord[-1] == 2:
                sub_p[coord] = color[2]
        brain[x_0:x_1, y_0:y_1, z_0:z_1,:] = sub_p
    except IndexError as ie:
        print(f"{ie} Brain {brain.shape} Components {components.shape} subp {sub_p.shape} {color}")
    except ValueError as ve:
        print(f"{ve} Brain {brain.shape} Components {components.shape} subp {sub_p.shape} {color}")

    return brain

def get_color_set(df):
    """Get df, select # unique RGB triplets, return list of triplets to color according to list
    """
    return sorted(list(df["color-hex-triplet"].unique()))

def blob_highlighter(settings):
    """Color blobs by their corresponding atlas region
    """

    path_binary         = settings["visualization"]["input_prediction_location"]
    if settings["visualization"]["input_size_location"] == "":
        settings["visualization"]["input_size_location"] = settings["postprocessing"]["output_location"]
    path_size_csv       = settings["visualization"]["input_size_location"]

    path_cell_csv       = settings["visualization"]["input_csv_location"]

    path_out            = settings["visualization"]["output_location"]
    path_cache          = settings["visualization"]["cache_location"]

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    if not os.path.exists(path_cache):
        os.mkdir(path_cache)

    # TODO Highlight matched areas based on csv
    brain_list = [
                    ("PBS_3418",""),("PBS_3402",""),("PBS_3400",""),\
                    ("c26_3409",""),\
                    ("nc26_3413",""),("nc26_3423","")
                ]
    RGB = False

    draw_ellipsoid = True

    for brain_item in brain_list:
        brain = brain_item[0]
        highlight_area = brain_item[1]
        dilation_kernel = 4
        if highlight_area != "":
            print(f"{datetime.datetime.now()} Highlighting {highlight_area} in {brain}")
        else:
            print(f"{datetime.datetime.now()} Highlighting everything in {brain}")

        path_brain_binary       = path_binary + brain + ".nii.gz"
        path_brain_size_csv           = path_size_csv + brain + ".csv"
        path_brain_cell_csv     = path_cell_csv + brain + ".csv"

        # Load csvs
        print(f"{datetime.datetime.now()} : Loading csv")
        size_csv    = pd.read_csv(path_brain_size_csv).set_index("Blob")
        cell_csv    = pd.read_csv(path_brain_cell_csv).set_index("connected_component_id")

        # Load brain
        print(f"{datetime.datetime.now()} : Loading brain")
        brain_nifti         = read_nifti(path_brain_binary)
        brain_nifti         = np.swapaxes(brain_nifti, 0, -1)

        # CC3D
        print(f"{datetime.datetime.now()} : CC3D")
        brain_exists = check_brain(path_cache, brain)
        if brain_exists != -1:
            print(f"{datetime.datetime.now()} : CC3D exists, reading...")
            labels, N = read_nifti(f"{path_cache}{brain_exists}"), int(brain_exists.split("_")[2])
        else:
            print(f"{datetime.datetime.now()} : Calculating CC3D")
            labels, N   = cc3d.connected_components(brain_nifti, return_N=True)
            print(f"{datetime.datetime.now()} : Writing cc3d nifti")
            write_nifti(path_cache + f"{brain}_{N}_cc3d.nii", labels)

        # Make np zero same size as brain
        print(f"{datetime.datetime.now()} : Creating empty volumes")
        if RGB:
            cell_nifti      = np.zeros_like(labels, dtype = np.uint8)
            cell_nifti      = np.stack((cell_nifti, cell_nifti, cell_nifti))
            cell_nifti      = np.swapaxes(cell_nifti, 0, -1)
            cell_nifti      = np.swapaxes(cell_nifti, 0, 1)
            cell_nifti      = np.swapaxes(cell_nifti, 1, 2)
        else:
            cell_nifti      = np.zeros_like(labels, dtype = np.uint8)



        if not os.path.exists(f"{path_cache}{brain}_statistics.pickledump"):
            print(f"{datetime.datetime.now()} : Calculating statistics")
            stats = cc3d.statistics(labels)
            with open(f"{path_cache}{brain}_statistics.pickledump", "xb") as file:
                pickle.dump(stats, file)
        else:
            print(f"{datetime.datetime.now()} : Reading statistics")
            with open(f"{path_cache}{brain}_statistics.pickledump", "rb") as file:
                stats = pickle.load(file)

        key_error_count = 0
        print(f"{datetime.datetime.now()} : Coloring")
        color_set = get_color_set(cell_csv)
        print(f"Len colorset {len(color_set)}")
        with open(f"{path_cache}{brain}_color_set.txt", "w") as file:
            file.write(str(color_set))


        print(f"Cell nifti {cell_nifti.shape}")
        print(f"labels {labels.shape}")
        for label in range(1,N):
            start = datetime.datetime.now()
            if label > 0:
                try:
                    size    = size_csv.loc[label]["Size"]
                    location= size_csv.loc[label]["Coords"]
                    if size < 200:
                        if RGB:
                            color_cell = [-1, -1, -1]
                            color_cell[0] = int(cell_csv.loc[label]["red"])
                            color_cell[1] = int(cell_csv.loc[label]["green"])
                            color_cell[2] = int(cell_csv.loc[label]["blue"])
                            cell_nifti = color_bb_rgb(cell_nifti, labels, stats, label, color_cell)
                        else:
                            color_cell = color_set.index(cell_csv.loc[label]["color-hex-triplet"])
                            if highlight_area != "":
                                if highlight_area == cell_csv.loc[label]["color-hex-triplet"]:
                                    if draw_ellipsoid:
                                        cell_nifti = color_sphere(cell_nifti, labels, stats, label, color_cell)
                                    else:
                                        cell_nifti = color_bb(cell_nifti, labels, stats, label, color_cell)
                            else:
                                if draw_ellipsoid:
                                    cell_nifti = color_sphere(cell_nifti, labels, stats, label, color_cell)
                                else:
                                    cell_nifti = color_bb(cell_nifti, labels, stats, label, color_cell)
                except KeyError as ke:
                    # if "key_error_count" in locals():
                    key_error_count += 1
                    if label == 1:
                        print("No first label found - Big oopsie! Loc vs iLoc?")
                        exit()
                    # print(f"\nKey Error {ke} Count {key_error_count}",end="\r",flush=True)
                except TypeError as te:
                    print(f"\nType Error {te} {label}")
                except ValueError as ve:
                    print(f"\nValue Error {ve} {label}")
                except IndexError as ie:
                    print(f"\nIndex Error {ie} {label}")

            if label % 50 == 0:
                d = datetime.datetime.now()
                print(f"{label} / {N} | KeyErrors: {key_error_count} ETA {((N - label) * (d-start).total_seconds())/60:.2f} Minutes remaining",end="\r",flush=True)
        cell_nifti = np.swapaxes(cell_nifti, 0, 2)

        print(f"{datetime.datetime.now()} : Writing cell nifti {np.amin(cell_nifti)} {np.amax(cell_nifti)}")
        output_name = f"{path_out}{brain}"
        if RGB:
            output_name += "_rgb"
        else:
            if dilation_kernel > 0:
                cell_nifti = fast_dilate(cell_nifti, dilation_kernel)
                output_name += "_dilated"
            output_name += "_singlecolor"
            if highlight_area != "":
                output_name += f"_{highlight_area}"

        if draw_ellipsoid:
            output_name += "_ellipsoid"
        output_name += ".nii"
        write_nifti(output_name, cell_nifti)
