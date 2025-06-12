#!/usr/bin/python
# -*- coding: utf-8 -*-

from altair import sample
import pandas as pd
import numpy as np
import os
from cProfile import label
from re import S, split
from spectral import envi
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from zmq import NULL

# get .hdr and .os files
def get_hdr_os_folders_files(root_folder):
    hdr_os_folders_files = []
    for subdir, _, files in os.walk(root_folder):
        parent_dir = os.path.basename(os.path.dirname(subdir))
        hdr_files = set()
        os_files = set()
        for file in files:
            if file.endswith('.hdr'):
                hdr_files.add(os.path.splitext(file)[0])
            elif file.endswith('.os'):
                os_files.add(os.path.splitext(file)[0])
        common_files = hdr_files & os_files
        for base_name in common_files:
            hdr_path = os.path.join(subdir, base_name + '.hdr')
            os_path = os.path.join(subdir, base_name + '.os')
            folder_name = os.path.basename(subdir)
            hdr_os_folders_files.append((parent_dir, hdr_path, os_path))

    return hdr_os_folders_files

# calcualte reflectance hyperspectrum
def reflc_hypspec_calc(grain_hdr, grain_os, white_hdr, white_os, reflc_hdr, reflc_os):

    # get grain's hyperspectrum
    spe_grain = envi.open(grain_hdr, grain_os)
    grain_hypspec = spe_grain.open_memmap(interleave='bsq')
    b_g, r_g, c_g = grain_hypspec.shape
    grain_hypspec = np.array(grain_hypspec, dtype=grain_hypspec.dtype)
    print(grain_hypspec.shape)

    # get whiteboard's hyperspectrum
    spe_white = envi.open(white_hdr, white_os)
    whiteboard_hypspec = spe_white.open_memmap(interleave='bsq')
    b_w, r_w, c_w = whiteboard_hypspec.shape
    whiteboard_hypspec = np.array(whiteboard_hypspec, dtype=whiteboard_hypspec.dtype)
    print(whiteboard_hypspec.shape)

    if r_g != r_w:

        # calculate reflectance hyperspecturm
        src_dtype = grain_hypspec.dtype
        reflc_hypspec = np.random.rand(grain_hypspec.shape[0], min(r_g, r_w), grain_hypspec.shape[2])
        reflc_hypspec = grain_hypspec[:, :min(r_g, r_w), :] / whiteboard_hypspec[:, :min(r_g, r_w), :]
        reflc_hypspec = 100 * reflc_hypspec
        reflc_hypspec[np.where(reflc_hypspec < 0)] = 0
        reflc_hypspec[np.isnan(reflc_hypspec)] = 0
        reflc_hypspec[np.isinf(reflc_hypspec)] = 0
        reflc_hypspec = np.array(reflc_hypspec, dtype=src_dtype)

                
        # abstract reflectance hyperspectrum's ROI
        band_subset = reflc_hypspec[99 : 200, :, :]
        mean_value = np.mean(band_subset, axis = 0)
        mask = (mean_value > 90) | (mean_value < 25)
        reflc_hypspec[:, mask] = 0  

        # wirte in .hdr and .os
        with open(reflc_os, 'wb+') as f:
            f.write(reflc_hypspec.tobytes())
        if src_dtype == np.float64:
            spe_grain.metadata['data type'] = 5
        elif src_dtype == np.uint16:
            spe_grain.metadata['data type'] = 12
        metadata = spe_grain.metadata
        metadata['interleave'] = 'bsq'
        envi.write_envi_header(reflc_hdr, metadata)

        return reflc_hypspec

    else:

        # calculate reflectance hyperspecturm
        src_dtype = grain_hypspec.dtype
        reflc_hypspec = grain_hypspec / whiteboard_hypspec
        reflc_hypspec = 100 * reflc_hypspec
        reflc_hypspec[np.where(reflc_hypspec < 0)] = 0
        reflc_hypspec[np.isnan(reflc_hypspec)] = 0
        reflc_hypspec[np.isinf(reflc_hypspec)] = 0
        reflc_hypspec = np.round(reflc_hypspec).astype(int)
        reflc_hypspec = np.array(reflc_hypspec, dtype=src_dtype)

                
        # abstract reflectance hyperspectrum's ROI
        band_subset = reflc_hypspec[99 : 200, :, :]
        mean_value = np.mean(band_subset, axis = 0)
        mask = (mean_value > 90) | (mean_value < 25)
        reflc_hypspec[:, mask] = 0  
        
        # wirte in .hdr and .os
        with open(reflc_os, 'wb+') as f:
            f.write(reflc_hypspec.tobytes())
        if src_dtype == np.float64:
            spe_grain.metadata['data type'] = 5
        elif src_dtype == np.uint16:
            spe_grain.metadata['data type'] = 12
        metadata = spe_grain.metadata
        metadata['interleave'] = 'bsq'
        envi.write_envi_header(reflc_hdr, metadata)

        return reflc_hypspec


if __name__ == '__main__':
    # files and folders
    root_folder = r'C:\Users\ap\Desktop\WorkSpace\new_old_grain_classification_based_hyperspectrum'
    hdr_os_dict = {
        "grain_hdr": r'',
        "garin_os": r'',
        "dark_hdr": r'',
        "dark_os": r'',
        "white_hdr": r'C:\Users\ap\Desktop\WorkSpace\new_old_grain_classification_based_hyperspectrum\hyperspectrum_whiteboard\whiteboard_2025.hdr',
        "white_os": r'C:\Users\ap\Desktop\WorkSpace\new_old_grain_classification_based_hyperspectrum\hyperspectrum_whiteboard\whiteboard_2025.os',
        "white_avg_spectrum": r'C:\Users\ap\Desktop\WorkSpace\new_old_grain_classification_based_hyperspectrum\hyperspectrum_whiteboard\whiteboard_avg_spectrum.txt'
    }
    reflc_folder = os.path.join(root_folder, 'hyperspectrum_reflectance')
    if not os.path.exists(reflc_folder):
        os.makedirs(reflc_folder)
    grain_folder = os.path.join(root_folder, 'hyperspectrum_grain')
    dataset_pixel_level = r'C:\Users\ap\Desktop\WorkSpace\new_old_grain_classification_based_hyperspectrum\dataset_200_pixels_level_2023.csv'

    # get reflectance hyperspectrum and pixel level dataset
    grain_hdr_os_folders_files = get_hdr_os_folders_files(grain_folder)
    with open(dataset_pixel_level, mode='w', newline='') as file:
        writer = csv.writer(file)
        num = 0
        for sample_name, hdr_file, os_file in grain_hdr_os_folders_files:
            hdr_os_dict["grain_hdr"] = hdr_file
            hdr_os_dict["grain_os"] = os_file
            sample_name = sample_name.split('_')[0]

            # create reflectance hyperspectrum's filenames
            base_path = os.path.abspath(root_folder) + os.path.sep + 'hyperspectrum_reflectance' + os.path.sep
            base_name = os.path.splitext(os.path.basename(hdr_os_dict["grain_os"]))[0]
            reflc_hdr = base_path + base_name + '_reflc.hdr'
            reflc_os = base_path + base_name + '_reflc.os'            

            # calcualte reflectance hyperspectrum
            num  = num + 1
            print(f"------------------{num} : sample {sample_name}'s reflectance hyperspectrum: -----------------------\n")
            reflc_hypspec = reflc_hypspec_calc(hdr_os_dict["grain_hdr"], hdr_os_dict["grain_os"], hdr_os_dict["white_hdr"], hdr_os_dict["white_os"], reflc_hdr, reflc_os)

            # remove zero pixels and downsampling 
            non_zero_mask = np.any(reflc_hypspec != 0, axis=0)
            reflc_hypspec_downsampled = reflc_hypspec[:, non_zero_mask]  # (b, w, h) ——> (b, n)
            print("hyperspectrum after downsampling: ", reflc_hypspec_downsampled.shape)

            # get the average value for every 10000 pixels
            bands, n_pixels = reflc_hypspec_downsampled.shape
            n_groups = int(np.ceil(n_pixels / 20000))  
            averaged_data = np.zeros((bands, n_groups))
            for group_idx in range(n_groups):
                start = group_idx * 20000
                end = min((group_idx + 1) * 20000, n_pixels)  
                group_pixels = reflc_hypspec_downsampled[:, start:end]
                averaged_data[:, group_idx] = np.mean(group_pixels, axis=1)  
            print("hyperspectrum after getting mean value: ", averaged_data.shape)

            # write in csv
            for pixel_idx in range(averaged_data.shape[1]):
                pixel_data = averaged_data[:, pixel_idx]  
                row = [sample_name] + pixel_data.tolist()  
                writer.writerow(row)
            print(f"The {averaged_data.shape[1]} mean pixels of the sample {sample_name} have been written to CSV.\n")