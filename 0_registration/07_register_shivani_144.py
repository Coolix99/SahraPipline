import os
import sys
import git
import numpy as np
from simple_file_checksum import get_checksum

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
from IO import *


def register_shivani_144hpf_reg_finmasks(skip_existing=True):
    """
    Register Shivani 144 hpf regeneration WT fin masks.

    Input:
        /home/max/Downloads/masks/*.tif

    Output structure:
        finmasks_path/
            image_name_without_extension/
                image_name.tif
                MetaData_finmasks.json
    """

    input_folder = "/home/max/Downloads/masks"

    condition = "Regeneration"
    time_hpf = 144
    experimentalist = "Shivani"
    genotype = "WT"

    # Input scaling is assumed to be X, Y, Z
    scale_x = 0.3459441
    scale_y = 0.3459441
    scale_z = 1.0

    scales_zyx = [scale_z, scale_y, scale_x]

    repo = git.Repo(gitPath, search_parent_directories=True)
    sha = repo.head.object.hexsha

    im_list = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".tif", ".tiff"))
    ]

    for img_name in im_list:
        print(f"Registering {img_name}")

        img_stem = os.path.splitext(img_name)[0]

        output_folder = os.path.join(finmasks_path, img_stem)
        make_path(output_folder)

        if skip_existing and get_JSON(output_folder) != {}:
            print("  skip existing")
            continue

        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        im = getImage(input_path)
        print(im.shape)
        # Ensure finmask is binary
        im = im > 0

        save_array_as_tiff(im, output_path)

        metadata = {}

        metadata["git hash"] = sha
        metadata["git repo"] = "Sahrapipline"

        metadata["finmasks file"] = img_name
        metadata["finmasks checksum"] = get_checksum(output_path, algorithm="SHA1")

        metadata["scales ZYX"] = scales_zyx
        metadata["condition"] = condition
        metadata["time in hpf"] = time_hpf
        metadata["experimentalist"] = experimentalist
        metadata["genotype"] = genotype

        writeJSON(output_folder, "MetaData_finmasks", metadata)

        print(f"  saved to {output_folder}")


if __name__ == "__main__":
    register_shivani_144hpf_reg_finmasks(skip_existing=True)