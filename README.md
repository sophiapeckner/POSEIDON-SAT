# POSEIDON

This fork is a modified version of the [POSEIDON data augmentation tool](https://github.com/pabloruizp/POSEIDON), described by P. Ruiz-Ponce, D. Ortez-Perez, J. Garcia-Rodriguez, and B. Kiefer (2023) in [POSEIDON: A Data Augmentation Tool for Small Object Detection Datasets in Maritime Environments](https://doi.org/10.3390/s23073691). It is modified to augment the [ShipRSImageNet](https://doi.org/10.1109/JSTARS.2021.3104230) dataset published by Z. Zhang, L. Zhang, Y. Wang, P. Feng and R. He.

The ShipRSImageNet dataset is avaialble in the [GitHub repository](https://github.com/zzndream/ShipRSImageNet) published by the authors of the dataset. As the ShipRSImageNet dataset is licensed for only academic use, an augmented dataset using this tool is subject to the same restrictions of use. See the GitHub repository for the ShipRSImageNet dataset for more information.

The [ShipRSImageNet devkit](https://github.com/zzndream/ShipRSImageNet_devkit) is useful for exploring the dataset, as are the various functions inside the `shiprsimagenet.py` file.

## Notable Modifications

As this fork of POSEIDON is interested in dataset augmentation of a class of interest, rather than the complete balacing of an entire dataset, and considers particular needs for applying POSEIDON to the ShipRSImageNet dataset.
The POSEIDON tool has been altered in several different ways to better suit this use case.

The modifications to the POSEIDON tool are as follows:

- Only a subset of images in the dataset are augmented using the method detailed by POSEIDON. These are specified as one of the inputs to this program. This was done to avoid potential issues that may arise from using POSEIDON on certain images without additional labeling of the original images, such as generated ships being placed on land. Docks are labeled in many parts of the ShipRSImageNet dataset, but land is not.
- Normalization is performed when inserting a new instance onto an existing image, based on the spatial resolution metadata included in many of the images in the ShipRSImageNet dataset. Additionally, the normalization method used is entirely different since resizing all images to match the lowest resolution of the images we are augmenting would not address the different spatial resolutions (meters/pixel resolution) across images and would also result in very low-resolution images, considering that 980 x 980 is the more common resolution in the dataset and lower resolutions are often around 350 x 350.

A more detailed explaination as to why these modifications were made can be found in the [dataset exploration notebook](dataset_exploration.ipynb), which further explores the ShipRSImageNet dataset and analyzes the potential issues that may arise from using POSEIDON on the dataset without these changes.

## Repository Structure

All the code involved in the actual dataset augmentation is contained in the `poseidon` folder. `main.py` in the root of the repository is used to run the augmentation process.

Other python scripts in the root of the repository are used to explore the dataset and to help select images for augmentation. These are used in the process outlined in the [dataset exploration Jupyter notebook](dataset_exploration.ipynb).

All code in this repository expects the ShipRSImageNet dataset to be extracted to the root of the repository, such that the `ShipRSImageNet_V1` folder is a sibling to the `poseidon` folder

## Running the code

Additionally, the augmentation tool requires that the list of images to be augmented is in a file called `selected_for_augmentation.txt` in the root of the repository. This file should contain the file names of the images to be augmented, one per line. See the [dataset exploration notebook](dataset_exploration.ipynb) for information on how the images in this repository's [`selected_for_augmentation.txt`](selected_for_augmentation.txt) file were selected from the ShipRSImageNet dataset, the tools and techniques used to build this list of images to augment, and for information on factors that were considered in the selection of augmentation candidate images.

After the `selected_for_augmentation.txt` file is created, the ShipRSImageNet dataset is extracted to the root of the repository, and dependencies are installed (see `requirements.txt` or `requirements-frozen.txt`), the code can be run by executing the `main.py` script in the root of the repository. To run this code using the same package versions that were used during the creation of this repository, use the `requirements-frozen.txt` file in a Python 3.8.10 environment on Linux.

To view the augmented images, you can use the `show_images_selected_for_augmentation.py` script and pass the path to the augmented images dataset directory. Ex. `python show_images_selected_for_augmentation.py augmented_image_folder/`

TODO: Add frozen requirements file after finishing everything
