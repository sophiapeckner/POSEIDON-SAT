# POSEIDON

This fork of the POSEIDON data augmentation tool, described by P. Ruiz-Ponce, D. Ortez-Perez, J. Garcia-Rodriguez, and B. Kiefer (2023) in [POSEIDON: A Data Augmentation Tool for Small Object Detection Datasets in Maritime Environments](https://doi.org/10.3390/s23073691); is modified to augment the [ShipRSImageNet](https://doi.org/10.1109/JSTARS.2021.3104230) dataset published by Z. Zhang, L. Zhang, Y. Wang, P. Feng and R. He.

The ShipRSImageNet dataset is avaialble in the [GitHub repository](https://github.com/zzndream/ShipRSImageNet) published by the authors. As the
ShipRSImageNet dataset is licensed for only academic use, an augmented dataset using this tool is subject to the same license. See the GitHub repository for the ShipRSImageNet dataset for more information.

The [ShipRSImageNet devkit](https://github.com/zzndream/ShipRSImageNet_devkit) is useful for exploring the dataset, as are the various functions inside the `poseidon/shiprsimagenet.py` file.

## Running the code

As this fork of POSEIDON is interested in dataset augmentation of a class of interest, rather than the complete balacing of an entire dataset, the POSEIDON tool has been altered in several different ways to better suit this use case. To better accomplish this goal and to accomodate the nature of the images contained in the ShipRSImageNet dataset, the POSEIDON tool has been modified to only augment a subset of specified images in the dataset to address potential issues that may arise from using POSEIDON on certain images without additional labeling of the original images.

The code in this repository expects the ShipRSImageNet dataset to be extracted to the root of the repository, such that the `ShipRSImageNet_V1` folder is in the same directory as the `poseidon` folder. Additionally, the augmentation tool requires that the list of images to be augmented is in a file called `selected_for_augmentation.txt` in the root of the repository. This file should contain the file names of the images to be augmented, one per line. See the [dataset exploration notebook](dataset_exploration.ipynb) for information on how the images in this repository's [`selected_for_augmentation.txt`](selected_for_augmentation.txt) file were selected from the ShipRSImageNet dataset, and for information on factors that were considered in the selection of augmentation candidate images.
