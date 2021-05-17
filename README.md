# Quick example for using HDF5 datasets with PyTorch DataLoader

## Converting existing dataset to HDF5

The file [create_hdf5.py](create_hdf5.py) contains an example of how to convert
a tar file with images to an HDF5 file. Usage example:

    python3 create_hdf5.py /path/to/image-data.tar

Converting a 26GB tar file with 1.3 million images took less than 10 minutes on
Mahti.

## Training with PyTorch

