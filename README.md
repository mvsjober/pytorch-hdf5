# Quick example for using HDF5 datasets with PyTorch DataLoader

## Converting existing dataset to HDF5

The file [create_hdf5.py](create_hdf5.py) contains an example of how to convert
a tar file with images to an HDF5 file. Usage example:

    python3 create_hdf5.py /path/to/image-data.tar

Converting a 26GB tar file with 1.3 million images took less than 10 minutes on
Mahti.

## Training with PyTorch

The file [pytorch_dvc_cnn_simple.py](pytorch_dvc_cnn_simple.py) together with
[pytorch_dvc_cnn_simple.py](pytorch_dvc_cnn_simple.py) shows a simple CNN image
training that uses an HDF5 dataset. The original dataset can be found from
<https://a3s.fi/mldata/dogs-vs-cats.tar>.

The main trick is that the HDF5 file needs to be opened in the `__getitem__`
method of the `Dataset`, as explained e.g. in [this blog post by Soumya
Tripathy](https://blade6570.github.io/soumyatripathy/hdf5_blog.html#orgc76c0fe)
