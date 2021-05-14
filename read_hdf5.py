import argparse
import h5py
import numpy as np
import os
import tarfile
from tqdm import tqdm


def main(args):
    hf = h5py.File(args.hdf5file, 'r')

    for gname, group in hf.items():
        print(gname)
        for dname, ds in group.items():
            print(dname, len(ds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read hdf5')
    parser.add_argument('hdf5file')
    args = parser.parse_args()
    main(args)
