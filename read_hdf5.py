import argparse
import h5py
import io
import numpy as np
from PIL import Image


def main(args):
    hf = h5py.File(args.hdf5file, 'r')

    for gname, group in hf.items():
        print(gname, len(group))
        for dname, data in group.items():
            print(dname, data.attrs['class'])
            break

    print("===")
    if args.get_dataset and args.get_filename:
        f = hf[args.get_dataset][args.get_filename]
        img = Image.open(io.BytesIO(np.array(f)))
        cls = f.attrs['class']
        print('image size:', img.size, 'class:', cls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read hdf5')
    parser.add_argument('hdf5file')
    parser.add_argument('get_dataset', nargs='?')
    parser.add_argument('get_filename', nargs='?')
    args = parser.parse_args()
    main(args)
