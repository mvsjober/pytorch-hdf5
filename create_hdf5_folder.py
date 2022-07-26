import argparse
import h5py
import numpy as np
import os
import tarfile
from tqdm import tqdm


def main(args):
    folder_path = args.folder_path
    hdf5_path = os.path.splitext(folder_path)[0] + '.hdf5'

    # NOTE: using libver='latest' seems to fix problem with big datasets, where
    # it suddenly slows down after reaching a certain point
    hf = h5py.File(hdf5_path, 'w', libver='latest')

    print('Reading {} ...'.format(folder_path))
    img_count = 0

    groups = {}

    # tfm = tf.getmembers()
    for path, subdirs, files in tqdm(os.walk(folder_path), total=1333169):
        for file in files:

            file_path = os.path.join(path,file)[len(folder_path):]

            path_parts = file_path.split('/')

            fn = path_parts[-1]
            class_name = path_parts[-2]
            dataset_name = os.path.basename(folder_path)

            if dataset_name not in groups:
                grp = hf.create_group(dataset_name)
                groups[dataset_name] = grp
            else:
                grp = groups[dataset_name]
                
            image = open(os.path.join(folder_path,path, file), 'rb')
            data_np = np.asarray(image.read())

            ds = grp.create_dataset(fn, data=data_np)
            ds.attrs['class'] = class_name

            img_count += 1

    hf.close()

    print('Created {} with {} images.'.format(hdf5_path, img_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert folder to hdf5')
    parser.add_argument('folder_path')
    args = parser.parse_args()
    main(args)
