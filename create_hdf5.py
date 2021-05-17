import argparse
import h5py
import numpy as np
import os
import tarfile
from tqdm import tqdm


def main(args):
    tar_path = args.tarfile
    hdf5_path = os.path.splitext(tar_path)[0] + '.hdf5'

    # NOTE: using libver='latest' seems to fix problem with big datasets, where
    # it suddenly slows down after reaching a certain point
    hf = h5py.File(hdf5_path, 'w', libver='latest')

    print('Reading {} ...'.format(tar_path))
    tf = tarfile.open(tar_path)
    img_count = 0

    groups = {}

    # tfm = tf.getmembers()
    for tarinfo in tqdm(tf, total=1333169):
        if not tarinfo.isreg():
            continue
        
        tn = tarinfo.name
        path_parts = tn.split('/')

        fn = path_parts[-1]
        class_name = path_parts[-2]
        dataset_name = path_parts[-3]

        if dataset_name not in groups:
            grp = hf.create_group(dataset_name)
            groups[dataset_name] = grp
        else:
            grp = groups[dataset_name]
            
        image = tf.extractfile(tarinfo)
        data_np = np.asarray(image.read())

        ds = grp.create_dataset(fn, data=data_np)
        ds.attrs['class'] = class_name

        img_count += 1

    tf.close()
    hf.close()

    print('Created {} with {} images.'.format(hdf5_path, img_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tar to hdf5')
    parser.add_argument('tarfile')
    args = parser.parse_args()
    main(args)
