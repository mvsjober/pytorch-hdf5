import argparse
import h5py
import numpy as np
import os
import tarfile
from tqdm import tqdm


def main(args):
    tar_path = args.tarfile
    hdf5_path = os.path.splitext(tar_path)[0] + '.hdf5'
    hf = h5py.File(hdf5_path, 'w')

    print('Reading {} ...'.format(tar_path))
    tf = tarfile.open(tar_path)
    img_count = 0

    groups = {}

    tfm = tf.getmembers()

    for tarinfo in tqdm(tfm):
        if not tarinfo.isreg():
            continue
        
        tn = tarinfo.name
        path_parts = tn.split('/')

        fn = path_parts[-1]
        cls = path_parts[-2]
        ds = path_parts[-3]

        if cls not in groups:
            grp = hf.create_group(cls)
            groups[cls] = grp
        else:
            grp = groups[cls]
            
        # print(fn, cls, ds)
        #dn = '/'.join(path_parts[:-1])
        #grp = hf[dn]

        
        # if tarinfo.isdir():
        #     if len(path_parts) > 0:
        #         grp = hf.create_group('/'.join(path_parts))

        image = tf.extractfile(tarinfo)
        data_np = np.asarray(image.read())
        # print('data_np', data_np.nbytes)

        # p = os.path.join(db_dir, dn, fn)
        # with open(p, 'rb') as fp:
        #     bd = fp.read()
        #     bd_np = np.asarray(bd)
        # print('bd_np', bd_np.nbytes)

        # img = Image.open(io.BytesIO(data_np))
        # print('image size:', img.size)

        grp.create_dataset(fn, data=data_np)
        img_count += 1
        if img_count > 10:
            break

    tf.close()

    ## Reading from files...
    # for dset in os.listdir(db_dir):
    #     dset_dir = os.path.join(db_dir, dset)
    #     dset_grp = hf.create_group(dset)

    #     for cls in os.listdir(dset_dir):
    #         cls_dir = os.path.join(dset_dir, cls)
    #         cls_grp = dset_grp.create_group(cls)

    #         for fn in tqdm(os.listdir(cls_dir), desc='{}/{}'.format(dset, cls)):
    #             fn_path = os.path.join(cls_dir, fn)

    #             with open(fn_path, 'rb') as fp:
    #                 bin_data = fp.read()
    #                 bin_data_np = np.asarray(bin_data)
    #                 ds = cls_grp.create_dataset(fn, data=bin_data_np)

    hf.close()

    print('Created {} with {} images.'.format(hdf5_path, img_count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert tar to hdf5')
    parser.add_argument('tarfile')
    args = parser.parse_args()
    main(args)
