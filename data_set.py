import numpy   as np
import os
import os.path as path
import shutil
import pickle

from keras.utils            import to_categorical
from keras.utils.data_utils import get_file
from operator               import attrgetter


def download_data():
    shutil.copytree(get_file('cifar-10-batches-py', origin='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', untar=True), './data')


def load_data(data_path='./data'):
    def load_batch(path):
        with open(path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')

        return (np.array(batch[b'data']).reshape(batch[b'data'].shape[0], 3, 32, 32).transpose(0, 2, 3, 1) / 255,
                to_categorical(np.array(batch[b'labels'])))

    def load_batches(paths):
        return tuple(map(np.concatenate, zip(*map(load_batch, paths))))

    if not path.exists('./data'):
        download_data()

    return (load_batches(sorted(map(attrgetter('path'), filter(lambda directory_entry: directory_entry.name.startswith('data_batch_'), os.scandir(data_path))))),
            load_batch('{0}/test_batch'.format(data_path)))
