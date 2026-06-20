from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import json
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


class ImageNet32(data.Dataset):
    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
    ]
    test_list = [
        ['val_data'],
    ]

    def __init__(self, root, train=True,transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        # now load the picked numpy arrays
        if self.train:
            train_data = []
            train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(root, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')

                train_data.append(entry['data'])
                if 'labels' in entry:
                    train_labels += entry['labels']
                else:
                    train_labels += entry['fine_labels']
                fo.close()
            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            train_labels[:] = [x - 1 for x in train_labels]

            train_data = np.concatenate(train_data)
            [picnum, pixel] = train_data.shape
            pixel = int(np.sqrt(pixel / 3))
            train_data = train_data.reshape((picnum, 3, pixel, pixel))
            self.data = train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.labels = train_labels
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            test_data = entry['data']
            [picnum,pixel]= test_data.shape
            pixel = int(np.sqrt(pixel/3))
            self.entry = entry
            if 'labels' in entry:
                test_labels = entry['labels']
            else:
                test_labels = entry['fine_labels']
            fo.close()

            # resize label range from [1,1000] to [0,1000),
            # This is required by CrossEntropyLoss
            test_labels[:] = [x - 1 for x in test_labels]
            test_data = test_data.reshape((picnum, 3, pixel, pixel))
            self.data = test_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.labels = test_labels
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(self.data[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[index]

    def __len__(self):
        return len(self.data)