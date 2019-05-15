from __future__ import print_function, division
import os

from torch.utils.data import Dataset

import pandas as pd
import numpy as np


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class MyDataset(Dataset):

    def __init__(self, root_dir, ids):

        self.root_dir = root_dir
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        pts_name = os.path.join(self.root_dir, id + '.csv')
        pts_bbox_name = os.path.join(self.root_dir, id + '.npy')

        pts_all = np.array(pd.read_csv(pts_name))[:, 1:]
        pts_xyz = pts_all[:, 0:3].transpose()  # xyz, and from N_points, N_channel to N_channel, N_points
        pts_label = pts_all[:, 9]  # point-wise label
        # 2 classes label
        binary = 1
        if binary:
            pts_label[pts_label==4] = 0  # non-bridge
            pts_label[pts_label!=0] = 1  # bridge
            pts_label_onehot = np.tile(np.zeros((pts_label.shape)), reps=(2, 1))
            pts_label_onehot[0, pts_label==0] = 1
            pts_label_onehot[1, pts_label==1] = 1

        pts_bbox = np.load(pts_bbox_name)[0:6]

        return {'pts_xyz': pts_xyz,
                'pts_label': pts_label,
                'pts_bbox': pts_bbox,
                'pts_all': pts_all
                }
