#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    datasets = [
        ('modelnet40_ply_hdf5_2048', 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'),
        ('3dmatch', 'http://web.tecnico.ulisboa.pt/sergio.agostinho/share/just-a-spoonful/3dmatch.zip'),
    ]
    for folder, www in datasets:
        if not os.path.exists(os.path.join(DATA_DIR, folder)):
            zipfile = os.path.basename(www)
            os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
            os.system('rm %s' % (zipfile))
    return DATA_DIR


def load_data(partition, prefix=None):
    if prefix:
        DATA_DIR = prefix
    else:
        download()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data', 'modelnet40_ply_hdf5_2048')

    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, 'ply_data_%s*.h5' % partition))):
        f = h5py.File(h5_name, mode="r", swmr=True)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', gaussian_noise=False, unseen=False, factor=4, prefix=None):
        self.data, self.label = load_data(partition, prefix=prefix)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label>=20]
                self.label = self.label[self.label>=20]
            elif self.partition == 'train':
                self.data = self.data[self.label<20]
                self.label = self.label[self.label<20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        if self.gaussian_noise:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


def generate_random_poses(N, factor):
    euler_ab = np.random.rand(N, 3) * np.pi / factor
    euler_ba = -euler_ab[:,::-1]

    rot = Rotation.from_euler("zyx", euler_ab)
    R_ab = rot.as_matrix()
    R_ba = R_ab.transpose(0, 2, 1)

    translation_ab = np.random.rand(N, 3) - 0.5
    translation_ba = - np.squeeze(translation_ab[:, None] @ R_ab, axis=1)
    return dict(
        R_ab=R_ab,
        R_ba=R_ba,
        translation_ab=translation_ab,
        translation_ba=translation_ba,
        euler_ab=euler_ab,
        euler_ba=euler_ba,
    )

class ThreeDMatch(Dataset):

    def __init__(self, prefix, partition, minimum_overlap=0.3, factor=4):
        super().__init__()

        self.overlap_options = set([0.3, 0.5, 0.7])

        if minimum_overlap is not None and minimum_overlap not in self.overlap_options:
            msg = f"Accepted minimum_overlap values are the following: {self.overlap_options}"
            raise ValueError(msg)

        # Check and download data if needed
        if not prefix:
            data_dir = download()
            self.prefix = os.path.join(data_dir, "3dmatch")
        else:
            self.prefix = prefix

        # use stage information to populate list of files belonging to the split
        scenes = self._parse_scenes(partition)

        # retrieve all valid point cloud pairs
        self.pairs = self._parse_sequences(scenes, minimum_overlap)

        # generate random poses
        self.poses = generate_random_poses(len(self.pairs), factor)


    def _parse_scenes(self, partition):
        file = os.path.join(self.prefix, "splits", f"{partition}_3dmatch.txt")
        with open(file) as f:
            scenes = [line[:-1] for line in f.readlines()]
        return scenes

    def _parse_sequences(self, scenes, minimum_overlap):

        path = os.path.join(self.prefix, "preprocessed")
        pattern = "@seq-[0-9][0-9].txt" if minimum_overlap is None else f"@seq-[0-9][0-9]-{minimum_overlap:0.2f}.txt"

        pairs = []
        for scene in scenes:
            # sorting to ensure reproduceability across OSes
            sequences = sorted(list(glob.glob(os.path.join(path, scene + pattern))))
            for seq in sequences:
                with open(seq) as f:
                    pairs += [tuple(line.split()) for line in f.readlines()]
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):

        # load both point clouds
        file0, file1 = self.pairs[index][:2]
        pointcloud0 = np.load(os.path.join(self.prefix, "preprocessed", file0))["pcd"]
        pointcloud1 = np.load(os.path.join(self.prefix, "preprocessed", file1))["pcd"]

        # Rescale both point clouds, to lie inside a unit sphere
        scale = np.max(np.linalg.norm(np.stack([pointcloud0, pointcloud1]), axis=-1))
        pointcloud0 /= scale
        pointcloud1 /= scale

        # pose data
        R_ab = self.poses["R_ab"][index]
        R_ba = self.poses["R_ba"][index]
        euler_ab = self.poses["euler_ab"][index]
        euler_ba = self.poses["euler_ba"][index]
        translation_ab = self.poses["translation_ab"][index]
        translation_ba = self.poses["translation_ba"][index]

        # Point cloud data in 3DMatch is perfectly superimposed
        # apply transformation to target point cloud
        pointcloud1 = pointcloud1 @ R_ba + translation_ab
        out = (
            pointcloud0.T.astype(np.float32),
            pointcloud1.T.astype(np.float32),
            R_ab.astype(np.float32),
            translation_ab.astype(np.float32),
            R_ba.astype(np.float32),
            translation_ba.astype(np.float32),
            euler_ab.astype(np.float32),
            euler_ba.astype(np.float32),
        )
        return out


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data in train:
        print(len(data))
        break
