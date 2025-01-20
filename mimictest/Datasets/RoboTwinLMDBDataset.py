import os
import json
import lmdb
from pickle import loads, dumps
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg
from torchvision.transforms.functional import resize
from mimictest.Utils.PreProcess import action_axis_to_6d, action_6d_to_axis, action_euler_to_6d, action_6d_to_euler
from tqdm import tqdm

CAMERAS = ['head_camera', 'left_camera', 'right_camera', 'front_camera']
RES = (320, 320)

class RoboTwinReader():

    def __init__(self, lmdb_dir):
        if isinstance(lmdb_dir, str):
            lmdb_dir = Path(lmdb_dir)
        self.lmdb_dir = lmdb_dir
        self.envs = []
        self.txns = []
        self.max_steps = json.load(open(lmdb_dir/'split.json', 'r'))
        split_num = len(self.max_steps)
        self.min_steps = [0] + [self.max_steps[split_id]+1 for split_id in range(split_num-1)]
        self.dataset_len = self.max_steps[-1] + 1

    def __len__(self):
        return self.dataset_len

    def open_lmdb(self, write=False):
        for split_id, split in enumerate(self.max_steps):
            split_path = self.lmdb_dir / str(split_id)
            env = lmdb.open(str(split_path), readonly=not write, create=False, lock=False, map_size=int(3e12))
            txn = env.begin(write=write)
            self.envs.append(env)
            self.txns.append(txn)
        self.res = {}
        for camera in CAMERAS:
            self.res[camera] = self.txns[0].get(f'{camera}_res'.encode())

    def close_lmdb(self):
        for txn in self.txns:
            txn.commit()
        for env in self.envs:
            env.close()
        self.envs = []
        self.txns = []

    def get_split_id(self, idx, array):
        left, right = 0, len(self.max_steps) - 1
        while left < right:
            mid = (left + right) // 2
            if array[mid] > idx:
                right = mid
            else:
                left = mid + 1
        return left

    def get_episode(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        cur_episode = loads(self.txns[split_id].get(f'cur_episode_{idx}'.encode()))
        return cur_episode

    def get_img(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        img = {}
        for camera in CAMERAS:
            img[camera] = decode_jpeg(loads(self.txns[split_id].get(f'{camera}_rgb_{idx}'.encode())))
        return img
    
    def get_pcd(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        pcd = loads(self.txns[split_id].get(f'pointcloud_{idx}'.encode()))
        return pcd
    
    def get_others(self, idx):
        if self.envs == []:
            self.open_lmdb()
        split_id = self.get_split_id(idx, self.max_steps)
        others = {}
        others['joint_action'] = loads(self.txns[split_id].get(f'joint_action_{idx}'.encode()))
        others['endpose'] = loads(self.txns[split_id].get(f'endpose_{idx}'.encode()))
        return others
    
class RoboTwinLMDBDataset(Dataset):

    def __init__(self, dataset_path, obs_horizon, chunk_size, start_ratio, end_ratio):
        self.obs_horizon = obs_horizon
        self.chunk_size = chunk_size
        self.reader = RoboTwinReader(dataset_path)
        self.dummy_rgb = torch.zeros((obs_horizon, 4, 3) + RES, dtype=torch.uint8) # (t v c h w)
        self.dummy_pos = torch.zeros((obs_horizon+chunk_size, 14)) 
        self.dummy_mask = torch.zeros(obs_horizon+chunk_size)
        self.start_step = int(self.reader.dataset_len * start_ratio)
        self.end_step = int(self.reader.dataset_len * end_ratio) - chunk_size - obs_horizon
    
    def __len__(self):
        return self.end_step - self.start_step
    
    def get_pos_range(self):
        pos_lst = []
        for idx in range(1000):
            others = self.reader.get_others(idx)
            pos = torch.from_numpy(others['joint_action']).to(torch.float32)
            pos_lst.append(pos)
        pose_lst = torch.stack(pos_lst)
        return {
            "pos_max": pose_lst.max(dim=0)[0],
            "pos_min": pose_lst.min(dim=0)[0],
        }

    def __getitem__(self, idx):
        idx = idx + self.start_step

        rgb = self.dummy_rgb.clone()
        pos = self.dummy_pos.clone()
        mask = self.dummy_mask.clone()
        
        episode_id = self.reader.get_episode(idx)

        for obs_idx in range(self.obs_horizon):
            imgs = self.reader.get_img(idx + obs_idx)
            if self.reader.get_episode(idx + obs_idx) == episode_id:
                for cam_id, camera in enumerate(CAMERAS):
                    img = imgs[camera]
                    rgb[obs_idx, cam_id] = resize(img, RES)
        
        for pos_idx in range(self.obs_horizon + self.chunk_size):
            others = self.reader.get_others(idx + pos_idx)
            if self.reader.get_episode(idx + pos_idx) == episode_id:
                pos[pos_idx] = torch.from_numpy(others['joint_action'])
                mask[pos_idx] = 1
        
        return {
            "rgb": rgb,
            "low_dim": pos[:self.obs_horizon],
            "action": pos[self.obs_horizon:],
            "mask": mask[self.obs_horizon:],
        }