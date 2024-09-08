import os
import math
import numpy as np
import torch
from einops import rearrange
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

class Evaluation():
    def __init__(self, envs, num_envs, preprcs, obs_horizon, chunk_size, num_actions, save_path, device):
        self.envs = envs
        self.num_envs = num_envs
        self.preprcs = preprcs
        self.device = device
        self.obs_horizon = obs_horizon
        self.chunk_size = chunk_size
        self.num_actions = num_actions
        self.save_path = save_path
        return None

    def evaluate_on_env(self, acc, policy, epoch, num_eval_ep, max_test_ep_len, record_video=False):
        if policy.use_ema:
            policy.ema_net.eval()
        else:
            policy.net.eval()
        total_rewards = np.zeros((self.num_envs)) 
        with torch.no_grad():
            for ep in range(num_eval_ep):
                rewards = np.zeros((self.num_envs))
                rgb_buffer = [] 
                low_dim_buffer = []
                obs = self.envs.reset()
                self.num_cameras = obs['rgb'].shape[1]
                if record_video:
                    videos = []
                    for camera_id in range(self.num_cameras):
                        videos.append([[] for i in range(self.num_envs)])
                
                for t in tqdm(range(max_test_ep_len), desc=f"run episode {ep+1} of {num_eval_ep}", disable=not acc.is_main_process):
                    # Add RGB image to placeholder 
                    rgb = torch.from_numpy(obs['rgb'])
                    rgb = rearrange(rgb, 'b v h w c -> b v c h w').contiguous()
                    rgb = self.preprcs.rgb_process(rgb, train=False).to(self.device)
                    low_dim = torch.from_numpy(obs['low_dim']).float()
                    low_dim = self.preprcs.low_dim_normalize(low_dim.to(self.device))
                    if len(rgb_buffer) == 0: # Fill the buffer 
                        for i in range(self.obs_horizon):
                            rgb_buffer.append(rgb)
                            low_dim_buffer.append(low_dim)
                    elif len(rgb_buffer) == self.obs_horizon: # Update the buffer
                        rgb_buffer.pop(0)
                        rgb_buffer.append(rgb)
                        low_dim_buffer.pop(0)
                        low_dim_buffer.append(low_dim)
                    else:
                        raise ValueError(f"Evaluation.py: buffer len {len(rgb_buffer)}")
                    pred_actions = policy.infer(torch.stack(rgb_buffer, dim=1), torch.stack(low_dim_buffer, dim=1))
                    pred_actions = self.preprcs.action_back_normalize(pred_actions).cpu().numpy()
                    for action_id in range(self.chunk_size):
                        obs, rw, done, info = self.envs.step(pred_actions[:, action_id])
                        for env_id in range(self.num_envs):
                            if rewards[env_id] == 0 and rw[env_id] == 1:
                                rewards[env_id] = 1
                                print(f'gpu{acc.process_index}_episode{ep}_env{env_id}: get reward! step {t*self.chunk_size+action_id}')
                        if record_video:
                            for env_id in range(self.num_envs):
                                if rewards[env_id] == 0:
                                    for camera_id in range(self.num_cameras):
                                        img = obs['rgb'][env_id, camera_id].astype(np.uint8).copy()
                                        videos[camera_id][env_id].append(img)

                # If there are multiple episodes in max_test_ep_len, only conut the 1st episode
                rewards = np.where(rewards > 0, 1, rewards)
                total_rewards += rewards 
                print(f'gpu{acc.process_index}_epidose{ep}: rewards {rewards}')
                if record_video:
                    for env_id in range(self.num_envs):
                        prefix = f'epoch{epoch}_gpu{acc.process_index}_episode{ep}_env{env_id}_reward{rewards[env_id]}'
                        for camera_id in range(self.num_cameras):
                            clip = ImageSequenceClip(videos[camera_id][env_id], fps=30)
                            clip.write_gif(self.save_path / (prefix+f'_camera{camera_id}.gif'), fps=30)
            total_rewards /= num_eval_ep
        return total_rewards.mean() 