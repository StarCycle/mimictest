# A wrapper of policy to satisfy the APIs in eval_policy.py                                                  
import numpy as np                                                                                           
import torch                                                                                                 
from torchvision.transforms.functional import resize, InterpolationMode
from mimictest.Datasets.RoboTwinLMDBDataset import RES, CAMERAS                                              
from mimictest.Utils.Deproject import deproject
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip                                             

class RoboTwinPolicy():                                                                                      
    def __init__(self, policy, preprcs, obs_horizon, action_horizon, save_path, record_video=False):         
        self.dummy_rgb = torch.zeros(len(CAMERAS), 3, RES[0], RES[1]) # (v c h w)                            
        self.dummy_coord = torch.zeros((len(CAMERAS), 3) + RES) # (v c h w)
        self.policy = policy                                                                                 
        self.preprcs = preprcs                                                                               
        self.obs_horizon = obs_horizon                                                                       
        self.action_horizon = action_horizon                                                                 
        self.save_path = save_path
        self.record_video = record_video
        self.reset_obs()

    def reset_obs(self):
        self.rgb_buffer = []
        self.coord_buffer = []
        self.low_dim_buffer = []
        if self.record_video:
            self.video = []

    def log_video(self, seed):
        if self.record_video:
            clip = ImageSequenceClip(self.video, fps=30)
            clip.write_gif(self.save_path / (f'seed{seed}.gif'), fps=30)                                 

    def fill_buffer(self, rgb, coord, low_dim):
        if len(self.rgb_buffer) == 0: # Fill the buffer
            for i in range(self.obs_horizon):
                self.rgb_buffer.append(rgb)
                self.coord_buffer.append(coord)
                self.low_dim_buffer.append(low_dim)
        elif len(self.rgb_buffer) == self.obs_horizon: # Update the buffer
            self.rgb_buffer.pop(0)
            self.rgb_buffer.append(rgb)
            self.coord_buffer.pop(0)
            self.coord_buffer.append(coord)
            self.low_dim_buffer.pop(0)
            self.low_dim_buffer.append(low_dim)
        else:
            raise ValueError(f"SequentialRoboTwin.py: buffer len {len(self.rgb_buffer)}")

    def get_action(self, obs):
        rgb = self.dummy_rgb.clone()
        coord = self.dummy_coord.clone()

        for cam_id, camera in enumerate(CAMERAS):
            img = torch.from_numpy(obs['observation'][camera]['rgb']).permute(2,0,1)
            rgb[cam_id] = resize(img, RES)
            points = deproject(
                torch.from_numpy(obs['observation'][camera]['intrinsic_cv']), 
                torch.from_numpy(obs['observation'][camera]['cam2world_gl']), 
                torch.from_numpy(obs['observation'][camera]['depth']).float() / 1000,
            )
            coord[cam_id] = resize(
                points, 
                RES,
                interpolation=InterpolationMode.NEAREST,
            )

        low_dim = torch.from_numpy(obs['joint_action']).float()
        rgb = rgb.unsqueeze(0)
        low_dim = low_dim.unsqueeze(0)
        coord = coord.unsqueeze(0)
        self.fill_buffer(rgb, coord, low_dim)
        with torch.no_grad():
            batch = {
                'rgb': torch.stack(self.rgb_buffer, dim=1).cuda(),
                'coord': torch.stack(self.coord_buffer, dim=1).cuda(),
                'low_dim': torch.stack(self.low_dim_buffer, dim=1).cuda(),
            }
            batch = self.preprcs.process(batch, train=False)
            pred = self.policy.infer(batch)
            self.pred_actions = self.preprcs.back_process(pred)['action'].cpu().numpy()

        pred_act = self.pred_actions[0]
        if self.record_video:
            self.video.append(obs['observation']['head_camera']['rgb'].astype(np.uint8).copy())
        return pred_act