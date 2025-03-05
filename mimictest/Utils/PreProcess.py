from collections import OrderedDict
import torch
from torchvision.transforms.functional import resize, crop, center_crop, InterpolationMode
import mimictest.Utils.RotationConversions as rot

def action_euler_to_6d(rot_euler):
    rot_mat = rot.euler_angles_to_matrix(rot_euler, 'XYZ')
    rot_6d = rot.matrix_to_rotation_6d(rot_mat)
    return rot_6d

def action_axis_to_6d(rot_axis):
    rot_mat = rot.axis_angle_to_matrix(rot_axis)
    rot_6d = rot.matrix_to_rotation_6d(rot_mat)
    return torch.nan_to_num(rot_6d, nan=0)

def action_6d_to_euler(rot_6d):
    rot_mat = rot.rotation_6d_to_matrix(rot_6d)
    rot_euler = rot.matrix_to_euler_angles(rot_mat, 'XYZ')
    return rot_euler

def action_6d_to_axis(rot_6d):
    rot_mat = rot.rotation_6d_to_matrix(rot_6d)
    rot_axis = rot.matrix_to_axis_angle(rot_mat)
    return rot_axis

class PreProcess(): 
    def __init__(
            self,
            process_configs,
            device,
        ):
        self.configs = process_configs
        for key in self.configs:
            if "max" in self.configs[key]:
                self.configs[key]['max'] = self.configs[key]['max'].to(device)
                self.configs[key]['min'] = self.configs[key]['min'].to(device)
    
    def process(self, batch, train=False):
        current_crop_params = None
        for key in batch:
            if 'img_shape' in self.configs[key]: # image data
                batch[key] = resize(
                    batch[key], 
                    self.configs[key]['img_shape'], 
                    antialias=True,
                    interpolation=InterpolationMode.NEAREST
                )
                if train:
                    if "crop_shape" in self.configs[key]:
                        if current_crop_params is None:
                            h, w = batch[key].shape[-2:]
                            crop_h, crop_w = self.configs[key]['crop_shape']
                            top = torch.randint(0, h - crop_h + 1, (1,)).item()
                            left = torch.randint(0, w - crop_w + 1, (1,)).item()
                            current_crop_params = (top, left, crop_h, crop_w)
                        batch[key] = crop(batch[key], *current_crop_params)
                else:
                    batch[key] = center_crop(batch[key], self.configs[key]['crop_shape'])
                if key == 'rgb':
                    batch[key] = batch[key].float() / 255.
            if 'enable_6d_rot' in self.configs[key]:
                if self.configs[key]['abs_mode']:
                    rot_axis = batch[key][..., 3:6]
                    rot_6d = action_axis_to_6d(rot_axis)
                else:
                    rot_euler = batch[key][..., 3:6]
                    rot_6d = action_euler_to_6d(rot_euler)
                batch[key] = torch.cat((batch[key][..., :3], rot_6d, batch[key][..., 6:]), dim=-1)
            if "max" in self.configs[key]:
                batch[key] = (batch[key] - self.configs[key]['min']) / (self.configs[key]['max'] - self.configs[key]['min'])
                batch[key] = batch[key] * 2 - 1 # from (0, 1) to (-1, 1)
        return batch

    def back_process(self, batch):
        for key in batch:
            if "max" in self.configs[key]:
                batch[key] = (batch[key] + 1) * 0.5 # from (-1, 1) to (0, 1)
                batch[key] = batch[key] * (self.configs[key]['max'] - self.configs[key]['min']) + self.configs[key]['min']
            if 'img_shape' in self.configs[key]:
                if key == 'rgb':
                    batch[key] = torch.clamp(batch[key], 0, 1)
                    batch[key] = batch[key] * 255.
            if 'enable_6d_rot' in self.configs[key]:
                rot_6d = batch[key][..., 3:9]
                if self.configs[key]['abs_mode']:
                    rot_axis = action_6d_to_axis(rot_6d)
                    batch[key] = torch.cat((batch[key][..., :3], rot_axis, batch[key][..., 9:]), dim=-1)
                else:
                    rot_euler = action_6d_to_euler(rot_6d)
                    batch[key] = torch.cat((batch[key][..., :3], rot_euler, batch[key][..., 9:]), dim=-1)
            if 'binary' in self.configs[key]:
                batch[key] = torch.nn.Sigmoid()(batch[key])
                batch[key] = batch[key] > 0.5
                batch[key] = batch[key].int().float()
                batch[key] = batch[key] * 2.0 - 1.0

        if "arm_action" in batch and "gripper_action" in batch:
            batch['action'] = torch.cat((
                batch['arm_action'],
                batch['gripper_action'],
            ), dim=-1)
        elif "arm_action" in batch: # no gripper action
            batch['action'] = batch['arm_action']
        elif "action" in batch: # no gripper action
            batch['action'] = batch['action']
        return batch
