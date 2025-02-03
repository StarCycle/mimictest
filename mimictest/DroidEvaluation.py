import os
import math
import numpy as np
import torch
from einops import rearrange

class Evaluation():
    def __init__(self, preprcs, prefetcher, save_path, device):
        self.preprcs = preprcs
        self.prefetcher = prefetcher
        self.save_path = save_path
        self.device = device
        return None

    def evaluate_on_env(self, acc, policy, batch_idx, num_eval_ep, max_test_ep_len, record_video=False):
        if policy.use_ema:
            policy.copy_ema_to_ema_net()
            policy.ema_net.eval()
        else:
            policy.net.eval()

        batch, _ = self.prefetcher.next_without_none()
        orig_actions = batch['action']
        batch = self.preprcs.process(batch, train=False)
        with torch.no_grad():
            pred = policy.infer(batch)
        pred_actions = self.preprcs.back_process(pred)['action']
        loss = torch.nn.functional.l1_loss(pred_actions, orig_actions)

        # Save the image in png
        '''
        import cv2
        imgs = ((batch['rgb'][0, 0] + 1)*0.5*255).permute(0, 2, 3, 1).cpu().numpy()
        for cam_id in range(4):
            img = imgs[cam_id].astype(np.uint8)
            cv2.imwrite(f'{self.save_path}/train_{cam_id}.png', img)
        import pdb; pdb.set_trace()
        '''
        # TODO: import pdb; pdb.set_trace()
        # TODO: loss = torch.nn.functional.l1_loss(pred['action'], batch['action'])

        return loss 
