import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import get_constant_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from mimictest.Utils.AccelerateFix import AsyncStep
from mimictest.Utils.PreProcess import PreProcess
from mimictest.Utils.RobomimicDataset import CustomMimicDataset, DataPrefetcher
from mimictest.Utils.ComputeLimit import ComputeLimit
from mimictest.Wrappers.BasePolicy import BasePolicy
from mimictest.Nets.FlorenceNet import FlorenceNet
from mimictest.Simulation.ParallelEnv import ParallelMimic
from mimictest.Train import train
from mimictest.Evaluation import Evaluation

if __name__ == '__main__':
    # Script-specific settings 
    mode = 'train' # 'train' or 'eval'

    # Saving path
    save_path = Path('./Save/')
    save_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    abs_mode = True # relative EE action space or absolute EE action space
    if abs_mode:
        file_name = 'image_abs.hdf5'
    else:
        file_name = 'image.hdf5'
    dataset_path = Path('/root/dataDisk/square/ph/') / file_name
    bs_per_gpu = 432
    desired_rgb_shape = 84
    crop_shape = 76
    workers_per_gpu = 8
    cache_ratio = 2

    # Space
    num_actions = 7
    num_actions_6d = 10
    obs_horizon = 2
    chunk_size = 16
    limits = ComputeLimit(dataset_path, abs_mode)

    # Network
    model_path = Path("/root/dataDisk/Florence-2-base")
    freeze_vision_tower = True
    do_compile = False
    do_profile = False

    # Training
    num_training_epochs = 1000
    save_interval = 50 
    load_epoch_id = 0
    gradient_accumulation_steps = 1
    lr_max = 1e-4
    warmup_steps = 5
    weight_decay = 1e-4
    print_interval = 66
    record_video = False

    # Testing (num_envs*num_eval_ep*num_GPU epochs)
    num_envs = 16
    num_eval_ep = 6
    test_chunk_size = 8
    max_test_ep_len = 50

    # Preparation
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="wandb",
        # kwargs_handlers=[kwargs],
    )
    device = acc.device
    preprocessor = PreProcess(
        desired_rgb_shape,
        crop_shape,
        limits['low_dim_max'],
        limits['low_dim_min'],
        limits['action_max'],
        limits['action_min'],
        abs_mode,
        device,
    )
    envs = ParallelMimic(dataset_path, num_envs, abs_mode)
    eva = Evaluation(
        envs, 
        num_envs, 
        preprocessor, 
        obs_horizon,
        test_chunk_size, 
        num_actions, 
        save_path,
        device,
    )
    dataset = CustomMimicDataset(dataset_path, obs_horizon, chunk_size, start_ratio=0, end_ratio=1)
    loader = DataLoader(
        dataset=dataset,
        sampler=None, 
        batch_size=bs_per_gpu,
        shuffle=True,
        num_workers=workers_per_gpu,
        drop_last=True,     
    )
    net = FlorenceNet(
        path=model_path,
        lowdim_obs_dim=len(limits['low_dim_max']),
        num_actions=num_actions_6d,
        chunk_size=chunk_size,
        freeze_vision_tower=freeze_vision_tower,
    ).to(device)
    policy = BasePolicy(
        net=net,
        loss_func=torch.nn.functional.l1_loss,
        do_compile=do_compile,
    )
    policy.load_pretrained(acc, save_path, load_epoch_id) # also set wandb here
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr_max, weight_decay=weight_decay, fused=True)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    policy.net, optimizer, loader = acc.prepare(
        policy.net, 
        optimizer, 
        loader, 
        device_placement=[True, True, False],
    )
    optimizer.step = AsyncStep
    prefetcher = DataPrefetcher(loader, device)

    if mode == 'train':
        train(
            acc=acc, 
            prefetcher=prefetcher, 
            preprocessor=preprocessor,
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            num_training_epochs=num_training_epochs,
            eva=eva,
            num_eval_ep=num_eval_ep, 
            max_test_ep_len=max_test_ep_len,
            device=device,
            save_path=save_path,
            load_epoch_id=load_epoch_id,
            save_interval=save_interval,
            print_interval=print_interval,
            bs_per_gpu=bs_per_gpu,
            record_video=record_video,
            do_profile=do_profile,
        )
    elif mode == 'eval':
        avg_reward = torch.tensor(eva.evaluate_on_env(
            acc=acc, 
            policy=policy, 
            epoch=0,
            num_eval_ep=num_eval_ep, 
            max_test_ep_len=max_test_ep_len, 
            record_video=True)
        ).to(device)
        avg_reward = acc.gather_for_metrics(avg_reward).mean(dim=0)
        acc.print(f'chunk size {test_chunk_size}, success rate {avg_reward}')
