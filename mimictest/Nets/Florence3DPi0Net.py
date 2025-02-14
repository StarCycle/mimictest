import os
import math
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.linalg import inv
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Florence3DPi0Net(nn.Module):
    def __init__(
            self,
            path,
            lowdim_obs_dim,
            num_actions,
            freeze_vision_tower,
        ):
        super().__init__()

        self.net = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
        if freeze_vision_tower:
            for param in self.net.vision_tower.parameters():
                param.requires_grad = False
        self.position_embedding_3d = nn.Linear(3, self.net.vision_tower.convs[-1].proj.out_channels)

        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self.tokenizer = AutoProcessor.from_pretrained(path, trust_remote_code=True).tokenizer
        prompt_token_ids = self.tokenizer(
            "<Action>",
            return_tensors="pt",
            padding=False,
            max_length=None,
            truncation=None,
            return_token_type_ids=False,
        )['input_ids']
        self.prompt_embeds = nn.Parameter(self.net.get_input_embeddings()(prompt_token_ids), requires_grad=False)

        token_dim = self.net.language_model.model.decoder.embed_tokens.embedding_dim
        if lowdim_obs_dim > 0:
            self.low_dim_encoder = nn.Linear(lowdim_obs_dim, token_dim)
        self.action_encoder = nn.Linear(num_actions, token_dim)
        self.action_timestep_mixer = nn.Sequential(
            nn.Linear(2*token_dim, token_dim),
            nn.SiLU(),
            nn.Linear(token_dim, token_dim),
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(token_dim, num_actions),
        )
        self.time_emb = SinusoidalPosEmb(token_dim)

    def _encode_image(self, rgb, coord):
        B, T, V, C, H, W = rgb.shape
        rgb_feature = self.net.vision_tower.forward_features_unpool(
            rgb.view(B*T*V, C, H, W),
        )
        B_T_V, N, D = rgb_feature.shape
        H1 = W1 = int(N ** 0.5)
        rgb_feature = rgb_feature.view(B*T, V*N, D)

        B, T, V, C, H, W = coord.shape
        coord = coord.reshape(B*T*V, C, H, W)
        coord = F.avg_pool2d(coord, kernel_size=(H//H1,H//H1), stride=(W//W1,W//W1))
        B_T_V, C, H, W = coord.shape
        coord = coord.permute(0, 2, 3, 1).reshape(B*T, V*H*W, C)
        pos_embed_3d = self.position_embedding_3d(coord)
        x = rgb_feature + pos_embed_3d

        if self.net.visual_temporal_embed is not None:
            visual_temporal_embed = self.net.visual_temporal_embed(x.view(B*T, 1, V*N, D)[:, :, 0])
            x = x.view(B*T, 1, V*N, D) + visual_temporal_embed.view(1, 1, 1, D)

        x_feat_dict = {}

        spatial_avg_pool_x = x.view(B*T, 1, V*N, D).mean(dim=2)
        x_feat_dict['spatial_avg_pool'] = spatial_avg_pool_x

        temporal_avg_pool_x = x.view(B*T, 1, V*N, D).mean(dim=1)
        x_feat_dict['temporal_avg_pool'] = temporal_avg_pool_x

        x = x.view(B*T, 1, V*N, D)[:, -1]
        x_feat_dict['last_frame'] = x

        new_x = []
        for _image_feature_source in self.net.image_feature_source:
            if _image_feature_source not in x_feat_dict:
                raise ValueError('invalid image feature source: {}'.format(_image_feature_source))
            new_x.append(x_feat_dict[_image_feature_source])

        x = torch.cat(new_x, dim=1)

        x = x @ self.net.image_projection
        x = self.net.image_proj_norm(x)

        return x 

    def forward(self, batch):
        if batch['obs_features'] is None:
            B, T, V, C, H, W = batch['rgb'].shape
            rgb_features = self._encode_image(batch['rgb'], batch['coord'])
            B_T, N, D = rgb_features.shape
            rgb_features = rgb_features.view(B, T*N, D)
            
            text_embeds = self.prompt_embeds.repeat(B, 1, 1) # (b n d)
            if "inst_token" in batch:
                inst_embeds = self.net.get_input_embeddings()(batch["inst_token"]) 
                text_embeds = torch.cat((text_embeds, inst_embeds), dim=1)
            inputs_embeds, attention_mask = self.net._merge_input_ids_with_image_features(rgb_features, text_embeds)

            obs_features = inputs_embeds
        else:
            inputs_embeds = batch['obs_features']
            obs_features = batch['obs_features']

        noisy_actions = self.action_encoder(batch['noisy_inputs']['action'])
        B, T, D = noisy_actions.shape
        time_emb = self.time_emb(batch["timesteps"]).unsqueeze(1).repeat(1, T, 1) # (b t d)
        noisy_actions = torch.cat((noisy_actions, time_emb), dim=-1) # (b t 2d)
        noisy_actions = self.action_timestep_mixer(noisy_actions) # (b t d)

        if "low_dim" in batch:
            low_dim = self.low_dim_encoder(batch['low_dim']) # (b, t, d)
            decoder_inputs_embeds = torch.cat((low_dim, noisy_actions), dim=1)
        else:
            decoder_inputs_embeds = noisy_actions
        decoder_outputs_embeds = self.net.language_model(
            inputs_embeds = inputs_embeds,
            decoder_inputs_embeds = decoder_inputs_embeds,
            output_hidden_states = True,
        )['decoder_hidden_states'][-1]

        # predict the noise residual
        pred_noise = {}
        pred_noise['action'] = self.action_decoder(decoder_outputs_embeds[:, -T:])
        return pred_noise, obs_features
