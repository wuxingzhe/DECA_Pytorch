import os, sys
import torch
import torch.nn.functional as F
import numpy as np
from skimage.io import imread
import pickle 
import cv2

from models.arcfacenet_ori import SEResNet_IR_ori_224
from utils.renderer import SRenderY
from utils import util

class UnsupervisedLosses(object):
    def  __init__(self, config, device):
        self.config = config
        self.device = device
        self.image_size = self.config.dataset.image_size
        # ldmk supervision config
        self.ldmk_weights = torch.from_numpy(np.load(self.config.train_params.ldmk_weights_file))[None,:,:].to(self.device)
        self.eye_closure_ldmk_idx = np.load(self.config.train_params.eye_closure_ldmk_idx_file)

        # identity recog network
        self.recog_network = SEResNet_IR_ori_224(50, feature_dim=self.config.recog_params.feature_dim, mode='ir')
        self.recog_network.load_state_dict(torch.load(self.config.pretrained_model.recog_network)['net_state_dict'])
        print('load recog model: ' + \
                    self.config.pretrained_model.recog_network)
        if len(self.config.gpus.split(',')) > 1:
            self.recog_network = torch.nn.DataParallel(self.recog_network).to(self.device)
        else:
            self.recog_network = self.recog_network.to(self.device)
        self.recog_network.eval()

        # render 
        self._setup_renderer(self.config.model)

    def _setup_renderer(self, model_cfg):
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def landmarks_2d_loss(self, ldmks_gt, ldmks_pred, norm_type = 'mse'):
        if norm_type == 'mse':
            return torch.mean(self.ldmk_weights * (ldmks_pred-ldmks_gt) ** 2)
        elif norm_type == 'l1':
            return torch.mean(self.ldmk_weights * torch.abs(ldmks_pred-ldmks_gt))

    def landmarks_eye_closure_loss(self, ldmks_gt, ldmks_pred, norm_type = 'mse'):
        sub_gt = ldmks_gt[:,self.eye_closure_ldmk_idx[0,:],:] - ldmks_gt[:,self.eye_closure_ldmk_idx[1,:],:]
        sub_pred = ldmks_pred[:,self.eye_closure_ldmk_idx[0,:],:] - ldmks_pred[:,self.eye_closure_ldmk_idx[1,:],:]

        if norm_type == 'mse':
            return torch.mean((sub_gt - sub_pred) ** 2)
        elif norm_type == 'l1':
            return torch.mean(torch.abs(sub_gt - sub_pred))

    def photometric_loss(self, imgs_input, output, lights, skin_seg_res, norm_type = 'mse'):
        ops = self.render(output['verts'], output['trans_verts'], output['albedo'], lights)
        imgs_render = ops['images']
        imgs_render = imgs_render[:,[2,1,0],:,:]
        # imgs_render /= 255.0
        if norm_type == 'mse':
            loss_photometric = torch.mean(skin_seg_res * (imgs_input - imgs_render) ** 2)
        elif norm_type == 'l1':
            loss_photometric = torch.mean(skin_seg_res * torch.abs(imgs_input - imgs_render))

        return loss_photometric, imgs_render

    def identity_loss(self, imgs_input, imgs_render):
        embs_input, _  = self.recog_network(imgs_input)
        embs_render, _ = self.recog_network(imgs_render)

        dot = torch.sum(embs_render * embs_input, 1)
        norm = torch.norm(embs_input, dim=1) * torch.norm(embs_render, dim=1)
        similarity = dot/norm
        loss_identity = torch.mean(torch.ones_like(similarity) - similarity) 
        return loss_identity

    def regular_loss(self, params, device, norm_type = 'mse'):
        losses_regular = []
        for param in params:
            if norm_type == 'mse':
                losses_regular.append(torch.mean(param ** 2))
            elif norm_type == 'l1':
                losses_regular.append(torch.mean(torch.abs(param)))

        return torch.sum(torch.tensor(losses_regular).float().to(device))
