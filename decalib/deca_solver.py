#!/usr/bin/env python3
# coding: utf-8
import os, sys
import pdb
import cv2
from skimage.io import imread
import pickle
import argparse
import time
import shutil
import logging
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable

from .utils.renderer import SRenderY
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis

from .datasets.datasets import datasets
from .datasets.train_datasets import TrainData
from .datasets.train_subject_datasets import TrainSubjectData

from .losses.unsupervised_losses import UnsupervisedLosses
from .utils.config import cfg
from .util.deca_utils import decompose_code, displacement2normal, displacement2vertex
torch.backends.cudnn.benchmark = True

class deca_solver(object):
    def __init__(self, config=None, mode = 'train_coarse',device='cuda'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.mode = mode
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self.save_path = os.path.join(self.cfg.save_dict_path, \
            '{}_{}'.format(self.config.config, datetime))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self._build()
        print('Build the code well for {}'.format(self.mode))
        # self._create_model(self.cfg.model)
        # self._setup_renderer(self.cfg.model)

    def encode(self, images):
        batch_size = images.shape[0]
        parameters = self.E_flame(images)
        codedict = decompose_code(parameters, self.param_dict)

        if self.mode == 'train_detail'
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode
        codedict['images'] = images

        return codedict

    def decode(self, codedict):
        images = codedict['images']
        batch_size = images.shape[0]
        
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], \
            expression_params=codedict['exp'], pose_params=codedict['pose'])
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device) 

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]
        landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]; 
        # landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
        landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]
        # landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_verts = util.batch_orth_proj(verts, codedict['cam'])
        trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        normals = util.vertex_normals(verts, self.faces.expand(batch_size, -1, -1))

        output = {'albedo': albedo, 'verts': verts, 'trans_verts': trans_verts, \
                    'landmarks2d': landmarks2d, 'landmarks3d': landmarks3d, 'normals': normals}

        if self.mode == 'train_detail':
            uv_z = self.D_detail(torch.cat([codedict['pose'][:,3:], codedict['exp'], \
                codedict['detail']], dim=1))
            output['displacement_map'] = uv_z+self.fixed_uv_dis[None,None,:,:]
            dense_vertices, dense_colors, dense_faces = util.upsample_mesh(verts, normals, \
                self.faces, output['displacement_map'], albedo, self.dense_template)

            output['detail_verts'] = dense_vertices
            output['detail_texture'] = dense_colors
            output['detail_faces'] = dense_faces

        return output

    def train(self, epoch):
        if self.mode == 'train_coarse':
            self.E_flame.train()
        elif self.mode == 'train_detail':
            self.E_detail.train()
            self.D_detail.train()

        for i, sample in enumerate(self.train_dataset):
            parameters = self.encode(sample['images'].to(self.device))
            output = self.decode(parameters)

            if self.mode == 'train_coarse':
                loss_ldmk = self.unsupervised_losses_conductor.landmarks_2d_loss( \
                    sample['kpts_gt'].to(self.device), output['landmarks2d'])
                loss_eye_closure = self.unsupervised_losses_conductor.landmarks_eye_closure_loss( \
                    sample['kpts_gt'].to(self.device), output['landmarks2d'])
                loss_total = self.cfg.train_params.ldmk_loss_factor * loss_ldmk +
                    self.cfg.train_params.eye_closure_loss_factor * loss_eye_closure

                if epoch > 2:
                    loss_total += self.unsupervised_losses_conductor. \
                        subject_consistency_loss(parameters['shape'])
                    loss_total += self.unsupervised_losses_conductor.photometric_loss( \
                        output['images'], output, parameters['light'])

            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            
            

    def trainval(self):
        config = self.config
        logging.info('Training ..... ')

        for epoch in range(self.start_epoch, config.train_param.scheduler.epochs + 1):
            self.lr_scheduler.step()
            if epoch > 2:
                self._build_train_loader(epoch)
            self.train(epoch)
            self._save_model_dict(epoch)

    # qualitive validation
    def validate(self):
        config = self.config

    def _save_model_dict(self, epoch):



    # render init
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

    # model init
    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape+model_cfg.n_tex+model_cfg.n_exp+model_cfg.n_pose+ \
            model_cfg.n_cam+model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3 # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, \
            model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i:model_cfg.get('n_' + i) for i in model_cfg.param_list}

        verts, faces, aux = load_obj(model_cfg.topology_path)
        self.faces = faces

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device) 
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail+self.n_cond, out_channels=1, \
            out_scale=model_cfg.max_z, sample_mode = 'bilinear').to(self.device)

        # resume model
        if hasattr(self.cfg, 'pretrained_model'):
            if hasattr(self.cfg.pretrained_model, 'encoder_coarse') \
                and os.path.exists(self.cfg.pretrained_model.encoder_coarse):
                util.copy_state_dict(self.E_flame.state_dict(), \
                    torch.load(self.cfg.pretrained_model.encoder_coarse))
                print('load encoder coarse pretrained model: ' + \
                    self.cfg.pretrained_model.encoder_coarse)

            if hasattr(self.cfg.pretrained_model, 'encoder_detail') \
                and os.path.exists(self.cfg.pretrained_model.encoder_detail):
                util.copy_state_dict(self.E_detail.state_dict(), \
                    torch.load(self.cfg.pretrained_model.encoder_detail))
                print('load encoder detail pretrained model: ' + \
                    self.cfg.pretrained_model.encoder_detail)

            if hasattr(self.cfg.pretrained_model, 'decoder_detail') \
                and os.path.exists(self.cfg.pretrained_model.encoder_detail):
                util.copy_state_dict(self.D_detail.state_dict(), \
                    torch.load(self.cfg.pretrained_model.decoder_detail))
                print('load decoder detail pretrained model: ' + \
                    self.cfg.pretrained_model.decoder_detail)

        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

	def _build_optimizer(self):		
		config = self.config.train_param.optimizer
		# model = self.model
		try:
			optim = getattr(torch.optim, config.type)
		except Exception:
			raise NotImplementedError('not implemented optim method ' + config.type)
        if self.mode == 'train_coarse':
            optimizer = optim(self.E_flame.parameters(), **config.kwargs)
        elif self.mode == 'train_detail':
            optimizer = optim([self.E_detail.parameters(), self.D_detail.parameters()], **config.kwargs)
        else:
            raise Warning('Invalid mode')
		# optimizer = optim(model.parameters(), **config.kwargs)
		self.optimizer = optimizer
		self.lr = config.kwargs.lr

		# if self.has_adv:
		# 	self.optimizer_discriminator = optim(self.discriminator.parameters(), **config.kwargs)

	def _build_scheduler(self):
		config = self.config.train_param
		self.lr_scheduler = MultiStepLR(self.optimizer,
										milestones = config.scheduler.milestones,
										gamma = config.scheduler.gamma,
										last_epoch = self.start_epoch - 1)

    def _build_criterion(self):
        self.unsupervised_losses_conductor = UnsupervisedLosses(self.config)

    def _build_train_loader(self, epoch):
        if self.mode == 'train_coarse':
            if epoch < 3:
                self.train_dataset = TrainData(self.config)
            else:
                self.train_dataset = TrainSubjectData(self.config)
    
    def _build_val_loader(self):
        # qualitative validation dataset
        self.qualitative_validate_dataset = datasets.TestData(self.config.test_params.qualitative_img_path, \
            iscrop=self.config.test_params.iscrop, face_detector=self.config.test_params.detector)

    def _build(self):
        config = self.cfg
        self.start_epoch = 0
        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

        if 'train' in self.mode:
            self._build_optimizer()
            self._build_scheduler()
            self._build_criterion()
            self._build_train_loader()