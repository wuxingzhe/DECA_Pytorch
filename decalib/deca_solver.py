#!/usr/bin/env python3
# coding: utf-8
import os, sys
import pdb
import cv2
from skimage.io import imread
import pickle
import argparse
import time
from datetime import datetime
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

from .datasets.test_datasets import QualitativeTestData
from .datasets.train_datasets import TrainData
from .datasets.train_subject_datasets import TrainSubjectData

from .losses.unsupervised_losses import UnsupervisedLosses
from .utils.config import cfg
from .util.deca_utils import decompose_code, displacement2normal, displacement2vertex
torch.backends.cudnn.benchmark = True

class deca_solver(object):
    def __init__(self, config=None, mode = 'train_coarse', device='cuda'):
        self.config = config
        self.device = device
        self.mode = mode
        self.image_size = self.config.dataset.image_size
        self.uv_size = self.config.model.uv_size

        if self.mode == 'train_coarse':
            self.epoch_phase = self.config.train_params.epoch_phase

        # torch.backends.cudnn.benchmark = True
        self.multi_gpus = False
        if len(self.config.gpus.split(',')) > 1:
            self.multi_gpus = True
        os.environ['CUDA_VISIBLE_DEVICES'] = self.config.gpus

        self.save_path = os.path.join(self.config.save_dict_path, \
            '{}_{}'.format(self.config.config, datetime.now().strftime('%Y%m%d_%H%M%S')))
        self.result_path = os.path.join(self.config.save_mesh_path, \
            '{}_{}'.format(self.config.config, datetime.now().strftime('%Y%m%d_%H%M%S')))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print('make save path: '+self.save_path)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
            print('make result path: '+self.result_path)

        self._build()
        print('Build the code well for {}'.format(self.mode))
        # self._create_model(self.cfg.model)
        # self._setup_renderer(self.cfg.model)

    def encode(self, images, epoch):
        batch_size = images.shape[0]
        parameters = self.E_flame(images)
        codedict = decompose_code(parameters, self.param_dict)

        if self.mode == 'train_detail':
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode
        codedict['images'] = images

        # shape consistency
        if epoch>self.epoch_phase and 'coarse' in self.mode:
            idx = np.arange(self.config.train_params.size_per_person_in_batch)
            random.shuffle(idx)
            idx_all = []
            for i in range(self.config.train_params.person_num_in_batch):
                idx_all.extend(idx+i*(self.config.train_params.size_per_person_in_batch))
            codedict['shape_shuffle'] = codedict['shape'][np.array(idx_all), :]

        if epoch>self.epoch_phase and self.config.eval_train:
            print(str(codedict['shape']))
            print(str(codedict['shape_shuffle']))

        return codedict

    def decode(self, codedict, epoch):
        images = codedict['images']
        batch_size = images.shape[0]
        
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], \
            expression_params=codedict['exp'], pose_params=codedict['pose'])
        if self.config.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device) 

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]
        landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]; 
        landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks2d /= (self.image_size - 1)
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
        landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]
        landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        landmarks3d /= (self.image_size - 1)
        trans_verts = util.batch_orth_proj(verts, codedict['cam'])
        trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        trans_verts = trans_verts*self.image_size/2 + self.image_size/2
        trans_verts /= (self.image_size - 1)
        normals = util.vertex_normals(verts, self.faces.expand(batch_size, -1, -1))

        output = {'albedo': albedo, 'verts': verts, 'trans_verts': trans_verts, \
                    'landmarks2d': landmarks2d, 'landmarks3d': landmarks3d, 'normals': normals}

        # shape consistency
        if 'coarse' in self.mode and epoch>self.epoch_phase:
            verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape_shuffle'], \
                expression_params=codedict['exp'], pose_params=codedict['pose'])

            ## projection
            landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]
            landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]
            landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
            landmarks2d /= (self.image_size - 1)
            landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam'])
            landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]
            landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
            landmarks3d /= (self.image_size - 1)
            trans_verts = util.batch_orth_proj(verts, codedict['cam'])
            trans_verts[:,:,1:] = -trans_verts[:,:,1:]
            trans_verts = trans_verts*self.image_size/2 + self.image_size/2
            trans_verts /= (self.image_size - 1)
            normals = util.vertex_normals(verts, self.faces.expand(batch_size, -1, -1))

            output['landmarks2d_shuffle'] = landmarks2d
            output['landmarks3d_shuffle'] = landmarks3d
            output['verts_shuffle'] = verts
            output['trans_verts_shuffle'] = trans_verts
            output['normals_shuffle'] = normals

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
            if i == self.half_iters:
                self._save_model_dict(epoch, half_epoch=True)
                self.validate(epoch, half_epoch = True)
            if epoch>self.epoch_phase and self.config.eval_train:
                print(str(sample['imagename']))
            parameters = self.encode(sample['images'].to(self.device), epoch)
            output = self.decode(parameters, epoch)

            if self.mode == 'train_coarse':
                loss_ldmk = self.unsupervised_losses_conductor.landmarks_2d_loss( \
                        sample['kpts_gt'].to(self.device), output['landmarks2d'], norm_type = self.config.train_params.norm_type_ldmk)
                loss_eye_closure = self.unsupervised_losses_conductor.landmarks_eye_closure_loss( \
                        sample['kpts_gt'].to(self.device), output['landmarks2d'], norm_type = self.config.train_params.norm_type_eye_closure)
                loss_regular = self.unsupervised_losses_conductor.regular_loss( \
                        [parameters['shape'], parameters['exp'], parameters['tex']], norm_type = self.config.train_params.norm_type_reg)

                if epoch <= self.epoch_phase:
                    loss_total = self.config.train_params.ldmk_loss_factors[0] * loss_ldmk + self.config.train_params.regular_loss_factors[0] * loss_regular
                        + self.config.train_params.eye_closure_loss_factors[0] * loss_eye_closure
                    print('Epoch: %d, Step: %d, Lr: %f, TL: %f, LdmkL: %f, EyeL: %f, RegL: %f' % (epoch, i, self.lr, \
                        loss_total.item(), loss_ldmk.item(), loss_eye_closure.item(), loss_regular.item()))
                        
                else:
                    loss_photometric = self.unsupervised_losses_conductor.photometric_loss( \
                        output['images'], output, parameters['light'], sample['seg_mask'].to(device), norm_type = self.config.train_params.norm_type_photometric)
                    loss_total = loss_photometric * self.config.train_params.photometric_loss_factor + \
                        self.config.train_params.ldmk_loss_factors[1] * loss_ldmk + self.config.train_params.regular_loss_factors[1] * loss_regular
                            + self.config.train_params.eye_closure_loss_factors[1] * loss_eye_closure

                    # shape consistency
                    loss_ldmk_consistency = self.unsupervised_losses_conductor.landmarks_2d_loss( \
                        sample['kpts_gt'].to(self.device), output['landmarks2d_shuffle'], norm_type = self.config.train_params.norm_type_ldmk)
                    loss_eye_closure_consistency = self.unsupervised_losses_conductor.landmarks_eye_closure_loss( \
                        sample['kpts_gt'].to(self.device), output['landmarks2d_shuffle'], norm_type = self.config.train_params.norm_type_eye_closure)
                    
                    output['verts'] = output['verts_shuffle']
                    output['trans_verts'] = output['trans_verts_shuffle']
                    loss_photometric_consistency = self.unsupervised_losses_conductor.photometric_loss( \
                        output['images'], output, parameters['light'], sample['seg_mask'].to(device))
                    loss_consistency =  ( \
                        loss_photometric_consistency * self.config.train_params.photometric_loss_factor + \
                        loss_ldmk_consistency * self.config.train_params.ldmk_loss_factors[1] + \
                        loss_eye_closure_consistency * self.config.train_params.eye_closure_loss_factors[1])
                    loss_total += self.config.train_params.shape_consistency_loss_factor * loss_consistency

                    print('Epoch: %d, Step: %d, Lr: %f, TL: %f, LdmkL: %f, EyeL: %f, RegL: %f, PhoL: %f, ConL: %f, LdConL: %f, EyeConL: %f, PhoConL: %f' \
                        % (epoch, i, self.lr, loss_total.item(), loss_ldmk.item(), loss_eye_closure.item(), loss_regular.item(), \
                        loss_photometric.item(), loss_consistency.item(), loss_ldmk_consistency.item(), \
                        loss_eye_closure_consistency.item(), loss_photometric_consistency.item()))

            self.lr_scheduler.optimizer.zero_grad()
            loss_total.backward()
            self.lr_scheduler.optimizer.step()
            
        if self.mode == 'train_coarse':
            self.E_flame.eval()
        elif self.mode == 'train_detail':
            self.E_detail.eval()
            self.D_detail.eval()
            

    def trainval(self):
        config = self.config
        print('Training ..... ')

        for epoch in range(self.start_epoch, config.train_params.scheduler.epochs + 1):
            self.lr_scheduler.step()
            self.lr = self.lr_scheduler.get_lr()[0]
            if epoch > self.epoch_phase:
                self._build_train_loader(epoch)
            self.train(epoch)
            self._save_model_dict(epoch)
            self.validate(epoch)

    # qualitive validation
    def validate(self, epoch, half_epoch = False):
        config = self.config
        if self.mode == 'train_coarse':
            self.E_flame.eval()
        elif self.mode == 'train_detail':
            self.E_detail.eval()
            self.D_detail.eval()

        for i, sample in enumerate(self.qualitative_validate_loader):
            parameters = self.encode(sample['images'].to(self.device), 0)
            output = self.decode(parameters, 0)

            for j in range(output['verts'].shape[0]):
                verts_j = output['verts'][j,:,:].cpu().numpy()
                texture_j = util.tensor2image(output['albedo'][j,:,:,:].cpu().numpy())
                img_name = str(epoch)+'_'+'_'.join(((sample['imagename'][j]).split('/'))[-3:])

                faces = self.faces[0].cpu().numpy()
                uvcoords = self.uvcoords[0].cpu().numpy()
                uvfaces = self.uvfaces[0].cpu().numpy()

                util.write_obj(img_name, verts_j, faces, 
                        texture=texture_j, 
                        uvcoords=uvcoords, 
                        uvfaces=uvfaces)


        if self.mode == 'train_coarse':
            self.E_flame.train()
        elif self.mode == 'train_detail':
            self.E_detail.train()
            self.D_detail.train()

    def _save_model_dict(self, epoch, half_epoch = False):
        print('save network in: '+self.save_path)
        if self.multi_gpus:
            if 'coarse' in self.mode:
                E_flame_state_dict = self.E_flame.module.state_dict()
            elif 'detail' in self.mode:
                E_detail_state_dict = self.E_detail.module.state_dict()
                D_detail_state_dict = self.D_detail.modele.state_dict()
        else:
            if 'coarse' in self.mode:
                E_flame_state_dict = self.E_flame.state_dict()
            elif 'detail' in self.mode:
                E_detail_state_dict = self.E_detail.state_dict()
                D_detail_state_dict = self.D_detail.state_dict()

        if half_epoch:
            path_prefix = 'Epoch_half'
        else:
            path_prefix = 'Epoch'
        if 'coarse' in self.mode:
            torch.save({
                'iters': epoch,
                'net_state_dict': E_flame_state_dict},
                os.path.join(self.save_path, path_prefix+'_%06d_E_flame.ckpt' % epoch))
        elif 'detail' in self.mode:
            torch.save({
                'iters': epoch,
                'net_state_dict': E_detail_state_dict},
                os.path.join(self.save_path, path_prefix+'_%06d_E_detail.ckpt' % epoch))
            torch.save({
                'iters': epoch,
                'net_state_dict': D_detail_state_dict},
                os.path.join(self.save_path, path_prefix+'_%06d_D_detail.ckpt' % epoch))

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
        self.faces = faces.verts_idx[None, ...] 
        self.uvcoords = aux.verts_uvs[None, ...]      # (N, V, 2)
        self.uvfaces = faces.textures_idx[None, ...] # (N, F, 3)

        # encoders
        if self.multi_gpus:
            self.E_flame = DataParallel(ResnetEncoder(outsize=self.n_param)).to(self.device)
        else:
            self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device) 
        if 'detail' in self.mode:
            if self.multi_gpus:
                self.E_detail = DataParallel(ResnetEncoder(outsize=self.n_detail)).to(self.device)
            else:
                self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)

        # decoders
        if self.multi_gpus:
            self.flame = DataParallel(FLAME(model_cfg)).to(self.device)
        else:
            self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            if self.multi_gpus:
                self.flametex = DataParallel(FLAMETex(model_cfg)).to(self.device)
            else:
                self.flametex = FLAMETex(model_cfg).to(self.device)
        if 'detail' in self.mode:
            if self.multi_gpus:
                self.D_detail = DataParallel(Generator(latent_dim=self.n_detail+self.n_cond, out_channels=1, \
                    out_scale=model_cfg.max_z, sample_mode = 'bilinear')).to(self.device)
            else:
                self.D_detail = Generator(latent_dim=self.n_detail+self.n_cond, out_channels=1, \
                    out_scale=model_cfg.max_z, sample_mode = 'bilinear').to(self.device)

        # resume model
        if hasattr(self.config, 'pretrained_model'):
            if hasattr(self.config.pretrained_model, 'encoder_coarse') \
                and os.path.exists(self.config.pretrained_model.encoder_coarse):
                self.E_flame.load_state_dict( \
                    torch.load(self.config.pretrained_model.encoder_coarse))
                print('load encoder coarse pretrained model: ' + \
                    self.config.pretrained_model.encoder_coarse)

            if hasattr(self.config.pretrained_model, 'encoder_detail') \
                and os.path.exists(self.config.pretrained_model.encoder_detail):
                self.E_detail.load_state_dict( \
                    torch.load(self.config.pretrained_model.encoder_detail))
                print('load encoder detail pretrained model: ' + \
                    self.config.pretrained_model.encoder_detail)

            if hasattr(self.config.pretrained_model, 'decoder_detail') \
                and os.path.exists(self.config.pretrained_model.encoder_detail):
                self.D_detail.load_state_dict( \
                    torch.load(self.config.pretrained_model.decoder_detail))
                print('load decoder detail pretrained model: ' + \
                    self.config.pretrained_model.decoder_detail)

        self.E_flame.eval()
        if 'detail' in self.mode:
            self.E_detail.eval()
            self.D_detail.eval()

	def _build_optimizer(self):		
		config = self.config.train_params.optimizer
		# model = self.model
		try:
			optim = getattr(torch.optim, config.type)
		except Exception:
			raise NotImplementedError('not implemented optim method ' + config.type)
        if self.mode == 'train_coarse':
            optimizer = optim(self.E_flame.parameters(), **config.kwargs)
        elif self.mode == 'train_detail':
            optimizer = optim(itertools.chain(self.E_detail.parameters(), self.D_detail.parameters()), **config.kwargs)
        else:
            raise Warning('Invalid mode')
		# optimizer = optim(model.parameters(), **config.kwargs)
		self.optimizer = optimizer
		self.lr = config.kwargs.lr

		# if self.has_adv:
		# 	self.optimizer_discriminator = optim(self.discriminator.parameters(), **config.kwargs)

	def _build_scheduler(self):
		config = self.config.train_params
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
                self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.train_params.batch_sizes[0],
									num_workers=self.config.train_params.workers,
									shuffle=True, pin_memory=True, drop_last=True)
                self.half_iters = len(self.train_dataset)/(self.config.train_params.batch_sizes[0])//2
            else:
                self.train_dataset = TrainSubjectData(self.config)
                self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.train_params.batch_sizes[1],
									num_workers=self.config.train_params.workers,
									shuffle=False, pin_memory=True, drop_last=True)
                self.half_iters = len(self.train_dataset)/(self.config.train_params.batch_sizes[1])//2
            print('train half iters: '+str(self.half_iters))

    
    def _build_val_loader(self):
        # qualitative validation dataset
        self.qualitative_validate_dataset = QualitativeTestData(self.config)
        self.qualitative_validate_loader = DataLoader(self.qualitative_validate_dataset, batch_size=self.config.test_params.batch_size,
									num_workers=self.config.test_params.workers,
									shuffle=True, pin_memory=True, drop_last=True)

    def _build(self):
        config = self.config
        self.start_epoch = 1
        self._create_model(self.config.model)
        self._setup_renderer(self.config.model)

        if 'train' in self.mode:
            self._build_optimizer()
            self._build_scheduler()
            self._build_criterion()
            self._build_train_loader()