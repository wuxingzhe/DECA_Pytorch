import os, sys
import torch

from .models.encoders import ResnetEncoder as RecogEncoder
from .utils.renderer import SRenderY

class UnsupervisedLosses(object):
    def  __init__(self, config):
        self.config = config
        self.image_size = self.config.image_size
        # ldmk supervision config
        self.ldmk_weights = np.load(self.config.train_params.ldmk_weights_file)
        self.eye_closure_ldmk_idx = np.load(self.config.train_params.eye_closure_ldmk_idx_file)

        # identity recog nertwork
        self.recog_network = RecogEncoder()
        # render 
        self._setup_renderer(self.config)

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

    def landmarks_2d_loss(self, ldmks_gt, ldmks_pred):
        return torch.mean(self.ldmk_weights * (ldmks_pred-ldmks_gt) ** 2)

    def landmarks_eye_closure_loss(self, ldmks_gt, ldmks_pred):
        sub_gt = ldmks_gt[self.eye_closure_ldmk_idx[0,:],:] - ldmks_gt[self.eye_closure_ldmk_idx[1,:],:]
        sub_pred = ldmks_pred[self.eye_closure_ldmk_idx[0,:],:] - ldmks_gt[self.eye_closure_ldmk_idx[1,:],:]

        return torch.mean((sub_gt - sub_pred) ** 2)

    def subject_consistency_loss(self, shape_codes):
        avg_shape_code = torch.mean(shape_codes, dims=0)
        return torch.mean((shape_codes - avg_shape_code) ** 2)

    def photometric_loss(self, imgs_input, output):
        imgs_render = self.render(output['verts'], output['trans_verts'], output['albedo'], lights)
        loss_photometric = torch.mean((imgs_input - imgs_render) ** 2)

        return loss_photometric

    def identity_loss(self, imgs_input, imgs_render):
