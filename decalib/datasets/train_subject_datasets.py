# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import random
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io

from . import detectors

def video2sequence(video_path):
    videofolder = video_path.split('.')[0]
    util.check_mkdir(videofolder)
    video_name = video_path.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TrainSubjectData(Dataset):
    def __init__(self, config, face_detector='mtcnn'):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        self.config = config
        self.image_size = self.config.dataset.image_size

        self.imagepath_list = []
        # paths
        # self.imgs_dir = self.config.train_params.imgs_dir
        self.kpts_gt_dir = self.config.train_params.kpts_gt_dir
        self.seg_masks_dir = self.config.train_params.seg_masks_dir

        # sizes about person subjects
        self.person_size = self.config.train_params.person_size
        self.size_per_person = self.config.train_params.size_per_person

        self.person_num_in_batch = self.config.train_params.person_num_in_batch
        self.size_per_person_in_batch = self.config.train_params.size_per_person_in_batch
        self.batch_size = self.size_per_person_in_batch * self.person_num_in_batch

        r1 = open(self.config.train_params.train_subject_list)
        files_list = r1.readlines()
        r1.close()
        for i,file_one in enumerate(files_list):
            file_one = file_one.strip()
            if i%(self.size_per_person) == 0:
                self.imagepath_list.append([])
            self.imagepath_list[i//(self.size_per_person)].append(file_one)
        # self.imagepath_list = [file_one.strip() for file_one in files_list]

        # shuffle list
        self.person_seq = np.arange(len(self.imagepath_list))
        random.shuffle(self.person_seq)
        self.seq_in_person = np.arange(self.size_per_person)
        random.shuffle(self.seq_in_person)

        print('total {} images'.format(len(self.imagepath_list) * self.size_per_person))
        print('total {} persons'.format(len(self.imagepath_list)))
        # self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = self.config.train_params.crop_size
        self.scale = self.config.train_params.scale
        self.iscrop = self.config.train_params.iscrop
        self.resolution_inp = self.config.train_params.crop_size

        if self.iscrop:
            if face_detector == 'fan':
                self.face_detector = detectors.FAN()
            # elif face_detector == 'mtcnn':
            #     self.face_detector = detectors.MTCNN()
            else:
                print(f'please check the detector: {face_detector}')
                exit()

    def __len__(self):
        return len(self.imagepath_list) * self.size_per_person

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type=='kpt68':
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def get_list_index(self, index):
        person_ind = index//(self.batch_size)//(self.size_per_person//self.size_per_person_in_batch) * \
            (self.person_num_in_batch) + (index%(self.batch_size))//(self.size_per_person_in_batch)
        ind_in_person = index//(self.batch_size)%(self.size_per_person//self.size_per_person_in_batch) * \
            (self.size_per_person_in_batch) + (index%(self.batch_size))%(self.size_per_person_in_batch)

        return self.person_seq[person_ind], self.seq_in_person[ind_in_person]

    def __getitem__(self, index):
        ind_x, ind_y = self.get_list_index(index)
        imagepath = self.imagepath_list[ind_x][ind_y]
        # imagename = imagepath.split('/')[-1].split('.')[0]
        root_path = '/'.join(imagepath.split('/')[:-2])
        imagename = os.path.basename(imagepath)
        kpts_gt = np.load(os.path.join(root_path, self.kpts_gt_dir, imagename[:-3]+'npy'))
        seg_mask = np.load(os.path.join(root_path, self.seg_masks_dir, imagename[:-3]+'npy'))

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = imagepath.replace('.jpg', '.mat').replace('.png', '.mat')
            kpt_txtpath = imagepath.replace('.jpg', '.txt').replace('.png', '.txt')
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T        
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kptpath)
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            else:
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print('no face detected! run original image')
                    left = 0; right = h-1; top=0; bottom=w-1
                else:
                    left = bbox[0]; right=bbox[2]
                    top = bbox[1]; bottom=bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        # else:
        #     src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        # DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        # tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.
        kpts_gt = kpts_gt/(self.image_size)

        # dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'kpts_gt': torch.tensor(kpts_gt).float(),
                'seg_mask': torch.tensor(seg_mask).float(),
                # 'tform': tform,
                # 'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }
