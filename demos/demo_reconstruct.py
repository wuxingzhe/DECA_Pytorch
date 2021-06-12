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
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.datasets.demo_datasets import DemoTestData
from torch.utils.data import DataLoader
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)
    print(str(args.iscrop))
    print(str(args.isOri))

    # load test images 
    if args.isOri:
        testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)
    else:
        testdata = DemoTestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)
    test_data_loader = DataLoader(testdata, batch_size=3,
                                    num_workers=4,
                                    shuffle=True, pin_memory=True, drop_last=True)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    if args.model_path != '':
        deca_cfg.pretrained_modelpath = args.model_path
    
    deca = DECA(config = deca_cfg, device=device, eval_detail = False, is_ori = args.isOri)
    # for i in range(len(testdata)):
    for i, sample in enumerate(test_data_loader):
        names = sample['imagename']
        images = sample['image'].to(device)
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict) #tensor
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            for name in names:
                os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_images = deca.render.render_depth(opdict['transformed_vertices']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_images
            for j in range(len(names)):
                cv2.imwrite(os.path.join(savefolder, names[j], names[j] + '_depth.jpg'), util.tensor2image(depth_image[j]))
        if args.saveKpt:
            for j in range(len(names)):
                np.savetxt(os.path.join(savefolder, names[j], names[j] + '_kpt2d.txt'), opdict['landmarks2d'][j].cpu().numpy())
                np.savetxt(os.path.join(savefolder, names[j], names[j] + '_kpt3d.txt'), opdict['landmarks3d'][j].cpu().numpy())
        if args.saveObj:
            deca.save_obj(savefolder, names, opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            vis_imgs = deca.visualize(visdict)
            for j in range(len(names)):
                cv2.imwrite(os.path.join(savefolder, names[j] + '_vis.jpg'), vis_imgs[j])
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                if vis_name not in visdict.keys():
                    continue
                image  =util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/vggface2', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/vggface2/results16', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('-m', '--model_path', default='', type=str,
                        help='path to the trained model')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--isOri', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether is origin model' )

    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())
