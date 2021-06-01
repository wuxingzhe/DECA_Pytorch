import cv2
import random
import numbers
import numpy as np
cv2.ocl.setUseOpenCL(False)

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms import Compose, Lambda

from utils.mesh_transform import rot_mat_to_euler


class TransformBuilder(object):
    def __init__(self, config):
        self.config = config

    @property
    def train_transforms(self):
        config = self.config

        train_transforms = []

        if hasattr(config, 'has_affine_aug') and config.has_affine_aug:
            train_transforms.append(RandomAffine(degrees=config.rotation, 
                                            translate=config.translate_rate,
                                            scale=(config.scale_min,
                                             config.scale_max), prop=0.7))
        if hasattr(config, 'has_blur_aug') and config.has_blur_aug:
            train_transforms.append(RandomBlur())
        if hasattr(config, 'has_noise_aug') and config.has_noise_aug:
            train_transforms.append(GaussianNoise())
        if hasattr(config, 'occ_ratio') and config.occ_ratio:
            train_transforms.append(RandomOcclude(config.crop_size, 
                                                occ_ratio=config.occ_ratio))
        if hasattr(config, 'has_channel_scale') and config.has_channel_scale:
            train_transforms.append(ChannelScale())
            
        return transforms.Compose(train_transforms)

    @property
    def test_transforms(self):
        config = self.config

        test_transforms = []
        if config.crop_size < 256:
            test_transforms.append(RandomAffine(degrees=0., translate=0.,
                                                scale=(config.crop_size / 256.,
                                                config.crop_size / 256.), prop=1.0))
        test_transforms.append(CenterCrop(config.crop_size))
        test_transforms.append(Normalize(config.normalize))
        test_transforms.append(ToTensor())

        return transforms.Compose(test_transforms)


class RandomBlur(object):
    def __call__(self, sample):
        random_blur = random.uniform(0., 1.)
        if random_blur < 0.4:
            image = sample['image']

            if random_blur > 0.2: # Gaussian Blur
                image = cv2.GaussianBlur(image, ksize=(0, 0), sigmaX=2, sigmaY=2)
            else:
                angle = random.randint(0, 180)
                image = self.motion_blur(image, ksize=5, angle=angle)

            sample['image'] = image
        return sample

    def motion_blur(self, src, ksize=5, angle=45):
        src = np.array(src)
        R = cv2.getRotationMatrix2D((ksize/2. - 0.5, ksize/2. - 0.5), angle, 1)
        kernel = np.diag(np.ones(ksize)) / ksize
        kernel = cv2.warpAffine(kernel, R, (ksize, ksize))
        dst = cv2.filter2D(src, -1, kernel)
        cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
        dst = np.array(dst, dtype=np.uint8)
        return dst


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees: sequence or float or int, Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate: float & optional, maximum absolute fraction for horizontal
            and vertical translations. For example translate=t, then horizontal 
            and vertical shift is randomly sampled in the range 
            -img_width * t < dx < img_width * t and
            -img_height * b < dy < img_height * b. Will not translate by default.
        scale: tuple & optional, scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. 
            Will keep original scale by default.
    """

    def __init__(self, degrees, translate=None, scale=None, prop=0.7):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate:
            if not (0.0 <= translate <= 1.0):
                raise ValueError("translation values should be between 0 and 1")
            translate = (translate, translate)
            
        self.translate = translate

        if scale:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        self.prop = prop

    @staticmethod
    def get_affine_matrix(center, angle, translations, zoom):
        """Compute affine matrix from affine transformation"""
        # Rotation & scale
        matrix = cv2.getRotationMatrix2D(center, angle, zoom).astype(np.float64)
        # translate
        matrix[0, 2] += translations[0] * zoom
        matrix[1, 2] += translations[1] * zoom

        return matrix

    @staticmethod
    def get_params(degrees, translate, scale_ranges, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        
        if translate:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges:
            if scale_ranges[0] == scale_ranges[1]:
                scale = scale_ranges[0]
            else:
                scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        return angle, translations, scale

    def __call__(self, sample):
        """
            img (OpenCV Image): Image to be transformed.

        Returns:
            OpenCV Image: Affine transformed image.
        """
        image, landmark, seg_mask = sample['image'], sample['kpts_gt'], sample['seg_mask']
        # print('before affine transform', len(landmarks))
        prop = random.uniform(0., 1.)
        if prop <= self.prop:
            h, w = image.shape[:2]
            center = (w/2 - 0.5, h/2 - 0.5)
            angle, translations, zoom = \
                self.get_params(self.degrees, self.translate, self.scale, [h, w])

            matrix = self.get_affine_matrix(center, angle, translations, zoom)
            dst_image = cv2.warpAffine(image, matrix, (w,h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
            dst_seg_mask = cv2.warpAffine(seg_mask, matrix, (w,h), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))

            # landmark transformation
            R2d = matrix[:2, :2]
            t2d = matrix[:, 2]
            landmark = landmark.dot(R2d.T) + t2d.reshape(1, 2)
            
            sample['image'] = dst_image
            sample['seg_mask'] = dst_seg_mask
            sample['kpts_gt'] = landmark

        return sample


class RandomOcclude(object):
    """Add random rectangle occlusion to image
    """
    def __init__(self, img_size, aspect_ratio=[0.4, 2.5], occ_ratio=0.35):
        self.img_size = img_size
        self.aspect_ratio = aspect_ratio
        self.occ_ratio_max = occ_ratio

    def __call__(self, sample):
        random_occ = random.uniform(0., 1.)
        if random_occ > 0.5:
            occ_ratio = random.uniform(0.15, self.occ_ratio_max)
            self.occ_area = occ_ratio * self.img_size * self.img_size
            x_c = random.uniform(0., self.img_size)
            y_c = random.uniform(0., self.img_size)
            aspect_ratio = random.uniform(self.aspect_ratio[0], self.aspect_ratio[1])
            
            w = (aspect_ratio * self.occ_area) ** 0.5
            h = w / aspect_ratio

            l = int(x_c - w/2)
            r = int(x_c + w/2)
            t = int(y_c - h/2)
            b = int(y_c + h/2)

            image, seg_mask = sample['image'], sample['seg_mask']
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            cv2.rectangle(image, (l, t), (r, b), color=color, thickness=-1)
            cv2.rectangle(seg_mask, (l, t), (r, b), color=(0, 0, 0), thickness=-1)
            sample['image'] = image
            sample['seg_mask'] = seg_mask

        return sample


class Normalize(object):
    """
    Normalize the image channel-wise for each input image,
    the mean and std. are calculated from each channel of the given image
    """
    def __init__(self, mode='zscore'):
        self.mode = mode
        self.name = ['image', 'perturb_image']

    def __call__(self, sample):

        for name in self.name:
            if name not in sample.keys():
                continue
            image = sample[name]

            if self.mode == 'zscore':
                mean, std = cv2.meanStdDev(image)
                mean, std = mean[:,0], std[:,0]
                std = np.where(std < 1e-6, 1, std)
                image = (image - mean)/std
            else:
                image = image/255. - 0.5
                
            image = image.astype(np.float32)
            sample[name] = image

        return sample


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self):
        self.name = ['image', 'perturb_image']

    def __call__(self, sample):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image (scale by 1/255).
        """
        for name in self.name:
            if name not in sample.keys():
                continue
            image = sample[name]
        
            image = TF.to_tensor(image)

            sample[name] = image
        return sample


class ChannelScale(object):
    def __init__(self, min_rate=0.8, max_rate=1.2):
        self.min_rate = min_rate
        self.max_rate = max_rate

    def __call__(self, sample):
        prop = random.uniform(0., 1.)
        if prop > 0.6:
            out = sample['image'].copy()
            seg_mask = sample['seg_mask'].copy()
            # global channel scale
            prop_global = random.uniform(0., 1.)
            if prop_global > 0.4:
                for i in range(3):
                    scale = np.random.uniform(self.min_rate, self.max_rate)
                    out[:, :, i] = out[:, :, i] * scale
            else: # local channel scale
                channel_scale_ratio = 0.4
                channel_scale_area = channel_scale_ratio * out.shape[0] * out.shape[1]
                aspect_ratio = random.uniform(0.4, 2.5)
                w = (aspect_ratio * channel_scale_area) ** 0.5
                h = w / aspect_ratio
                x_c = random.uniform(w/2., out.shape[0] - w/2.)
                y_c = random.uniform(h/2., out.shape[1] - h/2.)
                l = int(x_c - w/2.)
                r = int(x_c + w/2.)
                t = int(y_c - h/2.)
                b = int(y_c + h/2.)
                for i in range(3):
                    scale = np.random.uniform(self.min_rate, self.max_rate)
                    out[l:r, t:b, i] = out[l:r, t:b, i] * scale
                cv2.rectangle(seg_mask, (l, t), (r, b), color=(0, 0, 0), thickness=-1)
            out = np.maximum(out, 255)
            sample['image'] = out
            sample['seg_mask'] = seg_mask

        return sample


class GaussianNoise(object):
    def __init__(self, mean=0., sigma=0.3):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, sample):
        prop = random.uniform(0., 1.)
        if prop > 0.6:
            out = sample['image'].copy()
            h, w, c = out.shape
            noise = np.random.normal(self.mean, self.sigma, (h, w, c))
            out = out + noise
            out = np.maximum(out, 255)
            sample['image'] = out

        return sample


class CenterCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        pose = sample['pose']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[top: top + new_h, left: left + new_w]

        for i in range(landmarks.shape[0]):
            landmarks[i, 0] -= left
            landmarks[i, 1] -= top

        pose[3] -= left
        pose[4] -= top

        sample['image'] = image
        sample['landmarks'] = landmarks
        sample['pose'] = pose
        sample['image_cv2'] = image.astype(np.uint8)

        return sample


class TranslatePerturb(object):
    """Image translation. Perturb with a small shift"""

    def __init__(self, pixel):
        self.pixel = pixel
        self.min_rate = 0.6
        self.max_rate = 1.4
        self.mean = 0
        self.sigma = 0.4

    def channel_scale(self, image):
        out = image.copy()
        # global channel scale
        prop_global = random.uniform(0., 1.)
        if prop_global > 0.4:
            for i in range(3):
                scale = np.random.uniform(self.min_rate, self.max_rate)
                out[:, :, i] = out[:, :, i] * scale
        else: # local channel scale
            channel_scale_ratio = 0.4
            channel_scale_area = channel_scale_ratio * out.shape[0] * out.shape[1]
            aspect_ratio = random.uniform(0.4, 2.5)
            w = (aspect_ratio * channel_scale_area) ** 0.5
            h = w / aspect_ratio
            x_c = random.uniform(w/2., out.shape[0] - w/2.)
            y_c = random.uniform(h/2., out.shape[1] - h/2.)
            l = int(x_c - w/2.)
            r = int(x_c + w/2.)
            t = int(y_c - h/2.)
            b = int(y_c + h/2.)
            for i in range(3):
                scale = np.random.uniform(self.min_rate, self.max_rate)
                out[l:r, t:b, i] = out[l:r, t:b, i] * scale

        return out

    def add_noise(self, image):
        out = image.copy()
        h, w, c = out.shape
        noise = np.random.normal(self.mean, self.sigma, (h, w, c))
        out = out + noise

        return out

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        image = sample['image']
        landmarks = sample['landmarks']

        h, w = image.shape[:2]
        # hard code the perturb shift, usually it is a small value and need not be changed
        translations = (np.round(random.uniform(-self.pixel, self.pixel)),
                        np.round(random.uniform(-self.pixel, self.pixel)))
        
        t2d = np.array([translations[0], translations[1]]).reshape(1, 2)
        t3d = np.array([translations[0], translations[1], 0]).reshape(3, 1)
        perturb_landmarks = landmarks + t2d

        matrix = np.eye(3).astype(np.float32)

        matrix[0, 2] = translations[0]
        matrix[1, 2] = translations[1]
        dst_image = cv2.warpAffine(
            image, matrix[0:2, :], (h, w), cv2.INTER_LINEAR)
        if len(dst_image.shape) == 3:
            pass
        elif len(dst_image.shape) == 2:
            dst_image = dst_image.reshape(
                (dst_image.shape[0], dst_image.shape[1], 1))
        else:
            raise AssertionError('unexpected perturb_image_shape:', dst_image.shape)
        if random.uniform(0., 1.) > 0.5:
            dst_image = self.channel_scale(dst_image)
        if random.uniform(0., 1.) > 0.5:
            dst_image = self.add_noise(dst_image)

        # sample['perturb_image'] = TF.to_tensor(dst_image).float()
        sample['perturb_image'] = dst_image
        sample['perturb_landmarks'] = perturb_landmarks
        sample['shift_noise'] = np.array([translations[0], translations[1], 0])
        
        return sample


class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        image = sample['image']
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        sample['image'] = gray_image
        return sample

