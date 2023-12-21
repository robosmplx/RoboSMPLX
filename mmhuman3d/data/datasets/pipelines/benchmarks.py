import math
import random

# import albumentations
import mmcv
import numpy as np
import cv2

from ..builder import PIPELINES
from typing import Sequence
from mmhuman3d.core.conventions.keypoints_mapping import (
    get_keypoint_idx,
)


@PIPELINES.register_module()
class ControlledHardErasingKp:
    """Add random occlusion.

    Add random occlusion based on occlusion probability.

    Args:
        occlusion_prob (float): probability of the image having
        occlusion. Default: 0.5
    """

    def __init__(self, convention='smplx', keypoint='right_hand', p_size=500):
        self.convention = convention
        self.keypoint = keypoint
        self.synth_area = p_size

    def __call__(self, results):

        keypoints2d = results['keypoints2d'].copy()
        img = results['img']
        imgheight, imgwidth, _ = img.shape

        # get index of selected keypoint
        selected_kp_idx = get_keypoint_idx(
            self.keypoint, convention=self.convention)

        # get center of occluded box
        x, y, _ = keypoints2d.tolist()[selected_kp_idx]

        # get p size
        synth_ratio = 1
        synth_h = math.sqrt(self.synth_area * synth_ratio) # h
        synth_w = math.sqrt(self.synth_area / synth_ratio) # w

        # get p location
        synth_xmin = x - synth_w/2  + 1 # xmin
        synth_ymin = y - synth_h/2  + 1 # ymin

        if synth_xmin >= 0 and synth_ymin >= 0 and \
            synth_xmin + synth_w < imgwidth and \
                synth_ymin + synth_h < imgheight:
            synth_xmin = int(synth_xmin)
            synth_ymin = int(synth_ymin)
            synth_w = int(synth_w)
            synth_h = int(synth_h)
            img[synth_ymin:synth_ymin + synth_h, synth_xmin:synth_xmin +
                synth_w, :] = np.random.rand(synth_h, synth_w, 3) * 255


        results['img'] = img

        return results
    

@PIPELINES.register_module()
class ControlledTranslate(object):
    """Translate images by specified magnitude.
    Args:
        magnitude (int | float): The magnitude used for translate. Note that
            the offset is calculated by magnitude * size in the corresponding
            direction. With a magnitude of 1, the whole image will be moved out
            of the range.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing translate therefore should
             be in range [0, 1]. Defaults to 0.5.
        direction (str): The translating direction. Options are 'horizontal'
            and 'vertical'. Defaults to 'horizontal'.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
    """

    def __init__(self,
                 magnitude,
                 pad_val=128,
                 prob=0.5,
                 direction='horizontal',
                 random_negative_prob=0.5,
                 bordermode='wrap',
                 interpolation='nearest'):
        assert isinstance(magnitude, (int, float)), 'The magnitude type must '\
            f'be int or float, but got {type(magnitude)} instead.'
        if isinstance(pad_val, int):
            pad_val = tuple([pad_val] * 3)
        elif isinstance(pad_val, Sequence):
            assert len(pad_val) == 3, 'pad_val as a tuple must have 3 ' \
                f'elements, got {len(pad_val)} instead.'
            assert all(isinstance(i, int) for i in pad_val), 'pad_val as a '\
                'tuple must got elements of int type.'
        else:
            raise TypeError('pad_val must be int or tuple with 3 elements.')
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert direction in ('horizontal', 'vertical'), 'direction must be ' \
            f'either "horizontal" or "vertical", got {direction} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.magnitude = magnitude
        self.pad_val = tuple(pad_val)
        self.prob = prob
        self.direction = direction
        self.random_negative_prob = random_negative_prob
        self.interpolation = interpolation
        bordermode_dict = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'reflect_101': cv2.BORDER_REFLECT_101,
            'wrap': cv2.BORDER_WRAP,
        }
        self.bordermode = bordermode_dict[bordermode]
        
    def __call__(self, results):
        # if np.random.rand() > self.prob:
        #     return results
        # magnitude = random_negative(self.magnitude, self.random_negative_prob)
        for key in results.get('img_fields', ['img']):
            img = results[key]
            height, width = img.shape[:2]
            if self.direction == 'horizontal':
                offset = self.magnitude * width
            else:
                offset = self.magnitude * height
            img_translated = mmcv.imtranslate(
                img,
                offset,
                direction=self.direction,
                border_mode=self.bordermode,
                border_value=self.pad_val,
                interpolation=self.interpolation)
            results[key] = img_translated.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'direction={self.direction}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation})'
        return 
    

@PIPELINES.register_module()
class ControlledScale:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.
    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, scale_factor=0.25):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        s = results['scale']

        sf = self.scale_factor

        s_factor = sf + 1
        s = s * s_factor

        results['scale'] = s
        results['rotation'] = 0.

        return results
    

@PIPELINES.register_module()
class ControlledColorTransform(object):
    """Adjust images color balance.
    Args:
        magnitude (int | float): The magnitude used for color transform. A
            positive magnitude would enhance the color and a negative magnitude
            would make the image grayer. A magnitude=0 gives the origin img.
        prob (float): The probability for performing ColorTransform therefore
            should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.5.
    """

    def __init__(self, magnitude, prob=0.5, random_negative_prob=0.5, ctype='grayness'):
        assert isinstance(magnitude, (int, float)), 'The magnitude type must '\
            f'be int or float, but got {type(magnitude)} instead.'
        assert 0 <= prob <= 1.0, 'The prob should be in range [0,1], ' \
            f'got {prob} instead.'
        assert 0 <= random_negative_prob <= 1.0, 'The random_negative_prob ' \
            f'should be in range [0,1], got {random_negative_prob} instead.'

        self.magnitude = magnitude
        self.prob = prob
        self.random_negative_prob = random_negative_prob
        self.ctype = ctype

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.ctype == 'grayness':
                img_color_adjusted = mmcv.adjust_color(img, alpha=1 + self.magnitude)
            elif self.ctype == 'brightness':
                img_color_adjusted = mmcv.adjust_brightness(img, factor=1 + self.magnitude)
            elif self.ctype == 'contrast':
                img_color_adjusted = mmcv.adjust_contrast(img, factor=1 + self.magnitude)
            elif self.ctype == 'sharpness':
                img_color_adjusted = mmcv.adjust_sharpness(img, factor=1 + self.magnitude) # [0., 2.]
            elif self.ctype == 'hue':
                img_color_adjusted = mmcv.adjust_hue(img, hue_factor=self.magnitude) # [-0.5, 0.5]
            results[key] = img_color_adjusted.astype(img.dtype)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(magnitude={self.magnitude}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob})'
        return repr_str


@PIPELINES.register_module()
class ControlledRotation:
    """Data augmentation with random scaling & rotating.

    Required key: 'scale'. Modifies key: 'scale' and 'rotation'.
    Args:
        rot_factor (int): Rotating to ``[-2*rot_factor, 2*rot_factor]``.
        scale_factor (float): Scaling to ``[1-scale_factor, 1+scale_factor]``.
        rot_prob (float): Probability of random rotation.
    """

    def __init__(self, rot_factor=30):
        self.rot_factor = rot_factor

    def __call__(self, results):
        """Perform data augmentation with random scaling & rotating."""
        results['rotation'] = self.rot_factor

        return results
    
@PIPELINES.register_module()
class ControlledLowRes(object):

    def __init__(self,
                 dist: str = 'categorical',
                 factor: float = 1.0,
                 cat_factors=(1.0, ),
                 factor_min: float = 1.0,
                 factor_max: float = 1.0) -> None:
        self.factor=factor

    def _sample_low_res(self, image: np.ndarray) -> np.ndarray:
        """"""

        H, W, _ = image.shape
        downsampled_image = cv2.resize(image,
                                       (int(W // self.factor), int(H // self.factor)),
                                       cv2.INTER_NEAREST)
        resized_image = cv2.resize(downsampled_image, (W, H),
                                   cv2.INTER_LINEAR_EXACT)
        return resized_image

    def __call__(self, results):
        """"""
        img = results['img']
        img = self._sample_low_res(img)
        results['img'] = img

        return results
