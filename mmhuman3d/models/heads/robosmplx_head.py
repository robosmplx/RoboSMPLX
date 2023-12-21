import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmcv.runner.base_module import BaseModule
import pickle
from mmhuman3d.utils.geometry import rot6d_to_rotmat, rot6d_to_aa
# from mmhuman3d.utils.transforms import rot6d_to_aa
from .builder import HEADS
from mmhuman3d.models.heads.pare_head import softargmax2d, interpolate
from mmcv.cnn import (
    build_conv_layer,
    build_norm_layer
)
from mmhuman3d.models.body_models.builder import build_body_model

def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
                ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def make_linear_layers(feat_dims, relu_final=True, use_bn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_bn:
                layers.append(nn.BatchNorm1d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth*height*width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2,3))
    accu_y = heatmap3d.sum(dim=(2,4))
    accu_z = heatmap3d.sum(dim=(3,4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None,None,:]
    accu_y = accu_y * torch.arange(height).float().cuda()[None,None,:]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None,None,:]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out

@HEADS.register_module()
class RoboSMPLXHandHeadv5Dilated(BaseModule):

    def __init__(self,
                 feat_dim,
                 mean_pose_path=None,
                 joint_num=21,
                 npose=90,
                 norient=6,
                 nbeta=10,
                 ncam=3,
                 hdim=1024,
                 init_cfg=None,
                 output_hm_shape=[8, 8, 8],
                 use_heatmap_all=False,
                 use_heatmap_pose=False,
                 full=True):
        super(RoboSMPLXHandHeadv5Dilated, self).__init__(init_cfg=init_cfg)
        self.full = full

        self.joint_num = joint_num
        self.output_hm_shape = output_hm_shape

        int_feat_dim = hdim + 3

        self.use_heatmap_all = use_heatmap_all
        self.use_heatmap_pose = use_heatmap_pose

        
        if self.use_heatmap_all:
            pose_feat_dim =  feat_dim + self.joint_num*self.output_hm_shape[0]
            shapecam_feat_dim =hdim

        if self.use_heatmap_pose:
            pose_feat_dim =  feat_dim + self.joint_num*self.output_hm_shape[0]
            shapecam_feat_dim = feat_dim

        self.hand_conv = make_conv_layers([pose_feat_dim,hdim], kernel=1, stride=1, padding=0)
            
        # shape and camera from imge features
        if self.full:
            self.decshape = make_linear_layers([shapecam_feat_dim, nbeta], relu_final=False)
            self.deccam = make_linear_layers([shapecam_feat_dim, ncam], relu_final=False)
            self.decorient = make_linear_layers([hdim, norient], relu_final=False)
        self.decpose = make_linear_layers([hdim, npose], relu_final=False)

        
        self.position_conv = make_conv_layers([feat_dim, self.joint_num*self.output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)
        
        part_num = 16
        num_deconv_layers=3
        df_dim = 128
        self.num_input_features = 128
        self.norm_cfg = dict(type='BN')
        num_deconv_filters=(df_dim, df_dim, df_dim)
        final_conv_kernel = 1
        self.upsample_stage_4 = self._make_upsample_layer(1, num_channel=512)
        self.keypoint_deconv_layers = self._make_conv_layer(
            num_deconv_layers,
            num_deconv_filters,
            (3, ) * num_deconv_layers,
        )
        self.keypoint_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels = part_num + 1,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=0,
        )

    def _make_upsample_layer(self, num_layers, num_channel, kernel_size=3):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True))
            if i == int(num_layers - 1):
                num_out_channel = int(num_channel/4)
            else:
                num_out_channel = num_channel
            layers.append(
                build_conv_layer(
                    cfg=None,
                    in_channels=num_channel,
                    out_channels=num_out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                    bias=False,
                ))
            layers.append(build_norm_layer(self.norm_cfg, num_out_channel)[1])
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        """make convolution layers."""
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """get deconv padding, output padding according to kernel size."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
            
    def forward(self,
                img_feat,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                n_iter=3):

        if isinstance(img_feat, list) or isinstance(img_feat, tuple):
            img_feat = img_feat[-1]
        batch_size = img_feat.shape[0]

        # seg mask
        inputs = self.upsample_stage_4(img_feat)
        part_feats = self.keypoint_deconv_layers(inputs) 
        heatmaps = self.keypoint_final_layer(part_feats)

        pos_hm =  self.position_conv(img_feat)
        joint_hm = pos_hm.view(-1, self.joint_num, self.output_hm_shape[0], self.output_hm_shape[1], self.output_hm_shape[2]) # bs, 20-num_joints, 8, 8, 8
        joint_coord = soft_argmax_3d(joint_hm)

        if self.use_heatmap_all:
            img_feat = torch.cat([img_feat, pos_hm], 1)
            img_feat = self.hand_conv(img_feat)

        if self.full:
            pred_shape = self.decshape(img_feat.mean((2,3)))
            pred_cam = self.deccam(img_feat.mean((2,3)))

        if self.use_heatmap_pose:
            img_feat = torch.cat([img_feat, pos_hm], 1)
            img_feat = self.hand_conv(img_feat)

        pred_pose = self.decpose(img_feat.mean((2,3)))
        right_hand_pose = rot6d_to_rotmat(pred_pose).view(batch_size, 15, 3, 3)


        if self.full:
            pred_orient = self.decorient(img_feat.mean((2,3)))
            global_orient = rot6d_to_rotmat(pred_orient).view(batch_size, 1, 3, 3)



        if self.full:
            params_dict = {
                'global_orient': global_orient,
                'betas': pred_shape,
                'right_hand_pose': right_hand_pose
            }
            raw_dict = {
                'joint_coord': joint_coord,
                'pred_segm_mask': heatmaps,
            }
            return {
                'pred_param': params_dict,
                'pred_cam': pred_cam,
                'pred_raw': raw_dict
            }
        else:
            raw_dict = {
                'joint_coord': joint_coord,
            }
            params_dict = {
                'right_hand_pose': right_hand_pose
            }
            return {
                'pred_param': params_dict,
                'pred_raw': raw_dict
            }
        

@HEADS.register_module()
class RoboSMPLXFaceHeadv5Dilated(BaseModule):

    def __init__(self,
                 feat_dim,
                 mean_pose_path=None,
                 joint_num=73,
                 npose=6,
                 norient=6,
                 nexp=10,
                 nbeta=10,
                 ncam=3,
                 hdim=1024,
                 output_hm_shape=[32, 32, 32],
                 init_cfg=None,
                 use_heatmap_all=False,
                 use_heatmap_pose=False,
                 full=True):
        super(RoboSMPLXFaceHeadv5Dilated, self).__init__(init_cfg=init_cfg)

        self.full = full

        self.joint_num = joint_num
        self.output_hm_shape = output_hm_shape

        self.use_heatmap_all = use_heatmap_all
        self.use_heatmap_pose = use_heatmap_pose

        if self.use_heatmap_all:
            pose_feat_dim =  feat_dim + self.joint_num*self.output_hm_shape[0]
            shapecam_feat_dim = hdim

        if self.use_heatmap_pose:
            pose_feat_dim =  feat_dim + self.joint_num*self.output_hm_shape[0]
            shapecam_feat_dim = feat_dim

        self.face_conv = make_conv_layers([pose_feat_dim,hdim], kernel=1, stride=1, padding=0)

        if self.full:
            self.decorient = make_linear_layers([shapecam_feat_dim, norient], relu_final=False)
            self.decshape = make_linear_layers([shapecam_feat_dim, nbeta], relu_final=False)
            self.deccam = make_linear_layers([shapecam_feat_dim, ncam], relu_final=False)
            self.decexp = make_linear_layers([shapecam_feat_dim, nexp], relu_final=False)
        self.decpose = make_linear_layers([hdim, npose], relu_final=False)

        self.position_conv = make_conv_layers([feat_dim, self.joint_num*self.output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

        part_num = 14
        num_deconv_layers=3
        df_dim = 128
        self.num_input_features = 128
        self.norm_cfg = dict(type='BN')
        num_deconv_filters=(df_dim, df_dim, df_dim)
        final_conv_kernel = 1
        self.upsample_stage_4 = self._make_upsample_layer(1, num_channel=512)
        self.keypoint_deconv_layers = self._make_conv_layer(
            num_deconv_layers,
            num_deconv_filters,
            (3, ) * num_deconv_layers,
        )
        self.keypoint_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels = part_num + 1,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=0,
        )

    def _make_upsample_layer(self, num_layers, num_channel, kernel_size=3):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True))
            if i == int(num_layers - 1):
                num_out_channel = int(num_channel/4)
            else:
                num_out_channel = num_channel
            layers.append(
                build_conv_layer(
                    cfg=None,
                    in_channels=num_channel,
                    out_channels=num_out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                    bias=False,
                ))
            layers.append(build_norm_layer(self.norm_cfg, num_out_channel)[1])
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        """make convolution layers."""
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1)) 
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """get deconv padding, output padding according to kernel size."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
    
    def forward(self,
                img_feat,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                n_iter=3):

        # hmr head only support one layer feature
        if isinstance(img_feat, list) or isinstance(img_feat, tuple):
            img_feat = img_feat[-1] # bs, 2048, 8, 8
        batch_size = img_feat.shape[0]

        # seg mask
        inputs = self.upsample_stage_4(img_feat)
        part_feats = self.keypoint_deconv_layers(inputs) 
        heatmaps = self.keypoint_final_layer(part_feats)

        pos_hm = self.position_conv(img_feat)
        joint_hm = pos_hm.view(-1, self.joint_num, self.output_hm_shape[0], self.output_hm_shape[1], self.output_hm_shape[2]) # bs, 20-num_joints, 8, 8, 8
        joint_coord = soft_argmax_3d(joint_hm) 

        if self.use_heatmap_all:
            img_feat = torch.cat([img_feat, pos_hm], 1)
            img_feat = self.face_conv(img_feat)

        if self.full:
            pred_shape = self.decshape(img_feat.mean((2,3)))
            pred_cam = self.deccam(img_feat.mean((2,3)))
            pred_exp = self.decexp(img_feat.mean((2,3))) # expression parameter

        if self.use_heatmap_pose:
            img_feat = torch.cat([img_feat, pos_hm], 1)
            img_feat = self.face_conv(img_feat)
        
        jaw_pose = self.decpose(img_feat.mean((2,3))) # jaw pose parameter
        jaw_pose = rot6d_to_rotmat(jaw_pose).view(batch_size, 1, 3, 3)

        if self.full:
            pred_orient = self.decorient(img_feat.mean((2,3)))
            global_orient = rot6d_to_rotmat(pred_orient).view(batch_size, 1, 3, 3)


        if self.full:
            params_dict = {
                'global_orient': global_orient,
                'betas': pred_shape,
                'jaw_pose': jaw_pose,
                'expression': pred_exp
            }
            raw_dict = {
                'joint_coord_large': joint_coord,
                'pred_segm_mask': heatmaps,
            }
            return {
                'pred_param': params_dict,
                'pred_cam': pred_cam,
                'pred_raw': raw_dict
            }
        else:
            params_dict = {
                'jaw_pose': jaw_pose,
                'expression': pred_exp
            }
            return {
                'pred_param': params_dict,
            }
        
@HEADS.register_module()
class RoboSMPLXBodyHeadv5Dilated(BaseModule):

    def __init__(self,
                 feat_dim,
                 joint_num=24,
                 npose=138,
                 norient=6,
                 nbeta=10,
                 ncam=3,
                 hdim=1024,
                 init_cfg=None,
                 output_hm_shape=[8, 8, 8],
                 use_heatmap_all=False,
                 use_heatmap_pose=False,
                 full=True):
        super(RoboSMPLXBodyHeadv5Dilated, self).__init__(init_cfg=init_cfg)

        self.full = full

        self.joint_num = joint_num
        self.output_hm_shape = output_hm_shape

        int_feat_dim = hdim + 3

        self.use_heatmap_all = use_heatmap_all
        self.use_heatmap_pose = use_heatmap_pose

        
        if self.use_heatmap_all:
            pose_feat_dim =  feat_dim + self.joint_num*self.output_hm_shape[0]
            shapecam_feat_dim =hdim

        if self.use_heatmap_pose:
            pose_feat_dim =  feat_dim + self.joint_num*self.output_hm_shape[0]
            shapecam_feat_dim = feat_dim

        self.hand_conv = make_conv_layers([pose_feat_dim,hdim], kernel=1, stride=1, padding=0)
            
        if self.full:
            self.decshape = make_linear_layers([shapecam_feat_dim, nbeta], relu_final=False)
            self.deccam = make_linear_layers([shapecam_feat_dim, ncam], relu_final=False)
            self.decorient = make_linear_layers([hdim, norient], relu_final=False)
        self.decpose = make_linear_layers([hdim, npose], relu_final=False)

        
        self.position_conv = make_conv_layers([feat_dim, self.joint_num*self.output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

        part_num = 24
        num_deconv_layers=3
        df_dim = 128
        self.num_input_features = 128
        self.norm_cfg = dict(type='BN')
        num_deconv_filters=(df_dim, df_dim, df_dim)
        final_conv_kernel = 1
        self.upsample_stage_4 = self._make_upsample_layer(1, num_channel=512)
        self.keypoint_deconv_layers = self._make_conv_layer(
            num_deconv_layers,
            num_deconv_filters,
            (3, ) * num_deconv_layers,
        )
        self.keypoint_final_layer = nn.Conv2d(
            in_channels=num_deconv_filters[-1],
            out_channels = part_num + 1,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=0,
        )

    def _make_upsample_layer(self, num_layers, num_channel, kernel_size=3):
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Upsample(
                    scale_factor=2, mode='bilinear', align_corners=True))
            if i == int(num_layers - 1):
                num_out_channel = int(num_channel/4)
            else:
                num_out_channel = num_channel
            layers.append(
                build_conv_layer(
                    cfg=None,
                    in_channels=num_channel,
                    out_channels=num_out_channel,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1,
                    bias=False,
                ))
            layers.append(build_norm_layer(self.norm_cfg, num_out_channel)[1])
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_conv_layer(self, num_layers, num_filters, num_kernels):
        """make convolution layers."""
        assert num_layers == len(num_filters), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_conv_layers is different len(num_conv_filters)'
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                nn.Conv2d(
                    in_channels=self.num_input_features,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=1,
                    padding=padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1)) 
            layers.append(nn.ReLU(inplace=True))
            self.num_input_features = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel):
        """get deconv padding, output padding according to kernel size."""
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding
            
    def forward(self,
                img_feat,
                init_pose=None,
                init_shape=None,
                init_cam=None,
                n_iter=3):

        if isinstance(img_feat, list) or isinstance(img_feat, tuple):
            img_feat = img_feat[-1] 
        batch_size = img_feat.shape[0]

        inputs = self.upsample_stage_4(img_feat)
        part_feats = self.keypoint_deconv_layers(inputs) 
        heatmaps = self.keypoint_final_layer(part_feats)

        pos_hm =  self.position_conv(img_feat)
        joint_hm = pos_hm.view(-1, self.joint_num, self.output_hm_shape[0], self.output_hm_shape[1], self.output_hm_shape[2]) 
        joint_coord = soft_argmax_3d(joint_hm) 

        if self.use_heatmap_all:
            img_feat = torch.cat([img_feat, pos_hm], 1)
            img_feat = self.hand_conv(img_feat)

        if self.full:
            pred_shape = self.decshape(img_feat.mean((2,3)))
            pred_cam = self.deccam(img_feat.mean((2,3)))

        if self.use_heatmap_pose:
            img_feat = torch.cat([img_feat, pos_hm], 1)
            img_feat = self.hand_conv(img_feat)

        body_pose = self.decpose(img_feat.mean((2,3)))
        pred_orient = self.decorient(img_feat.mean((2,3)))


        pred_pose = torch.cat([pred_orient, body_pose], 1)

        raw_dict = {
            'joint_coord_large': joint_coord,
            'pred_segm_mask': heatmaps,
        }

        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, 24, 3, 3)

        output = {
            'pred_pose': pred_rotmat,
            'pred_shape': pred_shape,
            'pred_cam': pred_cam,
            'pred_raw': raw_dict
        }
        return output