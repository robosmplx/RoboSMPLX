_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(metric=['pa-mpjpe', 'mpjpe'])
# optimizer
optimizer = dict(
    backbone=dict(type='Adam', lr=2.5e-4),
    head=dict(type='Adam', lr=2.5e-4)
    )
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=50)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
checkpoint_config = dict(interval=10)

img_res = 256

# model settings
model = dict(
    type='ImageBodyModelEstimator',
    backbone=dict(
        type='DilatedResNet',
        block='bottleneck',
        layers=[1, 1, 3, 4, 6, 3, 1, 1],
        arch='D',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='expts/exp_20230313_dres53_coco_256x256/best_AP_epoch_180.pth',
            prefix='backbone'
            )
        ),
    head=dict(
        type='RoboSMPLXBodyHeadv5Dilated',
        feat_dim=512,
        hdim=512,
        output_hm_shape=[32, 32, 32],
        use_heatmap_all=True,
        ),
    body_model_train=dict(
        type='SMPL',
        keypoint_src='smpl_54',
        keypoint_dst='smpl_54',
        model_path='data/body_models/smpl',
        keypoint_approximate=True,
        extra_joints_regressor='data/body_models/J_regressor_extra.npy'),
    body_model_test=dict(
        type='SMPL',
        keypoint_src='h36m',
        keypoint_dst='h36m',
        model_path='data/body_models/smpl',
        joints_regressor='data/body_models/J_regressor_h36m.npy'),
    convention='smpl_54',
    loss_keypoints3d=dict(type='L1Loss', loss_weight=100),
    loss_keypoints2d=dict(type='L1Loss', loss_weight=10),
    loss_joint_img=dict(type='L1Loss', loss_weight=10),
    loss_proj_mask=dict(type='CrossEntropyLoss', loss_weight=10),
    loss_contrastive_smpl_keypoint=dict(type='L1Loss', loss_weight=2),
    loss_vertex=dict(type='L1Loss', loss_weight=2),
    loss_smpl_pose=dict(type='L1Loss', loss_weight=3),
    loss_smpl_betas=dict(type='L1Loss', loss_weight=0.02),
    get_positive_samples=True
    )
# dataset settings
dataset_type = 'HumanImageDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smpl', 'smpl_body_pose', 'smpl_global_orient', 'smpl_betas',
    'smpl_transl', 'keypoints2d', 'keypoints3d', 'sample_idx'
]
train_data_keys = data_keys + ['keypoints2d_contrastive']
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smpl_54'),
    dict(type='GetRandomScaleRotation', rot_factor=30, scale_factor=0.25),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'img_contrastive']),
    dict(type='ToTensor', keys=train_data_keys),
    dict(
        type='Collect',
        keys=['img', 'img_contrastive', *train_data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

inference_pipeline = [
    dict(type='MeshAffine', img_res=img_res),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]

cache_files = {
    'h36m': 'data/cache/h36m_mosh_train_smpl_54.npz',
    'eft_lspet': 'data/cache/eft_lspet_train_smpl_54.npz',
    'eft_mpii': 'data/cache/eft_mpii_train_smpl_54.npz',
    'eft_coco': 'data/cache/eft_coco_train_smpl_54.npz',
    'spin_mpi_inf_3dhp': 'data/cache/spin_mpi_inf_3dhp_train_smpl_54.npz'
}
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                dataset_name='h36m',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_54',
                cache_data_path=cache_files['h36m'],
                ann_file='h36m_mosh_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='coco',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_54',
                cache_data_path=cache_files['eft_coco'],
                ann_file='eft_coco_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='lspet',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_54',
                cache_data_path=cache_files['eft_lspet'],
                ann_file='eft_lspet_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='mpii',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_54',
                cache_data_path=cache_files['eft_mpii'],
                ann_file='eft_mpii_train.npz'),
            dict(
                type=dataset_type,
                dataset_name='mpi_inf_3dhp',
                data_prefix='data',
                pipeline=train_pipeline,
                convention='smpl_54',
                cache_data_path=cache_files['spin_mpi_inf_3dhp'],
                ann_file='spin_mpi_inf_3dhp_train.npz'),
        ],
        partition=[0.5, 0.233, 0.046, 0.021, 0.2],
    ),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='GenderedSMPL',
            keypoint_src='h36m',
            keypoint_dst='h36m',
            model_path='data/body_models/smpl',
            joints_regressor='data/body_models/J_regressor_h36m.npy'),
        dataset_name='pw3d',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='pw3d_test.npz'),
)

custom_imports = dict(
    imports=[
        'mmhuman3d.models.heads.robosmplx_head', 'mmhuman3d.models.backbones.dilated_resnet',
        'mmhuman3d.data.datasets.pipelines.nips',
        'mmhuman3d.data.datasets.pipelines.mmselfsup'
    ],
    allow_failed_imports=False)
