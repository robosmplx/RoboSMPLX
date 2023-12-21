_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(interval=5, metric=['3DRMSE'])

optimizer = dict(
    backbone=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    head=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[60, 100], gamma=0.1)

runner = dict(type='EpochBasedRunner', max_epochs=100)

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

checkpoint_config = dict(interval=10)

# model settings

find_unused_parameters = True

model = dict(
    type='SMPLXImageBodyModelEstimator',
    backbone=dict(
        type='DilatedResNet',
        block='bottleneck',
        layers=[1, 1, 3, 4, 6, 3, 1, 1],
        arch='D',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='data/checkpoints/drn_d_54-0e0534ff.pth',
            )
        ),
    head=dict(
        type='RoboSMPLXFaceHeadv5Dilated',
        feat_dim=512,
        hdim=512,
        output_hm_shape=[32, 32, 32],
        nexp=50, 
        nbeta=100,
        use_heatmap_all=True,
        mean_pose_path='data/body_models/smpl_mean_params.npz'),
    body_model_train=dict(
        type='flamelayer',
        num_expression_coeffs=50,
        num_betas=100,
        use_pca=False,
        use_face_contour=True,
        model_path='data/body_models/flame',
        keypoint_src='flame',
        keypoint_dst='flame',
    ),
    body_model_test=dict(
        type='flamelayer',
        num_expression_coeffs=50,
        num_betas=100,
        use_pca=False,
        use_face_contour=True,
        model_path='data/body_models/flame',
        keypoint_src='flame',
        keypoint_dst='face3d',
    ),
    loss_keypoints2d=dict(type='L1Loss', reduction='sum', loss_weight=1),
    loss_joint_img=dict(type='L1Loss', loss_weight=10),
    loss_proj_mask=dict(type='CrossEntropyLoss', loss_weight=10),
    loss_contrastive_smpl_keypoint=dict(type='L1Loss', loss_weight=2),
    loss_smplx_global_orient=dict(
        type='RotationDistance', reduction='sum', loss_weight=1),
    loss_smplx_jaw_pose=dict(
        type='RotationDistance', reduction='sum', loss_weight=1),
    loss_smplx_expression=dict(type='MSELoss', reduction='sum', loss_weight=1),
    loss_smplx_betas_prior=dict(
        type='ShapeThresholdPriorLoss', margin=3.0, norm='l2', loss_weight=1),
    convention='flame',
    get_positive_samples=True
    )

# dataset settings
dataset_type = 'HumanImageSMPLXDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smplx', 'has_keypoints3d', 'has_keypoints2d',
    'has_smplx_global_orient', 'has_smplx_jaw_pose', 'has_smplx_betas',
    'has_smplx_expression', 'smplx_jaw_pose', 'smplx_global_orient',
    'smplx_betas', 'keypoints2d', 'keypoints3d', 'sample_idx',
    'smplx_expression'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BBoxCenterJitter', factor=0.2, dist='uniform'),
    dict(type='RandomHorizontalFlip', flip_prob=0.5,
         convention='flame'),  # hand = 0,head = body = 0.5
    dict(
        type='GetRandomScaleRotation',
        rot_factor=30.0,
        scale_factor=0.2,
        rot_prob=0.6),
    dict(type='MeshAffine', img_res=256),  # hand = 224, body = head = 256
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'ori_img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation', 'ori_img'])
]
img_res = 256
inference_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    # dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', 'sample_idx'],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=['image_path', 'center', 'scale', 'rotation'])
]
cache_files = {
    'ffhq': 'data/cache/ffhq_train_flame.npz',
    'affectnet': 'data/cache/affectnet_train.npz',
    'bupt': 'data/cache/bupt_train.npz'
}
data = dict(
    samples_per_gpu=64,  # body 48, head = hand = 64
    workers_per_gpu=8,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                pipeline=train_pipeline,
                dataset_name='ffhq',
                data_prefix='data',
                ann_file='ffhq_flame_train.npz',
                convention='flame',
                num_betas=100,
                num_expression=50,
                cache_data_path=cache_files['ffhq'],
            ),
            dict(
                type=dataset_type,
                pipeline=train_pipeline,
                dataset_name='bupt',
                data_prefix='data',
                ann_file='bupt_train.npz',
                convention='flame',
                num_betas=100,
                num_expression=50,
                cache_data_path=cache_files['bupt'],
            ),
            dict(
                type=dataset_type,
                pipeline=train_pipeline,
                dataset_name='affectnet',
                data_prefix='data',
                ann_file='affectnet_train.npz',
                convention='flame',
                num_betas=100,
                num_expression=50,
                cache_data_path=cache_files['affectnet'],
            ),
        ],
        partition=[0.5, 0.2, 0.3],
    ),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='flamelayer',
            num_expression_coeffs=50,
            num_betas=100,
            use_pca=False,
            use_face_contour=True,
            model_path='data/body_models/flame',
            keypoint_src='flame',
            keypoint_dst='face3d',
        ),
        dataset_name='stirling',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='stirling_ESRC3D_HQ.npz',
        convention='face3d'),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='flamelayer',
            num_expression_coeffs=50,
            num_betas=100,
            use_pca=False,
            use_face_contour=True,
            model_path='data/body_models/flame',
            keypoint_src='flame',
            keypoint_dst='face3d',
        ),
        dataset_name='stirling',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='stirling_ESRC3D_HQ.npz',
        convention='face3d'),
)

custom_imports = dict(
    imports=[
        'mmhuman3d.models.heads.robosmplx_head', 'mmhuman3d.models.backbones.dilated_resnet',
        'mmhuman3d.data.datasets.pipelines.extra_transforms'
    ],
    allow_failed_imports=False)