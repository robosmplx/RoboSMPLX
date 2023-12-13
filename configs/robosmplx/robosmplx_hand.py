_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(interval=1, metric=['pa-mpjpe', 'pa-pve','mpjpe', 'pve'])

optimizer = dict(
    backbone=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    head=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[30, 60], gamma=0.1)

runner = dict(type='EpochBasedRunner', max_epochs=60)

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
        type='RoboSMPLXHandHeadv5Dilated',
        feat_dim=512,
        hdim=512,
        output_hm_shape=[32, 32, 32],
        use_heatmap_all=True,
        mean_pose_path='data/body_models/smpl_mean_params.npz'),
    body_model_train=dict(
        type='MANOLayer',
        num_expression_coeffs=10,
        num_betas=10,
        use_pca=False,
        flat_hand_mean=False,
        model_path='data/body_models/mano',
        keypoint_src='mano_right_reorder',
        keypoint_dst='mano_right_reorder',
    ),
    body_model_test=dict(
        type='MANOLayer',
        num_expression_coeffs=10,
        num_betas=10,
        use_pca=False,
        flat_hand_mean=False,
        model_path='data/body_models/mano',
        keypoint_src='mano_right_reorder',
        keypoint_dst='mano_right_reorder',
    ),
    loss_joint_img=dict(type='L1Loss', reduction='sum', loss_weight=1),
    loss_keypoints2d=dict(type='L1Loss', reduction='sum', loss_weight=1),
    loss_keypoints3d=dict(type='L1Loss', reduction='sum', loss_weight=1),
    loss_proj_mask=dict(type='CrossEntropyLoss', loss_weight=10),
    loss_contrastive_smpl_keypoint=dict(type='L1Loss', loss_weight=2),
    loss_smplx_global_orient=dict(
        type='RotationDistance', reduction='sum', loss_weight=1),
    loss_smplx_hand_pose=dict(
        type='RotationDistance', reduction='sum', loss_weight=1),
    loss_smplx_betas_prior=dict(
        type='ShapeThresholdPriorLoss', margin=3.0, norm='l2', loss_weight=1),
    convention='mano_right_reorder',
    get_positive_samples=True)

# dataset settings
dataset_type = 'HumanImageSMPLXDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smplx', 'has_keypoints3d', 'has_keypoints2d',
    'has_smplx_global_orient', 'has_smplx_right_hand_pose',
    'has_smplx_left_hand_pose', 'has_smplx_betas', 'smplx_right_hand_pose',
    'smplx_global_orient', 'smplx_betas', 'keypoints2d', 'keypoints3d',
    'sample_idx'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BBoxCenterJitter', factor=0.2, dist='uniform'),
    dict(type='RandomHorizontalFlip', flip_prob=0,
         convention='mano_right_reorder'),  # hand = 0,head = body = 0.5
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
    'freihand': 'data/cache/freihand_train_full.npz',
}
data = dict(
    samples_per_gpu=64,  # body 48, head = hand = 64
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        dataset_name='FreiHand',
        data_prefix='data',
        ann_file='freihand_train_full.npz',
        convention='mano_right_reorder',
        num_betas=10,
        num_expression=10,
        cache_data_path=cache_files['freihand']),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='MANOLayer',
            num_expression_coeffs=10,
            num_betas=10,
            use_pca=False,
            flat_hand_mean=False,
            model_path='data/body_models/mano',
            keypoint_src='mano_right_reorder',
            keypoint_dst='mano_right_reorder',
        ),
        dataset_name='FreiHand',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='freihand_test_full.npz',
        convention='mano_right_reorder',
    ),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='MANOLayer',
            num_expression_coeffs=10,
            num_betas=10,
            use_pca=False,
            flat_hand_mean=False,
            model_path='data/body_models/mano',
            keypoint_src='mano_right_reorder',
            keypoint_dst='mano_right_reorder',
        ),
        dataset_name='FreiHand',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='freihand_test_full.npz',
        convention='mano_right_reorder',
    ),
)

custom_imports = dict(
    imports=[
        'mmhuman3d.models.heads.robosmplx_part_head', 'mmhuman3d.models.backbones.dilated_resnet',
        'mmhuman3d.data.datasets.pipelines.nips'
    ],
    allow_failed_imports=False)