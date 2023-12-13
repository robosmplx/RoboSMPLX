_base_ = ['../_base_/default_runtime.py']
use_adversarial_train = True

# evaluate
evaluation = dict(
    interval=1,
    metric=['pa-mpjpe', 'mpjpe', 'pa-pve', 'pve'],
    body_part=[['body', 'right_hand', 'left_hand', 'J14'],
               ['body', 'right_hand', 'left_hand', 'J14'],
               ['', 'right_hand', 'left_hand', 'face'],
               ['', 'right_hand', 'left_hand', 'face']])

optimizer = dict(
    backbone=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    head=dict(type='Adam', lr=1.0e-4, weight_decay=1.0e-4),
    )
# optimizer = dict(
#     backbone=dict(type='Adam', lr=1.0e-5, weight_decay=1.0e-5),
#     head=dict(type='Adam', lr=1.0e-5, weight_decay=1.0e-5))
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[10, 20], gamma=0.1)

runner = dict(type='EpochBasedRunner', max_epochs=20)

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

img_res = 224

checkpoint_config = dict(interval=10)
face_vertex_ids_path = 'data/body_models/smplx/SMPL-X__FLAME_vertex_ids.npy'
hand_vertex_ids_path = 'data/body_models/smplx/MANO_SMPLX_vertex_ids.pkl'

model = dict(
    type='H4WImageBodyModelEstimator',
    backbone=dict(
        type='DilatedResNet',
        block='bottleneck',
        layers=[1, 1, 3, 4, 6, 3, 1, 1],
        arch='D',
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='expts/robosmplx_body.pth',
            prefix='backbone'
            )
        ),
    head=dict(
        type='RoboSMPLXHeadCombine',
        use_highres=False,
        feat_dim=512),
    body_model_train=dict(
        type='smplx', # SMPLXLayer
        num_expression_coeffs=10,
        num_betas=10,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=False,
        model_path='data/body_models/smplx',
        keypoint_src='smplx',
        keypoint_dst='smplx',
    ),
    body_model_test=dict(
        type='smplx',
        num_expression_coeffs=50,
        num_betas=10,
        use_face_contour=True,
        use_pca=False,
        flat_hand_mean=False,
        model_path='data/body_models/smplx',
        keypoint_src='lsp',
        keypoint_dst='lsp',
        joints_regressor='data/body_models/smplx/SMPLX_to_J14.npy'),
    loss_keypoints3d=dict(type='L1Loss', reduction='mean', loss_weight=1),
    loss_keypoints2d=dict(type='L1Loss', reduction='mean', loss_weight=1),
    loss_joint_img=dict(type='L1Loss', reduction='mean', loss_weight=1),
    loss_smplx_global_orient=dict(
        type='L1Loss', reduction='mean', loss_weight=1),
    loss_smplx_body_pose=dict(
        type='L1Loss', reduction='mean', loss_weight=1),
    loss_smplx_jaw_pose=dict(
        type='L1Loss', reduction='mean', loss_weight=1),
    loss_smplx_hand_pose=dict(
        type='L1Loss', reduction='mean', loss_weight=1),
    loss_smplx_betas=dict(type='L1Loss', reduction='mean', loss_weight=1),
    loss_smplx_expression=dict(type='L1Loss', reduction='mean', loss_weight=1),
    use_highres=False,
    convention='smplx')

# dataset settings
dataset_type = 'HumanImageH4WDataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data_keys = [
    'has_smplx', 'has_keypoints3d', 'has_keypoints2d',
    'has_smplx_global_orient', 'has_smplx_body_pose', 'has_smplx_jaw_pose',
    'has_smplx_right_hand_pose', 'has_smplx_left_hand_pose', 'has_smplx_betas',
    'has_smplx_expression', 'smplx_jaw_pose', 'smplx_body_pose',
    'smplx_right_hand_pose', 'smplx_left_hand_pose', 'smplx_global_orient',
    'smplx_betas', 'keypoints2d', 'keypoints3d', 'sample_idx',
    'smplx_expression', 'rhand_bbox_center', 'rhand_bbox_scale', 
    'lhand_bbox_center', 'lhand_bbox_scale', 'face_bbox_center', 'face_bbox_scale', 
    'has_rhand_bbox', 'has_lhand_bbox', 'has_face_bbox'
]
train_data_keys = data_keys + ['joint_img', 'smplx_joint_img',
    'smplx_joint_cam', 'joint_trunc', 'smplx_joint_trunc', 'smplx_joint_valid',
    'joint_cam', 'joint_valid', 'smplx_pose_valid'
]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='GenerateSMPLXtargets', mode='train'),  # hand = 0,head = body = 0.5 # this should be before flip??
    dict(type='RandomHorizontalFlip', flip_prob=0.5, convention='smplx'),  # hand = 0,head = body = 0.5
    dict(
        type='GetRandomScaleRotation',
        rot_factor=30.0,
        scale_factor=0.25,
        rot_prob=0.6),
    dict(type='MeshAffine', img_res=[512, 512], save_hr=False),
    dict(type='H4WMeshAffinePartv2', image_size=[512, 512], mode='train', normalize=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=train_data_keys),
    dict(
        type='Collect',
        keys=['img', *train_data_keys],
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation',
            'crop_transform'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BBoxCenterJitter', factor=0., dist='uniform'),
    dict(type='RandomHorizontalFlip', flip_prob=0., convention='smplx'),  # hand = 0,head = body = 0.5
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=[512, 512], save_hr=False),
    dict(type='ControlledColorTransform', magnitude=0.4, ctype='grayness'), # 0.5, -0.5
    dict(type='H4WMeshAffinePartv2', image_size=[512, 512], mode='test', normalize=True),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=data_keys),
    dict(
        type='Collect',
        keys=['img', *data_keys],
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation',
            'crop_transform'
        ])
]
inference_data_keys = [
    'sample_idx', 'inv_transform'
]
inference_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='GetRandomScaleRotation', rot_factor=0, scale_factor=0),
    dict(type='MeshAffine', img_res=[512, 512]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'ori_img']),
    dict(type='ToTensor', keys=inference_data_keys),
    dict(
        type='Collect',
        keys=['img', *inference_data_keys],
        meta_keys=[
            'image_path', 'center', 'scale', 'rotation', 'ori_img',
            'crop_transform'
        ])
]

cache_files = {
    'neural_annot_h4w_mpii_train': 'data/cache/neural_annot_h4w_mpii_train.npz',
    'neural_annot_h4w_coco_train_ar1': 'data/cache/neural_annot_h4w_coco_train_ar1.npz',
    'neural_annot_h4w_h36m_train': 'data/cache/neural_annot_h4w_h36m_train.npz',
}

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=1,
    train=dict(
        type='MixedDataset',
        configs=[
            dict(
                type=dataset_type,
                pipeline=train_pipeline,
                dataset_name='coco',
                data_prefix='data',
                ann_file='neural_annot_h4w_coco_train_ar1.npz',
                square_bbox=True,
                convention='smplx',
                num_betas=10,
                num_expression=10,
                cache_data_path=cache_files['neural_annot_h4w_coco_train_ar1'],
            ),
        ],
        partition=[1.0],
    ),
    val=dict(
        type=dataset_type,
        body_model=dict(
            type='smplx',
            keypoint_src='smplx',
            keypoint_dst='smplx',
            flat_hand_mean=False,
            model_path='data/body_models/smplx',
            joints_regressor='data/body_models/smplx/SMPLX_to_J14.npy'),
        dataset_name='EHF',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='h4w_ehf_val.npz',
        face_vertex_ids_path=face_vertex_ids_path,
        hand_vertex_ids_path=hand_vertex_ids_path,
        square_bbox=True,
        convention='smplx'),
    test=dict(
        type=dataset_type,
        body_model=dict(
            type='smplx',
            keypoint_src='smplx',
            keypoint_dst='smplx',
            flat_hand_mean=False,
            model_path='data/body_models/smplx',
            joints_regressor='data/body_models/smplx/SMPLX_to_J14.npy'),
        dataset_name='EHF',
        data_prefix='data',
        pipeline=test_pipeline,
        ann_file='h4w_ehf_val.npz',
        face_vertex_ids_path=face_vertex_ids_path,
        hand_vertex_ids_path=hand_vertex_ids_path,
        square_bbox=True,
        convention='smplx'),
)

custom_imports = dict(
    imports=[
        'mmhuman3d.models.heads.robosmplx_head',
        'mmhuman3d.models.architectures.robosmplx_mesh_estimator', 
        'mmhuman3d.data.datasets.human_image_h4w_dataset', 'mmhuman3d.data.datasets.pipelines.nips',
    ],
    allow_failed_imports=False)