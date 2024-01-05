dataset_type = 'CocoDataset'
data_root = '/home/safouane/Downloads/benchmark_aircraft/data/'
backend_args = None
max_epochs = 500
metainfo = dict(
    classes=('airplane', ), palette=[
        (
            220,
            20,
            60,
        ),
    ])
num_classes = 1
batch_size = 128
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        to_rgb=True,
        ratio_range=(
            1,
            4,
        )),
    dict(
        type='MinIoURandomCrop',
        min_ious=(
            0.1,
            0.3,
            0.5,
            0.7,
            0.9,
        ),
        min_crop_size=0.3),
    dict(type='Resize', scale=(
        320,
        320,
    ), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(
            0.5,
            1.5,
        ),
        saturation_range=(
            0.5,
            1.5,
        ),
        hue_delta=18),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(
        320,
        320,
    ), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        )),
]
train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='CocoDataset',
            metainfo=dict(classes=('airplane', ), palette=[
                (
                    220,
                    20,
                    60,
                ),
            ]),
            data_root=data_root,
            ann_file='train/__coco.json',
            data_prefix=dict(img='train/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Expand',
                    mean=[
                        123.675,
                        116.28,
                        103.53,
                    ],
                    to_rgb=True,
                    ratio_range=(
                        1,
                        4,
                    )),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(
                        0.1,
                        0.3,
                        0.5,
                        0.7,
                        0.9,
                    ),
                    min_crop_size=0.3),
                dict(type='Resize', scale=(
                    320,
                    320,
                ), keep_ratio=False),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(
                        0.5,
                        1.5,
                    ),
                    saturation_range=(
                        0.5,
                        1.5,
                    ),
                    hue_delta=18),
                dict(type='PackDetInputs'),
            ])))
val_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=('airplane', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        data_root=data_root,
        ann_file='val/__coco.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                320,
                320,
            ), keep_ratio=False),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=128,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=('airplane', ), palette=[
            (
                220,
                20,
                60,
            ),
        ]),
        data_root=data_root,
        ann_file='test/__coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(
                320,
                320,
            ), keep_ratio=False),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                )),
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root+'val/__coco.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root+'test/__coco.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=500, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=120,
        end=120,
        by_epoch=True,
        eta_min=0),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=4e-05))
auto_scale_lr = dict(enable=False, base_batch_size=64)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=20, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    bgr_to_rgb=True,
    pad_size_divisor=1)
model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='MobileNetV2',
        out_indices=(
            4,
            7,
        ),
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),
    neck=dict(
        type='SSDNeck',
        in_channels=(
            96,
            1280,
        ),
        out_channels=(
            96,
            1280,
            512,
            256,
            256,
            128,
        ),
        level_strides=(
            2,
            2,
            2,
            2,
        ),
        level_paddings=(
            1,
            1,
            1,
            1,
        ),
        l2_norm_scale=None,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(
            96,
            1280,
            512,
            256,
            256,
            128,
        ),
        num_classes=1,
        use_depthwise=True,
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='ReLU6'),
        init_cfg=dict(type='Normal', layer='Conv2d', std=0.001),
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            strides=[
                16,
                32,
                64,
                107,
                160,
                320,
            ],
            ratios=[
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
            ],
            min_sizes=[
                48,
                100,
                150,
                202,
                253,
                304,
            ],
            max_sizes=[
                100,
                150,
                202,
                253,
                304,
                320,
            ]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ])),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
input_size = 320
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW'),
]
launcher = 'none'
work_dir = './work_dirs/ssdlite'
