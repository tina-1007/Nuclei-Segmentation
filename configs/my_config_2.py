_base_ = './mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py'

data_root = '/home/ytliu/VRDL/Nuclei-Segmentation/data/'
dataset_type = 'CocoDataset'
classes = ('nucleus',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', img_scale=(1000, 1000), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,    
    train=dict(
        type=dataset_type, 
        ann_file=data_root + 'annotations/poly_nucleus_train.json',
        classes=classes,
        img_prefix=data_root + 'new_train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/poly_nucleus_val.json',
        classes=classes,        
        img_prefix=data_root + 'new_val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test_img_ids.json',
        classes=classes,        
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline))

model = dict(
    backbone=dict(
        depth=101,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
    mask_head=dict(num_classes=1)))

runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=5)
# load_from = 'pretrained_model/mask_rcnn_r101_fpn_2x_coco.pth'
resume_from = 'work_dirs/my_mask_rcnn_r101_2x/epoch_15.pth'