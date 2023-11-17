_base_ = [
    '../_base_/models/retinanet_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        alpha=0.9,
        local_ws=[0, 0, 2, 1],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_m.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5)
)

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001,
                 paramwise_cfg=dict({'norm': dict(decay_mult=0.)}))


lr_config = dict(step=[8, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)