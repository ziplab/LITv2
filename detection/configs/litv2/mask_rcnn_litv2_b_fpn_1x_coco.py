_base_ = [
    '../_base_/models/mask_rcnn_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[ 4, 8, 16, 32 ],
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=True,
        alpha=0.9,
        local_ws=[0, 0, 2, 1],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/ziplab/LITv2/releases/download/v1.0/litv2_b.pth')
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]))

optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.0001,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)})
                 )

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
