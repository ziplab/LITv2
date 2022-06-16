_base_ = [
    '../_base_/models/fpn_r50_lit.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='LITv2',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[ 4, 8, 16, 32 ],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        alpha=0.9,
        local_ws=[0, 0, 2, 1]
    ),
    neck=dict(
        type='FPN',
        in_channels=[128, 256, 512, 1024],
        out_channels=256,
        num_outs=4),
    decode_head=dict(num_classes=150))

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)