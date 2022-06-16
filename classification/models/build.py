from .litv2 import LITv2


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'litv2':
        model = LITv2(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.LIT.PATCH_SIZE,
                                in_chans=config.MODEL.LIT.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.LIT.EMBED_DIM,
                                depths=config.MODEL.LIT.DEPTHS,
                                num_heads=config.MODEL.LIT.NUM_HEADS,
                                mlp_ratio=config.MODEL.LIT.MLP_RATIO,
                                qkv_bias=config.MODEL.LIT.QKV_BIAS,
                                qk_scale=config.MODEL.LIT.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.LIT.APE,
                                patch_norm=config.MODEL.LIT.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                alpha=config.MODEL.LIT.ALPHA,
                                local_ws=config.MODEL.LIT.LOCAL_WS
                    )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
