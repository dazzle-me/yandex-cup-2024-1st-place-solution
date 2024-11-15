from configs.config_0 import cfg
import albumentations as A

# convnext_tiny.in12k_ft_in1k_384
cfg.model = 'finformer'
cfg.model_name = 'maxvit_rmlp_small_rw_224.sw_in1k'
cfg.do_resize = True
cfg.image_size = (224, 224)

cfg.output_features = 768
cfg.epochs = 100
cfg.lr = 1e-3
cfg.head_lr_multiple = 10.0
cfg.margin_offset = 0.5
cfg.margin_slope = 0.1

cfg.model_kwargs = {
    'drop_rate' : 0.3,
    'drop_path_rate' : 0.2
}

cfg.train_bs = 64
cfg.val_bs = 64

image_size = cfg.image_size

cfg.train_transforms = A.Compose([
    A.CoarseDropout(max_h_size=int(image_size[0] * 0.4), max_w_size=int(image_size[1] * 0.4), num_holes_range=(1, 1), p=0.5),
])