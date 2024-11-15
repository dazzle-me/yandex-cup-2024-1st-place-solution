import torch
import os
from pathlib import Path
import albumentations as A
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'


class CFG:
    def __init__(self):
        # General Configurations
        self.seed = 42 
        self.device = 'cuda:0'  
        self.debug = False  
        self.do_train = True  
        self.epochs = 10  
        self.ema_start_epoch = 5  
        self.exp_dir = Path('/storage/nvme1/hacks/yc-2024/exps')  
        
        # Dataset & Dataloaders
        self.dataset = 'simple_dataset'  
        self.train_bs = 512 
        self.val_bs = 512 
        self.num_workers = 12
        self.prefetch_factor = 2  
        self.storage_dir = Path('/home/slime/hacks/yc-2024/data/covers_detection_contest/dataset')  
        self.use_all_data = False
        
        # Optimizer & Scheduler
        self.optim = 'AdamW'  
        self.lr = 1e-3  
        self.wd = 1e-4  
        self.epochs_warmup = 1
        self.lr_min = 1e-6 
        self.head_lr_multiple = 1.0

        ## sched
        self.use_scheduler = True  
        self.use_t0 = False  
        self.T0 = 10  

        # Loss and Metrics
        self.criterion = 'ce'
        self.loss_threshold = 999999  
        self.grad_accum_steps = 1  
        self.max_grad_norm = None  
        self.set_grad_to_none = True 
        
        # AMP (Automatic Mixed Precision)
        self.enable_amp = True  
        self.autocast_dtype = torch.float16  
        
        # Model & Weights
        self.model = 'transformer'  
        self.weights = ''  
        self.load_opt = False  
        self.is_classification = False 
        self.point_dim = 84
        self.emb_dim = 128
        self.num_classes = 41617
        self.num_layers = 2
        ## misc model
        self.do_resize = False
        self.image_size = (224, 224)

        ## fformer
        self.model_name = 'tf_efficientnet_b0.ns_jft_in1k'
        self.output_features = 1280
        self.model_kwargs = {}
        self.in_chans = 1

        ## 2d-transformer
        self.use_rmsnorm = False
        self.patch_dim = 8
        self.nhead = 8
        
        ## Arcface
        self.margin_offset = 0.5
        self.margin_slope = 0.0
        self.margins = None
        # self.arcface_margin = 0.5
        self.arcface_scale = 30.0
        self.ls_eps = 0.0

        # Checkpoint & Evaluation
        self.checkpoint_mode = 'max'  
        self.save_checkpoints = True 
        self.do_eval_every = 1  
        
        # EMA (Exponential Moving Average)
        self.use_ema = False  
        self.ema_decay = 0.9999
        
        # Miscellaneous
        self.check_for_nan = False  
        self.max_steps_per_epoch = 999_999  

        ## augs
        self.flip_prob = 0.0
        self.train_transforms = None
        ##         albumentations.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
        self.label_flip_prob = 0.0
        self.val_transforms = None

cfg = CFG()