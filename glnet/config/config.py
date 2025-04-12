H = 900
W = 1600
final_dim = (224, 384)
img_conf = dict(img_mean=[123.675, 116.28, 103.53],
                img_std=[58.395, 57.12, 57.375],
                to_rgb=True)

nclt_pc_bev_conf = {
    'x_bound': [-70.0, 70.0],
    'y_bound': [-70.0, 70.0],
    'z_bound': [-20.0, -1.0],
    'x_grid': 160,
    'y_grid': 160,
    'z_grid': 20
}

oxford_pc_bev_conf = {
    'x_bound': [-70.0, 70.0],
    'y_bound': [-70.0, 70.0],
    'z_bound': [-16.0, 4.0],
    'x_grid': 160,
    'y_grid': 160,
    'z_grid': 20
}

backbone_conf = {
    'x_bound': [-51.2, 51.2, 0.8], # [-51.2, 51.2, 0.8]
    'y_bound': [-51.2, 51.2, 0.8], # [-51.2, 51.2, 0.8]
    'z_bound': [-15, 15, 30], # [-5, 3, 8] # [-15, 5, 20]
    'd_bound': [2.0, 58.0, 0.5],
    'final_dim':
    final_dim,
    'output_channels':
    80,
    'downsample_factor':
    16,
    'img_backbone_conf':
    dict(
        type='ResNet',
        depth=50,
        frozen_stages=0,
        out_indices=[0, 1, 2, 3],
        norm_eval=False,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    'img_neck_conf':
    dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128],
    ),
    'depth_net_conf':
    dict(in_channels=512, mid_channels=512)
}

ida_aug_conf = {
    'resize_lim': (0.386, 0.55),
    'final_dim':
    final_dim,
    'rot_lim': (-5.4, 5.4),
    'H':
    H,
    'W':
    W,
    'rand_flip':
    True,
    'bot_pct_lim': (0.0, 0.0),
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
}

bda_aug_conf = {
    'rot_lim': (-22.5, 22.5),
    'scale_lim': (0.95, 1.05),
    'flip_dx_ratio': 0.5,
    'flip_dy_ratio': 0.5
}
