# dataset settings
dataset_type = 'MoNuSegDataset'
data_root = '/data/rubyyao/Datasets/nuclei_ins_seg/MO/train33_val4_test14/monuseg'
train_processes = [
    dict(type='Affine', scale=(0.8, 1.2), shear=5, rotate_degree=[-180, 180], translate_frac=(0, 0.01)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomCrop', crop_size=(256, 256)),
    dict(type='Pad', pad_size=(256, 256)),
    dict(type='RandomBlur'),
    dict(
        type='ColorJitter', hue_delta=8, saturation_range=(0.8, 1.2), brightness_delta=26, contrast_range=(0.75, 1.25)),
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='HVLabelMake'),
    dict(type='Formatting', data_keys=['img'], label_keys=['sem_gt', 'hv_gt','inst_gt', 'inst_color_gt']),
]
test_processes = [
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='Formatting', data_keys=['img'], label_keys=[]),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/w512_s256',
        ann_dir='train/w512_s256',
        split='only-train_t12_v4_train_w512_s256.txt',
        processes=train_processes),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/w0_s0',
        ann_dir='test/w0_s0',
        split='only-train_t12_v4_test_w0_s0.txt',
        processes=test_processes),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/w0_s0',
        ann_dir='test/w0_s0',
        split='only-train_t12_v4_test_w0_s0.txt',
        processes=test_processes),
)

# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=8,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='train/w512_s256',
#         ann_dir='train/w512_s256',
#         split='only-train_t12_v4_train_w512_s256.txt',
#         processes=train_processes),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='val/w0_s0',
#         ann_dir='val/w0_s0',
#         split='only-train_t12_v4_val_w0_s0.txt',
#         processes=test_processes),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='test/w0_s0',
#         ann_dir='test/w0_s0',
#         split='only-train_t12_v4_test_w0_s0.txt',
#         processes=test_processes),
# )