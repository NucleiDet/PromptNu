# 数据处理命令
python tools/convert_dataset/monuseg.py data/monuseg only-train_t12_v4 -w 512 -s 256

# 训练命令
python tools/train.py configs/unet/unet_vgg16_adam-lr1e-4_bs8_256x256_300e_monuseg.py

# 测试命令

python tools/test.py configs/cdnet/cdnet_vgg16_radam-lr5e-4_bs16_300x300_300e_cpm17.py work_dirs/cdnet/cdnet_vgg16_radam-lr5e-4_bs16_300x300_300e_cpm17/best_mAji_epoch_300.pth --save-pred

bash tools/dist_train.sh configs/micronet/micronet_adam-lr1e-4_bs4_252x252_100e_conic.py 4

tensorboard --logdir=runs --port=8080
