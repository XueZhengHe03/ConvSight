# run_baselines.sh
python train_baseline.py --model resnet50
python train_baseline.py --model efficientnet_b0
# python train_baseline.py --model vit_small_patch16_224   #效果不好
# python train_baseline.py --model convnext_tiny    #效果不好
python train_baseline.py --model efficientnetv2_s
python train_baseline.py --model swin_tiny_patch4_window7_224
python evaluate_all.py