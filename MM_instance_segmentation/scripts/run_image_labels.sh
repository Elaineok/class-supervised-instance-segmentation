# Training BESTIE with image-level labels.

ROOT=/disk2/xjt/data/VOC/VOCdevkit/VOC2012/
SUP=cls
PSEUDO_THRESH=0.7
REFINE_THRESH=0.3
REFINE_WARMUP=0
SIZE=416
BATCH=16
WORKERS=4
TRAIN_ITERS=65000
BACKBONE=resnet50 # [resnet50, resnet101, hrnet32, hrnet48]
VAL_IGNORE=False

CUDA_VISIBLE_DEVICES=0,1 python3  -m torch.distributed.launch --nproc_per_node=2 --master_port 29031 main.py \
--root_dir ${ROOT} --PAM "Peak_points5" --sup ${SUP} --batch_size ${BATCH} --num_workers ${WORKERS} --crop_size ${SIZE} --train_iter ${TRAIN_ITERS} \
--refine True --refine_iter ${REFINE_WARMUP} --pseudo_thresh ${PSEUDO_THRESH} --refine_thresh ${REFINE_THRESH} \
--val_freq 1000 --val_ignore ${VAL_IGNORE} --val_clean False --val_flip False \
--seg_weight 1.0 --center_weight 200.0 --offset_weight 0.01 \
--lr 5e-5 --backbone ${BACKBONE} --random_seed 3407

####4：--ROOT
#####/disk2/xjt/data/VOC/VOCdevkit/VOC2012/

####5：--ROOT
#####/data2/xjt/home/data/VOC2012/

#####6:--ROOT
#####/data1/xjt/data/VOC/VOCdevkit/VOC2012/
