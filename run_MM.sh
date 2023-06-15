
#step1：train classification network
# you need change <root_dir> for your VOC2012 path
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --root_dir=/data/VOC/VOCdevkit/VOC2012/ \
    --lr=0.001 \
    --epoch=20 \
    --decay_points='5,10' \
    --save_folder=checkpoints/PAM \
    --show_interval=50


#step2：generate instance-aware localization
# you need change <root_dir>for your VOC2012 path
CUDA_VISIBLE_DEVICES=0,1 python point_extraction.py \
    --root_dir=/data/VOC/VOCdevkit/VOC2012/ \
    --checkpoint=checkpoints/PAM/ckpt_5.pth \
    --save_dir=MM_instance_segmentation/Peak_points5/


CUDA_VISIBLE_DEVICES=0,1 python point_extraction.py \
    --root_dir=/data/VOC/VOCdevkit/VOC2012/ \
    --checkpoint=checkpoints/PAM/ckpt_10.pth \
    --save_dir=MM_instance_segmentation/Peak_points10/
    
    
CUDA_VISIBLE_DEVICES=0,1 python point_extraction.py \
    --root_dir=/data/VOC/VOCdevkit/VOC2012/ \
    --checkpoint=checkpoints/PAM/ckpt_15.pth \
    --save_dir=MM_instance_segmentation/Peak_points15/
    
    
CUDA_VISIBLE_DEVICES=0,1 python point_extraction.py \
    --root_dir=/data/VOC/VOCdevkit/VOC2012/ \
    --checkpoint=checkpoints/PAM/ckpt_20.pth \
    --save_dir=MM_instance_segmentation/Peak_points20/




#step3：train displacement field for instance segmentation
# you need change <root_dir>for your VOC2012 path
cd MM_instance_segmentation/

CUDA_VISIBLE_DEVICES=0,1,2,3 python3  -m torch.distributed.launch --nproc_per_node=4 --master_port 29031 main.py \
--root_dir "/data/VOC/VOCdevkit/VOC2012/" --PAM "Peak_points5" --train_iter 65000 --save_folder "checkpoints/PAM5"

cd MM_instance_segmentation/
CUDA_VISIBLE_DEVICES=0,1,2,3 python3  -m torch.distributed.launch --nproc_per_node=4 --master_port 29031 main.py \
--root_dir "/data/VOC/VOCdevkit/VOC2012/" --PAM "Peak_points10" --train_iter 65000 --save_folder "checkpoints/PAM10"

cd MM_instance_segmentation/
CUDA_VISIBLE_DEVICES=0,1,2,3 python3  -m torch.distributed.launch --nproc_per_node=4 --master_port 29031 main.py \
--root_dir "/data/VOC/VOCdevkit/VOC2012/" --PAM "Peak_points15" --train_iter 65000 --save_folder "checkpoints/PAM15"

cd MM_instance_segmentation/
CUDA_VISIBLE_DEVICES=0,1,2,3 python3  -m torch.distributed.launch --nproc_per_node=4 --master_port 29031 main.py \
--root_dir "/data/VOC/VOCdevkit/VOC2012/" --PAM "Peak_points20" --train_iter 65000 --save_folder "checkpoints/PAM20"





#step3：train displacement field for instance segmentation
# you need change <root_dir>for your VOC2012 path
cd MM_instance_segmentation/

CUDA_VISIBLE_DEVICES=0,1,2,3 python3  -m torch.distributed.launch --nproc_per_node=4 --master_port 29031 main.py \
--root_dir "/data/VOC/VOCdevkit/VOC2012/" --PAM "Peak_points5" --validation True --resume "checkpoints/PAM5/best.pt"



CUDA_VISIBLE_DEVICES=0,1,2,3 python3  -m torch.distributed.launch --nproc_per_node=4 --master_port 29031 main.py \
--root_dir "/data/VOC/VOCdevkit/VOC2012/" --PAM "Peak_points10" --validation True --resume "checkpoints/PAM10/best.pt"


CUDA_VISIBLE_DEVICES=0,1,2,3 python3  -m torch.distributed.launch --nproc_per_node=4 --master_port 29031 main.py \
--root_dir "/data/VOC/VOCdevkit/VOC2012/" --PAM "Peak_points15" --validation True --resume "checkpoints/PAM15/best.pt"


CUDA_VISIBLE_DEVICES=0,1,2,3 python3  -m torch.distributed.launch --nproc_per_node=4 --master_port 29031 main.py \
--root_dir "/data/VOC/VOCdevkit/VOC2012/" --PAM "Peak_points20" --validation True --resume "checkpoints/PAM20/best.pt"
