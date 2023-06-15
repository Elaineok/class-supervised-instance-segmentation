# class-supervised-instance-segmentation

## Prerequisite
 - Python 3.8, PyTorch 1.8.1, and cuda 11.1
 - VOC dataset (10582 images for training; 1449 images for validation) 
 - NVIDIA GPU(such as:Two 1080 GPU or one 3090 GPU)
## Usage
### Install python dependencies：
 - conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.1
 - pip install yacs opencv-python chainercv scikit-learn imageio
 - pip install git+https://github.com/lucasb-eyer/pydensecrf.git
### Download pascal voc dataset
 - data link: [VOC](https://pan.baidu.com/s/1OPFD7R08ZRz6W3b4ZdeDrg?pwd=vf2j).
 - pass word:vf2j
 
 
 The downloaded data structure is shown:
 ```
data_root/
    --- VOC2012/
        --- Annotations/
        --- ImageSet/
        --- JPEGImages/
        --- SegmentationObject/
        --- WSSS_maps/
```

### Train or make your own script

 `sh run_MM.sh`
 
 1.Train classification network to generate instance-aware localization which saved in "Peak_points5" folder
 
 we also provide our own trained [instance localization](https://pan.baidu.com/s/1WO4KAGQaDrE_NstrBn83nw?pwd=cgqh).
 
pass word：cgqh

2.Train displacement field and eval instance segmentation
 
 - You can either mannually edit the file, or specify commandline arguments.
