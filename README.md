## DataSets

Dataset source can be downloaded here.

- [ImageNet](https://www.image-net.org/). The ILSVRC 2012 dataset as In-distribution (ID) dataset. The training subset we used is [this file](datalists/imagenet2012_train_random_200k.txt).
- [OpenImage-O](https://github.com/openimages/dataset/blob/main/READMEV3.md). The OpenImage-O dataset is a subset of the OpenImage-V3 testing set. The filelist is [here](datalists/openimage_o.txt). Please refer to [our paper of ViM](http://ooddetection.github.io) for details of dataset construction.
- [Texture](https://www.robots.ox.ac.uk/~vgg/data/dtd/). We rule out four classes that coincides with ImageNet. The filelist used in the paper is [here](datalists/texture.txt).
- [iNaturalist](https://arxiv.org/pdf/1707.06642.pdf). Follow the instructions in the [link](https://github.com/deeplearning-wisc/large_scale_ood) to prepare the iNaturalist OOD dataset.
- [ImageNet-O](https://github.com/hendrycks/natural-adv-examples). Follow the guide to download the ImageNet-O OOD dataset.
- [SUN](https://faculty.cc.gatech.edu/~hays/papers/sun.pdf). . Follow the instructions in the [line](https://github.com/YongHyun-Ahn/LINe-Out-of-Distribution-Detection-by-Leveraging-Important-Neurons)
- [Places](http://olivalab.mit.edu/Papers/Places-PAMI2018.pdf). Follow the instructions in the [line](https://github.com/YongHyun-Ahn/LINe-Out-of-Distribution-Detection-by-Leveraging-Important-Neurons)

## Pretrained Model Preparation

### VIT

1. GEN: install mmclassification
   ViM: install mmpretrain
2. download checkpoint
   ```bash
   mkdir checkpoints
   cd checkpoints
   wget https://download.openmmlab.com/mmclassification/v0/vit/finetune/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth
   cd ..
   ```
3. extract features
    ```bash
    ./extract_feature_vit.py data/imagenet outputs/vit_imagenet_val.pkl --img_list datalists/imagenet2012_val_list.txt
    ./extract_feature_vit.py data/imagenet outputs/vit_train_200k.pkl --img_list datalists/imagenet2012_train_random_200k.txt
    ./extract_feature_vit.py data/openimage_o outputs/vit_openimage_o.pkl --img_list datalists/openimage_o.txt
    ./extract_feature_vit.py data/texture outputs/vit_texture.pkl --img_list datalists/texture.txt
    ./extract_feature_vit.py data/inaturalist outputs/vit_inaturalist.pkl
    ./extract_feature_vit.py data/imagenet_o outputs/vit_imagenet_o.pkl
    ./extract_feature_bit.py data/sun outputs/vit_sun.pkl
    ./extract_feature_bit.py data/places outputs/vit_places.pkl
    ```
4. extract w and b in fc
    ```bash
    ./extract_feature_vit.py a b --fc_save_path outputs/vit_fc.pkl
    ```
5. evaluation
    ```bash
    ./benchmark.py outputs/vit_fc.pkl outputs/vit_train_200k.pkl outputs/vit_imagenet_val.pkl outputs/vit_openimage_o.pkl outputs/vit_texture.pkl outputs/vit_inaturalist.pkl outputs/vit_imagenet_o.pkl
    ```

### BIT

1. download checkpoint
   ```bash
   mkdir checkpoints
   cd checkpoints
   wget https://storage.googleapis.com/bit_models/BiT-S-R101x1.npz
   cd ..
   ```
2. extract features
    ```bash
    ./extract_feature_bit.py data/imagenet outputs/bit_imagenet_val.pkl --img_list datalists/imagenet2012_val_list.txt
    ./extract_feature_bit.py data/imagenet outputs/bit_train_200k.pkl --img_list datalists/imagenet2012_train_random_200k.txt
    ./extract_feature_bit.py data/openimage_o outputs/bit_openimage_o.pkl --img_list datalists/openimage_o.txt
    ./extract_feature_bit.py data/texture outputs/bit_texture.pkl --img_list datalists/texture.txt
    ./extract_feature_bit.py data/inaturalist outputs/bit_inaturalist.pkl
    ./extract_feature_bit.py data/imagenet_o outputs/bit_imagenet_o.pkl
    ./extract_feature_bit.py data/sun outputs/bit_sun.pkl
    ./extract_feature_bit.py data/places outputs/bit_places.pkl
    ```
3. extract w and b in fc
    ```bash
    ./extract_feature_bit.py a b --fc_save_path outputs/bit_fc.pkl
    ```
4. evaluation
    ```bash
    ./benchmark.py outputs/bit_fc.pkl outputs/bit_train_200k.pkl outputs/bit_imagenet_val.pkl outputs/bit_openimage_o.pkl outputs/bit_texture.pkl outputs/bit_inaturalist.pkl outputs/bit_imagenet_o.pkl
    ```

### RepVGG, Res50d, Swin, DeiT

1. extract features, use repvgg_b3, resnet50d, swin, deit as model
    ```bash
    # choose one of them
    export MODEL=repvgg_b3 && export NAME=repvgg
    export MODEL=resnet50d && export NAME=resnet50d
    export MODEL=swin_base_patch4_window7_224 && export NAME=swin
    export MODEL=deit_base_patch16_224 && export NAME=deit

    ./extract_feature_timm.py data/imagenet outputs/${NAME}_imagenet_val.pkl ${MODEL} --img_list datalists/imagenet2012_val_list.txt
    ./extract_feature_timm.py data/imagenet outputs/${NAME}_train_200k.pkl ${MODEL} --img_list datalists/imagenet2012_train_random_200k.txt
    ./extract_feature_timm.py data/openimage_o outputs/${NAME}_openimage_o.pkl ${MODEL} --img_list datalists/openimage_o.txt
    ./extract_feature_timm.py data/texture outputs/${NAME}_texture.pkl ${MODEL} --img_list datalists/texture.txt
    ./extract_feature_timm.py data/inaturalist outputs/${NAME}_inaturalist.pkl ${MODEL}
    ./extract_feature_timm.py data/imagenet_o outputs/${NAME}_imagenet_o.pkl ${MODEL}
    ./extract_feature_bit.py data/sun outputs/${NAME}_sun.pkl ${MODEL}
    ./extract_feature_bit.py data/places outputs/${NAME}_places.pkl ${MODEL}
    ```
2. extract w and b in fc
    ```bash
    ./extract_feature_timm.py a b ${MODEL} --fc_save_path outputs/${NAME}_fc.pkl
    ```
3. evaluation
    ```bash
    ./benchmark.py outputs/${NAME}_fc.pkl outputs/${NAME}_train_200k.pkl outputs/${NAME}_imagenet_val.pkl outputs/${NAME}_openimage_o.pkl outputs/${NAME}_texture.pkl outputs/${NAME}_inaturalist.pkl outputs/${NAME}_imagenet_o.pkl
    ```

Note: To reproduce ODIN baseline, please refer to [this repo](https://github.com/deeplearning-wisc/large_scale_ood).

## Acknowledgement

Our code is built on [GEN](https://github.com/XixiLiu95/GEN) repo, thanks a lot for their great work!

## Cite

```
@article{lu2025musia,
  title={MuSIA: Exploiting multi-source information fusion with abnormal activations for out-of-distribution detection},
  author={Lu, Heng-yang and Guo, Xin and Jiang, Wenyu and Fan, Chenyou and Du, Yuntao and Shao, Zhenhao and Fang, Wei and Wu, Xiaojun},
  journal={Neural Networks},
  volume={188},
  pages={107427},
  year={2025},
  publisher={Elsevier}
}
```

contact: 2674507042@qq.com

