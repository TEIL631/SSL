# Files description

## Python files

1. data.py: preparation of train/valid/test datasets and dataloaderss
2. S.py: supervised training with Wide ResNet
3. S_Mixup.py: S.py + Mixup
4. P.py: S.py + pseudolabeling (no Mixup)
5. R.py: P.py + representation learning (i.e., + concatenating supervised vectors (trained) and unsuperivsed vectors(pretrained and freezed)) (no Mixup)
6. R_Mixup.py: R.py + Mixup
7. R_SA_Mixup.py: R_Mixup.py + Smart Augmentation
8. run_ssl.py: an script for repeatitive training
9. resnet_vae.py: the script defining the autoencoders for unsupervised training (see Checkpoints)
10. statistics.py: contains global variables related to statistics

## Checkpoints (i.e., pretrained models)

1. best_u_model_VAE_CNN_AUG_640: a pretrained unsupervised model for WM811K
2. unsupervised_model_mitbih: a pretrained unsupervised model for MITBIH

## Datasets

1. MITBIH (dir): the electrocardiogram dataset
2. WM811K (dir): the wafer dataset

這兩個資料夾包含已經下載並切好的資料，若有需要可以按需求重新切分資料集。

## Models

1. models (dir): contains a lot of model architectures

## Other files

1. config.yaml: the configuration file for training

## Other methods

1. Curriculum-Labeling
2. EnAET

# Models we use

In R.py and its variant, there are three models:

1. Two modes for generating represention vectors:
    1. The model for supervised vectors: called the supervised model -> Wide_ResNet in ./model/wide_resnet.py
    2. The model for unsupervised vectors: called the unsupervised model -> defined in resnet_vae.py
2. One model for prediction
    3. Called the main classifier -> SimpleNN1 and SimpleNN3 in pl_model.py

Note: In S.py, there are only one model: the supervised model (Wide_ResNet), which at the same time is the model for prediction.