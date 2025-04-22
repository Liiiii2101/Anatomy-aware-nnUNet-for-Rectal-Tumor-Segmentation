## Improving Rectal Tumor Segmentation with Anomaly Fusion Derived from Anatomical Inpainting: A Multicenter Study
You can find the article [here](https://www.medrxiv.org/content/10.1101/2024.10.15.24315517v1)


This repo is a PyTorch-based framework for rectal tumor and rectal structures (rectum and mesorectum) segmentation, whose goal is to provide a comprehensive  comparison of various deep learning models. It provides fair evaluation and comparison of CNNs and Transformers on rectal cancer MRI (T2WI). 

## Data and Model Weights
We publish the rectal structure including rectum and mesorectum segmenting model [here](https://drive.google.com/drive/folders/1ontHBRMZia00npnt5YVQoDFzeW92MyOd?usp=drive_link). The model was trained on our own rectal dataset (T2WI)

Also, the rectum and mesorectum masks of 100 T2WI scans of PI-CAI dataset can be found [here](https://drive.google.com/drive/folders/1dbKnbT8pQxC3fBrAGKWnlV_argsa3Ysf?usp=drive_link). They were annotated by a radiologist with more than 6-year experience.



### Features
- Cover the whole process of model design, including dataset processing, model definition, model configuration, training and evaluation.
- Provide SOTA models as baseline for comparison for rectal tumor segmentation. Model definition, training and evaluation code are simple with no complex code encapsulation.
- Provide models, losses, metrics, augmentation and etc. for 2D, 3D data.



### Supporting models
- [MedFormer](https://arxiv.org/abs/2203.00131) (Official implementation)
- UNet. Including 2D, 3D with different building block, e.g. double conv, Residual BasicBlock, Bottleneck, MBConv, or ConvNeXt block.
- [UNet++](https://arxiv.org/abs/1807.10165)
- [Attention UNet](https://arxiv.org/abs/1804.03999)
- [TransUNet](https://arxiv.org/abs/2102.04306)
- [SwinUNet](https://arxiv.org/abs/2105.05537)
- [UNETR](https://arxiv.org/abs/2103.10504)
- [nnFormer](https://arxiv.org/abs/2109.03201)
- [SwinUNETR](https://arxiv.org/abs/2201.01266)




### Usage
If you just want to use the models in your own framework, you can directly find the corresponding models in the `model/` folder and use them in your own framework. For the definition of specific models, please refer to `model/utils.py` in the `get_model` function. The models we provide do not have complex dependencies and encapsulation. The modules used by each model are defined in its own `xxx_utils.py` file. For example, the model definition of UNet only depend on `unet.py`, `unet_utils.py` and `conv_layers.py`.

If you want to use our framework, please follow below steps.

#### Install requirements
Create a new virtual environment and install all dependencies by:
```
pip install -r requirement.txt
```
#### Data preparation
Download the origin dataset from their corresponding official website.

Enter the `dataset_conversion` fold and find the dataset you want to use and the corresponding dimension (2d or 3d)

Edit the `src_path` and `tgt_path` the in `xxxdataset.py`, where the `src_path` is the path to the origin dataset, and `tgt_path` is the target path to store the processed dataset.

Then, `python xxxdataset.py`

After processing is finished, put the processed dataset into `dataset/` folder or use a soft link.

#### Configuration
Enter `config/xxxdataset/` and find the model and dimension (2d or 3d) you want to use. The training details, e.g. model hyper-parameters, training epochs, learning rate, optimizer, data augmentation, etc., can be altered here. You can try your own congiguration or use the default configure, the one we used in the MedFormer paper, which should have a decent performance. The only thing to care is the `data_root`, make sure it points to the processed dataset directory.

#### Training
We can start training after the data and configuration is done. Several arguments can be parsed in the command line, see in the `get_parser()` function in the `train.py` and `train_ddp.py`. You need to specify the model, the dimension, the dataset, whether use pretrain weights, batch size, and the unique experiment name. Our code will find the corresponding configuration and dataset for training.

Here is an example to train 3D MedFormer with one gpu on our dataset:

`python train.py --model medformer --dimension 3d --dataset rectal --batch_size 3 --unique_name rectal_3d_medformer --gpu 0`

This command will start the cross validation on ACDC. The training loss and evaluation performance will be logged by tensorboard. You can find them in the `log/dataset/unique_name` folder. All the standard output, the configuration, and model weigths will be saved in the `exp/dataset/unique_name` folder. The results of cross validation will be saved in `exp/dataset/unique_name/cross_validation.txt`.

Besides training with a single GPU, we also provide distributed training (DDP) and automatic mixed precision (AMP) training in the `train_ddp.py`. The `train_ddp.py` is the same as `train.py` except it supports DDP and AMP. We recomend you to start with `train.py` to make sure the whole train and eval pipeline is correct, and then use `train_ddp.py` for faster training or larger batch size.

Example of using DDP:

`python train_ddp.py --model medformer --dimension 3d --dataset acdc --batch_size 16 --unique_name acdc_3d_medformer_ddp --gpu 0,1,2,3`

Example of using DDP and AMP:

`python train_ddp.py --model medformer --dimension 3d --dataset acdc --batch_size 32 --unique_name acdc_3d_medformer_ddp_amp --gpu 0,1,2,3 --amp`

We have not fully benchmark if AMP can speed up training, but AMP can reduce the GPU memory consumption a lot.

### Acknowledgement
The code is built on top of https://github.com/DL4mHealth/Medformer
Also, many thanks to https://github.com/MIC-DKFZ/nnUNet and https://github.com/fiy2W/mri_seq2seq


### Citation
If you find this repo helps or if you are using our model or dataset, please kindly cite our paper, thanks!
```
@@article{cai2024improving,
  title={Improving Rectal Tumor Segmentation with Anomaly Fusion Derived from Anatomical Inpainting: A Multicenter Study},
  author={Cai, Lishan and Abdelatty, Mohamed A and Han, Luyi and Lambregts, Doenja MJ and van Griethuysen, Joost and Pooch, Eduardo and Beets-Tan, Regina GH and Benson, Sean and Brunekreef, Joren and Teuwen, Jonas},
  journal={medRxiv},
  pages={2024--10},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}



```
