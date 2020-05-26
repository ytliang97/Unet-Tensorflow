# Under re-generating...

To see change detail, refer to [CHANGELOG.md](./CHANGELOG.md)

Change Category:

+ **A**: edit coding style (follow PEP 8 and tensorflow/research/deeplab)

+ **B**: update code/function for using 

+ **C**: update repository architecture

## Getting-Start
### e.g. Carvana data
+ Step1. Download Carvana dataset from kaggle
+ Step2. transform format of Carvana mask from `gif` to `jpg`
```shell
python3 transform_images.py --dir ./Carvana/train_masks
```
+ Step3. build tfrecord of Carvana dataset
```shell
python3 build_Carvana_to_tfrecord.py --train_image_folder ./Carvana/train --train_label_folder ./Carvana/train_masks_jpg --build_datadir ./Carvana
```
```shell
python3 tfrecord2image_carvana_format.py --tfrecord_path datasets/Carvana/Carvana_train.tfrecord
```
+ Step4. setting parameter to train the model
Recommend models directory naming: Target_Arch_lossName_datasetName_InputDimenstion_InputWidth_InputHeight_version
e.g. CarSeg_UNet_CE_Carvana_2D_512_512_v1
```shell
# this is for testing, not the best settings
python3 train.py --train_logdir models/CarSeg_UNet_CE_Carvana_2D_512_512_v1 --phase train --dataset_dir ./datasets/Carvana --dataset Carvana_train.tfrecord --train_batch_size 2 --training_number_of_steps 30 --summary_steps 100 --save_steps 100 --checkpoint_steps 2
```

### another dataset
+ Step1. Download your dataset, and figure out data type and what data you want to store in tfrecord
+ Step2. create a new build function `build_*New Dataset Name*_to_tfrecord.py` in `datasets/`
+ Step3. follow build function of other dataset to generate tfreocrd of new dataset.


# Unet 
* Tensorflow implement of U-Net: Convolutional Networks for Biomedical Image Segmentation..[[Paper]](https://arxiv.org/abs/1505.04597)
* Borrowed code and ideas from zhixuhao's unet: https://github.com/zhixuhao/unet.

## Install Required Packages
First ensure that you have installed the following required packages:
* TensorFlow1.4.0 ([instructions](https://www.tensorflow.org/install/)). Maybe other version is ok.
* Opencv ([instructions](https://github.com/opencv/opencv)). Here is opencv-2.4.9.

## Datasets
* In this implementation of the Unet, we use Carvana Image Masking Challenge data.[[download]](https://www.kaggle.com/c/carvana-image-masking-challenge/data) We download train.zip and train_masks.zip. You can put all the datasets in datasets folder.
* Run **scripts/transform_images.py** to transform all the image to gray JPEG image.
* Run **scripts/build_tfrecords.py** to generate training data, data format is tfrecords.

## Training and Testing Model
* Run the following script to train the model, in the process of training, will save the training images every 500 steps. See the **model/unet.py** for details.
```shell
sh train.sh
```
You can change the arguments in train.sh depend on your machine config.
* Run the following script to test the trained model. The test.sh will transform the datasets.
```shell
sh test.sh
```
The script will load the trained StarGAN model to generate the transformed images. You could change the arguments in test.sh depend on your machine config.

## Downloading data/trained model
* Pretrained model: [[download]](https://drive.google.com/open?id=14_8ZthgcpIXdEQEzIENueXv7dGVzHvjK). The model-8500 is better.