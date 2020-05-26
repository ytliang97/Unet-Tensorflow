# CHANGELOG


## Model
### 05-26-2020
#### [CarSeg_UNet_CE_2D_512_512_v1](./logs/CarSeg_UNet_CE_2D_512_512_v1.log)

Dataset: Carvana

## code logs
Change Category:
+ **A**: edit coding style (follow PEP 8 and tensorflow/research/deeplab)
+ **B**: update code/function for using 
+ **C**: update repository architecture

### May 26, 2020
+A-1: 模型資料夾名稱改為 目的_架構_lossName_datasetName_InputDimenstion_InputWidth_InputHeight_(InputDepth_)version
+B-1: 把每次參數設定加進logging info
+C-1: 把`Unet.log`獨立至`log/`

Future work:
+ B-2: 修改input image比例, 並對應修改input mask, output mask的shape