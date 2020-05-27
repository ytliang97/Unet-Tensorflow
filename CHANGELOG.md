# CHANGELOG


## Model
### 05-26-2020
#### [CarSeg_UNet_CE_2D_512_512_v1](./logs/CarSeg_UNet_CE_Carvana_2D_512_512_v1.log)

Dataset: Carvana

## code logs
Change Category:
+ **A**: edit coding style (follow PEP 8 and tensorflow/research/deeplab)
+ **B**: update code/function for using 
+ **C**: update repository architecture
+ **D**: DEBUG


### May 27, 2020
+ D-1: 發現transform_image.py並沒有將gif正確轉至jpg，label不是[0, 255]，而是漸變(不論使用pyplot, pillow, cv2儲存再讀取皆會出錯)，原因是因為jpg是有損壓縮，儲存後再載入會有差異，將label格式轉為png
+ C-2: 把`train_results/`資料夾移至`models/*Train Model*/`
+ B-4: 改變`model/utils.py`儲存訓練快照function(cv2 -> plt)
+ B-5: 將`model/models.py`最後一層輸出加入sigmoid activation function
+ B-6: 將`scripts/transform_images.py`中，label值255改為1

Future work:
+ B-3(C-3): 把`train_results/`改為用tensorboard顯示

### May 26, 2020
+ A-1: 模型資料夾名稱改為 目的_架構_lossName_datasetName_InputDimenstion_InputWidth_InputHeight_(InputDepth_)version
+ B-1: 把每次參數設定加進logging info
+ C-1: 把`Unet.log`獨立至`log/`


Future work:
+ B-2: 修改input image比例, 並對應修改input mask, output mask的shape