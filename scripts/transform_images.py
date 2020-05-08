
import os
# import cv2 # Opencv can not read GIF image directly!
from PIL import Image

if __name__ == '__main__':
    # change data_root for different datasets.
    # First, we can use os.listdir() to get every image name.
    label_dir = "/Users/yenciliang/Documents/DataCenter/Carvana/train_masks"
    label_names = os.listdir(label_dir) # return JPEG image names.
    # os.listdir() return a list, which includes the name of folder or file in a appointed foldr.
    # Do not include "." and "..".
    output_dir = label_dir + '_jpg'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for label_name in label_names:
        label_path = os.path.join(label_dir, label_name)
        image = Image.open(label_path).convert('L')
        label_path = os.path.join(output_dir, label_name[:-4] + '.jpg')
        image.save(label_path)
        
    '''
    for filename in ["train", "train_masks"]:
        for image_name in image_names:
            if filename is "train":
                image_file = os.path.join(data_root, filename, image_name)
                image = Image.open(image_file).convert("L")
                if not os.path.exists(os.path.join("../datasets/CarvanaImages", filename)):
                    os.makedirs(os.path.join("../datasets/CarvanaImages", filename))
                image.save(os.path.join("../datasets/CarvanaImages", filename, image_name))

            if filename is "train_masks":
                image_file = os.path.join(data_root, filename, image_name[:-4] + "_mask.gif")
                image = Image.open(image_file).convert("L")
                if not os.path.exists(os.path.join("../datasets/CarvanaImages", filename)):
                    os.makedirs(os.path.join("../datasets/CarvanaImages", filename)
                image.save(os.path.join("../datasets/CarvanaImages", filename, image_name[:-4] + "_mask.jpg"))
    '''