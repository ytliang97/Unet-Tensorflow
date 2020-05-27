import os
import argparse

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage.io as skio

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='DataCenter/Carvana/train_masks', 
                    help='The directory contains images that want to \
                    change to jpg data type.')
FLAGS = parser.parse_args()

def main():
    image_names = os.listdir(FLAGS.dir)
    image_names = [img for img in image_names if not img.startswith('.')]
    
    transform_datadir = '/'.join(FLAGS.dir.split('/')[:-1]) + '/' +FLAGS.dir.split('/')[-1] + '_png'
    if not os.path.exists(transform_datadir):
        os.makedirs(transform_datadir)

    for image_name in image_names:
        image_path = os.path.join(FLAGS.dir, image_name)
        print(image_path)
        #image = Image.open(image_path)
        image = skio.imread(image_path)
        #image = plt.imread(image_path)
        #image = cv2.imread(image_path)
        #print('plt.imread: ', np.unique(image))
        #image = image.convert('L')
        #print('convert mode: ', np.unique(image))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #print('cvtColor: ', np.unique(image))
        image[image == 255] = 1
        image_path = os.path.join(transform_datadir, image_name[:-4] + '.png')
        #image = np.uint8(image)
        skio.imsave(image_path, image)
        #print('uint8: ', np.unique(image))
        #print(image.shape)
        #image.save(image_path)
        #plt.imsave(image_path, image)
        #cv2.imwrite(image_path, image)

if __name__ == '__main__':
    main()