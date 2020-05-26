import os
import argparse

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='DataCenter/Carvana/train_masks', 
                    help='The directory contains images that want to \
                    change to jpg data type.')
FLAGS = parser.parse_args()

def main():
    image_names = os.listdir(FLAGS.dir)
    image_names = [img for img in image_names if not img.startswith('.') 
                    and os.path.isfile(img)]
    
    transform_datadir = FLAGS.dir + '_jpg'
    if not os.path.exists(transform_datadir):
        os.makedirs(transform_datadir)

    for image_name in image_names:
        image_path = os.path.join(FLAGS.dir, image_name)
        image = Image.open(image_path).convert('L')
        image_path = os.path.join(transform_datadir, image_name[:-4] + '.jpg')
        image.save(image_path, quality=100)

if __name__ == '__main__':
    main()