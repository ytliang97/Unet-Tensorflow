"""Converts Carvana data to TFRecord file format with Example protos.
Generate TFRecords file, for training."""

import os

import tensorflow as tf

import build_data

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_image_folder', '',
    'Folder containing trainng images')

tf.app.flags.DEFINE_string(
    'train_label_folder', '',
    'Folder containing annotations for trainng images')

tf.app.flags.DEFINE_string(
    'output_dir', './Carvana/',
    'Path to save converted tfrecord of Tensorflow example')

def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir):
    
    image_reader = build_data.ImageReader('jpeg', channels=3)
    image_names = os.listdir(FLAGS.train_image_folder)
    image_names[:] = [x for x in image_names if not x.startswith('.')]
    image_names.sort()

    output_filename = '%s_%s.tfrecord' % ('Carvana', dataset_split)
    output_filename = os.path.join(FLAGS.output_dir, output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        total = 0
        for idx, image_name in enumerate(image_names):
            if idx % 5 == 0:
                print('total', total, 'file(s), process', idx, 'file(s).')

            data_path = os.path.join(dataset_dir, image_name)
            label_path = os.path.join(dataset_label_dir, image_name[:-4] + '_mask.jpg')

            image_data = tf.gfile.GFile(data_path, 'rb').read()
            seg_data = tf.gfile.GFile(label_path, 'rb').read()
            height, width = image_reader.read_image_dims(image_data)
            try:
                tf_example = build_data.image_seg_to_tfexample(
                    image_data, image_name, height, width, seg_data)
                if tf_example is not None:
                    tfrecord_writer.write(tf_example.SerializeToString())
            except ValueError:
                tf.logging.warning('Invalid example:', image_name ,', ignorig.')

            total += 1
    print('total', total, 'file(s), process', idx, 'file(s).')

def main(_):
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    _convert_dataset('train', FLAGS.train_image_folder, FLAGS.train_label_folder)
    

    # change data_root for different datasets.
    # First, we can use os.listdir() to get every image name.
    #data_root = "../datasets/CarvanaImages"
    #image_names = os.listdir(os.path.join(data_root, "train")) # return JPEG image names.
    # os.listdir() return a list, which includes the name of folder or file in a appointed foldr.
    # Do not include "." and "..".

    # TFRecordWriter, dump to tfrecords file
    # TFRecord file name, change save_name for different datasets.
    # Create one proto buffer, then add two Features.
    ##if not os.path.exists(os.path.join("../datasets", "tfrecords")):
    ##    os.makedirs(os.path.join("../datasets", "tfrecords"))
    ##writer = tf.python_io.TFRecordWriter(os.path.join("../datasets", "tfrecords", 
    ##    "Carvana.tfrecords"))

    ##for image_name in image_names:
        # image_name
        ##image_raw_file = os.path.join(data_root, "train", image_name)
        ##image_label_file = os.path.join(data_root, "train_masks", 
        ##    image_name[:-4] + "_mask.jpg")

        # The first method to load image.
        
        #image_raw = Image.open(image_file) # It image is RGB, then mode=RGB; otherwise, mode=L.
        # reszie image. In reading the TFRecords file, if you want resize the image, you could put image 
        # height and width into the Feature.
        # In this way, when reading the TFRecords file, it can use width and height.
        #width = image_raw.size[0]
        #height = image_raw.size[1]
        # put image height and width into the Feature.
        #"height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height]))
        #"width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width]))

        #image_raw = image_raw.tobytes()
        # Transform image to byte.
        

        # Second method to load image.
        ##image_raw = tf.gfile.FastGFile(image_raw_file, 'rb').read() # image data type is string. 
        # read and binary.
        ##image_label = tf.gfile.FastGFile(image_label_file, 'rb').read()

        # write bytes to Example proto buffer.
        ##example = tf.train.Example(features=tf.train.Features(feature={
        ##    "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
         ##   "image_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label]))
          ##  }))
        
        ##writer.write(example.SerializeToString()) # Serialize To String
    
    ##writer.close()


if __name__ == '__main__':
    tf.app.run()
    