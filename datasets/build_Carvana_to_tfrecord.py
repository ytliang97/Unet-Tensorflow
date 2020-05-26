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
    'build_datadir', './Carvana/',
    'Path to save converted tfrecord of Tensorflow example')

def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir):
    
    image_reader = build_data.ImageReader('jpeg', channels=3)
    image_names = os.listdir(FLAGS.train_image_folder)
    image_names[:] = [x for x in image_names if not x.startswith('.')]
    image_names.sort()

    output_filename = '%s_%s.tfrecord' % ('Carvana', dataset_split)
    output_filename = os.path.join(FLAGS.build_datadir, output_filename)
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
    if not tf.gfile.IsDirectory(FLAGS.build_datadir):
        tf.gfile.MakeDirs(FLAGS.build_datadir)
    _convert_dataset('train', FLAGS.train_image_folder, FLAGS.train_label_folder)


if __name__ == '__main__':
    tf.app.run()
    