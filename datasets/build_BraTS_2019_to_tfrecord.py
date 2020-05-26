"""Converts BraTS data to TFRecord file format with Example protos."""
import io
import os
import gzip

import numpy as np
import tensorflow as tf
import nibabel as nib
import matplotlib.pyplot as plt

import build_data


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string(
    'train_image_folder', '',
    'Folder containing trainng images')

tf.app.flags.DEFINE_string(
    'train_label_folder', '',
    'Folder containing annotations for trainng images')

tf.app.flags.DEFINE_string(
    'build_datadir', './BraTS_2019',
    'Path to save converted tfrecord of Tensorflow example')


def _convert_dataset(dataset_split, dataset_dir, dataset_label_dir):
    datas = os.listdir(dataset_dir)
    datas[:] = [x for x in datas if not x.startswith('.')]
    datas.sort()
    
    output_filename = '%s_%s.tfrecord' % ('BraTS_2019', dataset_split)
    output_filename = os.path.join(FLAGS.build_datadir, output_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        count = 0
        total = 0
        for idx, data in enumerate(datas):
            if idx % 5 == 0:
                print('total', total, 'file(s), process', count, 'file(s).')
            entry = {}
            entry['filename'] = data
            label = data.split('.')[0][:-4] + 'seg.nii.gz'
            entry['data_path'] = os.path.join(dataset_dir, data)
            entry['label_path'] = os.path.join(dataset_label_dir, label)
            if os.path.isfile(entry['label_path']):
                count += 1
                with gzip.open(entry['data_path'], 'rb') as fd, \
                    gzip.open(entry['label_path'], 'rb') as fl:
                    try:
                        
                        nii_data = fd.read()
                        seg_data = fl.read()
                        
                        tf_example = build_data.nii_seg_to_tf_example(nii_data, entry['filename'], seg_data)
                        if tf_example is not None:
                            tfrecord_writer.write(tf_example.SerializeToString())
                        
                    except ValueError:
                        tf.logging.warning('Invalid example: %s, ignoring.')
                
            total += 1
        print('total', total, 'file(s), process', count, 'file(s).')




def main(_):
    if not tf.gfile.IsDirectory(FLAGS.build_datadir):
        tf.gfile.MakeDirs(FLAGS.build_datadir)
    _convert_dataset('train', FLAGS.train_image_folder, FLAGS.train_label_folder)

if __name__ == '__main__':
    tf.app.run()