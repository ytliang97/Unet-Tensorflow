"""Convert a tf record to image, and use cv2 to show the image.
"""
import os
import copy 

import tensorflow as tf
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as macolors

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('tfrecord_path', '', 'tfrecord file path.')

def brats_2019_format_tfrecord_parse(example_proto):

    features = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/filename': tf.io.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string),
        'image/segmentation/class/encoded': tf.FixedLenFeature((), tf.string),
        'image/segmentation/class/format': tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    return parsed_features


def brats_2019_format_tfrecord_preprocess(parsed_features):
    f = parsed_features['image/filename']
    d = parsed_features['image/encoded']
    s = parsed_features['image/segmentation/class/encoded']

    return d, s, f


def main(_):
    count = 0
    for record in tf.python_io.tf_record_iterator(FLAGS.tfrecord_path):
        count += 1
    print('There are', count, 'examples in the tfrecord.')

    batch_size = 1
    dataset = tf.data.TFRecordDataset(filenames = FLAGS.tfrecord_path)
    dataset = dataset.map(brats_2019_format_tfrecord_parse)
    dataset = dataset.map(brats_2019_format_tfrecord_preprocess)
    dataset = dataset.batch(batch_size).prefetch(2 * batch_size)
    iterator = dataset.make_one_shot_iterator()
    
    _d, _s, _f = iterator.get_next()

    stop = False
    idx = 0
    with tf.Session() as sess:
        while True:
            try:
                datas, segmentations, filenames = sess.run([_d, _s, _f])
            except tf.errors.OutOfRangeError:
                print('End of', FLAGS.tfrecord_path)
                break
            except tf.errors.DataLossError:
                print('last tf example corrupt')
                break

            for i in range(0, datas.shape[0]):
                idx += 1
                filename = filenames[i].decode()
                nii_img = nib.Nifti1Image.from_bytes(datas[i])
                nii_seg = nib.Nifti1Image.from_bytes(segmentations[i])
                nii_img = nii_img.get_data()
                nii_seg = nii_seg.get_data()
                
                print('({idx}/{total})'.format(idx=idx, total=count),filename, 'label number: ', np.unique(nii_seg))
                plt.subplot(1,2,1)
                plt.imshow(nii_img[:,:,90], cmap='gray')
                plt.subplot(1,2,2)
                cmap = macolors.ListedColormap(['black', 'yellow', 'cyan', 'red', 'blue'])
                label_name = ['bg', 'ncr/net', 'edema', 'ventricle', 'enh_tumor']
                im = plt.imshow(nii_seg[:,:,90], vmin=0, vmax=len(cmap.colors),cmap=cmap)
                patches = [ mpatches.Patch(
                                color=cmap.colors[i], 
                                label="{c} {l}: {n}".format(l=i, n=label_name[i], c=cmap.colors[i]) 
                            ) for i in range(len(cmap.colors)) ]
                plt.legend(handles=patches, bbox_to_anchor=(0.5, 1.4), 
                           loc='upper left', borderaxespad=0., prop={'size': 6} )
                plt.show()
                

                

    print('There are', count, 'examples in the tfrecord.')


if __name__ == '__main__':
    tf.app.run()
