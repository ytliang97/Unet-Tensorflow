"""Convert a tf record to image, and use cv2 to show the image.
"""
import os
import copy 

import matplotlib.pyplot as plt
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS
tf.flags.DEFINE_string('tfrecord_path', '', 'tfrecord file path.')


def carvana_format_tfrecord_parse(example_proto):

    features = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/filename': tf.io.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string),
        'image/height': tf.FixedLenFeature((), tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
        'image/channels': tf.FixedLenFeature((), tf.int64),
        'image/segmentation/class/encoded': tf.FixedLenFeature((), tf.string),
        'image/segmentation/class/format': tf.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.parse_single_example(example_proto, features)

    return parsed_features


def carvana_format_tfrecord_preprocess(parsed_features):
    o_img = parsed_features['image/encoded']
    img = tf.image.decode_jpeg(parsed_features['image/encoded'], 
                               channels=3, 
                               dct_method='INTEGER_ACCURATE')
    h = parsed_features['image/height']
    w = parsed_features['image/width']
    f = parsed_features['image/filename']
    c = parsed_features['image/channels']
    o_lb = parsed_features['image/segmentation/class/encoded']
    lb = tf.image.decode_png(
        parsed_features['image/segmentation/class/encoded'],
        channels=1)

    return o_img, img, h, w, f, c, o_lb, lb


def main(_):
    count = 0
    for record in tf.python_io.tf_record_iterator(FLAGS.tfrecord_path):
        count += 1
    print('There are', count, 'examples in the tfrecord.')

    batch_size = 100
    dataset = tf.data.TFRecordDataset(filenames = FLAGS.tfrecord_path)
    dataset = dataset.map(carvana_format_tfrecord_parse)
    dataset = dataset.map(carvana_format_tfrecord_preprocess)
    dataset = dataset.batch(batch_size).prefetch(2 * batch_size)
    iterator = dataset.make_one_shot_iterator()
    
    _o_img, _img, _h, _w, _f, _c, _o_lb, _lb = iterator.get_next()

    stop = False
    with tf.Session() as sess:
        while True:
            try:
                origin_images, images, heights, widths, filenames, channels, origin_labels, labels = sess.run([_o_img, _img, _h, _w, _f, _c, _o_lb, _lb])

            except tf.errors.OutOfRangeError:
                print('End of', FLAGS.tfrecord_path)
                break
            except tf.errors.DataLossError:
                print('last tf example corrupt')
                break

            for i in range(0, images.shape[0]):

                filename = filenames[i].decode()
                height = heights[i]
                width = widths[i]
                image = images[i]
                label = labels[i]

                plt.subplot(1,2,1)
                plt.imshow(image)
                plt.subplot(1,2,2)
                plt.imshow(label[:,:,0])
                plt.show()
                print(filename, height, width)

                break

    print('There are', count, 'examples in the tfrecord.')


if __name__ == '__main__':
    tf.app.run()
