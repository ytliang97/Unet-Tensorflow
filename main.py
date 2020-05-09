'''An implementation of CycleGan using TensorFlow (work in progress).'''

import tensorflow as tf
import numpy as np
from model import unet
import cv2
import scipy.misc # save image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('output_dir', 'model_output', 
                           'checkpoint and summary directory.')

tf.app.flags.DEFINE_string('phase', 'train', 
                           'model phase: train/test.')

tf.app.flags.DEFINE_string('training_set', './datasets', 
                           'dataset path for training.')

tf.app.flags.DEFINE_string('testing_set', './datasets/test', 
                           'dataset path for testing one image pair.')

tf.app.flags.DEFINE_integer('batch_size', 64, 
                            'batch size for training.')

tf.app.flags.DEFINE_integer('training_steps', 100000, 
                            'total training steps.')

tf.app.flags.DEFINE_integer('summary_steps', 100, 
                            'summary period.')

tf.app.flags.DEFINE_integer('checkpoint_steps', 1000, 
                            'checkpoint period.')

tf.app.flags.DEFINE_integer('save_steps', 500, 
                            'checkpoint period.')

tf.app.flags.DEFINE_string('checkpoint', None, 
                           'checkpoint name for restoring.')


def main(_):
    #tf_flags = tf.app.flags.FLAGS
    # gpu config.
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth = True

    if FLAGS.phase == 'train':
        with tf.Session(config=config) as sess: 
        # when use queue to load data, not use with to define sess
            train_model = unet.UNet(sess, FLAGS)
            train_model.train(FLAGS.batch_size, FLAGS.training_steps, 
                              FLAGS.summary_steps, FLAGS.checkpoint_steps, FLAGS.save_steps)
    else:
        with tf.Session(config=config) as sess:
            # test on a image pair.
            test_model = unet.UNet(sess, FLAGS)
            test_model.load(FLAGS.checkpoint)
            image, output_masks = test_model.test()
            # return numpy ndarray.
            
            # save two images.
            filename_A = 'input.png'
            filename_B = 'output_masks.png'
            
            cv2.imwrite(filename_A, np.uint8(image[0].clip(0., 1.) * 255.))
            cv2.imwrite(filename_B, np.uint8(output_masks[0].clip(0., 1.) * 255.))

            # Utilize cv2.imwrite() to save images.
            print('Saved files: {}, {}'.format(filename_A, filename_B))

if __name__ == '__main__':
    tf.app.run()
