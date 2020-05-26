'''An implementation of CycleGan using TensorFlow (work in progress).'''
import os
import logging

import tensorflow as tf
import numpy as np
import cv2
import scipy.misc # save image

from model import unet

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_logdir', 
                           'models/Target_Arch_lossName_datasetName_InputDimenstion_InputWidth_InputHeight_version', 
                           'Where the checkpoint and logs are stored.')

tf.app.flags.DEFINE_string('phase', 'train', 
                           'model phase: train/test.')

tf.app.flags.DEFINE_string('dataset_dir', './datasets', 
                           'dataset path for training.')

tf.app.flags.DEFINE_string('dataset', 'Carvana_train.tfrecord',
                           'Name of the segmentation dataset.')

tf.app.flags.DEFINE_string('test_dataset', './datasets/test', 
                           'dataset path for testing one image pair.')

tf.app.flags.DEFINE_integer('train_batch_size', 64, 
                            'The number of images in each batch during training.')

tf.app.flags.DEFINE_integer('training_number_of_steps', 100000, 
                            'The number of steps used for training')

tf.app.flags.DEFINE_integer('summary_steps', 100, 
                            'summary period.')

tf.app.flags.DEFINE_integer('checkpoint_steps', 1000, 
                            'checkpoint period.')

tf.app.flags.DEFINE_integer('save_steps', 500, 
                            'checkpoint period.')

tf.app.flags.DEFINE_string('checkpoint', None, 
                           'checkpoint name for restoring.')

tf.app.flags.DEFINE_string('tf_initial_checkpoint_dir', None,
                           'pretrained model directory.')

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # gpu config.
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.allow_growth = True

    if FLAGS.phase == 'train':
        with tf.Session(config=config) as sess: 
        # when use queue to load data, not use with to define sess
            train_model = unet.UNet(sess, FLAGS, 
                                    is_training=True)
            train_model.train(batch_size=FLAGS.train_batch_size, 
                              training_number_of_steps=FLAGS.training_number_of_steps, 
                              summary_steps=FLAGS.summary_steps, 
                              checkpoint_steps=FLAGS.checkpoint_steps, 
                              save_steps=FLAGS.save_steps,
                              dataset_dir=FLAGS.dataset_dir,
                              dataset=FLAGS.dataset,
                              tf_initial_checkpoint_dir=FLAGS.tf_initial_checkpoint_dir)
    else:
        with tf.Session(config=config) as sess:
            # test on a image pair.
            test_model = unet.UNet(sess, FLAGS)
            test_model.load(FLAGS.checkpoint)
            image, output_masks = test_model.test()
            # return numpy ndarray.
            
            # save two images.
            print('\n\n\n\n???\n\n\n\n')
            filename_A = 'input.png'
            filename_B = 'output_masks.png'
            
            cv2.imwrite(filename_A, np.uint8(image[0].clip(0., 1.) * 255.))
            cv2.imwrite(filename_B, np.uint8(output_masks[0].clip(0., 1.) * 255.))

            # Utilize cv2.imwrite() to save images.
            print('Saved files: {}, {}'.format(filename_A, filename_B))

if __name__ == '__main__':
    if not os.path.exists('./models'):
        os.makedirs('./models')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename='./logs/' + FLAGS.train_logdir.split('/')[-1] + '.log',
        level=logging.DEBUG,
        filemode='w'
    )
    logging.info(
        (
            '--train_logdir={0} \\\n' +
            '--phase={1} \\\n' +
            '--dataset_dir={2} \\\n' +
            '--dataset={3} \\\n' +
            '--test_dataset={4} \\\n' +
            '--train_batch_size={5} \\\n' +
            '--training_number_of_steps={6} \\\n' +
            '--summary_steps={7} \\\n' +
            '--checkpoint_steps={8} \\\n' +
            '--save_steps={9} \\\n' +
            '--checkpoint={10} \\\n' +
            '--tf_initial_checkpoint_dir={10} \\\n'
        ).format(
            FLAGS.train_logdir,
            FLAGS.phase,
            FLAGS.dataset_dir,
            FLAGS.dataset,
            FLAGS.test_dataset,
            FLAGS.train_batch_size,
            FLAGS.training_number_of_steps,
            FLAGS.summary_steps,
            FLAGS.checkpoint_steps,
            FLAGS.save_steps,
            FLAGS.checkpoint,
            FLAGS.tf_initial_checkpoint_dir
        )
    )
    #logging.getLogger().addHandler(logging.StreamHandler())
    tf.app.run()
