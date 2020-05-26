"""Load data.Train and test CycleGAN."""
import os
import sys
import logging
import time
import glob
from datetime import datetime

import tensorflow as tf
import numpy as np
import cv2

from model import read_tfrecords
from model import models
from model.utils import save_images

class UNet(object):
    def __init__(self, sess, FLAGS, is_training=False):
        """
        In training, it will create directory to save training result, 
        and crate model object.
        """
        self.sess = sess
        self.dtype = tf.float32

        self.checkpoint_dir = os.path.join(FLAGS.train_logdir, 'checkpoint')
        self.checkpoint_prefix = 'model'
        self.saver_name = 'checkpoint'
        self.summary_dir = os.path.join(FLAGS.train_logdir, 'summary')

        self.learning_rate = 0.001
        
        # data parameters
        self.image_w = 512
        self.image_h = 512 # The raw and mask image is 1918 * 1280.
        self.image_c = 1 # Gray image.

        self.input_data = tf.placeholder(self.dtype, 
                            [None, self.image_h, self.image_w, self.image_c])
        self.input_masks = tf.placeholder(self.dtype, 
                            [None, 324, 324, self.image_c])
        # TODO: The shape of image masks. Refer to the Unet in model.py, the output image is
        # 324 * 324 * 1. But is not good.
        # learning rate
        self.lr = tf.placeholder(self.dtype)

        # train
        if is_training:
            #self.training_set = FLAGS.dataset_dir
            self.sample_dir = 'train_results'

            # makedir aux dir
            self._make_aux_dirs()
            # compute and define loss
            self._build_training()
            # logging, only use in training
            
            ##log_file = FLAGS.train_logdir + '/Unet.log'
            ##logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
            ##                    filename=log_file,
            ##                    level=logging.DEBUG,
            ##                    filemode='w')
            ##logging.getLogger().addHandler(logging.StreamHandler())
        else:
            # test
            self.test_dataset = FLAGS.test_dataset
            # build model
            self.output = self._build_test()

    def _build_training(self):
        """
        use models Unet function to generate UNet object.
        """
        # Unet
        self.output = models.Unet(name='UNet', in_data=self.input_data, reuse=False)

        # loss.
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=self.input_masks, logits=self.output))
        # self.loss = tf.reduce_mean(tf.squared_difference(self.input_masks,
        #     self.output))
        # Use Tensorflow and Keras at the same time.
        self.loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            self.input_masks, self.output))
        
        # optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.loss, name='opt')
        
        # summary
        tf.summary.scalar('loss', self.loss)
        
        self.summary = tf.summary.merge_all()
        # summary and checkpoint
        self.writer = tf.summary.FileWriter(
            self.summary_dir, graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name)
        self.summary_proto = tf.Summary()


    def train(self, batch_size, training_number_of_steps, summary_steps, 
          checkpoint_steps, save_steps, dataset_dir, dataset, 
          tf_initial_checkpoint_dir):
        """
        1. Choose to train with pretrained model or not.
        2. Load dataset(tfrecord).

        Args:
          tf_initial_checkpoint_dir: train with pretrained model or not.

        """

        # 1.
        step_num = 0
        if tf_initial_checkpoint_dir:
            pretrained_ckpt = os.path.join(tf_initial_checkpoint_dir, 'checkpoint')
            latest_checkpoint = tf.train.latest_checkpoint(pretrained_ckpt)
            step_num = int(os.path.basename(latest_checkpoint).split('-')[1])
            assert step_num > 0, 'Please ensure checkpoint format is model-*.*.'
            self.saver.restore(self.sess, latest_checkpoint)
            logging.info('{}: Resume training from step {}. Loaded \
                checkpoint {}'.format(datetime.now(), step_num, latest_checkpoint))
        else:
            self.sess.run(tf.global_variables_initializer()) # init all variables
            logging.info('{}: Init new training'.format(datetime.now()))

        # 2.
        tfrecord_paths = os.path.join(dataset_dir, dataset)
        tf_reader = read_tfrecords.Read_TFRecords(
                        filename=tfrecord_paths,
                        batch_size=batch_size, 
                        image_h=self.image_h, 
                        image_w=self.image_w, 
                        image_c=self.image_c)

        images, images_masks = tf_reader.read()

        logging.info('{}: Done init data generators'.format(datetime.now()))

        self.coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        try:
            # train
            c_time = time.time()
            lrval = self.learning_rate
            for c_step in range(step_num + 1, training_number_of_steps + 1):
                # learning rate, adjust lr
                if c_step % 5000 == 0:
                    lrval = self.learning_rate * .5
                
                batch_images, batch_images_masks = self.sess.run([images, images_masks])
                c_feed_dict = {
                    # TFRecord
                    self.input_data: batch_images,
                    self.input_masks: batch_images_masks,
                    self.lr: lrval
                }
                print('batch_images shape: ', np.shape(batch_images))
                print('batch_images_masks shape: ', np.shape(batch_images_masks))
                self.sess.run(self.opt, feed_dict=c_feed_dict)
                print('output mask shape: ', np.shape(self.output))

                # save summary
                if c_step % summary_steps == 0:
                    c_summary = self.sess.run(self.summary, feed_dict=c_feed_dict)
                    self.writer.add_summary(c_summary, c_step)

                    e_time = time.time() - c_time
                    time_periter = e_time / summary_steps
                    logging.info('{}: Iteration_{} ({:.4f}s/iter) {}'.format(
                        datetime.now(), c_step, time_periter,
                        self._print_summary(c_summary)))
                    c_time = time.time() # update time

                # save checkpoint
                if c_step % checkpoint_steps == 0:
                    self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, self.checkpoint_prefix),
                        global_step=c_step)
                    logging.info('{}: Iteration_{} Saved checkpoint'.format(
                        datetime.now(), c_step))

                if c_step % save_steps == 0:
                    _, output_masks, input_masks = self.sess.run(
                        [self.input_data, self.output, self.input_masks],
                        feed_dict=c_feed_dict)
                    save_images(None, output_masks, input_masks,
                        input_path = './{}/input_{:04d}.png'.format(self.sample_dir, c_step),
                        image_path = './{}/train_{:04d}.png'.format(self.sample_dir, c_step))
        except KeyboardInterrupt:
            print('Interrupted')
            self.coord.request_stop()
        except Exception as e:
            self.coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            self.coord.request_stop()
            self.coord.join(threads)

        logging.info('{}: Done training'.format(datetime.now()))

    def _build_test(self):
        # network.
        output = models.Unet(name='UNet', in_data=self.input_data, reuse=False)

        self.saver = tf.train.Saver(max_to_keep=10, name=self.saver_name) 
        # define saver, after the network!

        return output

    def load(self, checkpoint_name=None):
        # restore checkpoint
        print('{}: Loading checkpoint...'.format(datetime.now())),
        if checkpoint_name:
            checkpoint = os.path.join(self.checkpoint_dir, checkpoint_name)
            self.saver.restore(self.sess, checkpoint)
            print(' loaded {}'.format(checkpoint_name))
        else:
            # restore latest model
            latest_checkpoint = tf.train.latest_checkpoint(
                self.checkpoint_dir)
            if latest_checkpoint:
                self.saver.restore(self.sess, latest_checkpoint)
                print(' loaded {}'.format(os.path.basename(latest_checkpoint)))
            else:
                raise IOError(
                    'No checkpoints found in {}'.format(self.checkpoint_dir))

    def test(self):
        # Test only in a image.
        image_name = glob.glob(os.path.join(self.test_dataset, '*.jpg'))
        
        
        # In tensorflow, test image must divide 255.0.
        print(image_name)
        image = np.reshape(cv2.resize(cv2.imread(image_name[0], 0), 
            (self.image_h, self.image_w)), (1, self.image_h, self.image_w, self.image_c)) / 255.
        # OpenCV load image. the data format is BGR, w.t., (H, W, C). The default load is channel=3.

        print('{}: Done init data generators'.format(datetime.now()))

        c_feed_dict = {
            self.input_data: image
        }

        output_masks = self.sess.run(
            self.output, feed_dict=c_feed_dict)

        return image, output_masks
        # image: 1 * 512 * 512 * 1
        # output_masks: 1 * 324 * 342 * 1.

    def _make_aux_dirs(self):
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)

    def _print_summary(self, summary_string):
        self.summary_proto.ParseFromString(summary_string)
        result = []
        for val in self.summary_proto.value:
            result.append('({}={})'.format(val.tag, val.simple_value))
        return ' '.join(result)
