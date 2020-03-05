import functools
import logging
import os
import os.path as osp

import numpy as np
import tensorflow as tf

from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet
from embeddings.attention_module import attach_attention_module
from utils.infer_utils import get_exemplar_images
from utils.misc_utils import get_center
from matting.matting_biulder import matting_select

slim = tf.contrib.slim


class InferenceWrapper():
  """Model wrapper class for performing inference with a siamese model."""

  def __init__(self):
    self.image = None
    self.matting = None
    self.target_bbox_feed = None
    self.search_images = None
    self.embeds = None
    self.original_feature = None
    self.matting_feature = None
    self.templates = None
    self.init = None
    self.model_config = None
    self.track_config = None
    self.response_up = None
    self.is_matted = True
    self.search_matting = None
    self.feature_weight = 0.3

  def build_graph_from_config(self, model_config, track_config, checkpoint_path):
    """Build the inference graph and return a restore function."""
    self.build_model(model_config, track_config)
    ema = tf.train.ExponentialMovingAverage(0)
    variables_to_restore = ema.variables_to_restore(moving_avg_variables=[])

    # Filter out State variables
    variables_to_restore_filterd = {}
    for key, value in variables_to_restore.items():
      if key.split('/')[1] != 'State':
        variables_to_restore_filterd[key] = value

    saver = tf.train.Saver(variables_to_restore_filterd)

    if osp.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_path:
        raise ValueError("No checkpoint file found in: {}".format(checkpoint_path))

    def _restore_fn(sess):
      logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      logging.info("Successfully loaded checkpoint: %s", os.path.basename(checkpoint_path))

    return _restore_fn

  def build_model(self, model_config, track_config):
    self.model_config = model_config
    self.track_config = track_config

    self.build_inputs()
    self.build_search_images()
    self.build_template()
    self.build_detection()
    self.build_upsample()
    self.dumb_op = tf.no_op('dumb_operation')

  def build_inputs(self):
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.to_float(image)
    self.image = image

    matted_filename = tf.placeholder(tf.string, [], name='matted_filename')
    matting_file = tf.read_file(matted_filename)
    matting = tf.image.decode_jpeg(matting_file, channels=3, dct_method="INTEGER_ACCURATE")
    matting = tf.to_float(matting)
    self.matting = matting

    self.target_bbox_feed = tf.placeholder(dtype=tf.float32,
                                           shape=[4],
                                           name='target_bbox_feed')  # center's y, x, height, width

  def build_search_images(self):
    """Crop search images from the input image based on the last target position

    1. The input image is scaled such that the area of target&context takes up to (scale_factor * z_image_size) ^ 2
    2. Crop an image patch as large as x_image_size centered at the target center.
    3. If the cropped image region is beyond the boundary of the input image, mean values are padded.
    """
    model_config = self.model_config
    track_config = self.track_config

    size_z = model_config['z_image_size']
    size_x = track_config['x_image_size']
    context_amount = 0.5

    num_scales = track_config['num_scales']
    scales = np.arange(num_scales) - get_center(num_scales)
    assert np.sum(scales) == 0, 'scales should be symmetric'
    search_factors = [track_config['scale_step'] ** x for x in scales]

    frame_sz = tf.shape(self.image)
    target_yx = self.target_bbox_feed[0:2]
    target_size = self.target_bbox_feed[2:4]
    avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')

    # Compute base values
    base_z_size = target_size
    base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size)
    base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
    base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
    d_search = (size_x - size_z) / 2.0
    base_pad = tf.div(d_search, base_scale_z)
    base_s_x = base_s_z + 2 * base_pad
    base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

    boxes = []
    for factor in search_factors:
      s_x = factor * base_s_x
      frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
      topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
      bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
      box = tf.concat([topleft, bottomright], axis=0)
      boxes.append(box)
    boxes = tf.stack(boxes)

    scale_xs = []
    for factor in search_factors:
      scale_x = base_scale_x / factor
      scale_xs.append(scale_x)
    self.scale_xs = tf.stack(scale_xs)

    # Note we use different padding values for each image
    # while the original implementation uses only the average value
    # of the first image for all images.
    matting_dims = tf.expand_dims(self.matting, 0)
    matting_cropped = tf.image.crop_and_resize(matting_dims, boxes,
                                             box_ind=tf.zeros((track_config['num_scales']), tf.int32),
                                             crop_size=[size_x, size_x])
    self.search_matting = matting_cropped

    image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
    image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                             box_ind=tf.zeros((track_config['num_scales']), tf.int32),
                                             crop_size=[size_x, size_x])
    self.search_images = image_cropped + avg_chan

  def get_image_embedding(self, images, reuse=True):
    config = self.model_config['embed_config']
    arg_scope = convolutional_alexnet_arg_scope(config,
                                                trainable=config['train_embedding'],
                                                is_training=False)

    @functools.wraps(convolutional_alexnet)
    def embedding_fn(images, reuse=False):
      with slim.arg_scope(arg_scope):
        return convolutional_alexnet(images, reuse=reuse)

    embed, _ = embedding_fn(images, reuse)

    return embed

  def build_matting_feature(self):
    # Exemplar image lies at the center of the search image in the first frame
    matted_exemplar_images = get_exemplar_images(self.search_matting, [self.model_config['z_image_size'],
                                                               self.model_config['z_image_size']])
    templates = self.get_image_embedding(matted_exemplar_images, reuse=tf.AUTO_REUSE)
    templates = attach_attention_module(templates, attention_module='se_block')
    center_scale = int(get_center(self.track_config['num_scales']))
    center_template = tf.identity(templates[center_scale])
    matting_feature = tf.stack([center_template for _ in range(self.track_config['num_scales'])])
    self.matting_feature = matting_feature
    return matting_feature

  def build_original_featrue(self):
    # Exemplar image lies at the center of the search image in the first frame
    exemplar_images = get_exemplar_images(self.search_images, [self.model_config['z_image_size'],
                                                               self.model_config['z_image_size']])
    templates = self.get_image_embedding(exemplar_images, reuse=tf.AUTO_REUSE)
    templates = attach_attention_module(templates, attention_module='se_block')
    center_scale = int(get_center(self.track_config['num_scales']))
    center_template = tf.identity(templates[center_scale])
    original_feature = tf.stack([center_template for _ in range(self.track_config['num_scales'])])
    self.original_feature = original_feature
    return original_feature

  def build_template(self):
    matting_feature = self.build_matting_feature()
    original_feature = self.build_original_featrue()
    templates = self.feature_weight * matting_feature + (1 - self.feature_weight) * original_feature

    with tf.variable_scope('target_template'):
      # Store template in Variable such that we don't have to feed this template every time.
      with tf.variable_scope('State'):
        state = tf.get_variable('exemplar',
                                initializer=tf.zeros(templates.get_shape().as_list(), dtype=templates.dtype),
                                trainable=False)
        with tf.control_dependencies([templates]):
          self.init = tf.assign(state, templates, validate_shape=True)
        self.templates = state

  def build_detection(self):
    self.embeds = self.get_image_embedding(self.search_images, reuse=True)
    with tf.variable_scope('detection'):
      def _translation_match(x, z):
        x = tf.expand_dims(x, 0)  # [batch, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, out_channels]
        return tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')

      output = tf.map_fn(
        lambda x: _translation_match(x[0], x[1]),
        (self.embeds, self.templates), dtype=self.embeds.dtype)  # of shape [3, 1, 17, 17, 1]
      output = tf.squeeze(output, [1, 4])  # of shape e.g. [3, 17, 17]

      bias = tf.get_variable('biases', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
      response = self.model_config['adjust_response_config']['scale'] * output + bias
      self.response = response

  def build_upsample(self):
    """Upsample response to obtain finer target position"""
    with tf.variable_scope('upsample'):
      response = tf.expand_dims(self.response, 3)
      up_method = self.track_config['upsample_method']
      methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                 'bicubic': tf.image.ResizeMethod.BICUBIC}
      up_method = methods[up_method]
      response_spatial_size = self.response.get_shape().as_list()[1:3]
      up_size = [s * self.track_config['upsample_factor'] for s in response_spatial_size]
      response_up = tf.image.resize_images(response,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up = tf.squeeze(response_up, [3])
      self.response_up = response_up

  def initialize(self, sess, input_feed, matting_method, logdir):
    image_path, target_bbox = input_feed
    matted_path = matting_select(image_path, target_bbox, matting_method, logdir)
    scale_xs, _ = sess.run([self.scale_xs, self.init],
                           feed_dict={'filename:0': image_path,
                                      'matted_filename:0': matted_path,
                                      "target_bbox_feed:0": target_bbox})
    return scale_xs

  def inference_step(self, sess, input_feed):
    image_path, target_bbox = input_feed
    log_level = self.track_config['log_level']
    image_cropped_op = self.search_images if log_level > 0 else self.dumb_op
    image_cropped, scale_xs, response_output = sess.run(
      fetches=[image_cropped_op, self.scale_xs, self.response_up],
      feed_dict={
        "filename:0": image_path,
        "target_bbox_feed:0": target_bbox, })

    output = {
      'image_cropped': image_cropped,
      'scale_xs': scale_xs,
      'response': response_output}
    return output, None
