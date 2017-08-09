# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import tensorflow as tf
from spatial_transformer_alpha import transformer
import numpy as np
import shutil
import os
import cv2
import sys
import math
import time
from tf_utils import weight_variable, bias_variable, dense_to_one_hot
from nets import inception_v2
from nets import inception_v1
from nets.inception_v3 import inception_v3_arg_scope
from guuker import prt
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import transformer_factory
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.training.python.training import training
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.training import supervisor
from tensorflow.python.training import sync_replicas_optimizer
from tensorflow.python.training import training_util

slim = tf.contrib.slim
cuda_devices = os.environ['CUDA_VISIBLE_DEVICES']
NUM_GPUS = len(cuda_devices.split(','))
prt("NUM_GPUS %d" % NUM_GPUS)
NUM_CLASSES = 120
NUM_ATTRIBS = 10654
BATCH_PER_GPU = 16
# assert BATCH_SIZE%NUM_GPUS==0
SAVE_EVERY_N_EPOCH = 2
DEFAULT_IMAGE_SIZE = 448
IMAGE_SIZE = DEFAULT_IMAGE_SIZE
if len(sys.argv) > 1:
    IMAGE_SIZE = int(sys.argv[1])

STN_OUT_SIZE = 224

prt("IMAGE_SIZE %d" % IMAGE_SIZE)
INIT_LR = 0.01
LOC_LR = 0.00001
EXCLUDE_AUX = False
MAX_TRAIN_EPOCH = 120
MODEL_NAME = "inception_v2"
CKPT_DIR = "./inception_v2.ckpt"
TRAIN_DIR = "./taobao_train"
BATCH_SIZE = BATCH_PER_GPU * NUM_GPUS
NUM_STN = 2

inputs = ["/home/deepinsight/jiaguo/taobao_train.csv3", "/home/deepinsight/jiaguo/taobao_val.csv3"]

if not os.path.exists(TRAIN_DIR):
    os.makedirs(TRAIN_DIR)

USE_VAL = False
train_image_paths = []
train_labels = []
train_labels_a = []
val_image_paths = []
val_labels = []
for i in xrange(len(inputs)):
    with open(inputs[i], 'r') as f:
        for line in f:
            filepath, label, nid, alabel_text = line.split(",")
            label = int(label)
            alabels = alabel_text.split()
            alabels = [int(x) for x in alabels]
            labels_a = np.array([0] * NUM_ATTRIBS, dtype=np.int64)
            for a in alabels:
                labels_a[a] = 1
            if i == 0 or USE_VAL:
                train_image_paths.append(filepath)
                train_labels.append(label)
                train_labels_a.append(labels_a)
            else:
                val_image_paths.append(filepath)
                val_labels.append(label)

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', TRAIN_DIR,
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', NUM_GPUS,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('num_classes', NUM_CLASSES,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_integer('num_samples', len(train_labels),
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_integer('max_train_epoch', MAX_TRAIN_EPOCH,
                            '')
tf.app.flags.DEFINE_integer('save_every_n_steps',
                            int(math.ceil(float(SAVE_EVERY_N_EPOCH) * len(train_labels) / BATCH_SIZE)),
                            '')
tf.app.flags.DEFINE_integer('steps_in_epoch', int(math.ceil(float(len(train_labels)) / BATCH_SIZE)),
                            '')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 20,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 7200,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 0,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'sgd',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', MODEL_NAME, 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', BATCH_SIZE, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'batch_size_in_clone', int(BATCH_SIZE / NUM_GPUS), 'The number of samples in each batch in each clone.')

tf.app.flags.DEFINE_integer(
    'train_image_size', IMAGE_SIZE, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps',
                            int(math.ceil(float(MAX_TRAIN_EPOCH) * len(train_labels) / BATCH_SIZE)),
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', CKPT_DIR,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', 'InceptionResnetV2/AuxLogits' if EXCLUDE_AUX else None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

prt("save every %d steps" % FLAGS.save_every_n_steps)
prt("steps in epoch %d" % FLAGS.steps_in_epoch)
prt("max number of steps %d" % FLAGS.max_number_of_steps)


def preprocessing(image):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [IMAGE_SIZE, IMAGE_SIZE],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    image = tf.image.random_flip_left_right(image)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    return image


def _configure_learning_rate(num_samples_per_epoch, global_step, init_lr):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(init_lr,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(init_lr, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(init_lr,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


savers = []


def _init_fn(sess):
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    for saver in savers:
        saver.restore(sess, checkpoint_path)


def _get_init_fn():
    return _init_fn


def _get_variables_to_train():
    variables_to_train = []
    for v in tf.trainable_variables():
        if not v.name.startswith("loc/"):
            variables_to_train.append(v)
    return variables_to_train


def _get_variables_to_train_lower():
    variables_to_train = []
    for v in tf.trainable_variables():
        if v.name.startswith("loc/"):
            variables_to_train.append(v)
    return variables_to_train


def network_fn(inputs):
    # return transformer_factory.transform(inputs, BATCH_PER_GPU, NUM_STN, (224, 224), NUM_CLASSES, FLAGS.weight_decay, True)
    end_points = {}
    # with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=True):
    # with slim.arg_scope(inception_v3_arg_scope(weight_decay=FLAGS.weight_decay)):
    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with slim.arg_scope(inception_v3_arg_scope(weight_decay=weight_decay)):
            with tf.variable_scope("loc") as scope:
                with tf.variable_scope("net") as scope2:
                    # _, _end_points = inception_resnet_v2.inception_resnet_v2(inputs, num_classes=2, is_training=True, scope = scope2)
                    loc_net, _ = inception_v2.inception_v2_base(inputs, scope=scope2)
                # loc_net = _end_points['Conv2d_7b_1x1']
                loc_net = slim.conv2d(loc_net, 128, [1, 1], scope='Loc_1x1')
                default_kernel_size = [14, 14]
                # kernel_size = _reduced_kernel_size_for_small_input(loc_net, default_kernel_size)
                loc_net = slim.conv2d(loc_net, 128, loc_net.get_shape()[1:3], padding='VALID', activation_fn=tf.nn.tanh,
                                      scope='Loc_fc1')
                loc_net = slim.flatten(loc_net)
                iv = 4.
                initial = np.array([iv, 0, iv, 0] * NUM_STN, dtype=np.float32)
                b_fc_loc = tf.get_variable("Loc_fc_b", shape=[4 * NUM_STN],
                                           initializer=init_ops.constant_initializer(initial), dtype=dtypes.float32)
                W_fc_loc = tf.get_variable("Loc_fc_W", shape=[128, 4 * NUM_STN],
                                           initializer=init_ops.constant_initializer(np.zeros((128, 4 * NUM_STN))),
                                           dtype=dtypes.float32)
                theta = tf.nn.tanh(tf.matmul(loc_net, W_fc_loc) + b_fc_loc)
            _finals = []
            for i in xrange(NUM_STN):
                scope_name = "stn%d" % i
                with tf.variable_scope(scope_name) as scope1:
                    _theta = tf.slice(theta, [0, 4 * i], [-1, 4 * (i + 1)])
                    # loc_net = slim.conv2d(loc_net, 6, [1,1], activation_fn=tf.nn.tanh, scope='Loc_fc', biases_initializer = init_ops.constant_initializer([4.0,0.0,0.0,0.0,4.0,0.0]*128,dtype=dtypes.float32))
                    # loc_net = slim.conv2d(loc_net, 6, [1,1], activation_fn=tf.nn.tanh, scope='Loc_fc', biases_initializer = init_ops.constant_initializer([4.0],dtype=dtypes.float32))
                    # loc_net = slim.flatten(loc_net)
                    stn_output_size = (STN_OUT_SIZE, STN_OUT_SIZE)
                    x = transformer(inputs, _theta, stn_output_size)
                    x.set_shape([BATCH_PER_GPU, stn_output_size[0], stn_output_size[1], 3])
                    # x.set_shape(tf.shape(inputs))
                    # tf.reshape(x, tf.shape(inputs))
                    end_points['x'] = x
                    # with tf.variable_scope("net") as scope2:
                    #  return inception_resnet_v2.inception_resnet_v2(x, num_classes=NUM_CLASSES, is_training=True, scope = scope2)
                    with tf.variable_scope("net") as scope2:
                        net, _ = inception_v2.inception_v2_base(x, scope=scope2)
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a')
                    net = slim.dropout(net, keep_prob=0.7, scope='Dropout_1b')
                    _finals.append(net)
            with tf.variable_scope('Logits'):
                net = tf.concat(axis=3, values=_finals)
                logits = slim.conv2d(net, NUM_CLASSES, [1, 1], activation_fn=None,
                                     normalizer_fn=None, scope='Conv2d_1c_1x1')
                logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                predictions = slim.softmax(logits, scope='Predictions')
                end_points['Predictions'] = predictions

                logits_a = slim.conv2d(net, NUM_ATTRIBS, [1, 1], activation_fn=None,
                                       normalizer_fn=None, scope='Conv2d_1c_1x1_a')
                logits_a = tf.squeeze(logits_a, [1, 2], name='SpatialSqueeze_a')
                predictions_a = slim.sigmoid(logits_a, scope='Predictions_a')
                end_points['Predictions_a'] = predictions_a
                return logits, logits_a, end_points


def train_step(sess, train_op, global_step, lr, train_step_kwargs):
    """Function that takes a gradient step and specifies whether to stop.

    Args:
      sess: The current session.
      train_op: An `Operation` that evaluates the gradients and returns the
        total loss.
      global_step: A `Tensor` representing the global training step.
      train_step_kwargs: A dictionary of keyword arguments.

    Returns:
      The total loss and a boolean indicating whether or not to stop training.

    Raises:
      ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
    """
    start_time = time.time()

    trace_run_options = None
    run_metadata = None
    if 'should_trace' in train_step_kwargs:
        if 'logdir' not in train_step_kwargs:
            raise ValueError('logdir must be present in train_step_kwargs when '
                             'should_trace is present')
        if sess.run(train_step_kwargs['should_trace']):
            trace_run_options = config_pb2.RunOptions(
                trace_level=config_pb2.RunOptions.FULL_TRACE)
            run_metadata = config_pb2.RunMetadata()

    total_loss, lr_value, np_global_step = sess.run([train_op, lr, global_step],
                                                    options=trace_run_options,
                                                    run_metadata=run_metadata)
    time_elapsed = time.time() - start_time

    if run_metadata is not None:
        tl = timeline.Timeline(run_metadata.step_stats)
        trace = tl.generate_chrome_trace_format()
        trace_filename = os.path.join(train_step_kwargs['logdir'],
                                      'tf_trace-%d.json' % np_global_step)
        logging.info('Writing trace to %s', trace_filename)
        file_io.write_string_to_file(trace_filename, trace)
        if 'summary_writer' in train_step_kwargs:
            train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
                                                                 'run_metadata-%d' %
                                                                 np_global_step)

    if 'should_log' in train_step_kwargs:
        if sess.run(train_step_kwargs['should_log']):
            logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                         np_global_step, total_loss, time_elapsed)
            prt('global step %d with lr %.4f: loss = %.4f (%.3f sec/step)' %
                (np_global_step, lr_value, total_loss, time_elapsed))

    # TODO(nsilberman): figure out why we can't put this into sess.run. The
    # issue right now is that the stop check depends on the global step. The
    # increment of global step often happens via the train op, which used
    # created using optimizer.apply_gradients.
    #
    # Since running `train_op` causes the global step to be incremented, one
    # would expected that using a control dependency would allow the
    # should_stop check to be run in the same session.run call:
    #
    #   with ops.control_dependencies([train_op]):
    #     should_stop_op = ...
    #
    # However, this actually seems not to work on certain platforms.
    if 'should_stop' in train_step_kwargs:
        should_stop = sess.run(train_step_kwargs['should_stop'])
    else:
        should_stop = False

    return total_loss, np_global_step, should_stop


def do_training(train_op, init_fn=None, summary_op=None, lr=None):
    global savers
    graph = ops.get_default_graph()
    with graph.as_default():
        global_step = variables.get_or_create_global_step()
        saver = tf_saver.Saver(max_to_keep=0)

        with ops.name_scope('init_ops'):
            init_op = tf_variables.global_variables_initializer()

            ready_op = tf_variables.report_uninitialized_variables()

            local_init_op = control_flow_ops.group(
                tf_variables.local_variables_initializer(),
                data_flow_ops.tables_initializer())

        summary_writer = supervisor.Supervisor.USE_DEFAULT
        with ops.name_scope('train_step'):
            train_step_kwargs = {}

            if not FLAGS.max_number_of_steps is None:
                should_stop_op = math_ops.greater_equal(global_step, FLAGS.max_number_of_steps)
            else:
                should_stop_op = constant_op.constant(False)
            train_step_kwargs['should_stop'] = should_stop_op
            if FLAGS.log_every_n_steps > 0:
                train_step_kwargs['should_log'] = math_ops.equal(
                    math_ops.mod(global_step, FLAGS.log_every_n_steps), 0)
        prefix = "loc/net"
        lp = len(prefix)
        vdic = {"InceptionV2" + v.op.name[lp:]: v for v in tf.trainable_variables() if
                v.name.startswith(prefix) and v.name.find("Logits/") < 0}
        _saver = tf_saver.Saver(vdic)
        savers.append(_saver)
        for i in xrange(NUM_STN):
            prefix = "stn%d/net" % i
            lp = len(prefix)
            vdic = {"InceptionV2" + v.op.name[lp:]: v for v in tf.trainable_variables() if
                    v.name.startswith(prefix) and v.name.find("Logits/") < 0}
            # saver = tf.train.Saver(vdic)
            _saver = tf_saver.Saver(vdic)
            savers.append(_saver)
    prt("savers %d" % len(savers))

    is_chief = True
    logdir = FLAGS.train_dir

    sv = supervisor.Supervisor(
        graph=graph,
        is_chief=is_chief,
        logdir=logdir,
        init_op=init_op,
        init_feed_dict=None,
        local_init_op=local_init_op,
        ready_for_local_init_op=None,
        ready_op=ready_op,
        summary_op=summary_op,
        summary_writer=summary_writer,
        global_step=global_step,
        saver=saver,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_model_secs=FLAGS.save_interval_secs,
        init_fn=init_fn)

    if summary_writer is not None:
        train_step_kwargs['summary_writer'] = sv.summary_writer

    with sv.managed_session('', start_standard_services=False, config=None) as sess:
        logging.info('Starting Session.')
        if is_chief:
            if logdir:
                sv.start_standard_services(sess)
        elif startup_delay_steps > 0:
            _wait_for_step(sess, global_step,
                           min(startup_delay_steps, number_of_steps or
                               sys.maxint))
        sv.start_queue_runners(sess)
        logging.info('Starting Queues.')
        try:
            while not sv.should_stop():
                total_loss, global_step_value, should_stop = train_step(
                    sess, train_op, global_step, lr, train_step_kwargs)
                current_epoch = int(math.ceil(float(global_step_value) / FLAGS.steps_in_epoch))
                if global_step_value > 0 and global_step_value % FLAGS.save_every_n_steps == 0:
                    sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

                if should_stop:
                    logging.info('Stopping Training.')
                    break
        except errors.OutOfRangeError:
            # OutOfRangeError is thrown when epoch limit per
            # tf.train.limit_epochs is reached.
            logging.info('Caught OutOfRangeError. Stopping Training.')
        if logdir and sv.is_chief:
            logging.info('Finished training! Saving model to disk.')
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


def main(_):
    # if not FLAGS.dataset_dir:
    #  raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=NUM_GPUS,
            clone_on_cpu=False,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        ######################
        # Select the dataset #
        ######################
        # dataset = dataset_factory.get_dataset(
        #    FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        ######################
        # Select the network #
        ######################
        # network_fn = nets_factory.get_network_fn(
        #    FLAGS.model_name,
        #    num_classes=NUM_CLASSES,
        #    weight_decay=FLAGS.weight_decay,
        #    is_training=True)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################
        with tf.device(deploy_config.inputs_device()):
            train_image_size = FLAGS.train_image_size
            # provider = slim.dataset_data_provider.DatasetDataProvider(
            #    dataset,
            #    num_readers=FLAGS.num_readers,
            #    common_queue_capacity=20 * FLAGS.batch_size,
            #    common_queue_min=10 * FLAGS.batch_size)
            # [image, label] = provider.get(['image', 'label'])
            _images = tf.convert_to_tensor(train_image_paths, dtype=tf.string)
            _labels = tf.convert_to_tensor(train_labels, dtype=tf.int64)
            _labels_a = tf.convert_to_tensor(train_labels_a, dtype=tf.int64)
            input_queue = tf.train.slice_input_producer([_images, _labels, _labels_a], shuffle=True)
            file_path = input_queue[0]
            tf.Print(file_path, [file_path], "image path:")
            file_content = tf.read_file(file_path)
            image = tf.image.decode_jpeg(file_content, channels=3)
            image = preprocessing(image)
            # image = image_preprocessing_fn(image, train_image_size, train_image_size)
            label = input_queue[1]
            label -= FLAGS.labels_offset
            label_a = input_queue[2]

            images, labels, labels_a = tf.train.batch(
                [image, label, label_a],
                batch_size=FLAGS.batch_size_in_clone,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=(NUM_GPUS + 2) * FLAGS.batch_size_in_clone)

            # [images], labels = tf.contrib.training.stratified_sample(
            #    [input_queue[0]], input_queue[1], target_probs,
            #    batch_size=FLAGS.batch_size_in_clone,
            #    init_probs = init_probs,
            #    threads_per_queue=FLAGS.num_preprocessing_threads,
            #    queue_capacity=(NUM_GPUS+2)*FLAGS.batch_size_in_clone, prob_dtype=dtypes.float64)
            # labels -= FLAGS.labels_offset
            labels = slim.one_hot_encoding(
                labels, FLAGS.num_classes - FLAGS.labels_offset)
            # images_ = []
            # im_dtype = dtypes.int32
            # for i in xrange(FLAGS.batch_size_in_clone):
            #  image = images[i]
            #  file_content = tf.read_file(image)
            #  image = tf.image.decode_jpeg(file_content, channels=3)
            #  image = image_preprocessing_fn(image, train_image_size, train_image_size)
            #  im_dtype = image.dtype
            #  images_.append(image)
            # images = tf.convert_to_tensor(images_, dtype=im_dtype)

            batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels, labels_a], capacity=8 * deploy_config.num_clones)

            # _images_val = tf.convert_to_tensor(val_image_paths,dtype=tf.string)
            # _labels_val = tf.convert_to_tensor(val_labels,dtype=tf.int64)
            # input_queue_val = tf.train.slice_input_producer([_images_val, _labels_val], shuffle=False)
            # file_content_val = tf.read_file(input_queue_val[0])
            # image_val = tf.image.decode_jpeg(file_content_val, channels=3)
            # label_val = input_queue_val[1]
            # label_val -= FLAGS.labels_offset

            # image_size = FLAGS.train_image_size or network_fn.default_image_size

            # image_val = image_preprocessing_fn(image_val, image_size, image_size)

            # images_val, labels_val = tf.train.batch(
            #    [image_val, label_val],
            #    batch_size=FLAGS.batch_size_in_clone,
            #    num_threads=FLAGS.num_preprocessing_threads,
            #    capacity=5 * FLAGS.batch_size_in_clone)
            # labels_val = slim.one_hot_encoding(
            #    labels_val, FLAGS.num_classes - FLAGS.labels_offset)
            # batch_queue_val = slim.prefetch_queue.prefetch_queue(
            #    [images_val, labels_val], capacity=2 * deploy_config.num_clones)

        ####################
        # Define the model #
        ####################
        def clone_fn(batch_queue):
            """Allows data parallelism by creating multiple clones of network_fn."""
            images, labels, labels_a = batch_queue.dequeue()
            logits, logits_a, end_points = network_fn(images)

            #############################
            # Specify the loss function #
            #############################
            if not EXCLUDE_AUX and 'AuxLogits' in end_points:
                tf.losses.softmax_cross_entropy(
                    logits=end_points['AuxLogits'], onehot_labels=labels,
                    label_smoothing=FLAGS.label_smoothing, weights=0.4, scope='aux_loss')
            tf.losses.softmax_cross_entropy(
                logits=logits, onehot_labels=labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)
            tf.losses.sigmoid_cross_entropy(
                logits=logits_a, multi_class_labels=labels_a,
                label_smoothing=FLAGS.label_smoothing, weights=1.0)
            return end_points

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        # Gather update_ops from the first clone. These contain, for example,
        # the updates for the batch_norm variables created by network_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Add summaries for end_points.
        end_points = clones[0].outputs
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        #################################
        # Configure the moving averages #
        #################################
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = _configure_learning_rate(FLAGS.num_samples, global_step, INIT_LR)
            optimizer = _configure_optimizer(learning_rate)
            learning_rate_lower = _configure_learning_rate(FLAGS.num_samples, global_step, LOC_LR)
            optimizer_lower = _configure_optimizer(learning_rate_lower)
            # summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        variables_to_train = _get_variables_to_train()
        variables_to_train_lower = _get_variables_to_train_lower()

        total_loss_lower, clones_gradients_lower = model_deploy.optimize_clones(
            clones,
            optimizer_lower,
            var_list=variables_to_train_lower)

        total_loss, clones_gradients = model_deploy.optimize_clones(
            clones,
            optimizer,
            var_list=variables_to_train)

        # print(len(clones_gradients))
        # print(len(variables_to_train))
        # print(len(variables_to_train_lower))
        # assert(len(clones_gradients)==len(variables_to_train)+len(variables_to_train_lower))

        grad_updates = optimizer.apply_gradients(clones_gradients)
        update_ops.append(grad_updates)
        grad_updates_lower = optimizer_lower.apply_gradients(clones_gradients_lower, global_step=global_step)
        update_ops.append(grad_updates_lower)

        update_op = tf.group(*update_ops)
        train_tensor = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or _gather_clone_loss().
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        ###########################
        # Kicks off the training. #
        ###########################
        # slim.learning.train(
        #    train_tensor,
        #    logdir=FLAGS.train_dir,
        #    master=FLAGS.master,
        #    is_chief=(FLAGS.task == 0),
        #    init_fn=_get_init_fn(),
        #    summary_op=summary_op,
        #    number_of_steps=FLAGS.max_number_of_steps,
        #    log_every_n_steps=FLAGS.log_every_n_steps,
        #    save_summaries_secs=FLAGS.save_summaries_secs,
        #    save_interval_secs=FLAGS.save_interval_secs,
        #    sync_optimizer=optimizer if FLAGS.sync_replicas else None)

        do_training(train_tensor, init_fn=_get_init_fn(), summary_op=summary_op, lr=learning_rate)


# data = np.random.normal( size=(32, 224, 224, 3))
if __name__ == '__main__':
    tf.app.run()
    sys.exit(0)

data = np.zeros((32, 224, 224, 3))
data1 = np.zeros((32, 224, 224, 3))
inputs = tf.convert_to_tensor(data, dtype=dtypes.float32)
inputs1 = tf.convert_to_tensor(data1, dtype=dtypes.float32)
end_points = network_fn(inputs)
end_points1 = network_fn(inputs1, reuse=True)
for v in tf.global_variables():
    print(v.op.name)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    evalue = sess.run(end_points)
    x_value = evalue['x']
    prt(x_value.shape)
    # prt(x_value)
    # evalue = sess.run(end_points1)
    # x_value = evalue['x']
    # prt(x_value.shape)
    # prt(x_value)
#  saver_1 = tf.train.Saver({"InceptionV2"+v.op.name[4:] : v for v in tf.global_variables() if v.name.startswith("inc1") and v.name.find("Logits/")<0})
#  saver_1.restore(sess, ckpt_file)
#  saver_2 = tf.train.Saver({"InceptionV2"+v.op.name[4:] : v for v in tf.global_variables() if v.name.startswith("inc2") and v.name.find("Logits/")<0})
#  saver_2.restore(sess, ckpt_file)


