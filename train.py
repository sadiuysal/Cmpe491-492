"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer


#import tensorflow as tf
import numpy as np
#import imageio


from model import Model
import cifar10_input

from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

def convert_to_onehot(y, nb_classes):
    y_onehot = np.zeros([len(y), nb_classes])
    for i in range(len(y)):
        y_onehot[i, y[i]] = 1.0
    return y_onehot


# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()

model = Model(mode='train', epsilon=config['epsilon'])


# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)


mean_xent, xent, weight_decay_loss, accuracy_nat, _ = model.loss_func()
total_loss = mean_xent + weight_decay * weight_decay_loss 

train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(total_loss, global_step=global_step)



# TODO set up adversary with GAN 
# Set up adversary
attack = LinfPGDAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])


# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=200)
#tf.summary.scalar('accuracy adv train', accuracy_adv)
tf.summary.scalar('xent adv train', xent / batch_size)
#tf.summary.image('images adv train', x_adv)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)



with tf.Session() as sess:
 #TODO Already Augmented or Clean examples ? ASK 
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  output_dir = os.path.join(model_dir, 'samples')
  if not os.path.exists(output_dir):
      os.makedirs(output_dir)


  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)
    y_batch = convert_to_onehot(y_batch, 10)
    #x_batch = preprocess(x_batch)
    
    #x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    ##added 
    #generator = model.generator(x, f_dim, output_size, c_dim, is_training=True)
    # f_dim = nof filters to use for generator

    train_dict = {model.x_input: x_batch, model.y_input: y_batch} 
     
    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(accuracy_nat, feed_dict=train_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()), flush=True)
      print('training nat accuracy {:.4}%'.format(nat_acc * 100), flush=True)
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=train_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict=train_dict)
    end = timer()
    training_time += end - start
