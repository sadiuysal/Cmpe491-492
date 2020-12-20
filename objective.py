# coding=utf-8
# Copyright 2020 The SimCLR Authors.
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
# See the License for the specific simclr governing permissions and
# limitations under the License.
# ==============================================================================
"""Contrastive loss functions."""

from absl import flags
import data_util
#import tensorflow.compat.v2 as tf
import tensorflow as tf
import model as model_class

 
def contrastive_loss(ind , output , temperature=1 ): 
  print("***************-1")
  print(tf.trainable_variables())
  x = tf.gather(model_class.x_train, ind)[0]
  #batch_size=model_class.batch_size
  #mask=tf.one_hot(ind, depth = batch_size , on_value=0, off_value=1)[0]
  #t_x=model_class.data_augmentation(x)
  t_prime_x=model_class.data_augmentation(x)
  print("index : ")
  print(ind)
  #print("x shape : ")
  #print(tf.shape(t_x))
  #z_x = model_class.model(t_x)  # shape [None, 10] for now
  z_prime_x = model_class.model(t_prime_x)  # shape [None, 10] for now
  #print("z(x) shape : ")
  #print(tf.shape(z_x))
  #print("z'(x) shape: ")
  #print(tf.shape(z_prime_x))

  #print(model_class.vector_mappings)
  function_to_map = lambda y : tf.math.exp(data_util.sim_with_temperature(output ,y ,temperature)) 
  exponential_sim_matrix = tf.map_fn(function_to_map, model_class.vector_mappings , dtype=tf.float32)
  #print(tf.shape(exponential_sim_matrix))
  z_neg_sum = tf.reduce_sum(exponential_sim_matrix) - exponential_sim_matrix[ind[0,0]]
  z_pos_sum = tf.math.exp(data_util.sim_with_temperature(output ,z_prime_x ,temperature))
  loss = - tf.math.log(z_pos_sum / (z_pos_sum+z_neg_sum) )
  #print(z_neg_sum)
  #print(z_pos_sum)
  print("Loss  : " + str(loss))
  return loss

def RoCL_contrastive_loss(y_train_sets,outputs, temperature=0.5):
  batch_size=int(tf.shape(outputs)[0])
  print()
  print(tf.shape(y_train_sets))
  print(tf.shape(outputs))
  #LOOK HERE
  for ind in range(batch_size):
    z_prime = y_train_sets[ind]
    z_positive_set = [z_prime]
    z = outputs[ind]
    sum_z_pos = 0
    sum_z_neg = 0
    for z_pos in z_positive_set:
      sum_z_pos += tf.math.exp(data_util.sim_with_temperature(z,z_pos,temperature) , name=None)
    for temp in range(batch_size):
      if temp == ind:
        continue
      z_neg = outputs[ind]
      sum_z_neg += tf.math.exp(data_util.sim_with_temperature(z,z_neg,temperature) , name=None)
    return - tf.math.log(sum_z_pos/(sum_z_neg+sum_z_pos), name=None)
