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

import data_util
import tensorflow as tf


# outputs is 2N*d
def contrastive_Loss( outputs , adversarial_selection =False , temperature= 1):
  
  N=int(tf.shape(outputs)[0]/2)
  #print("N : " + str(N))
  #print("outputs.shape : " + str(tf.shape(outputs)))
  z_x=tf.slice(outputs,[ 0,0 ],[ N ,-1 ])
  #print(tf.shape(z_x))
  z_prime_x= tf.slice(outputs,[ N,0 ],[ -1 , -1 ])

  #print(str(tf.shape(z_x)) + "  &&&&&&&&&&&&  " + str(tf.shape(z_prime_x) ))
  sim_matrix=data_util.sim_matrix_with_temperature(z_x,z_prime_x,temperature)
  diagonal = tf.linalg.diag_part(sim_matrix)
  diagonal = tf.math.exp(diagonal)
  upper = tf.linalg.band_part(sim_matrix, 0, -1)
  upper = tf.math.exp(upper)
  pos_set = tf.math.reduce_sum(diagonal)
  neg_set = tf.math.reduce_sum(upper)-pos_set
  loss = - tf.math.log(pos_set / (pos_set+neg_set) )
  return loss

