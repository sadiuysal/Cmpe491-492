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
#TODO temperature parameter
def contrastive_Loss( output , adversarial_selection =False , temperature= 0.5 , _lambda = 256):
  if adversarial_selection:
    N = int( tf.shape(output)[0]/3 )
    # RESNET returns tensor with shapes [N,1,1,2048], so I reshaped it.
    outputs = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[-1]], name=None)
    # print("N : " + str(N))
    # print("outputs.shape : " + str(tf.shape(outputs)))
    z_x = tf.slice(outputs, [0, 0], [N, -1])
    z_prime_x = tf.slice(outputs, [N, 0], [2*N, -1])
    z_adversaries = tf.slice(outputs, [2*N, 0], [-1, -1])
    #print(str(tf.shape(z_x)) + "  &&&&&&&&&&&&  " + str(tf.shape(z_prime_x)) + "  &&&&&&&&&&&&  " + str(tf.shape(z_adversaries) ))
    sim_matrix_1 = data_util.sim_matrix_with_temperature(z_x, z_prime_x, temperature)
    sim_matrix_2 = data_util.sim_matrix_with_temperature(z_x, z_adversaries, temperature)
    sim_matrix_3 = data_util.sim_matrix_with_temperature(z_adversaries,z_x , temperature)
    sim_matrix_4 = data_util.sim_matrix_with_temperature(z_adversaries, z_prime_x, temperature)
    loss = data_util.find_loss_from_sim_matrix(sim_matrix_1) + data_util.find_loss_from_sim_matrix(sim_matrix_2)
    loss_adv = data_util.find_loss_from_sim_matrix(sim_matrix_3) + data_util.find_loss_from_sim_matrix(sim_matrix_4)
    loss += (1/_lambda)*loss_adv
  else:
    N=int( tf.shape(output)[0]/2 )
    # RESNET returns tensor with shapes [N,1,1,2048], so I reshaped it.
    outputs=tf.reshape(output, [ tf.shape(output)[0] ,tf.shape(output)[-1] ], name=None)
    #print("N : " + str(N))
    #print("outputs.shape : " + str(tf.shape(outputs)))
    z_x=tf.slice(outputs,[ 0,0 ],[ N ,-1 ])
    #print(tf.shape(z_x))
    z_prime_x= tf.slice(outputs,[ N,0 ],[ -1 , -1 ])
    #print(str(tf.shape(z_x)) + "  &&&&&&&&&&&&  " + str(tf.shape(z_prime_x) ))
    sim_matrix=data_util.sim_matrix_with_temperature(z_x,z_prime_x,temperature)
    loss = data_util.find_loss_from_sim_matrix(sim_matrix)

  return loss

