
import tensorflow as tf
import objective as obj_lib
import data_util
import numpy as np
"""
def get_adversaries( model, x ,target,epsilon,itr_count):
  #print(tf.shape(x))
  #print(tf.shape(target))
  with tf.GradientTape() as tape:
    tape.watch(x)
    inputs = tf.concat([x, target], 0)
    outputs=model(inputs, training=False)
    loss = obj_lib.contrastive_Loss(outputs)

  #print("current loss : "+ str(loss))
  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, x)
  #print(gradient)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  adv = x + (epsilon * signed_grad)
  adversaries = data_util.cast_to_float32(tf.clip_by_value(adv, 0, 1)) #TODO change to 0-255
  if itr_count>1:
    return get_adversaries(model,adversaries,target,epsilon,itr_count-1)
  else:
    inputs = tf.concat([adv, x], 0)
    outputs = model(inputs, training=False)
    loss = obj_lib.contrastive_Loss(outputs)
    # print(tf.shape(adversaries))
    return adversaries,loss"""

def get_adversaries_2( model, x ,target,epsilon,itr_count,step_size):  # not completed
  perturbation = np.random.uniform(-epsilon,epsilon,tf.shape(x))
  adv = x + perturbation
  adv = data_util.cast_to_float32(tf.clip_by_value(adv, 0, 255))
  for itr in range(itr_count):
    with tf.GradientTape() as tape:
      tape.watch(adv)
      inputs = tf.concat([adv, target], 0)
      outputs=model(inputs, training=False)
      loss = obj_lib.contrastive_Loss(outputs)

    #print("Adversarial example finding itr count: "+str(itr)+" Loss: " + str(loss))
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, adv)
    #print(gradient)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    perturbation += (step_size * signed_grad)
    perturbation = data_util.cast_to_float32(tf.clip_by_value(perturbation, -epsilon, epsilon))
    adv = x + perturbation
    adversaries = data_util.cast_to_float32(tf.clip_by_value(adv, 0, 255))


  inputs = tf.concat([adv, x], 0)
  outputs = model(inputs, training=False)
  loss = obj_lib.contrastive_Loss(outputs)
  # print(tf.shape(adversaries))
  return adversaries,loss
