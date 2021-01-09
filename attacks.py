
import tensorflow as tf
import objective as obj_lib

loss_object = obj_lib.contrastive_Loss

"""
def create_adversarial_pattern(input_images, input_label):
  with tf.GradientTape() as tape:
    tape.watch(input_images)
    prediction = pretrained_model(input_images)
    loss = loss_object(input_label, prediction)

  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, input_images)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  return signed_grad
"""
def get_loss(target):
  return