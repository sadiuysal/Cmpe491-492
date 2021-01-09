
import tensorflow as tf
import objective as obj_lib


def get_adversaries( model, x ,target,epsilon):
  print(tf.shape(x))
  with tf.GradientTape() as tape:
    tape.watch(x)
    inputs = tf.concat([x, target], 0)
    outputs=model(inputs, training=False)
    loss = obj_lib.contrastive_Loss(outputs)

  print(loss)
  # Get the gradients of the loss w.r.t to the input image.
  gradient = tape.gradient(loss, x)
  print(gradient)
  # Get the sign of the gradients to create the perturbation
  signed_grad = tf.sign(gradient)
  adv = x + (epsilon * signed_grad)
  adversaries = tf.clip_by_value(adv, 0, 1)

  inputs=tf.concat([adv, x], 0)
  outputs = model(inputs, training=False)
  loss = obj_lib.contrastive_Loss(outputs)
  print(tf.shape(adversaries))
  return adversaries,loss
