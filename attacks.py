
import tensorflow as tf
import objective as obj_lib
import data_util

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
  adversaries = data_util.cast_to_float32(tf.clip_by_value(adv, 0, 1))
  if itr_count>1:
    return get_adversaries(model,adversaries,target,epsilon,itr_count-1)
  else:
    inputs = tf.concat([adv, x], 0)
    outputs = model(inputs, training=False)
    loss = obj_lib.contrastive_Loss(outputs)
    # print(tf.shape(adversaries))
    return adversaries,loss


