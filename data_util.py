"""Data preprocessing and augmentation."""


import tensorflow as tf
import numpy as np

#cosine similarity_with_temperature
def sim_matrix_with_temperature(x,y,temperature):
  # normalize each row
  normalized_x = tf.math.l2_normalize(
    x, axis=1
  )
  normalized_y = tf.math.l2_normalize(
    y, axis=1
  )
  # multiply row i with row j using transpose
  prod = tf.matmul(normalized_x, normalized_y,
                   adjoint_b=True  # transpose second matrix
                   )
  return prod / temperature

#similarity matrix finder with temperature
def sim_matrix_finder(x,temperature):
  # normalize each row
  normalized_x = tf.math.l2_normalize(
    x, axis=1
  )
  # multiply row i with row j using transpose
  prod = tf.matmul(normalized_x, normalized_x,
                   adjoint_b=True  # transpose second matrix
                   )
  return prod / temperature

def find_loss_from_sim_matrix(sim_matrix):
    diagonal = tf.linalg.diag_part(sim_matrix)
    diagonal = tf.math.exp(diagonal)
    upper = tf.linalg.band_part(sim_matrix, 0, -1)
    upper = tf.math.exp(upper)
    pos_set = tf.math.reduce_sum(diagonal)
    neg_set = tf.math.reduce_sum(upper) - pos_set
    loss = - tf.math.log(pos_set / (pos_set + neg_set))
    return loss

def cast_to_float32(data):
  return tf.cast(data,tf.float32)



class ColorJitter_and_GrayScale(tf.keras.layers.Layer):
    def __init__(self, colorJitter_prob,grayScale_prob, **kwargs):
        super(ColorJitter_and_GrayScale, self).__init__(**kwargs)
        self.colorJitter_prob = colorJitter_prob
        self.grayScale_prob = grayScale_prob


    def call(self, images, training=None):
        if not training:
            return images
        # colorJitter
        rand = np.random.uniform()
        if rand < self.colorJitter_prob:
            images = tf.image.random_hue(images, 0.08)
            images = tf.image.random_saturation(images, 0.6, 1.6)
            images = tf.image.random_brightness(images, 0.05)

        #grayScale
        rand = np.random.uniform()
        if rand < self.grayScale_prob:
            images = tf.image.rgb_to_grayscale(images)

        images = tf.clip_by_value(images, -1, 1)
        return images