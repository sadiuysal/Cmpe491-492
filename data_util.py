"""Data preprocessing and augmentation."""


import tensorflow as tf

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
