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

