"""Data preprocessing and augmentation."""

import functools
import numpy as np
import tensorflow as tf




#cosine similarity_with_temperature
def sim_with_temperature(x,y,temperature):
  print(tf.shape(x))
  print(tf.shape(y))
  print(x)
  print(y)
  cos_sim=np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
  """m = tf.keras.metrics.CosineSimilarity(axis=1)
  m.update_state(x, y)"""
  return cos_sim/temperature

