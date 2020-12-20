"""Data preprocessing and augmentation."""

import functools
import numpy as np
import tensorflow as tf
from scipy import spatial



#cosine similarity_with_temperature
def sim_with_temperature(x,y,temperature):
  #print(x)
  #print(y)
  #cos_sim=1 - spatial.distance.cosine(x, y)
  m = tf.keras.metrics.CosineSimilarity(axis=1)
  m.update_state(x, y)
  return m.result().numpy()
  #return cos_sim/temperature

