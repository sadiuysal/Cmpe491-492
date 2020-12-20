# -*- coding: utf-8 -*-
"""beginner.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb

##### Copyright 2019 The TensorFlow Authors.
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
from tensorflow.keras import layers
import objective as obj_lib
import data_util
import tensorflow as tf
import numpy as np
import sys
import model as model_class


"""Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

#mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, y_train, x_test, y_test = x_train[-1000:] , y_train[-1000:], x_test[-1000:], y_test[-1000:]
x_train, x_test = x_train / 255.0, x_test / 255.0

batch_size=tf.shape(x_train)[0]
indicies=np.array([i for i in range(batch_size)]).reshape((batch_size, 1))


IMG_SIZE=tf.shape(x_train)[1]
#resize_and_scale with layers
resize_and_rescale = model_class.resize_and_rescale
#data augmentation layers
data_augmentation = model_class.data_augmentation




"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""

model = model_class.model



def prepare(ds=x_train, shuffle=False, augment=True,batch_size = 32):
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  # Resize and rescale all datasets
  ds = ds.map(lambda x, y: (augment(x), y), 
              num_parallel_calls=AUTOTUNE)
  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefecting on all datasets
  return ds.prefetch(buffer_size=AUTOTUNE)

def display_aug_images():
  image=x_train[0]
  # Add the image to a batch
  image = tf.expand_dims(image, 0)
  plt.figure(figsize=(10, 10))
  for i in range(9):
    augmented_image = data_augmentation(image)
    print(augmented_image)
    ax = plt.subplot(3, 3, i + 1)
    plt.show(augmented_image[0],block=True)
    plt.axis("off")

"""For each example the model returns a vector of "[logits](https://developers.google.com/machine-learning/glossary#logits)" or "[log-odds](https://developers.google.com/machine-learning/glossary#log-odds)" scores, one for each class."""

#predictions = model(tf.expand_dims(x_train_sets[0],0)).numpy()
#predictions

"""The `tf.nn.softmax` function converts these logits to "probabilities" for each class: """

#tf.nn.softmax(predictions).numpy()

"""Note: It is possible to bake this `tf.nn.softmax` in as the activation function for the last layer of the network. While this can make the model output more directly interpretable, this approach is discouraged as it's impossible to
provide an exact and numerically stable loss calculation for all models when using a softmax output.

The `losses.SparseCategoricalCrossentropy` loss takes a vector of logits and a `True` index and returns a scalar loss for each example.
"""



#loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn = obj_lib.contrastive_loss

"""This loss is equal to the negative log probability of the true class:
It is zero if the model is sure of the correct class.

This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to `-tf.log(1/10) ~= 2.3`.
"""

def run_model():
  #tf.compat.v1.enable_eager_execution()
  model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
  """ The `Model.fit` method adjusts the model parameters to minimize the loss: """
  model.fit(x_train, indicies, epochs=5)
  """The `Model.evaluate` method checks the models performance, usually on a "[Validation-set](https://developers.google.com/machine-learning/glossary#validation-set)" or "[Test-set](https://developers.google.com/machine-learning/glossary#test-set)"."""

  #model.evaluate(x_test,  y_test, verbose=2)

  """The image classifier is now trained to ~98% accuracy on this dataset. To learn more, read the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

  If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:
  """
  """
  probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
  ])

  probability_model(x_test[:5])
  """


#print(loss_fn(indicies[0], x_train[0]))
run_model()



