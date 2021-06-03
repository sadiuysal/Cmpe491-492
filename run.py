# -*- coding: utf-8 -*-

# @title Licensed under the Apache License, Version 2.0 (the "License");
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
import tensorflow as tf
from tensorflow import keras
import objective as loss
import config as cfg
import data_util
import models

"""Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

#mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = data_util.cast_to_float32 ( x_train ) , data_util.cast_to_float32( x_test )
(x_train, y_train), (x_test, y_test) = (x_train[:1000,:]/255, y_train[:1000,:]), (x_test[:100,:]/255, y_test[:100,:])

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(cfg.BUFFER_SIZE).batch(cfg.BATCH_SIZE)

inputs = keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
"""
base_model = applications.resnet50.ResNet50(weights=None, include_top=False,
                                            input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
discriminator = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet',
    input_shape=(32,32,3), input_tensor=inputs,pooling=None
)"""


noise = tf.random.normal([1, 100])
generator = models.make_generator_model(latent_dim=100)
# summarize the model
generator.summary()

#generated_image = generator(noise, training=False)
#print(generated_image[0, :, :, 0].shape)
#print(x_train[0,:].shape)
#plt.imshow(generated_image[0, :, :, 0], cmap='gray')
#plt.show()


