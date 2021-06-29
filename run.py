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
import glob
import PIL
import imageio as imageio
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils

import config as cfg
import data_util
import models
import os
import tensorflow_docs.vis.embed as embed

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
logical_devices_CPU = tf.config.list_logical_devices('CPU')
print("Num CPUs:", len(logical_devices_CPU))

logical_devices_GPU = tf.config.list_logical_devices('GPU')
print("Num GPUs:", len(logical_devices_GPU))
#print("GPU devices: \n", logical_devices_GPU)

logical_devices_TPU = tf.config.list_logical_devices('TPU')
print("Num TPUs:", len(logical_devices_TPU))

# TO SET SPECIFIC DEVICE TYPE
device = logical_devices_CPU

print("Using device :", device[0].name)
#strategy = tf.distribute.MirroredStrategy()

with tf.device(device[0].name):
#with strategy.scope():

    """Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

    #mnist = tf.keras.datasets.mnist
    cifar10 = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = data_util.cast_to_float32 ( x_train ) , data_util.cast_to_float32( x_test )
    #(x_train, y_train), (x_test, y_test) = (x_train[:128*1,:], y_train[:128*1,:]), (x_test[:128,:], y_test[:128,:])

    x_train = x_train / 255.0  #Normalize to [0,1]
    x_test = x_test / 255.0

    x_train = (x_train - 0.5) * 2  # Normalize the images to [-1, 1]
    x_test = (x_test - 0.5) * 2   # Normalize the images to [-1, 1]


    # Change labels to binary class labels
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)



    #generator = models.make_generator_model(latent_dim=100)

    #backbone, classifier = models.make_discriminator_model()

    #test_accuracy_2(x_test,y_test,classifier)
    #backbone,classifier = models.fineTuneonCifar10(backbone,classifier,x_train,y_train,x_test,y_test,isLoad=True)




    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(cfg.BUFFER_SIZE).batch(
        cfg.BATCH_SIZE,drop_remainder=True,num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(cfg.BUFFER_SIZE).batch(
        cfg.BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)


    generator_optimizer = tf.keras.optimizers.Adam(1e-4) #optimizer
    g_loss_tracker = tf.keras.metrics.Mean(name="loss")          #loss tracker
    EPOCHS = cfg.nof_epochs
    noise_dim = 100
    num_examples_to_generate = 9


    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    x_test_sample=x_test[:num_examples_to_generate,:]
    y_test_sample=y_test[:num_examples_to_generate,:]

    backbone, classifier = models.make_discriminator_model()
    backbone, classifier = models.fineTuneonCifar10(backbone, classifier, isLoad=True)
    generator = models.make_generator_model(latent_dim=100)
    augmentation_layer = models.data_augmentation


    # Save model with callback
    checkpoint_path_generator = "output/custom_GAN/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path_generator)
    # Create a callback that saves the model's weights
    cp_callback_ = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_generator,
                                                     save_weights_only=True,
                                                     verbose=1,save_freq="epoch")


    gan = models.GAN(backbone, classifier, generator, augmentation_layer, [x_test_sample,y_test_sample])
    gan.compile(generator_optimizer,g_loss_tracker,models.generator_loss)

    test_acc = gan.evaluate(test_dataset)
    print("Initial test set accuracy: "+ str(test_acc))

    gan.fit(train_dataset,epochs=EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=[cp_callback_,models.CustomCallback()])

    # The model weights (that are considered the best) are loaded into the model.
    gan.load_weights(checkpoint_path_generator)
    test_acc = gan.evaluate(test_dataset)
    print("Current test set accuracy: "+ str(test_acc))
    #gan.fit(train_dataset,epochs=EPOCHS, batch_size=cfg.BATCH_SIZE, callbacks=[cp_callback_])



    #Image display

    def display_image(epoch_no):
      return PIL.Image.open('output/images/image_at_epoch_{:04d}.png'.format(epoch_no))



    # Display a single image using the epoch number
    #display_image(EPOCHS)
    # create Animation
    anim_file = 'output/gan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
      filenames = glob.glob('output/images/image*.png')
      filenames = sorted(filenames)
      for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)


embed.embed_file(anim_file)




