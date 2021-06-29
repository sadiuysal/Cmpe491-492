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
import time

import PIL
import imageio as imageio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils import np_utils

import config as cfg
from keras import metrics
import data_util
import models
import os
from models import data_augmentation
from IPython import display
import numpy as np



logical_devices = tf.config.list_logical_devices('CPU')
print("Num CPUs:", len(logical_devices))



def test_accuracy_2(images, labels, model):
    dataset = tf.data.Dataset.from_tensor_slices(images).batch(1024)
    for (ind,batch) in enumerate(dataset):
        if ind==0:
            predictions = model(batch)
        else:
            predictions = tf.concat([predictions, model(batch)], 0)


    prediction_labels = tf.math.argmax(predictions, 1)
    prediction_labels = np_utils.to_categorical(prediction_labels, 10)
    print("Accuracy score: ")
    m = tf.keras.metrics.Accuracy()
    m.update_state(labels, prediction_labels)
    print(m.result())



with tf.device(logical_devices[0].name):

    """Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

    #mnist = tf.keras.datasets.mnist
    cifar10 = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = data_util.cast_to_float32 ( x_train ) , data_util.cast_to_float32( x_test )
    (x_train, y_train), (x_test, y_test) = (x_train[:128*100,:], y_train[:128*100,:]), (x_test[:,:], y_test[:,:])

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

    #print("Initial train set acc:\n")
    #test_accuracy_2(x_train, y_train, classifier)
    #print("Initial test set acc:\n")
    #test_accuracy_2(x_test, y_test, classifier)

    #


    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(cfg.BUFFER_SIZE).batch(
        cfg.BATCH_SIZE,drop_remainder=True,num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(cfg.BUFFER_SIZE).batch(
        cfg.BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)


    generator_optimizer = tf.keras.optimizers.Adam(1e-4) #optimizer
    g_loss_tracker = metrics.Mean(name="loss")          #loss tracker
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




    """
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(images):
        noise = tf.random.normal([images.shape[0], noise_dim])

        with tf.GradientTape() as gen_tape: #, tf.GradientTape() as disc_tape:
            image_noises=tf.clip_by_value(generator(noise, training=True),-1,1)
            adv_images = tf.clip_by_value(images+cfg.epsilon*image_noises,-1,1)
            aug_images = tf.clip_by_value(data_augmentation(images), -1, 1)

            #real_output = discriminator(images, training=False)
            #fake_output = discriminator(adv_images, training=False)

            gen_loss = models.generator_loss(backbone,images,aug_images,adv_images)
            #disc_loss = models.discriminator_loss(real_output, fake_output)


        #print("Generator loss: ", gen_loss)
        #print("Discriminator loss: ", disc_loss)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        #gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        #discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        g_loss_tracker.update_state(gen_loss)
        #d_loss_tracker.update_state(disc_loss)
        # Return a dict mapping metric names to current value
        return {"g_loss": [g_loss_tracker.result() for m in [g_loss_tracker]] }


    def train(dataset, epochs):
      for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            models.train_step(image_batch)


        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)



        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
          print("Epoch "+str(epoch+1)+" Generator loss: " + str(g_loss_tracker.result()))
          test_accuracy(x_test,y_test,classifier,generator)


        if epoch == 0:
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            test_accuracy(x_test, y_test, classifier,generator)

      # Generate after the final epoch
      display.clear_output(wait=True)
      generate_and_save_images(generator,
                               epochs,
                               seed)



    def generate_and_save_images(model, epoch, test_input):
        image_noises = tf.clip_by_value(model(test_input, training=False), -1, 1)
        adv_images = tf.clip_by_value(x_test_sample + cfg.epsilon * image_noises, -1, 1)
        fig = plt.figure(figsize=(8, 8))

        for i in range(adv_images.shape[0]):
          plt.subplot(3, 3, i+1)
          plt.imshow(((adv_images[i, :, :, :] + 1 ) / 2 )) #cmap='gray'
          plt.axis('off')

        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
        #plt.show()


    def test_accuracy(images,labels,classifier_model,generator_model):
        _noise = tf.random.normal([images.shape[0], noise_dim])
        _image_noises = tf.clip_by_value(generator_model(_noise, training=False), -1, 1)
        _adv_images = tf.clip_by_value(images + cfg.epsilon * _image_noises, -1, 1)

        _adv_images = (_adv_images + 1) / 2 #change range to [0,1]
        predictions = classifier_model(_adv_images)

        print("Current Accuracy: ")
        m = tf.keras.metrics.Accuracy()
        m.update_state(labels,predictions)
        print(m.result())"""



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

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)




