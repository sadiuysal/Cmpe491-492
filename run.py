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
import config as cfg
from keras import metrics
import data_util
import models
import os
from models import data_augmentation
from IPython import display



logical_devices = tf.config.list_logical_devices('CPU')
print("Num CPUs:", len(logical_devices))


def test_accuracy_2(images, labels, model):
    predictions = model(images)

    prediction_labels = tf.math.argmax(predictions, 1)
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
    #(x_train, y_train), (x_test, y_test) = (x_train[:300,:], y_train[:300,:]), (x_test[:100,:], y_test[:100,:])

    #print(x_train.shape)
    train_images = (x_train - 127.5) / 127.5  # Normalize the images to [-1, 1]
    #TODO change later on
    test_images = x_test
    test_labels = y_test

    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(cfg.BUFFER_SIZE).batch(cfg.BATCH_SIZE)

    noise = tf.random.normal([1, 100])
    generator = models.make_generator_model(latent_dim=100)

    backbone,classifier = models.make_discriminator_model()
    test_accuracy_2(test_images,test_labels,classifier)
    classifier = models.fineTuneonCifar10(classifier,x_train,y_train,isLoad=False)
    test_accuracy_2(test_images, test_labels, classifier)




    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    #discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    ## Save Checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     generator=generator)

    #loss tracker
    g_loss_tracker = metrics.Mean(name="loss")
    #d_loss_tracker = metrics.Mean(name="loss")

    ## Training Loop

    EPOCHS = cfg.nof_epochs
    noise_dim = 100
    num_examples_to_generate = 9


    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    x_test_sample=test_images[:num_examples_to_generate,:]
    y_test_sample=test_labels[:num_examples_to_generate,:]

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
      epoch_avg_time = 0
      for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)


        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)


        epoch_avg_time += time.time() - start

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
          print("Epoch "+str(epoch+1)+" Generator loss: " + str(g_loss_tracker.result()))
          epoch_avg_time /= 5
          print('Avg time for 5 epoch is {} sec'.format(epoch_avg_time))
          epoch_avg_time = 0
          test_accuracy(test_images,test_labels,classifier)


        if epoch == 0:
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            test_accuracy(test_images, test_labels, classifier)

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
          plt.imshow((adv_images[i, :, :, :] * 127.5 + 127.5 )/255) #cmap='gray'
          plt.axis('off')

        plt.savefig('images/image_at_epoch_{:04d}.png'.format(epoch))
        #plt.show()


    def test_accuracy(images,labels,model):
        predictions = model(images)

        prediction_labels = tf.math.argmax(predictions, 1)
        #print(predictions[:10,:])
        #print(prediction_labels[:10])
        #print("True labels: ", labels[:10])
        print("Accuracy score on given test data: ")
        m = tf.keras.metrics.Accuracy()
        m.update_state(labels,prediction_labels)
        print(m.result())
        #accuracy_score(test_labels, prediction_labels)



    #Image display

    def display_image(epoch_no):
      return PIL.Image.open('images/image_at_epoch_{:04d}.png'.format(epoch_no))




    if not os.path.exists('images'):
        os.makedirs('images')

    #call train
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""
    train(train_dataset, EPOCHS)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # Display a single image using the epoch number
    display_image(EPOCHS)
    # create Animation
    anim_file = 'gan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
      filenames = glob.glob('images/image*.png')
      filenames = sorted(filenames)
      for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)"""




