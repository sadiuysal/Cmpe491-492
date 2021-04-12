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
from keras import applications
import matplotlib.pyplot as plt
import tensorflow as tf
import model as model_class
from tensorflow import keras
import attacks
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sn
import config as cfg
import data_util
import os

import objective
#import tensorflow_addons as tfa

tf.config.run_functions_eagerly(True)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#TODO LOOK


def loadModel(epoch):
    return keras.models.load_model("outputs/modelAtEpoch_" + str(epoch))

def build_compile_model():
    """Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

    # mnist = tf.keras.datasets.mnist
    cifar10 = tf.keras.datasets.cifar10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = data_util.cast_to_float32(x_train), data_util.cast_to_float32(x_test)
    inputs = keras.Input(shape=(32, 32, 3))
    base_model = applications.resnet50.ResNet50(weights=None, include_top=False,
                                                input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
    """Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""

    model_layers = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dropout(0.2),
        # tf.keras.layers.Dense(10)
    ])
    """Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""
    model = model_class.CustomModel(inputs, model_layers(inputs))
    # model = model_class.CustomModel(base_model.input,base_model.output)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #opt=tfa.optimizers.LAMB(learning_rate=lr_schedule,epsilon=,weight_decay_rate=1e-6)
    model.compile(optimizer=opt,loss=objective.contrastive_Loss)  # ['accuracy'])
    """ The `Model.fit` method adjusts the model parameters to minimize the loss: """
    return model , (x_train, y_train), (x_test, y_test)


@tf.function
def train_epoch(dataset,model,strategy):
    #tf.print("Epoch Dataset shape : ", tf.shape(dataset))
    #for input in dataset:
    #tf.print("Input shape : ", tf.shape(input))
    l = strategy.experimental_run(model.replica_step, dataset.make_one_shot_iterator())
    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, l, axis=0)
    tf.print("Training loss: ", loss)

def execute():
    strategy=tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    with strategy.scope():
        model, (x_train, y_train), (x_test, y_test) = build_compile_model()
        # x_train, y_train = x_train[-(2*cfg.batch_size):], y_train[-(2*cfg.batch_size):] # IF NEEDED GET SLICE OF THE DATASET
        # create the dataset and distrubute
        train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=1024).batch(
            cfg.GLOBAL_BATCH_SIZE)
        dataset = strategy.experimental_distribute_dataset(train_dataset)

        # create and use callback:
        #saver = model_class.CustomSaver()
        #model.fit(training_data, epochs=cfg.nof_epochs, batch_size=cfg.batch_size, callbacks=[saver],steps_per_epoch=2)
        for epoch in range(cfg.nof_epochs):
            print("Starting epoch : ",epoch + 1 )
            train_epoch(dataset,model,strategy)
            #Checkpoint (do whatever you want, ex: eval or save the model )




def load_predict_save(original_imgs,n_of_epoch):

    original_imgs_concat = np.tile(original_imgs,[20,1,1,1])
    #print(tf.shape(original_imgs))
    #print(tf.shape(original_imgs_concat))
    #original_imgs_concat=np.concatenate((original_imgs,original_imgs,original_imgs,original_imgs,original_imgs),axis=0)
    augmented_imgs = model_class.data_augmentation(original_imgs_concat)
    np.save("outputs/augmented_imgs", augmented_imgs)
    for i in range(n_of_epoch):
        model_loaded = loadModel(epoch=i + 1)
        adversary_imgs, adv_loss = attacks.get_adversaries_2(model_loaded, x=augmented_imgs,
                                                             target=model_class.data_augmentation(original_imgs_concat),
                                                             epsilon=cfg.epsilon,
                                                             itr_count=cfg.attacks_itr_count,
                                                             step_size=cfg.attacks_step_size)

        results = model_loaded(augmented_imgs, training=False)
        results_adversaries = model_loaded(adversary_imgs, training=False)
        results_original_imgs = model_loaded(original_imgs, training=False)

        np.save("outputs/results_adversaries_epoch_" + str(i + 1), results_adversaries)
        np.save("outputs/results_epoch_" + str(i + 1), results)
        np.save("outputs/results_original_epoch_" + str(i + 1), results_original_imgs)
        # results_all = np.concatenate((results, results_adversaries), axis=0)
        print("Epoch " + str(i + 1) + " model results is COMPLETED.")

def loadEmbeddings(epoch):
    results_adversaries = np.load("outputs/results_adversaries_epoch_" + str(epoch) + ".npy")
    results_aug = np.load("outputs/results_epoch_" + str(epoch) + ".npy")
    results_original_imgs = np.load("outputs/results_original_epoch_" + str(epoch) + ".npy")
    return results_original_imgs,results_aug, results_adversaries

"""
def visualize(results_orig,results_aug,results_adv, plot_title=""):
    lbl_size=tf.shape(y_test)
    labels_orig = np.full(shape=lbl_size, fill_value="original")
    labels_aug = np.full(shape=(lbl_size[0]*20,lbl_size[1]), fill_value="augmented")
    labels_adv = np.full(shape=(lbl_size[0]*20,lbl_size[1]), fill_value="adversary")
    #results_all = np.concatenate((results_orig, results_aug, results_adv), axis=0)
    #labels_all = np.concatenate((labels_orig,labels_aug,labels_adv ), axis=0)
    results_all = np.concatenate((results_orig, results_adv), axis=0)
    labels_all = np.concatenate((labels_orig,labels_adv ), axis=0)

    tsne_model = TSNE(random_state=0, perplexity=20, n_iter=5000*4 )
    tsne_data = tsne_model.fit_transform(results_all)
    # creating a new data frame which help us in ploting the result data
    tsne_data = np.vstack((tsne_data.T, labels_all.T)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
    # using dictionary to convert specific columns
    convert_dict = {'Dim_1': float,
                    'Dim_2': float,
                    'label': str
                    }

    tsne_df = tsne_df.astype(convert_dict)
    # Ploting the result of tsne
    sn.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, "Dim_1", "Dim_2").add_legend()
    # displaying the title
    plt.title(label="Epoch " + str(plot_title),
              fontsize=20,
              color="green")
    plt.subplots_adjust(top=0.9)
    #plt.xlim([-400, 400])
    #plt.ylim([-400, 400])
    plt.show()


def get_selected_label_data(labels, max_row_count_per_label):
    res = []
    res_labels = []
    for label_ind in range(len(labels)):
        indicies = np.nonzero(y_train == labels[label_ind])[0]
        for i in range(tf.shape(indicies)[0]):
            if i >= max_row_count_per_label:
                break
            ind = indicies[i]
            if i == 0 and label_ind == 0:
                res = np.expand_dims(x_train[ind], axis=0)
                res_labels = np.expand_dims(y_train[ind], axis=0)
            else:
                res = np.concatenate((res, np.expand_dims(x_train[ind], axis=0)), axis=0)
                res_labels = np.concatenate((res_labels, np.expand_dims(y_train[ind], axis=0)), axis=0)
    return res, res_labels

"""
#x_train, y_train = x_train[-(2*cfg.batch_size):], y_train[-(2*cfg.batch_size):]

#test_dataset = tf.data.Dataset.from_tensor_slices(x_test).batch(GLOBAL_BATCH_SIZE)


execute()


