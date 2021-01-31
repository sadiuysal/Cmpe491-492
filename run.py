# -*- coding: utf-8 -*-

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
from keras import applications
import matplotlib.pyplot as plt
import tensorflow as tf
import model as model_class
from tensorflow import keras
import attacks
import numpy as np
#from yellowbrick.text import TSNEVisualizer
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sn


#TODO correction for PGD (DONE, step_size,epsilon check added, 0-255 , loss values checked)
#TODO loss check via writing matricies ()
#TODO Train>>Save>>check space location difference between adversarial-augmented embeddings  (T-SNE library)


tf.config.run_functions_eagerly(True)

x_train, x_test = model_class.x_train , model_class.x_test
y_train, y_test = model_class.y_train, model_class.y_test


#resize_and_scale with layers
resize_and_rescale = model_class.resize_and_rescale
#data augmentation layers
data_augmentation = model_class.data_augmentation
#resize-scale+augmentation
preprocess_layer = model_class.preprocess


inputs=keras.Input(shape=(32, 32, 3))
base_model = applications.resnet50.ResNet50(weights= None, include_top=False, input_shape= (model_class.IMG_SIZE,model_class.IMG_SIZE,3))
"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""
model = model_class.CustomModel(inputs,model_class.model_layers(inputs))
#model = model_class.CustomModel(base_model.input,base_model.output)

"""
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
    plt.axis("off")"""


def loadModel(epoch):
  return keras.models.load_model("modelAtEpoch_" + str(epoch))

def train():
  model.compile(optimizer='adam') # ['accuracy'])
  """ The `Model.fit` method adjusts the model parameters to minimize the loss: """
  # create and use callback:
  saver = model_class.CustomSaver()
  model.fit(x_train, epochs=model_class.nof_epochs, batch_size=model_class.batch_size, callbacks=[saver])
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
def load_predict_save(augmented_imgs):
  np.save("augmented_imgs", augmented_imgs)
  for i in range(5):
    model_loaded = loadModel(epoch=i + 1)
    adversary_imgs, adv_loss = attacks.get_adversaries_2(model_loaded, x=augmented_imgs,
                                                         target=data_augmentation(x_train), epsilon=model_class.epsilon,
                                                         itr_count=5, step_size=0.01)
    results = model_loaded(augmented_imgs, training=False)
    results_adversaries = model_loaded(adversary_imgs, training=False)
    np.save("results_adversaries_epoch_" + str(i + 1), results_adversaries)
    np.save("results_epoch_" + str(i + 1), results)
    print("Epoch " + str(i + 1) + " model results is COMPLETED")

def loadEmbeddings(epoch):
  results_adversaries=np.load("results_adversaries_epoch_" + str(epoch)+".npy")
  results= np.load("results_epoch_" + str(epoch)+".npy")
  return results,results_adversaries

def visualize(results,results_adversaries,plot_title):
  results_all = np.concatenate((results, results_adversaries), axis=0)
  labels_all = np.concatenate((y_train, y_train), axis=0)
  print("Results shape: " + str(np.shape(results)))
  print("Results_all shape: " + str(np.shape(results_all)))
  print("labels_all shape: " + str(np.shape(labels_all)))
  tsne_model = TSNE(n_components=2, random_state=0)
  tsne_data = tsne_model.fit_transform(results_all)
  # creating a new data frame which help us in ploting the result data
  tsne_data = np.vstack((tsne_data.T, labels_all.T)).T
  tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
  # Ploting the result of tsne
  sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, "Dim_1", "Dim_2").add_legend()
  # displaying the title
  plt.title(label="Epoch "+str(plot_title),
            fontsize=40,
            color="green")
  plt.show()

#train()
x_train,y_train = x_train[-50:] , y_train[-50:]
augmented_imgs=data_augmentation(x_train)
load_predict_save(augmented_imgs)
results,results_adversaries=loadEmbeddings(epoch=1)
visualize(results,results_adversaries,1)
results,results_adversaries=loadEmbeddings(epoch=2)
visualize(results,results_adversaries,2)
results,results_adversaries=loadEmbeddings(epoch=3)
visualize(results,results_adversaries,3)







