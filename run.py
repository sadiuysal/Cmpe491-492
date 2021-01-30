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

#TODO correction for PGD (DONE, step_size,epsilon check added, 0-255 , loss values checked)
#TODO loss check via writing matricies ()
#TODO Train>>Save>>check space location difference between adversarial-augmented embeddings  (T-SNE library)


tf.config.run_functions_eagerly(True)

x_train, x_test = model_class.x_train , model_class.x_test


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
  model.fit(x_train, epochs=5, batch_size=model_class.batch_size, callbacks=[saver])
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


train()
model_1=loadModel(epoch=1)



