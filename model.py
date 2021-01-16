import matplotlib.pyplot as plt
from tensorflow.keras import layers
import objective as obj_lib
import data_util
import tensorflow as tf
from tensorflow import keras
import attacks

# Do the adversarial part
# Migrate to resnet 


"""Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

#mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
#x_train, y_train, x_test, y_test = x_train[-1000:] , y_train[-1000:], x_test[-1000:], y_test[-1000:]
x_train, x_test = data_util.cast_to_float32 (x_train * 1./255) , data_util.cast_to_float32( x_test * 1.255 )



#TODO 
batch_size=64
IMG_SIZE=32
adversarial_attack=True
epsilon=0.01
_lambda=1

#resize_and_scale with layers
resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255)
])
#data augmentation layers
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])
#resize-scale+augmentation
preprocess = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.3),
])

"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""

model_layers = tf.keras.Sequential([
  #preprocess,
  #resize_and_rescale,
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.Flatten()
  #tf.keras.layers.Dense(64, activation='relu'),
  #tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(10)
])

loss_tracker = keras.metrics.Mean(name="loss")
class CustomModel(keras.Model):
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x = data
        x_2N = tf.concat([data_augmentation(x), data_augmentation(x)], 0)
        if adversarial_attack:
            adversaries, adv_loss = attacks.get_adversaries(self, x=x, target=data_augmentation(x), epsilon=epsilon ,itr_count=3)
            x_3N = tf.concat([x_2N, adversaries], 0)
            #print(adv_loss)
            with tf.GradientTape() as tape:
                y_pred = self(x_3N, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = obj_lib.contrastive_Loss(y_pred,adversarial_selection=True)
        else:
            with tf.GradientTape() as tape:
                y_pred = self(x_2N, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = obj_lib.contrastive_Loss(y_pred)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        #self.compiled_metrics.update_state(y, y_pred)
        loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {"loss": loss_tracker.result() for m in self.metrics}
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [loss_tracker]