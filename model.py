import matplotlib.pyplot as plt
from tensorflow.keras import layers
import objective as obj_lib
import data_util
import tensorflow as tf
from tensorflow import keras
import attacks
import config as cfg

# TODO List
# create config class
# parametrize temperature parameter
#
# TODO END
# loss check via writing matricies ()


# ROCL: Inception crop, [horizontal flip, color jitter, and grayscale](DONE) for random augmentations
#data augmentation layers
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.1),
  layers.experimental.preprocessing.RandomZoom(0.1),
  data_util.ColorJitter_and_GrayScale(colorJitter_prob=0.8,grayScale_prob=0.2),
])

"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""

model_layers = tf.keras.Sequential([
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Flatten(),
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
        if cfg.adversarial_attack:
            adversaries, adv_loss = attacks.get_adversaries_2(self, x=x, target=data_augmentation(x), epsilon=cfg.epsilon ,itr_count=cfg.attacks_itr_count,step_size=cfg.attacks_step_size)
            x_3N = tf.concat([x_2N, adversaries], 0)
            #print(adv_loss)
            with tf.GradientTape() as tape:
                y_pred = self(x_3N, training=True)  # Forward pass
                # Compute the loss value
                # (the loss function is configured in `compile()`)
                loss = obj_lib.contrastive_Loss(y_pred,adversarial_selection=True,_lambda=cfg._lambda)
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

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        self.model.save("outputs/modelAtEpoch_"+str(epoch+1))

########################



