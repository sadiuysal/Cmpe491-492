import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
import objective as losses
from tensorflow.keras.layers.experimental import preprocessing
import data_util
from tensorflow.keras import layers, Model, Input, optimizers
import config as cfg
import os
import matplotlib.pyplot as plt


#tf.config.threading.set_inter_op_parallelism_threads(4)
#tf.config.threading.set_intra_op_parallelism_threads(5)



# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
m = tf.keras.metrics.Accuracy()

#custom accuracy metric
def custom_acc(y_true, y_pred):
    prediction_labels = tf.math.argmax(y_pred, 1)
    prediction_labels = tf.one_hot(prediction_labels, 10)
    m.update_state(y_true, prediction_labels)
    return m.result()




def make_generator_model(latent_dim):
    model = tf.keras.models.Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 8 * 8
    model.add(layers.Dense(n_nodes, input_dim=latent_dim,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))

    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # upsample to 32x32
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #print("Before Out layer dimension : ",model.output_shape )
    # output layer
    model.add(layers.Conv2DTranspose(3, (4, 4),strides=(2, 2), activation='tanh', padding='same',use_bias=False))
    #print("After Out layer dimension : ", model.output_shape)


    return model


def make_discriminator_model():

    in_shape = (32, 32, 3)
    model_backbone,model_classifier = Sequential(), Sequential()

    model_classifier.add(layers.InputLayer(input_shape=in_shape))
    model_classifier.add(layers.UpSampling2D((2, 2)))
    model_classifier.add(layers.UpSampling2D((2, 2)))
    #print("End of upsampling: ", model_classifier.output_shape)
    model_backbone.add(layers.InputLayer(input_shape=in_shape))
    model_backbone.add(layers.UpSampling2D((2, 2)))
    model_backbone.add(layers.UpSampling2D((2, 2)))

    ENetB0 = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=True, weights=None,input_shape=(128,128,3),
        classes=10,classifier_activation="softmax")

    ENetB0_backbone = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, weights=None,input_shape=(128,128,3),pooling='avg')

    for l in ENetB0.layers[:-6]:
        l.trainable = False

    model_classifier.add(ENetB0)
    model_backbone.add(ENetB0_backbone)

    #return backbone_ENetB0,classifier_ENetB0
    return model_backbone,model_classifier


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(backbone_model,org_images,aug_images,adv_images):

    out = backbone_model(tf.concat([org_images,aug_images,adv_images], 0))
    return -losses.contrastive_Loss(out)
    #return cross_entropy(tf.ones_like(fake_output), fake_output)


def fineTuneonCifar10(backbone,classifier,x_train=None ,y_train=None ,x_test=None ,y_test=None, isLoad=True):



    # Save finetuned model with callback
    checkpoint_path = "output/finetuned_classifier/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    if isLoad:
        # Loads the weights
        checkpoint = tf.train.Checkpoint(classifier)
        checkpoint.restore(checkpoint_path).expect_partial()
        classifier.trainable = False
        checkpoint = tf.train.Checkpoint(backbone)
        checkpoint.restore(checkpoint_path).expect_partial()
        backbone.trainable = False
    else:
        x_train = (x_train + 1) / 2  # Normalize the images to [0, 1]
        classifier.trainable = True
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[custom_acc])
        history = classifier.fit(x_train, y_train, epochs=2, callbacks=[cp_callback],batch_size=128 )

    return backbone, classifier





#data augmentation layers
data_augmentation = tf.keras.Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.1),
  preprocessing.RandomZoom(0.1),
  data_util.ColorJitter_and_GrayScale(colorJitter_prob=0.8,grayScale_prob=0.2),
])



class GAN(keras.Model):
    def __init__(self,backbone, classifier,generator, augmentation_layer, test_sample):
        super(GAN, self).__init__()
        self.latent_dim = 100
        self.backbone, self.classifier = backbone, classifier
        self.generator = generator
        self.data_augmentation = augmentation_layer
        self.test_sample = test_sample



    def compile(self, g_optimizer,metric, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.g_loss_tracker = metric
        self.acc_metric = custom_acc
        self.loss_fn = loss_fn

    def call(self, data, training=False):
        # Unpack the data
        images, labels = data[0], data[1]
        _noise = tf.random.normal([images.shape[0], self.latent_dim])
        _image_noises = tf.clip_by_value(self.generator(_noise, training), -1, 1)
        _adv_images = tf.clip_by_value(images + cfg.epsilon * _image_noises, -1, 1)

        _adv_images = (_adv_images + 1) / 2  # change range to [0,1]
        return _adv_images


    @tf.function
    def test_step(self, data):
        labels = data[1]
        _adv_images = self.call(data)
        # Compute predictions
        predictions = self.classifier(_adv_images)

        acc = custom_acc(labels, predictions)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {"acc":acc}



    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    @tf.function
    def train_step(self,data):

        images = data[0]
        labels = data[1]
        noise = tf.random.normal([cfg.BATCH_SIZE, self.latent_dim])

        with tf.GradientTape() as gen_tape: #, tf.GradientTape() as disc_tape:
            image_noises=tf.clip_by_value(self.generator(noise, training=True),-1,1)
            adv_images = tf.clip_by_value(images+cfg.epsilon*image_noises,-1,1)
            aug_images = tf.clip_by_value(self.data_augmentation(images), -1, 1)


            gen_loss = self.loss_fn(self.backbone,images,aug_images,adv_images)
            #disc_loss = models.discriminator_loss(real_output, fake_output)


        adv_images_embedding = self.classifier(adv_images + 1 / 2)
        acc = self.acc_metric(labels,adv_images_embedding)



        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        #discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        self.g_loss_tracker.update_state(gen_loss)


        #d_loss_tracker.update_state(disc_loss)
        # Return a dict mapping metric names to current value
        return {"g_loss": self.g_loss_tracker.result() , "training acc": acc }



class CustomCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        adv_images = self.model.call(self.model.test_sample)
        #print(adv_images.shape)
        fig = plt.figure(figsize=(8, 8))

        if not os.path.exists('output/images'):
            os.makedirs('output/images')

        for i in range(adv_images.shape[0]):
            plt.subplot(3, 3, i + 1)
            plt.imshow((adv_images[i, :, :, :] ))  # cmap='gray'
            plt.axis('off')

        plt.savefig('output/images/image_at_epoch_{:04d}.png'.format(epoch))