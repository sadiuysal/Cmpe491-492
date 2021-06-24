import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils import np_utils

import objective as losses
from tensorflow.keras.layers.experimental import preprocessing
import data_util
from tensorflow.keras import layers, Model, Input, optimizers
import config as cfg
import os
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy


tf.config.threading.set_inter_op_parallelism_threads(3)
tf.config.threading.set_intra_op_parallelism_threads(3)



# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
m = tf.keras.metrics.Accuracy()
#custom accuracy metric
def custom_acc(y_true, y_pred):

    #print(y_pred)
    prediction_labels = tf.math.argmax(y_pred, 1)
    #print(prediction_labels)
    #prediction_labels = np_utils.to_categorical(prediction_labels[1], 10)
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
    #TODO change kernel size to 4
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



    #model.add(layers.BatchNormalization())
    #assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

    return model


def make_discriminator_model():
    """in_shape = (32, 32, 3)
    inputs = Input(shape=(32, 32, 3))
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet',input_shape=in_shape,input_tensor=inputs,
        pooling="avg"
    )"""
    """backbone_ENetB0 = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=True, input_tensor=inputs,
        input_shape=in_shape, pooling="avg", classes=10, activation="softmax"
    )"""

    # Freeze the pretrained weights
    #backbone_ENetB0.trainable = False
    #print("ENetB0 out shape: ",ENetB0.output_shape)
    in_shape = (32, 32, 3)
    model_backbone,model_classifier = Sequential(), Sequential()

    model_classifier.add(layers.InputLayer(input_shape=in_shape))
    model_classifier.add(layers.UpSampling2D((2, 2)))
    model_classifier.add(layers.UpSampling2D((2, 2)))
    print("End of upsampling: ", model_classifier.output_shape)
    model_backbone.add(layers.InputLayer(input_shape=in_shape))
    model_backbone.add(layers.UpSampling2D((2, 2)))
    model_backbone.add(layers.UpSampling2D((2, 2)))

    """
    ENetB0 = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, weights='imagenet',
        pooling="avg"
    )"""
    ENetB0 = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=True, weights=None,input_shape=(128,128,3),
        classes=10,classifier_activation="softmax")

    ENetB0_backbone = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, weights=None,input_shape=(128,128,3),pooling='avg')

    #ENetB0.summary()
    #ENetB0_backbone.summary()

    #ENetB0.trainable = False
    for l in ENetB0.layers[:-6]:
        l.trainable = False

    model_classifier.add(ENetB0)
    model_backbone.add(ENetB0_backbone)
    #print(model_backbone.layers())
    #model.summary()
    """
    model_classifier = Sequential()
    #model_classifier.add(keras.Input(shape=(256,256,3)))
    model_classifier.add(model_backbone)
    model_classifier.add(layers.Flatten())
    model_classifier.add(layers.BatchNormalization())
    #model_classifier.add(layers.Dense(128, activation='relu'))
    #model_classifier.add(layers.Dropout(0.5))
    #model_classifier.add(layers.BatchNormalization())
    model_classifier.add(layers.Dense(64, activation='relu'))
    model_classifier.add(layers.Dropout(0.5))
    model_classifier.add(layers.BatchNormalization())
    model_classifier.add(layers.Dense(10, activation='softmax'))
    #model_classifier.summary()
    #x = GlobalAveragePooling2D()(backbone_ENetB0.output)
    # TODO Lrelu(0.1) Dense (check dimension ex. 1080) (class:512)
    # use this feature #"""

    #return backbone_ENetB0,classifier_ENetB0
    return model_backbone,model_classifier

    #Flatten output layer of Resnet
    #flattened = Flatten()(base_model.output)
    #dense_layer =layers.Dense(cfg.class_num,activation="softmax")(base_model.output)
    #model = Model(inputs=base_model.input, outputs=dense_layer)

    #return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(backbone_model,org_images,aug_images,adv_images):
    #TODO use Resnet backbone to get features
    out = backbone_model(tf.concat([org_images,aug_images,adv_images], 0))
    return -losses.contrastive_Loss(out)
    #return cross_entropy(tf.ones_like(fake_output), fake_output)


def fineTuneonCifar10(backbone,classifier,x_train,y_train,x_test,y_test,isLoad):
    # Save finetuned model with callback
    checkpoint_path = "finetuned/cp.ckpt"
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
        # Loads the weights
        #classifier.load_weights(checkpoint_path)
        # Finetuning on cifar10
        #loss_fn = SparseCategoricalCrossentropy(from_logits=False)
        #classifier.summary()
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