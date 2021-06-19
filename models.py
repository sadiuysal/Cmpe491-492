import tensorflow as tf
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam

import objective as losses
from tensorflow.keras.layers.experimental import preprocessing
import data_util
from tensorflow.keras import layers, Model, Input
import config as cfg
import os
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

tf.config.threading.set_inter_op_parallelism_threads(6)
tf.config.threading.set_intra_op_parallelism_threads(6)



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
    in_shape = (32, 32, 3)
    inputs = Input(shape=(32, 32, 3))
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet',input_shape=in_shape,input_tensor=inputs,
        pooling="avg"
    )
    backbone_ENetB0 = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, weights='imagenet', input_tensor=inputs,
        input_shape=in_shape
    )
    # Freeze the pretrained weights
    backbone_ENetB0.trainable = False
    print("Backbone ENetB0 out shape: ",backbone_ENetB0.output_shape)


    x = GlobalAveragePooling2D()(backbone_ENetB0.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(cfg.class_num,activation="softmax")(x)
    classifier_ENetB0 = Model(inputs=backbone_ENetB0.input, outputs=outputs)
    print("Classifier ENetB0 out shape: ",classifier_ENetB0.output_shape)

    return backbone_ENetB0,classifier_ENetB0

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


def fineTuneonCifar10(classifier,x_train,y_train,isLoad):
    # Save finetuned model with callback
    checkpoint_path = "finetuned/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    if isLoad:
        # Loads the weights
        classifier.load_weights(checkpoint_path)
        classifier.trainable = False
    else:
        # Loads the weights
        #classifier.load_weights(checkpoint_path)
        # Finetuning on cifar10
        loss_fn = SparseCategoricalCrossentropy(from_logits=False)
        #classifier.summary()
        classifier.trainable = True
        for l in classifier.layers[:-3]:
            l.trainable = False
        classifier.compile(optimizer=Adam(learning_rate=1e-4, decay=1e-4 / 150), loss=loss_fn)
        classifier.fit(x_train, y_train, callbacks=[cp_callback],epochs=150)

    return classifier



#data augmentation layers
data_augmentation = tf.keras.Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.1),
  preprocessing.RandomZoom(0.1),
  data_util.ColorJitter_and_GrayScale(colorJitter_prob=0.8,grayScale_prob=0.2),
])