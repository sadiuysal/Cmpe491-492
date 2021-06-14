import tensorflow as tf
import objective as losses
from tensorflow.keras.layers.experimental import preprocessing
import data_util
from tensorflow.keras import layers, Model, Input


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

tf.config.threading.set_inter_op_parallelism_threads(3)
tf.config.threading.set_intra_op_parallelism_threads(3)



def make_generator_model(latent_dim):
    model = tf.keras.models.Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 8 * 8
    model.add(layers.Dense(n_nodes, input_dim=latent_dim,use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))

    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (6, 6), strides=(1, 1), padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # upsample to 32x32
    model.add(layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding='same',use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #print("Before Out layer dimension : ",model.output_shape )
    # output layer
    model.add(layers.Conv2DTranspose(3, (6, 6),strides=(2, 2), activation='tanh', padding='same',use_bias=False))
    #print("After Out layer dimension : ", model.output_shape)



    #model.add(layers.BatchNormalization())
    #assert model.output_shape == (None, 4, 4, 256)  # Note: None is the batch size

    return model


def make_discriminator_model():
    in_shape = (32, 32, 3)
    #model = Sequential()
    """
    # normal
    model.add(Conv2D(64, (6, 6),strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    # downsample
    model.add(Conv2D(128, (6, 6), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))
    print("1-Out layer dimension : ",model.output_shape )
    # classifier
    model.add(Flatten())
    print("2-Out layer dimension : ",model.output_shape )
    model.add(Dense(1))
    print("3-Out layer dimension : ", model.output_shape)
    #return model"""

    """"# compile model ##LOOK HERE
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])"""
    inputs = Input(shape=(32, 32, 3))
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet',input_shape=in_shape,input_tensor=inputs,
        pooling="avg"
    )
    #Flatten output layer of Resnet
    #flattened = Flatten()(base_model.output)
    dense_layer =layers.Dense(1,activation="sigmoid")(base_model.output)
    model = Model(inputs=base_model.input, outputs=dense_layer)

    #print("*1-Out layer dimension : ", model.output_shape)
    #x = base_model.output
    #x = Dropout(0.3)(x)
    #x = Flatten()(x)
    #predictions = Dense(1, activation='softmax')(x)
    #model_pretrained = Model(inputs=base_model.input, outputs=predictions)

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(org_images,aug_images,adv_images):
    return -losses.contrastive_Loss(tf.concat([org_images,aug_images,adv_images], 0))
    #return cross_entropy(tf.ones_like(fake_output), fake_output)

#data augmentation layers
data_augmentation = tf.keras.Sequential([
  preprocessing.RandomFlip("horizontal"),
  preprocessing.RandomRotation(0.1),
  preprocessing.RandomZoom(0.1),
  data_util.ColorJitter_and_GrayScale(colorJitter_prob=0.8,grayScale_prob=0.2),
])