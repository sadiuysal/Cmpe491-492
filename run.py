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
import matplotlib.pyplot as plt
import tensorflow as tf
#import model as model_class
from tensorflow import keras
import objective as loss
import config as cfg
import data_util
import models
#import tensorflow_addons as tfa


"""Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

#mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = data_util.cast_to_float32 ( x_train ) , data_util.cast_to_float32( x_test )
(x_train, y_train), (x_test, y_test) = (x_train[:1000,:]/256, y_train[:1000,:]/256), (x_test[:100,:]/256, y_test[:100,:]/256)
"""with tf.Session() as sess:
    print(x_train[0,:].eval())"""

#tf.config.run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()



# data augmentation layers
#data_augmentation = model_class.data_augmentation

inputs = keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))
"""
base_model = applications.resnet50.ResNet50(weights=None, include_top=False,
                                            input_shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3))"""
discriminator = tf.keras.applications.ResNet50(
    include_top=False, weights='imagenet',
    input_shape=(32,32,3), input_tensor=inputs,pooling=None
)
layer_to_logits=tf.keras.layers.Dense(10,activation='relu')

generator = models.get_generator(c_dim=cfg.c_dim, f_dim=cfg.gf_dim)
#print(generator(x_train[:5,:]).shape)

"""Build the `tf.keras.Sequential` model by stacking layers. Choose an optimizer and loss function for training:"""
#model = model_class.CustomModel(inputs, model_class.model_layers(inputs))

#model = model_class.CustomModel(inputs,discriminator(generator))



def train(generator,discriminator,data,test_data):
    tf.set_random_seed(cfg.random_seed)

    batch_size = cfg.batch_size
    epsilon = cfg.epsilon    # size of perturbation
    class_num = cfg.class_num    # number of output classes
    weight_decay = cfg.weight_decay
    learning_rate = cfg.learning_rate
    gamma = cfg.gamma    # gradient regularization parameter
    train_size = data[0].shape[0]    # training set size

    print_iter = cfg.print_iter
    save_iter = cfg.save_iter


    x_clean = data[0]
    label = data[1]

    # Data augmentation for CIFAR10
    x_real = tf.map_fn(lambda x: tf.image.resize_image_with_crop_or_pad(x, 32+4, 32+4), x_clean)
    x_real = tf.map_fn(lambda x: tf.random_crop(x, [32,32,3]), x_real)
    x_real = tf.map_fn(lambda x: tf.image.random_flip_left_right(x), x_real)

    # Normalize to range [-1,1]
    x_real = 2.*x_real - 1.
    x_noise = generator(x_real)
    x_perturbed = x_real + epsilon * x_noise
    x_all=tf.concat([x_clean,x_real, x_perturbed], 0)



    """d_out_real = layer_to_logits(discriminator(x_real))
    d_out_noise = layer_to_logits(discriminator(x_perturbed))
    d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out_real, labels=tf.one_hot(label, class_num)))
    d_loss_noise = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=d_out_noise, labels=tf.one_hot(label, class_num)))"""

    output = discriminator(x_all)
    """print("x_real: ",x_real.shape)
    print("x_all: ",x_all.shape)
    print("x_all_out: ",output.shape)"""
    # Losses for the discriminator network and generator network
    # d_loss = d_loss_real + d_loss_noise
    #g_loss = -d_loss_noise
    g_loss = loss.contrastive_Loss(output)

    # Weight decay: assume weights are named 'kernel' or 'weights'
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    #g_vars = tf.trainable_variables(scope='generator')

    g_decay = weight_decay * 0.5 * sum(
        tf.reduce_sum(tf.square(v)) for v in g_vars if (v.name.find('kernel') > 0 or v.name.find('weights') > 0)
    )





    g_optimizer = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5, beta2=0.999)


    #g_grads=tape.gradient(g_loss, g_vars)
    g_grads = tf.gradients(g_loss + g_decay, g_vars, stop_gradients=g_vars)
    g_train_op = g_optimizer.apply_gradients(zip(g_grads,g_vars))

    acc_g, acc_update_g, acc_init_g = build_test_G(discriminator, generator, test_data)
    #acc_g, acc_init_g = build_test_G(discriminator, generator, test_data)
    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    saverG = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

    # Supervisor
    sv = tf.train.Supervisor(
        logdir=cfg.log_dir, global_step=global_step,
        summary_op=None,
    )

    model_filenameG = cfg.model_fileG

    with sv.managed_session() as sess:

        num_steps_per_epoch = int(train_size // (batch_size)) + 1

        for batch_idx in range(cfg.nsteps):
            if sv.should_stop():
                break

            g_loss_out = sess.run(
                [g_loss])

            if batch_idx % print_iter == 0:
                #test_acc = run_test(acc, acc_update, acc_init, sess, config)
                #test_acc_fgs = run_test(acc_fgs, acc_update_fgs, acc_init_fgs, sess, config)
                #test_acc_pgd = run_test(acc_pgd, acc_update_pgd, acc_init_pgd, sess, config)
                test_acc_g = run_test(acc_g, acc_update_g, acc_init_g, sess)

                print('i=%d, Loss_g: %4.4f, acc_g: %.4f' % (batch_idx, g_loss_out[0], test_acc_g),
                      flush=True)

            if batch_idx % save_iter == 0:
                #saverD.save(sess, model_filenameD)
                saverG.save(sess, model_filenameG)

            sess.run(global_step_op)

        #saverD.save(sess, model_filenameD)
        saverG.save(sess, model_filenameG)


# construct graph for testing on examples perturbed by G
def build_test_G(discriminator, generator, test_data):
    x = test_data[0]
    x = 2.0*x - 1.0
    y = test_data[1]
    epsilon = cfg.epsilon

    x_noise = tf.stop_gradient(generator(x))
    x_perturbed = x + epsilon*x_noise

    d_out = layer_to_logits(discriminator(x_perturbed))
    #d_out = tf.squeeze(d_out, [2])
    predictions = tf.argmax(d_out, 1)
    y = tf.one_hot(y, 10)

    tf_metric, tf_metric_update = tf.metrics.accuracy(y, predictions, name='g_metric')
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="g_metric")
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)


    return (tf_metric, tf_metric_update, running_vars_initializer)

# run actual test to collect performance statistics
def run_test(tf_metric, tf_metric_update, running_vars_initializer, sess):
    #print("run_test1")
    sess.run(running_vars_initializer)

    batch_size = cfg.batch_size
    test_size = x_test.shape[0]
    num_steps = int(test_size//batch_size)+1
    #print("nof steps: ", num_steps)
    for i in range(num_steps):
        #print(i)
        sess.run(tf_metric_update)
    #print("run_test3")
    accuracy = sess.run(tf_metric)
    #print("run_test4")
    return accuracy



#x_train, y_train = x_train[-(2*cfg.batch_size):], y_train[-(2*cfg.batch_size):]
#train(training_data=x_train)

train(generator,discriminator,(x_train,y_train),(x_test,y_test))


