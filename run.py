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
import models
#import tensorflow_addons as tfa


"""Load and prepare the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Convert the samples from integers to floating-point numbers:"""

#mnist = tf.keras.datasets.mnist
cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = data_util.cast_to_float32 ( x_train ) , data_util.cast_to_float32( x_test )
(x_train, y_train), (x_test, y_test) = (x_train[:5,:], y_train[:5,:]), (x_test, y_test)

#tf.config.run_functions_eagerly(True)
#tf.compat.v1.disable_eager_execution()



# data augmentation layers
data_augmentation = model_class.data_augmentation

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


def loadModel(epoch):
    return keras.models.load_model("outputs/modelAtEpoch_" + str(epoch))


def train(generator,discriminator,data,test_data):
    tf.compat.v1.set_random_seed(cfg.random_seed)

    batch_size = cfg.batch_size
    epsilon = cfg.epsilon    # size of perturbation
    class_num = cfg.class_num    # number of output classes
    weight_decay = cfg.weight_decay
    learning_rate = cfg.learning_rate
    gamma = cfg.gamma    # gradient regularization parameter
    train_size = data[0].shape[0]    # training set size

    print_iter = cfg.print_iter
    save_iter = cfg.save_iter


    x_real = data[0]
    label = data[1]

    # Data augmentation for CIFAR10
    x_real = tf.map_fn(lambda x: tf.compat.v1.image.resize_image_with_crop_or_pad(x, 32+4, 32+4), x_real)
    x_real = tf.map_fn(lambda x: tf.compat.v1.random_crop(x, [32,32,3]), x_real)
    x_real = tf.map_fn(lambda x: tf.compat.v1.image.random_flip_left_right(x), x_real)

    # Normalize to range [-1,1]
    x_real = 2.*x_real - 1.





    g_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.002, beta1=0.5, beta2=0.999)


    with tf.GradientTape() as tape:

        x_noise = generator(x_real)
        x_perturbed = x_real + epsilon * x_noise
        d_out_real = layer_to_logits(discriminator(x_real))
        d_out_noise = layer_to_logits(discriminator(x_perturbed))
        d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=d_out_real, labels=tf.one_hot(label, class_num)))
        d_loss_noise = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=d_out_noise, labels=tf.one_hot(label, class_num)))

        # Losses for the discriminator network and generator network
        # d_loss = d_loss_real + d_loss_noise
        g_loss = -d_loss_noise


        # Weight decay: assume weights are named 'kernel' or 'weights'
        #g_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        g_vars = tf.compat.v1.trainable_variables(scope='generator')

        g_decay = weight_decay * 0.5 * sum(
            tf.reduce_sum(tf.square(v)) for v in g_vars if (v.name.find('kernel') > 0 or v.name.find('weights') > 0)
        )
        ###

    g_grads=tape.gradient(g_loss, g_vars)
    #g_grads = tf.gradients(g_loss + g_decay, g_vars, stop_gradients=g_vars)
    g_train_op = g_optimizer.apply_gradients(zip(g_grads,g_vars))

    #acc_g, acc_update_g, acc_init_g = build_test_G(discriminator, generator, test_data)
    acc_g, acc_init_g = build_test_G(discriminator, generator, test_data)
    # Global step
    global_step = tf.Variable(0, trainable=False)
    global_step_op = global_step.assign_add(1)

    saverG = tf.train.Saver(tf.compat.v1.trainable_variables(scope='generator'))

    # Supervisor
    sv = tf.train.Supervisor(
        logdir=cfg.log_dir, global_step=global_step,
        summary_op=None,
    )

    model_filenameG = cfg.model_fileG

    with sv.managed_session() as sess:

        num_steps_per_epoch = int(train_size / (batch_size)) + 1

        for batch_idx in range(cfg.nsteps):
            if sv.should_stop():
                break

            g_loss_out = sess.run(
                [g_loss])

            if batch_idx % print_iter == 0:
                #test_acc = run_test(acc, acc_update, acc_init, sess, config)
                #test_acc_fgs = run_test(acc_fgs, acc_update_fgs, acc_init_fgs, sess, config)
                #test_acc_pgd = run_test(acc_pgd, acc_update_pgd, acc_init_pgd, sess, config)
                test_acc_g = run_test(acc_g, acc_init_g, sess)

                print('i=%d, Loss_g: %4.4f, acc_g: %.4f' % (batch_idx, g_loss_out, test_acc_g),
                      flush=True)

            if batch_idx % save_iter == 0:
                #saverD.save(sess, model_filenameD)
                saverG.save(sess, model_filenameG)

            sess.run(global_step_op)

        #saverD.save(sess, model_filenameD)
        saverG.save(sess, model_filenameG)



    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    #opt=tfa.optimizers.LAMB(learning_rate=lr_schedule,epsilon=,weight_decay_rate=1e-6)
    model.compile(optimizer=opt)  # ['accuracy'])
    # The `Model.fit` method adjusts the model parameters to minimize the loss:
    # create and use callback:
    saver = model_class.CustomSaver()
    model.fit(training_data, epochs=cfg.nof_epochs, batch_size=cfg.batch_size, callbacks=[saver])
    #The `Model.evaluate` method checks the models performance, usually on a "[Validation-set](https://developers.google.com/machine-learning/glossary#validation-set)" or "[Test-set](https://developers.google.com/machine-learning/glossary#test-set)"."""

    # model.evaluate(x_test,  y_test, verbose=2)

# construct graph for testing on examples perturbed by G
def build_test_G(discriminator, generator, test_data):
    x = test_data[0]
    x = 2.0*x - 1.0
    y = test_data[1]
    epsilon = cfg.epsilon

    x_noise = tf.stop_gradient(generator(x))
    x_perturbed = x + epsilon*x_noise

    d_out = layer_to_logits(discriminator(x_perturbed))
    d_out = tf.squeeze(d_out, [2])
    #predictions = tf.argmax(d_out, 1)
    y = tf.one_hot(y, 10)


    loss_tracker = tf.keras.metrics.Accuracy()
    loss_tracker.update_state(y,d_out)
    loss_tracker_result = loss_tracker.result()
    # Return a dict mapping metric names to current value
    #return {"loss": loss_tracker.result()}

    #tf_metric, tf_metric_update = tf.compat.v1.metrics.accuracy(y, predictions, name='g_metric')

    #running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="g_metric")
    running_vars = tf.compat.v1.trainable_variables(scope='generator')
    running_vars_initializer = tf.compat.v1.variables_initializer(var_list=running_vars)

    return (loss_tracker_result, running_vars_initializer)

# run actual test to collect performance statistics
def run_test(tf_metric, running_vars_initializer, sess):
    sess.run(running_vars_initializer)

    batch_size = cfg.batch_size
    test_size = x_test.shape[0]
    num_steps = int(test_size/batch_size)
    """for i in range(num_steps):
        sess.run(tf_metric_update)"""

    accuracy = sess.run(tf_metric)

    return accuracy



def load_predict_save(original_imgs,n_of_epoch):

    original_imgs_concat = np.tile(original_imgs,[20,1,1,1])
    #print(tf.shape(original_imgs))
    #print(tf.shape(original_imgs_concat))
    #original_imgs_concat=np.concatenate((original_imgs,original_imgs,original_imgs,original_imgs,original_imgs),axis=0)
    augmented_imgs = data_augmentation(original_imgs_concat)
    np.save("outputs/augmented_imgs", augmented_imgs)
    euclidean_distances=[[] for i in range(5)]
    for i in range(n_of_epoch):
        model_loaded = loadModel(epoch=i + 1)
        adversary_imgs, adv_loss = attacks.get_adversaries_2(model_loaded, x=augmented_imgs,
                                                             target=data_augmentation(original_imgs_concat),
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
        #dst = [distance.euclidean(x, y) for x, y in zip(results, results_adversaries)]
        #l1_norm = tf.norm(tf.math.subtract(results, results_adversaries), ord=1, axis=1)
        #l2_norm = tf.norm(tf.math.subtract(results, results_adversaries), ord=2, axis=1)
        #l_inf_norm = tf.norm(tf.math.subtract(results, results_adversaries), ord=np.inf, axis=1)
        #euclidean_distances[i] = l_inf_norm

        """
        print("L1 " + " : " + " L2 " + " : " + " L_inf" + " euclidean")
        for x, y, z  in zip(l1_norm[-50:], l2_norm[-50:], l_inf_norm[-50:] ):
            print(str(x) + "   :   " + str(y) + "  :  " + str(z))"""
"""
    print("Euclidean distances: ")
    print("change1to2 " + " : " + " change2to3 " + " : " + " change3to4" + " : " +"change4to5")
    for i in range(len(euclidean_distances[0])):
        debug_str=""
        for j in range(4):
            debug_str += str(euclidean_distances[j+1][i]-euclidean_distances[j][i]) + "  :  "
        print(debug_str)
        
"""
def loadEmbeddings(epoch):
    results_adversaries = np.load("outputs/results_adversaries_epoch_" + str(epoch) + ".npy")
    results_aug = np.load("outputs/results_epoch_" + str(epoch) + ".npy")
    results_original_imgs = np.load("outputs/results_original_epoch_" + str(epoch) + ".npy")
    return results_original_imgs,results_aug, results_adversaries


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


#x_train, y_train = x_train[-(2*cfg.batch_size):], y_train[-(2*cfg.batch_size):]
#train(training_data=x_train)

train(generator,discriminator,(x_train,y_train),(x_test,y_test))


