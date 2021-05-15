
adversarial_attack = True
epsilon = 0.0625  # 0-255>>8
_lambda = 256
nof_epochs = 500 # 
attacks_itr_count = 7
attacks_step_size = 1.8

##
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
IMG_SIZE = 32
c_dim = 3 # nof channels
gf_dim = 64 # Number of filters to use for generator.
random_seed = 20180526 #
batch_size = 64
class_num = 10
weight_decay = 1e-5
learning_rate = 0.1
gamma = 1e-2 #Regularization parameter for gradient regularization
print_iter = 100 #Number of iterations to print results.
save_iter = 20000 # Number of iterations to save model.
nsteps = 200000 #Number of steps to run training.
model_fileG = './modelG_cifar10.ckpt' #
log_dir = './logs_cifar10'#
