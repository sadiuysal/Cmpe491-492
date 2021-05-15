
adversarial_attack = True
_lambda = 64 #256 in RoCL
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
batch_size = 256 # default 64
class_num = 10
epsilon = 0.0625  # 0-255>>8
weight_decay = 1e-5
learning_rate = 0.1
gamma = 1e-2 #Regularization parameter for gradient regularization
print_iter = 100 #Number of iterations to print results.
save_iter = 5 #20000 # Number of iterations to save model.
nsteps = 20000#0 #Number of steps to run training.
model_fileG = './savedModels/modelG_cifar10.ckpt' #
log_dir = './logs_cifar10'#
