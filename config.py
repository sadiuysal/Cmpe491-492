_lambda = 64 #256 in RoCL
nof_epochs = 25 # Generator training nof epochs
nof_epochs_classifier = 20 # Classifier training nof epochs
##
BATCH_SIZE = 128 # default 64
class_num = 10
epsilon = 0.0625  # 0-255>>16
## buffer size for shuffling the data and creating batches
BUFFER_SIZE = 60000
