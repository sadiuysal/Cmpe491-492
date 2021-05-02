import tensorflow as tf


def contrastive_Loss( output , temperature= 0.5 , _lambda = 256):
  N = int( tf.shape(output)[0]/3 )
  # RESNET returns tensor with shapes [N,1,1,2048], so I reshaped it.
  outputs = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[-1]], name=None)
  # print("N : " + str(N))
  # print("outputs.shape : " + str(tf.shape(outputs)))
  z_x = tf.slice(outputs, [0, 0], [N, -1])
  z_prime_x = tf.slice(outputs, [N, 0], [2*N, -1])
  z_adversaries = tf.slice(outputs, [2*N, 0], [-1, -1])
  #print(str(tf.shape(z_x)) + "  &&&&&&&&&&&&  " + str(tf.shape(z_prime_x)) + "  &&&&&&&&&&&&  " + str(tf.shape(z_adversaries) ))
  sim_matrix_1 = sim_matrix_with_temperature(z_x, z_prime_x, temperature)
  sim_matrix_2 = sim_matrix_with_temperature(z_x, z_adversaries, temperature)
  sim_matrix_3 = sim_matrix_with_temperature(z_adversaries,z_x , temperature)
  sim_matrix_4 = sim_matrix_with_temperature(z_adversaries, z_prime_x, temperature)
  loss = find_loss_from_sim_matrix(sim_matrix_1) + find_loss_from_sim_matrix(sim_matrix_2)
  loss_adv = find_loss_from_sim_matrix(sim_matrix_3) + find_loss_from_sim_matrix(sim_matrix_4)
  loss += (1/_lambda)*loss_adv

  return loss

def sim_matrix_with_temperature(x,y,temperature):
  # normalize each row
  normalized_x = tf.math.l2_normalize(
    x, axis=1
  )
  normalized_y = tf.math.l2_normalize(
    y, axis=1
  )
  # multiply row i with row j using transpose
  prod = tf.matmul(normalized_x, normalized_y,
                   adjoint_b=True  # transpose second matrix
                   )
  return prod / temperature


def find_loss_from_sim_matrix(sim_matrix):
    diagonal = tf.linalg.diag_part(sim_matrix)
    diagonal = tf.math.exp(diagonal)
    upper = tf.linalg.band_part(sim_matrix, 0, -1)
    upper = tf.math.exp(upper)
    pos_set = tf.math.reduce_sum(diagonal)
    neg_set = tf.math.reduce_sum(upper) - pos_set
    loss = - tf.math.log(pos_set / (pos_set + neg_set))
    return loss