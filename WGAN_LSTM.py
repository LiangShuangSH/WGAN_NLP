import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import argparse
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import operator

# Random Seed
# All weights, bias, random values rely on this seed
seed = 50
np.random.seed(seed)
tf.set_random_seed(seed)
# Training text address
FILE_ADDRESS = "./train"

# Get Vocabulary
tokenizer = Tokenizer(num_words=None,
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                              lower=True,
                              split=" ",
                              char_level=False)
tokenizer.fit_on_texts([open(FILE_ADDRESS, "r", encoding='utf-8').read()])
vocabulary = [tup[0] for tup in sorted(tokenizer.word_index.items(), key=operator.itemgetter(1))]
# Insert empty tag in head of vacab
vocabulary.insert(0, " ")
len_dic = len(vocabulary)

TIMESTEPS = 11  # Timesteps of the input sequence, which is actually the length of sentences
NUM_UNITS = 128 # Num of hidden units inside LSTM
LAMDA = 10


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


class GeneratorDistribution(object):
    def sample(self, N):
        samples = np.random.randn(N, TIMESTEPS, len_dic)
        return samples


# Used to sample from training data
class DataDistribution(object):
    def __init__(self):
        file = open(FILE_ADDRESS, "r", encoding='utf-8')
        sequences = tokenizer.texts_to_sequences(file)
        p_sequence = pad_sequences(sequences=sequences, maxlen=TIMESTEPS, dtype='int32', padding='post', value=0)
        for n in p_sequence[0]:
            print(vocabulary[n], end="")
            print(" ", end="")
        self.data_flat = np.array(p_sequence).reshape([-1, TIMESTEPS])

    def sample_sentences(self, batch_size):
        idx = np.random.randint(0, len(self.data_flat), size=batch_size)
        samples_idx = self.data_flat[idx, :]
        samples = to_categorical(samples_idx, len_dic)
        samples = samples.reshape([-1, TIMESTEPS, len_dic])
        # Add noise to samples
        noise = 0.01 * np.random.rand(batch_size, TIMESTEPS, len_dic)
        samples += noise
        samples = softmax(samples, 1.0, axis=2)
        return samples


# The network structure for generator
def generator(noise):
    # Unstack the input into time sequence
    input = tf.unstack(noise, TIMESTEPS, 1)
    # Create LSTM cell
    lstm = rnn.BasicLSTMCell(NUM_UNITS)
    # Define the final outout layer
    W_out = tf.Variable(initial_value=tf.random_uniform(shape=[NUM_UNITS, len_dic]))
    b_out = tf.Variable(initial_value=tf.random_uniform(shape=[len_dic]))
    # LSTM Network
    outputs, states = rnn.static_rnn(cell=lstm, inputs=input, dtype=tf.float32)
    # Feeding the output of each LSTM cell to the same output layer
    for i in range(len(outputs)):
        outputs[i] = tf.matmul(outputs[i], W_out) + b_out
        #outputs[i] = tf.nn.softmax(outputs[i], dim=-1)
    # Reassemble the sequence
    outputs = tf.stack(outputs, axis=1)
    return outputs


def discriminator(input, batch_size):
    input = tf.unstack(input, TIMESTEPS, 1)
    lstm = rnn.BasicLSTMCell(NUM_UNITS)
    outputs, states = rnn.static_rnn(cell=lstm, inputs=input, dtype=tf.float32)
    outputs = tf.stack(outputs, axis=1)
    outputs = tf.layers.dense(inputs=outputs, units=1)
    return outputs


def optimizer(loss, var_list):
    learning_rate = 0.00001
    step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.9).minimize(
        loss,
        global_step=step,
        var_list=var_list
    )
    return optimizer


class GAN(object):
    def __init__(self, params):
        # This defines the generator network - it takes samples from a noise
        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, TIMESTEPS, len_dic))
            self.G = generator(self.z)
        # Reusing the generator at last, to generate files for evaluation
        with tf.variable_scope('G', reuse=True):
            self.z_eval = tf.placeholder(tf.float32, shape=(1000, TIMESTEPS, len_dic))
            self.G_eval = generator(self.z_eval)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, TIMESTEPS, len_dic))
        with tf.variable_scope('D'):
            self.D1 = discriminator(self.x, batch_size=params.batch_size)
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(self.G, batch_size=params.batch_size)
        with tf.variable_scope('D', reuse=True):
            # Define Gradient Penalty
            alpha = tf.random_uniform(shape=[params.batch_size, 1, 1], minval=0., maxval=1.)
            differences = self.G - self.x
            interpolates = self.x + (alpha * differences)
            D_interpolates = discriminator(interpolates, batch_size=params.batch_size)
            gradients = tf.gradients(D_interpolates, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
            self.gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
            #self.gradient_penalty = tf.reduce_mean(tf.maximum(0.0, slopes-1.0))


        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both
        self.loss_d = -tf.reduce_mean(self.D1) + tf.reduce_mean(self.D2) + LAMDA*self.gradient_penalty
        self.loss_g = -tf.reduce_mean(self.D2)

        # Retrieve parameters used in NN: weights, bias
        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)

        # Using argmax to transform the results into one-hot-vector
        self.one_hot_integer = tf.argmax(self.G, axis=2)
        self.one_hot_integer_eval = tf.argmax(self.G_eval, axis=2)


def train(model, data, gen, params):

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps):
            # update discriminator
            for i in range(1):
                # Sample from real data
                x = data.sample_sentences(params.batch_size)
                # Sample from noise
                z = gen.sample(N=params.batch_size)
                # Train discriminator only
                loss_d, _, gp = session.run([model.loss_d, model.opt_d, model.gradient_penalty], {
                    model.x: x,
                    model.z: z
                })

            # update generator
            z = gen.sample(params.batch_size)
            loss_g, _, G, one_hot_integer = session.run([model.loss_g, model.opt_g, model.G, model.one_hot_integer], {
                model.z: z
            })

            # Print loss of discriminator and generator, also with gradient penalty
            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
                print('{}'.format(gp))

                # print 1 batch
                one_poem = one_hot_integer[25, :]
                for i in range(TIMESTEPS):
                    print(vocabulary[one_poem[i]], end='')
                    print(" ", end='')
                print("")

        # Write final batch of results to file
        f = open('./results/LSTM_results_' + str(TIMESTEPS), 'w')
        z_eval = gen.sample(N=1000)
        one_hot_integer_eval = session.run(model.one_hot_integer_eval, {model.z_eval: z_eval})
        for i in range(1000):
            line = one_hot_integer_eval[i, :]
            for j in range(TIMESTEPS):
                f.write(vocabulary[line[j]])
                f.write(" ")
            f.write("\n")
        f.close()

        _ = input("Press [enter] to continue.")


def main(args):
    model = GAN(args)
    train(model, DataDistribution(), GeneratorDistribution(), args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=20000,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='the batch size')
    parser.add_argument('--log-every', type=int, default=10,
                        help='print loss after this many steps')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())