import argparse
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import operator
import gensim

# Random Seed
# All weights, bias, random values rely on this seed
seed = 25
np.random.seed(seed)
tf.set_random_seed(seed)

# Training text address
FILE_ADDRESS = "./train"
# Using Keras Tokenizer
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

# Pre-trained word2vec model loaded
w2v_model = gensim.models.KeyedVectors.load_word2vec_format("./word2vec-GoogleNews-vectors/GoogleNews-vectors-negative300.bin", binary=True)

DIM = 128
SEQ_LEN = 10 # Num of words in 1 sentence
W2V_DIM = 300
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


def ResBlock(inputs):
    relu1 = tf.nn.relu(inputs)
    conv1 = tf.layers.conv1d(inputs=relu1, filters=DIM, kernel_size=3, strides=1, padding="SAME", activation=tf.nn.relu)
    output = tf.layers.conv1d(inputs=conv1, filters=DIM, kernel_size=3, strides=1, padding="SAME")
    return inputs + (0.3 * output)


# Used to sample from training data
class DataDistribution(object):
    def __init__(self):
        file = open(FILE_ADDRESS, "r", encoding='utf-8')
        sequences = tokenizer.texts_to_sequences(file)
        # Pad the training sentences with empty tag
        p_sequence = pad_sequences(sequences=sequences, maxlen=SEQ_LEN, dtype='int32', padding='post', value=0)
        self.data_flat = np.array(p_sequence).reshape([-1, SEQ_LEN])
        self.unk = np.random.rand(W2V_DIM)

    def sample_sentences(self, batch_size):
        # Sample random batch_size number of sentences out
        # idx used to indicate which sentences are chosen
        idx = np.random.randint(0, len(self.data_flat), size=batch_size)
        # For each word in the sentence, they are just integers yet
        samples_idx = self.data_flat[idx, :]
        # One-hot them to vector
        samples = np.zeros((batch_size, SEQ_LEN, W2V_DIM))
        # Word to Vec
        for i in range(batch_size):
            for j in range(SEQ_LEN):
                word_str = vocabulary[samples_idx[i, j]]
                # If word is not contained in w2v_model, use random vector instead
                if word_str in w2v_model:
                    samples[i, j, :] = w2v_model[word_str]
                else:
                    samples[i, j, :] = self.unk
        return samples


class GeneratorDistribution(object):
    def __init__(self):
        self.shape = [SEQ_LEN, W2V_DIM]

    def sample(self, N):
        samples = np.random.randn(N, DIM)
        return samples


# The network structure for generator
def generator(noise, len_dic):
    h0 = tf.layers.dense(inputs=noise, units=SEQ_LEN*DIM)
    h0_cube = tf.reshape(h0, [-1, SEQ_LEN, DIM])
    res1 = ResBlock(h0_cube)
    res2 = ResBlock(res1)
    res3 = ResBlock(res2)
    res4 = ResBlock(res3)
    res5 = ResBlock(res4)
    conv = tf.layers.conv1d(inputs=res5, filters=W2V_DIM, kernel_size=3, strides=1, padding="SAME")
    output = conv
    return output


# The network structure for discriminator
def discriminator(input):
    conv1 = tf.layers.conv1d(inputs=input, filters=DIM, kernel_size=3, strides=1, padding="SAME")
    res1 = ResBlock(conv1)
    res2 = ResBlock(res1)
    res3 = ResBlock(res2)
    res4 = ResBlock(res3)
    res5 = ResBlock(res4)
    res_flat = tf.reshape(res5, [-1, SEQ_LEN*DIM])
    output = tf.layers.dense(inputs=res_flat, units=1)
    return output


def optimizer(loss, var_list):
    learning_rate = 0.0001
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
            self.z = tf.placeholder(tf.float32, shape=(params.batch_size, DIM))
            self.G = generator(self.z, len_dic=len_dic)
        # Reusing the generator at last, to generate files for evaluation
        with tf.variable_scope('G', reuse=True):
            self.z_eval = tf.placeholder(tf.float32, shape=(1000, DIM))
            self.G_eval = generator(self.z_eval, len_dic=len_dic)

        # The discriminator tries to tell the difference between samples from
        # the true data distribution (self.x) and the generated samples
        # (self.z).
        #
        # Here we create two copies of the discriminator network
        # that share parameters, as you cannot use the same network with
        # different inputs in TensorFlow.
        self.x = tf.placeholder(tf.float32, shape=(params.batch_size, SEQ_LEN, W2V_DIM))
        with tf.variable_scope('D'):
            self.D1 = discriminator(self.x)
        with tf.variable_scope('D', reuse=True):
            self.D2 = discriminator(self.G)
        with tf.variable_scope('D', reuse=True):
            # Define Gradient Penalty
            alpha = tf.random_uniform(shape=[params.batch_size, 1, 1], minval=0., maxval=1.)
            differences = self.G - self.x
            interpolates = self.x + (alpha * differences)
            D_interpolates = discriminator(interpolates)
            gradients = tf.gradients(D_interpolates, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
            self.gradient_penalty = tf.reduce_mean(tf.maximum(0.0, slopes-1.0))


        # Define the loss for discriminator and generator networks
        # (see the original paper for details), and create optimizers for both
        self.loss_d = -tf.reduce_mean(self.D1) +  tf.reduce_mean(self.D2) + LAMDA*self.gradient_penalty
        self.loss_g = -tf.reduce_mean(self.D2)

        # Retrieve parameters used in NN: weights, bias
        vars = tf.trainable_variables()
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params)
        self.opt_g = optimizer(self.loss_g, self.g_params)


def train(model, data, gen, params):

    with tf.Session() as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        for step in range(params.num_steps + 1):
            # update discriminator
            for i in range(5):
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
            loss_g, _, G = session.run([model.loss_g, model.opt_g, model.G], {
                model.z: z
            })

            if step % params.log_every == 0:
                print('{}: {:.4f}\t{:.4f}'.format(step, loss_d, loss_g))
                print('{}'.format(gp))

                # print 1 batch
                one_sentence = G[25, :, :]
                for i in range(SEQ_LEN):
                    word = w2v_model.similar_by_vector(one_sentence[i, :], topn=5)
                    print(word[0][0], end='')
                    print(" ", end='')
                print("")

        # Write final batch of results to file
        f = open('./results/CNN_w2v_results_'+str(SEQ_LEN), 'w', encoding='utf-8')
        z_eval = gen.sample(1000)
        G_eval = session.run(model.G_eval, {model.z_eval: z_eval})
        for i in range(1000):
            line = G_eval[i, :, :]
            for j in range(SEQ_LEN):
                # Vector -> Word, by finding the most similar vector in the model
                word = w2v_model.similar_by_vector(line[j, :], topn=5)
                f.write(word[0][0])
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