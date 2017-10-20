import tensorflow as tf


class EventAutoEncoder(object):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skips, shared=True, alpha=1.0):
        self.num_entities = num_entites
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_skips = num_skips
        self.shared = shared
        self.alpha = alpha

    def create_graph(self):
        pass

    def loss(self, num_sampled, train_labels, train_indices, embeddings=None):
        if embeddings is None:
            embeddings = self.V
        train_embeddings = tf.nn.embedding_lookup(embeddings, train_indices)

        decoding = self.encode_decode(train_embeddings)
        cost = tf.reduce_sum(tf.square(decoding - train_embeddings))
        return cost

    def variables(self):
        pass


class ConvolutionalAutoEncoder(EventAutoEncoder):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skips, shared=True, alpha=1.0):
        super(ConvolutionalAutoEncoder, self).__init__(num_entites, vocab_size, embedding_size, num_skips, shared, alpha)

    def create_graph(self):
        if not self.shared:
            # own input layer
            self.V = tf.Variable(tf.truncated_normal([self.num_entities, self.embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32)), name="V"))

            self.update = tf.scatter_update(self.V, range(self.vocab_size, self.num_entities),
                                            tf.zeros([self.num_entities-self.vocab_size, self.embedding_size]))
        filter_size = 3     # 3 x 3
        n_filters = 10
        self.Wfilter = tf.Variable(
            tf.random_uniform([
                filter_size,
                filter_size,
                1, n_filters],
                -1.0),
            1.0, name='Wfilter')
        self.bfilter = tf.Variable(tf.zeros([n_filters]), name='bfilter')
        self.bfilter2 = tf.Variable(tf.zeros([self.Wfilter.get_shape().as_list()[2]]), name='btransposefilter')

    def encode_decode(self, x):
        stacked_embeddings = tf.expand_dims(x, axis=3)
        output = tf.nn.relu(
            tf.add(tf.nn.conv2d(
                stacked_embeddings, self.Wfilter, strides=[1, 2, 2, 1], padding='SAME'), self.bfilter))
        stacked_embeddings = output

        output = tf.nn.relu(tf.add(
            tf.nn.conv2d_transpose(
                stacked_embeddings, self.Wfilter,
                tf.stack([x.get_shape()[0], 2 * self.num_skips + 1, x.get_shape()[2], 1]),
                strides=[1, 2, 2, 1], padding='SAME'), self.bfilter2))
        stacked_embeddings = output
        y = tf.squeeze(stacked_embeddings, 3)
        return y

    def variables(self):
        vars = [self.Wfilter, self.bfilter, self.bfilter2]
        if self.shared:
            return vars
        else:
            return vars + [self.V]

    @staticmethod
    def name():
        return "Conv"


class LSTMAutoencoder(EventAutoEncoder):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skips, shared=True, alpha=1.0):
        super(LSTMAutoencoder, self).__init__(num_entites, vocab_size, embedding_size, num_skips, shared, alpha)

    def create_graph(self):
        if not self.shared:
            # own input layer
            self.V = tf.Variable(tf.truncated_normal([self.num_entities, self.embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32)), name="V"))

            self.update = tf.scatter_update(self.V, range(self.vocab_size, self.num_entities),
                                            tf.zeros([self.num_entities-self.vocab_size, self.embedding_size]))
        hidden_num = 16
        self._enc_cell = tf.contrib.rnn.LSTMCell(hidden_num, cell_clip=1.0)
        self._dec_cell = tf.contrib.rnn.LSTMCell(hidden_num, cell_clip=1.0)
        with tf.variable_scope('decoder') as vs:
            self.dec_weight_ = tf.Variable(
                            tf.truncated_normal([hidden_num, self.embedding_size], dtype=tf.float32),
                            name="dec_weight")
            self.dec_bias_ = tf.Variable(
                            tf.constant(0.1, shape=[self.embedding_size], dtype=tf.float32),
                            name="dec_bias")

    def encode_decode(self, inputs):
        reverse = False
        inputs = tf.unstack(inputs, num=self.num_skips, axis=1)
        self.batch_num = inputs[0].get_shape().as_list()[0]
        self.elem_num = inputs[0].get_shape().as_list()[1]
        self.z_codes, self.enc_state = tf.nn.dynamic_rnn(
            self._enc_cell, tf.stack(inputs, axis=0), dtype=tf.float32, scope='LSTM'
        )
        # enctoder state [batch, num_hidden] last output state of the encoder
        if True:
            dec_inputs = [tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)
                          for _ in range(len(inputs))]
            dec_outputs, dec_state = tf.nn.dynamic_rnn(
              self._dec_cell, tf.stack(dec_inputs, axis=0),
              initial_state=self.enc_state, dtype=tf.float32)
            dec_output_ = tf.transpose(tf.stack(dec_outputs), [1,0,2])
            dec_weight_ = tf.tile(tf.expand_dims(self.dec_weight_, 0), [self.batch_num,1,1])
            self.output = tf.matmul(dec_output_, dec_weight_) + self.dec_bias_
        else:
            # assign state as above
            dec_state = self.enc_state
            dec_input_ = tf.zeros(tf.shape(inputs[0]), dtype=tf.float32)    # this should be the actual input?
            dec_outputs = []

            for step in range(len(inputs)):
                if step > 0: tf.get_variable_scope().reuse_variables()
                dec_input_, dec_state = self._dec_cell(dec_input_, dec_state)
                dec_input_ = tf.matmul(dec_input_, self.dec_weight_) + self.dec_bias_
                dec_outputs.append(dec_input_)
            if reverse:
                dec_outputs = dec_outputs[::-1]
            self.output = tf.transpose(tf.stack(dec_outputs), [1, 0, 2])

        return self.output
        #self.input_ = tf.transpose(tf._pack(inputs), [1, 0, 2])
        #self.loss = tf.reduce_mean(tf.square(self.input_ - self.output_))

    def variables(self):
        vars = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES, scope='LSTM')]
        vars += [self.dec_weight_, self.dec_bias_]
        if self.shared:
            return vars
        else:
            return vars + [self.V]

    @staticmethod
    def name():
        return "LSTM"


if __name__ == '__main__':
    import numpy as np
    batch_num = 16
    hidden_num = 12
    step_num = 8
    elem_num = 1
    iteration = 10000
    # placeholder list
    p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
    p_inputs = [tf.squeeze(t, [1]) for t in tf.split(p_input, step_num, 1)]
    ae = LSTMAutoencoder(100, 10, elem_num, step_num)
    ae.create_graph()
    loss = ae.loss(None, None, [range(8) for i in range(16)], p_inputs)
    train = tf.train.AdamOptimizer(0.05).minimize(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sum_l = 0
        for i in range(iteration):
            """Random sequences.
              Every sequence has size batch_num * step_num * elem_num
              Each step number increases 1 by 1.
              An initial number of each sequence is in the range from 0 to 19.
              (ex. [8. 9. 10. 11. 12. 13. 14. 15])
            """
            r = np.random.randint(20, size=batch_num).reshape([batch_num, 1, 1])
            r = np.tile(r, (1, step_num, elem_num))
            d = np.linspace(0, step_num, step_num, endpoint=False).reshape([1, step_num, elem_num])
            d = np.tile(d, (batch_num, 1, 1))
            random_sequences = r + d

            l,_ = sess.run([loss, train], {p_input : random_sequences})
            sum_l += l
            if i % 1000 == 0:
                print("iter %d:" % (i + 1), 1.0 * sum_l / 1000.0)
                sum_l = 0
                test = r + d
                output_ = sess.run([ae.output], {p_input: test})
                print("train result :")
                print("input :", test[0, :, :].flatten())
                print("output :", output_[0][0,:,:].flatten())