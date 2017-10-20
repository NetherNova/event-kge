import tensorflow as tf
import numpy as np
import math
import pickle


def dot_similarity(x, y, broadcast=False, expand=False):
    """
    Dot similarity across batch
    :param x:
    :param y:
    :return:
    """
    return tf.batch_matmul(x, y)


def dot(x, y):
    return tf.reduce_sum(tf.multiply(x, y), 1, keep_dims=True)


def l2_similarity(x, y, broadcast=False, expand=True):
    """
    L2 similairty across batch
    :param x:
    :param y:
    :return:
    """
    if broadcast:
        if expand:
            x = tf.expand_dims(x, 1)
            diff = x - y
        else:
            diff = x - y
            diff = tf.transpose(diff, [1,0,2])
        return -tf.sqrt(tf.reduce_sum(diff ** 2, axis=2))
    else:
        diff = x - y
        return -tf.sqrt(tf.reduce_sum(diff ** 2, axis=1))


def l1_similarity(x, y):
    return - tf.reduce_sum(tf.abs(x - y))


def trans(x, y):
    return x+y


def ident_entity(x, y):
    return x


def max_margin(pos, neg, marge=1.0):
    cost = marge - pos + neg
    return tf.reduce_sum(tf.maximum(0., cost))


def rescal_similarity():
    pass


def normalize(W):
    return W / tf.expand_dims(tf.sqrt(tf.reduce_sum(W ** 2, axis=1)), 1)


def rank_left_fn_idx(simfn, embeddings_ent, embeddings_rel, leftop, rightop, inpr, inpo):
    """
    compute similarity score of all 'left' entities given 'right' and 'rel' members
    return *batch_size* rank lists for all entities [all_entities, batch_size] similarity
    :param simfn:
    :param embeddings_ent:
    :param embeddings_rel:
    :param leftop:
    :param rightop:
    :param inpr:
    :param inpo:
    :return:
    """
    lhs = embeddings_ent # [num_entities, d]
    rell = tf.nn.embedding_lookup(embeddings_rel, inpo) # [num_test, d], RESCAL : [num_test, d, d]
    rhs = tf.nn.embedding_lookup(embeddings_ent, inpr) # [num_test, d]
    expanded_lhs = tf.expand_dims(lhs, 1) # [num_ents, 1, d]
    if simfn == l2_similarity:
        batch_lhs = tf.transpose(leftop(expanded_lhs, rell), [0, 1, 2])
        simi = simfn(batch_lhs, rhs, broadcast=True, expand=False)
    elif simfn == rescal_similarity:
        # TODO: only use unique relations in rell / do in loop for rescal outside of TF
        expanded_lhs = tf.expand_dims(expanded_lhs, 2) # [entity_size, 1, 1, d] # TODO: was ist zeile und was ist Spalte in rell?
        # [entity_size, test_size, d]
        expanded_lhs = tf.reduce_sum(tf.multiply(expanded_lhs, rell), 3) # TODO: which dim to reduce? 2 or 3
        # [entity_size, test_size, d] * [test_size, d]
        simi = tf.nn.sigmoid(tf.transpose(tf.reduce_sum(tf.multiply(expanded_lhs, rhs), 2)))
    else:
        batch_lhs = tf.transpose(leftop(expanded_lhs, rell), [1, 0, 2])
        batch_rhs = tf.transpose(tf.expand_dims(rhs, 1), [0, 2, 1])
        simi = tf.squeeze(simfn(batch_lhs, batch_rhs), 2)
    return simi


def rank_right_fn_idx(simfn, embeddings_ent, embeddings_rel, leftop, rightop, inpl, inpo):
    """
    compute similarity score of all 'right' entities given 'left' and 'rel' members (test_size)
    :param simfn:
    :param embeddings_ent:
    :param embeddings_rel:
    :param leftop:
    :param rightop:
    :return:
    """
    rhs = embeddings_ent
    rell = tf.nn.embedding_lookup(embeddings_rel, inpo)
    lhs = tf.nn.embedding_lookup(embeddings_ent, inpl)
    if simfn == rescal_similarity:
        lhs = tf.expand_dims(lhs, 1)
        # [test_size, 1, d]
        lhs = tf.expand_dims(tf.reduce_sum(tf.multiply(lhs, rell), 2), 1)
        # [test_size, 1, d] x [entity, d]
        simi = tf.reduce_sum(tf.multiply(lhs, rhs), 2)
        return tf.nn.sigmoid(simi)
    elif simfn == dot_similarity:
        rhs = tf.transpose(rhs)
    simi = simfn(leftop(lhs, rell), rhs, broadcast=True)
    return simi


class SuppliedEmbedding(object):
    def __init__(self, W, dictionary):
        self._W = W
        self._dictionary = dictionary   # {'id' : id}

    def get_embeddings(self):
        return self._W

    def get_dictionary(self):
        return self._dictionary

    def save_embedding(self, file_name):
        pickle.dump(self, open(file_name, "wb"))


class Softmax(object):
    def __init__(self, context, labels, vocabulary_size, negative_sample_size, hidden_dim):
        """ Class needs input of sequence activaction vector,
        the context embeddings (previous lookup) and the actual labels
        :param context:
        :param labels:
        :param vocabulary_size:
        :param negative_sample_size:
        :param hidden_dim:
        """
        self._context = context
        self._labels = labels
        self._vocabulary_size = vocabulary_size
        self._negative_sample_size = negative_sample_size
        self._nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_dim],
                      stddev=1.0 / math.sqrt(hidden_dim)))
        self._nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    def loss(self):
        return tf.reduce_mean(tf.nn.nce_loss(self._nce_weights, self._nce_biases, self._context, self._labels,
                                             self._negative_sample_size, self._vocabulary_size))


class SkipgramModel(object):
    def __init__(self, embedding_size, batch_size, num_sampled, vocab_size):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.vocab_size = vocab_size

    def create_graph(self):
        w_bound = 0.5 * self.embedding_size
        self.E = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -w_bound, w_bound))
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        sg_embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
        self.loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size, self.train_labels)
        self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        return self.E


def incremental_concat_layer(embeddings, train_dataset, batch_size, embedding_size, len_sequence,
                             num_words_per_sequence):
    """Deprecated: simply use tf.reshape()"""
    def body(i, x):
        a = tf.reshape(tf.nn.embedding_lookup(embeddings, tf.slice(train_dataset, [0, (num_words_per_sequence+1)*i],
                        [batch_size, (num_words_per_sequence+1)])),
                       [batch_size, (num_words_per_sequence+1)*embedding_size])
        return i+1, tf.concat(1, [x, a])

    def condition(i, x):
        return i < len_sequence
    
    i = tf.constant(1)
    init = tf.reshape(tf.nn.embedding_lookup(embeddings, tf.slice(train_dataset, [0,0], [batch_size,
                            (num_words_per_sequence+1)])), [batch_size, (num_words_per_sequence+1)*embedding_size])
    _, result = tf.while_loop(condition, body, [i, init],
                              shape_invariants=[i.get_shape(), tf.TensorShape([None, None])])
    return tf.reshape(result, [batch_size, len_sequence*(num_words_per_sequence+1)*embedding_size])


def concat_layer(left, right):
    """
    Concat two layers alongside axis 1
    :param left:
    :param right:
    :return:
    """
    return tf.concat(1, [left, right])


def average_layer(tensor, axis):
    """
    for 3-dim tensor with batches --> use axis=1
    :param tensor:
    :param axis:
    :return:
    """
    return tf.reduce_mean(tensor, axis)


def skipgram_loss(vocab_size, num_sampled, embed, embedding_size, train_labels):
    nce_weights = tf.Variable(
        tf.truncated_normal([vocab_size, embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
    nce_biases = tf.Variable(tf.truncated_normal([vocab_size]))

    skipgram_loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
                           biases=nce_biases,
                           labels=train_labels,
                           inputs=embed,
                           num_sampled=num_sampled,
                           num_classes=vocab_size,
                           remove_accidental_hits=True))
    return skipgram_loss


def lstm_loss(vocab_size, num_sampled, embed, embedding_size, train_labels):
    W = tf.Variable(
        tf.truncated_normal([vocab_size, embedding_size],
                            stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
    bias = tf.Variable(tf.truncated_normal([vocab_size]))

    # [batch_size, num_steps, embedding_size] into list of [batch_size, embedding_size]
    event_list = tf.unstack(embed, axis=1)
    cell = tf.nn.rnn_cell.LSTMCell(embedding_size)
    outputs, state = tf.nn.rnn(cell, event_list, dtype=tf.float32)
    embed_context = state[1]  # take last hidden state (h) state[0] = (cell state c)

    logits = tf.matmul(embed_context, tf.transpose(W)) + bias
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, tf.reshape(train_labels, [-1])))
    return loss


def composition_rnn_loss(vocab_size, num_sampled, embed, embedding_size, train_labels):
    """

    :param vocab_size:
    :param num_sampled:
    :param embed: [batch_size, len_sequence, embedding_size]
    :param embedding_size:
    :param train_labels:
    :return:
    """
    event_list = tf.unstack(embed, axis=1)
    cell = ComposistionRNN(50, embedding_size)
    outputs, state = tf.nn.rnn(cell, event_list, dtype=tf.float32)
    embed_context = state  # take last hidden state


def rnn_loss(vocab_size, num_sampled, embed, embedding_size, train_labels):
    embedding_size = 50
    W = tf.Variable(
        tf.truncated_normal([vocab_size, embedding_size],
                            stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
    bias = tf.Variable(tf.truncated_normal([vocab_size]))

    # [batch_size, num_steps, embedding_size] into list of [batch_size, embedding_size]
    # event_list = tf.unstack(embed, axis=1)
    cell = tf.contrib.rnn.BasicLSTMCell(embedding_size)
    outputs, state = tf.nn.dynamic_rnn(cell, embed, dtype=tf.float32)
    last = tf.slice(outputs, [0, embed.get_shape()[1].value - 1, 0], [embed.get_shape()[0].value, 1,
                                                                  embedding_size])
    last = tf.squeeze(last, axis=1)
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=W,
                       biases=bias,
                       labels=train_labels,
                       inputs=last,
                       num_sampled=num_sampled,
                       num_classes=vocab_size,
                       remove_accidental_hits=True))
    return loss


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def cnn_loss(vocab_size, num_sampled, embed, embedding_size, train_labels):
    # [batch_size, num_steps, embedding_size] into list of [batch_size, embedding_size]
    num_filters = 20
    embed_context = tf.expand_dims(embed, axis=3)
    W_conv = tf.Variable(tf.random_normal([embed.get_shape()[1].value, embed.get_shape()[1].value, 1, num_filters]))
    b_conv = tf.Variable(tf.random_normal([num_filters]))

    net = tf.nn.conv2d(input=embed_context, name='layer_conv2',
                         filter=W_conv, strides=[1, 1, 1, 1],
                         padding='VALID')
    act = tf.nn.relu(tf.nn.bias_add(net, b_conv))
    act = tf.contrib.layers.flatten(act)

    W = tf.Variable(
        tf.truncated_normal([vocab_size, act.get_shape()[1].value],
                            stddev=1.0 / tf.sqrt(tf.constant(act.get_shape()[0].value, dtype=tf.float32))))
    bias = tf.Variable(tf.truncated_normal([vocab_size]))

    # Softmax of cnn activation vectors
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=W,
                       biases=bias,
                       labels=train_labels,
                       inputs=act,
                       num_sampled=num_sampled,
                       num_classes=vocab_size,
                       remove_accidental_hits=True))
    return loss


def concat_window_loss(vocab_size, num_sampled, embed, embedding_size, train_labels):
    """

    :param vocab_size:
    :param num_sampled:
    :param embed: tensor of size [batch_size, num_events, embedding_size]
    :param embedding_size:
    :param train_labels:
    :param sequence_id:
    :return:
    """
    # concatenate everything in *embed*
    embed_context = tf.reshape(embed, [embed.get_shape()[0].value, embed.get_shape()[1].value * embedding_size])

    #embed_context = tf.concat_v2([embed_context, sequence_vectors], axis=1)

    #embed_context = tf.Print(embed_context.get_shape(),[embed_context], "context: ")
    W = tf.Variable(
        tf.truncated_normal([vocab_size, embed_context.get_shape()[1].value],
                            stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
    bias = tf.Variable(tf.truncated_normal([vocab_size]))

    # Softmax of concatenated vectors
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=W,
                       biases=bias,
                       labels=train_labels,
                       inputs=embed_context,
                       num_sampled=num_sampled,
                       num_classes=vocab_size,
                       remove_accidental_hits=True))
    return loss


def ranking_error_triples(filter_triples, scores_l, scores_r, left_ind, o_ind, right_ind):
    errl = []
    errr = []
    for i, (l, o, r) in enumerate(
            zip(left_ind, o_ind, right_ind)):
        # find those triples that have <*,o,r> and * != l
        rmv_idx_l = [l_rmv for (l_rmv, rel, rhs) in filter_triples if
                     rel == o and r == rhs and l_rmv != l]
        # *l* is the correct index
        scores_l[i, rmv_idx_l] = -np.inf
        errl += [np.argsort(np.argsort(-scores_l[i, :]))[l] + 1]
        # since index start at 0, best possible value is 1
        rmv_idx_r = [r_rmv for (lhs, rel, r_rmv) in filter_triples if
                     rel == o and lhs == l and r_rmv != r]
        # *l* is the correct index
        scores_r[i, rmv_idx_r] = -np.inf
        errr += [np.argsort(np.argsort(-scores_r[i, :]))[r] + 1]
    return errl, errr


def insight_error_triples(filter_triples, scores_l, scores_r, left_ind, o_ind, right_ind, r_ent_dict, r_rel_dict):
    for i, (l, o, r) in enumerate(
            zip(left_ind, o_ind, right_ind)):
        # find those triples that have <*,o,r> and * != l
        rmv_idx_l = [l_rmv for (l_rmv, rel, rhs) in filter_triples if
                     rel == o and r == rel and l_rmv != l]
        # *l* is the correct index
        scores_l[i, rmv_idx_l] = -np.inf
        top_ents_l = np.argsort(-scores_l[i, :])[:10]
        # since index start at 0, best possible value is 1
        rmv_idx_r = [r_rmv for (lhs, rel, r_rmv) in filter_triples if
                     rel == o and lhs == l and r_rmv != r]
        # *l* is the correct index
        scores_r[i, rmv_idx_r] = -np.inf
        top_ents_r = np.argsort(-scores_r[i, :])[:10]
        print(r_ent_dict[l], r_rel_dict[o], r_ent_dict[r], "Left Recommendations: ", [r_ent_dict[left] for left in top_ents_l])
        print(r_ent_dict[l], r_rel_dict[o], r_ent_dict[r], "Right Recommendations: ", [r_ent_dict[left] for left in top_ents_r])


def bernoulli_probs(ontology, relation_dictionary):
    """
    Obtain bernoulli probabilities for each relation
    :param ontology:
    :param relation_dictionary:
    :return:
    """
    probs = dict()
    relations = set(ontology.predicates(None, None))
    for r in relations:
        heads = set(ontology.subjects(r, None))
        tph = 0
        for h in heads:
            tails = set(ontology.objects(h, r))
            tph += len(tails)
        tph = tph / (1.0 * len(heads))

        tails = set(ontology.objects(None, r))
        hpt = 0
        for t in tails:
            heads = set(ontology.subjects(r, t))
            hpt += len(heads)
        hpt = hpt / (1.0 * len(tails))
        probs[relation_dictionary[str(r)]] = tph / (tph + hpt)
    return probs


class ComposistionRNN(object):#tf.contrib.rnn_cell.BasicRNNCell):
    """
    Semantic compposition RNN using recursive concatentation
    """
    def __init__(self, num_units, embedding_size, activation=None, reuse=None):
        super(tf.nn.rnn_cell.BasicRNNCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or tf.ops.math_ops.tanh

        self.W = tf.Variable(
            tf.truncated_normal([num_units, num_units + embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(num_units + embedding_size, dtype=tf.float32))))

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self.num_units

    def call(self, inputs, state):
        output = tf.multiply(self.W, tf.concat([state, inputs], axis=1))
        return output, output