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
    return tf.reduce_sum(tf.mul(x, y), 1, keep_dims=True)


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
    cost = 1. - pos + neg
    return tf.reduce_mean(tf.maximum(0., cost))


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
    lhs = embeddings_ent
    rell = tf.nn.embedding_lookup(embeddings_rel, inpo)
    rhs = tf.nn.embedding_lookup(embeddings_ent, inpr)
    # [num_entities, embedding_size, batch_size], [batch_size , embedding_size]
    expanded_lhs = tf.expand_dims(lhs, 1)

    if simfn == l2_similarity:
        batch_lhs = tf.transpose(leftop(expanded_lhs, rell), [0, 1, 2])
        simi = simfn(batch_lhs, rhs, broadcast=True, expand=False)
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
    if simfn == dot_similarity:
        rhs = tf.transpose(rhs)
    simi = simfn(leftop(lhs, rell), rhs, broadcast=True)
    return simi


class SuppliedEmbedding(object):
    def __init__(self, W, dictionary):
        self._W = W
        self._dictionary = dictionary

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
    def __init__(self, label, num_entities, num_hidden, num_hidden_softmax):
        self.num_dim = num_hidden
        self.num_entities = num_entities
        self.num_hidden_softmax = num_hidden_softmax
        self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_hidden), name="W-"+label))

    def loss(self, lookup_entities, labels):
        # TODO: embedding_lookup, sum over context
        # concatentation of previous layer --> num_hidden_softmax needs to be the size of the concatentation
        context_embeddings = tf.nn.embedding_lookup(self.W, lookup_entities)
        loss = Softmax(context_embeddings, labels, self.num_entities, self.num_hidden_softmax).loss()
        return loss

    def get_normalized_embeddings(self):
        return self.W / tf.sqrt(tf.reduce_sum(tf.square(self.W), 1, keep_dims=True))

    def get_embeddings(self):
        return self.W     


class EventsWithWordsModel(object):
    def __init__(self, len_sequence, num_words_per_sequence, num_entities, embedding_size, num_label_entities, num_neg_samples):
        self._len_sequence = len_sequence
        self._num_words_per_sequence = num_words_per_sequence
        self._num_entities = num_entities
        self._embedding_size = embedding_size
        self._num_label_entities = num_label_entities
        self._num_neg_samples = num_neg_samples
        self.W = EmbeddingLayer("EventWords", num_entities, embedding_size)

    def loss(self, train_dataset, train_labels, batch_size):
        concat = incremental_concat_layer(self.W.get_embeddings(), train_dataset, batch_size, self._embedding_size,
                                          self._len_sequence, self._num_words_per_sequence)
        loss = Softmax(concat, train_labels, self._num_label_entities, self._num_neg_samples, (self._len_sequence) *
                       (self._num_words_per_sequence+1) * self._embedding_size).loss()
        return loss


class EventsWithWordsAndVariantModel(object):
    def __init__(self, len_sequence, num_words_per_sequence, num_entities, embedding_size, num_label_events, num_label_variants, variant_index, num_neg_samples):
        self._len_sequence = len_sequence
        self._num_words_per_sequence = num_words_per_sequence
        self._num_entities = num_entities
        self._embedding_size = embedding_size
        self._num_label_events = num_label_events
        self._num_label_variants = num_label_variants
        self._num_neg_samples = num_neg_samples
        self._variant_index = variant_index
        self.W = EmbeddingLayer("EventWords", num_entities, embedding_size)

    def loss(self, train_dataset, train_labels_events, train_labels_variants, batch_size):
        concat = incremental_concat_layer(self.W.get_embeddings(), train_dataset, batch_size, self._embedding_size,
                                          self._len_sequence, self._num_words_per_sequence)
        variant_embeddings = tf.reshape(tf.nn.embedding_lookup(self.W.get_embeddings(), tf.slice(train_dataset,
                                [0, self._variant_index], [batch_size, 1])), [batch_size, self._embedding_size])
        concat_variant = concat_layer(concat, variant_embeddings)
        concat_last_event = concat_layer(concat, tf.reshape(tf.nn.embedding_lookup(self.W.get_embeddings(),
                                                                                   train_labels_events), [batch_size, self._embedding_size]))
        loss1 = Softmax(concat_variant, train_labels_events, self._num_label_events, self._num_neg_samples,
                        (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size +
                        self._embedding_size).loss()
        loss2 = Softmax(concat_last_event, train_labels_variants, self._num_label_variants, self._num_neg_samples - 3,
                        (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size +
                        self._embedding_size).loss()
        return loss1 + loss2

    def get_model(self, train_dataset, dataset_size):
        concat = incremental_concat_layer(self.W.get_embeddings(), train_dataset, dataset_size, self._embedding_size,
                                          self._len_sequence, self._num_words_per_sequence)
        variant_embeddings = tf.reshape(tf.nn.embedding_lookup(self.W.get_embeddings(), tf.slice(train_dataset,
                            [0, self._variant_index], [dataset_size, 1])), [dataset_size, self._embedding_size])
        concat_variant = concat_layer(concat, variant_embeddings)
        return concat_variant

    def get_embeddings(self, dataset):
        return self.W.get_embeddings()
    

class EventsWithWordsAndVariantComposedModel(object):
    def __init__(self, len_sequence, num_words_per_sequence, num_entities, embedding_size, num_label_events,
                 num_label_variants, variant_index, num_neg_samples, num_variant_parts):
        self._len_sequence = len_sequence
        self._num_words_per_sequence = num_words_per_sequence
        self._num_entities = num_entities
        self._embedding_size = embedding_size
        self._num_label_events = num_label_events
        self._num_label_variants = num_label_variants
        self._num_neg_samples = num_neg_samples
        self._variant_index = variant_index
        self._num_variant_parts = num_variant_parts
        self.W = EmbeddingLayer("EventWords", num_entities, embedding_size)

    def loss(self, train_dataset, train_labels_events, train_labels_variants, batch_size):
        # last entries in train_dataset (..., variant, part, part, ..., part)
        concat = incremental_concat_layer(self.W.get_embeddings(), train_dataset, batch_size,
                                          self._embedding_size, self._len_sequence, self._num_words_per_sequence)
        var_avg = average_layer(tf.nn.embedding_lookup(self.W.get_embeddings(), tf.slice(train_dataset,
                                            [0, self._variant_index], [batch_size, self._num_variant_parts])), axis=1)
        concat_variant_parts = concat_layer(concat, var_avg)
        concat_last_event = concat_layer(concat, tf.reshape(tf.nn.embedding_lookup(self.W.get_embeddings(),
                                                        train_labels_events), [batch_size, self._embedding_size]))
        loss1 = Softmax(concat_variant_parts, train_labels_events, self._num_label_events, self._num_neg_samples,
                        (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size +
                        self._embedding_size).loss()
        loss2 = Softmax(concat_last_event, train_labels_variants, self._num_label_variants, self._num_neg_samples - 3,
                        (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size +
                        self._embedding_size).loss()
        return loss1 + loss2

    def get_model(self, train_dataset, dataset_size):
        concat = incremental_concat_layer(self.W.get_embeddings(), train_dataset, dataset_size, self._embedding_size,
                                          self._len_sequence, self._num_words_per_sequence)
        var_avg = average_layer(tf.nn.embedding_lookup(self.W.get_embeddings(), tf.slice(train_dataset,
                            [0, self._variant_index], [dataset_size, self._num_variant_parts])), axis=1)
        concat_variant_parts = concat_layer(concat, var_avg)
        return concat_variant_parts

    def get_embeddings(self, dataset):
        return self.W.get_embeddings()


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


class EmbeddingLayer(object):
    def __init__(self, label, num_entities, num_hidden):
        self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_hidden), name="W-"+label))

    def get_embeddings(self):
        return self.W


class EventEmbedding(object):
    def __init__(self, label, num_entities, num_events, num_dim):
        self.num_dim = num_dim
        self.num_entities = num_entities
        self.num_events = num_events
        self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_dim), name="W-"+label))
        self.W_events = tf.Variable(tf.truncated_normal(shape=(num_events, num_dim), name="W-events"))

    def loss(self, lookup_entity, negative_entity, event_entities):
        embed_pos = tf.nn.embedding_lookup(self.W, lookup_entity)
        embed_neg = tf.nn.embedding_lookup(self.W, negative_entity)
        embed_context = tf.reduce_sum(tf.nn.embedding_lookup(self.W_events, event_entities), 1)
        sim_pos = tf.matmul(embed_pos, tf.transpose(embed_context))
        sim_neg = tf.matmul(embed_neg, tf.transpose(embed_context))
        loss = max_margin(sim_pos, sim_neg).loss() + 0.01*tf.nn.l2_loss(self.W) + 0.01*tf.nn.l2_loss(self.W_events)
        return loss

    def get_normalized_embeddings(self):
        return normalize(self.W)

    def evaluate_cosine_similarity(self, valid_dataset):
        normalized_embeddings = self.get_normalized_embeddings()
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        return tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


class RecurrentEventEmbedding(EventEmbedding):
    def __init__(self, label, num_entities, num_dim):
        # here num_entities is only number of variants
        super(RecurrentEventEmbedding, self).__init__(label, num_entities, 0, num_dim)

    def loss(self, lookup_entity, negative_entity, context_entities):
        # cell = tf.nn.rnn_cell.BasicRNNCell(self.num_dim)
        cell = tf.nn.rnn_cell.LSTMCell(self.num_dim)
        # context_embedding = tf.nn.embedding_lookup(self.W, context_entities)
        # context_embedding =tf.unstack(context_embeddings)
        context_embedding = tf.pack(context_entities)
        context_embedding = tf.one_hot(context_embedding, 3)
        context_embedding = tf.unstack(context_embedding)
        outputs, state = tf.nn.rnn(cell, context_embedding, dtype=tf.float32)
        embed_context = state[1] # take last state only
        embed_pos = tf.nn.embedding_lookup(self.W, lookup_entity)
        embed_neg = tf.nn.embedding_lookup(self.W, negative_entity)
        score_pos = tf.matmul(embed_pos, tf.transpose(embed_context))
        score_neg = tf.matmul(embed_neg, tf.transpose(embed_context))

        loss = tf.reduce_mean(tf.maximum(0., 1. - score_pos + score_neg))
        return loss


class TranslationEmbeddings(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, leftop, rightop, fnsim):
        """
        Implements translation-based triplet scoring from negative sampling (TransE)
        :param num_entities:
        :param num_relations:
        :param embedding_size:
        :param batch_size_kg:
        :param batch_size_sg:
        :param num_sampled:
        :param vocab_size:
        :param leftop:
        :param rightop:
        :param fnsim:
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.batch_size_kg = batch_size_kg
        self.batch_size_sg = batch_size_sg
        self.leftop = leftop
        self.rightop = rightop
        self.fnsim = fnsim

    def run(self, tg, sg, test_tg, test_size, num_steps, init_lr=1.0, skipgram=True, store_embeddings=False):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            # Translation Model initialisation
            w_bound = np.sqrt(6. / self.embedding_size)
            self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))
            self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))

            normalize_E = self.E.assign(tf.nn.l2_normalize(self.E, 1))
            normalize_R = self.R.assign(tf.nn.l2_normalize(self.R, 1))

            inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            test_inpr = tf.placeholder(tf.int32, [test_size], name="test_rhs")
            test_inpl = tf.placeholder(tf.int32, [test_size], name="test_lhs")
            test_inpo = tf.placeholder(tf.int32, [test_size], name="test_rell")

            lhs = tf.nn.embedding_lookup(self.E, inpl)
            rhs = tf.nn.embedding_lookup(self.E, inpr)
            rell = tf.nn.embedding_lookup(self.R, inpo)
            relr = tf.nn.embedding_lookup(self.R, inpo)

            lhsn = tf.nn.embedding_lookup(self.E, inpln)
            rhsn = tf.nn.embedding_lookup(self.E, inprn)
            relln = tf.nn.embedding_lookup(self.R, inpon)
            relrn = tf.nn.embedding_lookup(self.R, inpon)

            if self.fnsim == dot_similarity:
                simi = tf.diag_part(self.fnsim(self.leftop(lhs, rell), tf.transpose(self.rightop(rhs, relr)),
                                               broadcast=False))
                simin = tf.diag_part(self.fnsim(self.leftop(lhsn, relln), tf.transpose(self.rightop(rhsn, relrn)),
                                                broadcast=False))
            else:
                simi = self.fnsim(self.leftop(lhs, rell), self.rightop(rhs, relr), broadcast=False)
                simin = self.fnsim(self.leftop(lhsn, relln), self.rightop(rhsn, relrn), broadcast=False)

            kg_loss = max_margin(simi, simin)

            # Skipgram Model
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

            embed = tf.nn.embedding_lookup(self.E, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, self.embedding_size],
                                    stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32))))
            nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
            skipgram_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size,
                               remove_accidental_hits=True))

            if skipgram:
                loss = kg_loss + skipgram_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
            else:
                loss = kg_loss
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = init_lr
            # tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.98, staircase=True)
            learning_rate = tf.constant(starter_learning_rate)
            # grads_E = tf.reduce_mean(tf.gradients(loss, self.E)[0])
            # grads_R = tf.reduce_mean(tf.gradients(loss, self.R)[0])

            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

            ranking_error_l = rank_left_fn_idx(self.fnsim, self.E, self.R, self.leftop, self.rightop, test_inpr,
                                               test_inpo)
            ranking_error_r = rank_right_fn_idx(self.fnsim, self.E, self.R, self.leftop, self.rightop, test_inpl,
                                                test_inpo)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            eval_step_size = 10
            best_hits = -np.inf
            best_rank = np.inf
            mean_rank_list = []
            hits_10_list = []
            loss_list = []
            if store_embeddings:
                entity_embs = []
                relation_embs = []
            for b in range(1, num_steps + 1):
                batch_pos, batch_neg = tg.next()
                test_batch_pos, _ = test_tg.next()
                batch_x, batch_y = sg.next()
                batch_y = np.array(batch_y).reshape((self.batch_size_sg, 1))
                session.run([normalize_E, normalize_R])
                # calculate valid indices for scoring
                feed_dict = {inpl: batch_pos[1, :], inpr: batch_pos[0, :], inpo: batch_pos[2, :],
                             inpln: batch_neg[1, :], inprn: batch_neg[0, :], inpon: batch_neg[2, :],
                             train_inputs: batch_x, train_labels: batch_y,
                             global_step : b
                             }
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)

                average_loss += l
                if b % eval_step_size == 0:
                    feed_dict = {test_inpl: test_batch_pos[1, :], test_inpo: test_batch_pos[2, :],
                                 test_inpr: test_batch_pos[0, :]}
                    scores_l, scores_r = session.run([ranking_error_l, ranking_error_r], feed_dict=feed_dict)

                    errl = []
                    errr = []
                    for i, (l, o, r) in enumerate(
                            zip(test_batch_pos[1, :], test_batch_pos[2, :], test_batch_pos[0, :])):
                        # find those triples that have <*,o,r> and * != l
                        rmv_idx_l = [l_rmv for (l_rmv, rel, rhs) in test_tg.all_triples if
                                     rel == o and r == rel and l_rmv != l]
                        # *l* is the correct index
                        scores_l[i, rmv_idx_l] = -np.inf
                        errl += [np.argsort(np.argsort(-scores_l[i, :]))[l] + 1]
                        # since index start at 0, best possible value is 1
                        rmv_idx_r = [r_rmv for (lhs, rel, r_rmv) in test_tg.all_triples if
                                     rel == o and lhs == l and r_rmv != r]
                        # *l* is the correct index
                        scores_r[i, rmv_idx_r] = -np.inf
                        errr += [np.argsort(np.argsort(-scores_r[i, :]))[r] + 1]

                    hits_10 = np.mean(np.asarray(errl + errr) <= 10) * 100
                    mean_rank = np.mean(np.asarray(errl + errr))
                    mean_rank_list.append(mean_rank)
                    hits_10_list.append(hits_10)

                    if best_hits < hits_10:
                        best_hits = hits_10
                    if best_rank > mean_rank:
                        best_rank = mean_rank

                    if b > 0:
                        average_loss = average_loss / eval_step_size
                    loss_list.append(average_loss)

                    if store_embeddings:
                        entity_embs.append(session.run(self.E))
                        relation_embs.append(session.run(self.R))

                    # The average loss is an estimate of the loss over the last eval_step_size batches.
                    print('Average loss at step %d: %f' % (b, average_loss))
                    print "Hits10: ", hits_10
                    print "MeanRank: ", mean_rank
                    average_loss = 0
                if not store_embeddings:
                    entity_embs = [session.run(self.E)]
                    relation_embs = [session.run(self.R)]
            return entity_embs, relation_embs, best_hits, best_rank, mean_rank_list, hits_10_list, loss_list



class TransH(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size):
        """
        Implements translation-based triplet scoring from negative sampling (TransH)
        :param num_entities:
        :param num_relations:
        :param embedding_size:
        :param batch_size_kg:
        :param batch_size_sg:
        :param num_sampled:
        :param vocab_size:
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.batch_size_kg = batch_size_kg
        self.batch_size_sg = batch_size_sg

    def run(self, tg, sg, test_tg, test_size, num_steps, init_lr=1.0, skipgram=True, store_embeddings=False):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            # Translation Model initialisation
            w_bound = np.sqrt(6. / self.embedding_size)
            self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))
            self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))

            self.W = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))

            normalize_E = self.E.assign(tf.nn.l2_normalize(self.E, 1))
            normalize_R = self.R.assign(tf.nn.l2_normalize(self.R, 1))
            normalize_W = self.W.assign(tf.nn.l2_normalize(self.W, 1))

            inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            test_inpr = tf.placeholder(tf.int32, [test_size], name="test_rhs")
            test_inpl = tf.placeholder(tf.int32, [test_size], name="test_lhs")
            test_inpo = tf.placeholder(tf.int32, [test_size], name="test_rell")

            lhs = tf.nn.embedding_lookup(self.E, inpl)
            rhs = tf.nn.embedding_lookup(self.E, inpr)
            rell = tf.nn.embedding_lookup(self.R, inpo)
            wr = tf.nn.embedding_lookup(self.W, inpo)

            lhsn = tf.nn.embedding_lookup(self.E, inpln)
            rhsn = tf.nn.embedding_lookup(self.E, inprn)
            relln = tf.nn.embedding_lookup(self.R, inpon)
            wrn = tf.nn.embedding_lookup(self.W, inpon)

            lhs_proj = lhs - dot(dot(lhs, wr), wr)
            rhs_proj = rhs - dot(dot(rhs, wr), wr)

            lhs_proj_n = lhsn - dot(dot(lhsn, wrn), wrn)
            rhs_proj_n = rhsn - dot(dot(rhsn, wrn), wrn)

            simi = l2_similarity(trans(lhs_proj, rell), ident_entity(rhs_proj, rell))
            simin = l2_similarity(trans(lhs_proj_n, relln), ident_entity(rhs_proj_n, relln))

            kg_loss = max_margin(simi, simin) + 0.25 * (tf.reduce_sum(tf.sqrt(tf.reduce_sum(self.E ** 2, axis=1)) - 1) +
                      tf.reduce_sum(tf.batch_matmul(self.W, tf.transpose(self.R)) ** 2 /
                      tf.expand_dims(tf.sqrt(tf.reduce_sum(self.R ** 2, axis=1)), 1)))

            # Skipgram Model
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

            embed = tf.nn.embedding_lookup(self.E, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, self.embedding_size],
                                    stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32))))
            nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
            skipgram_loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size,
                               remove_accidental_hits=True))

            if skipgram:
                loss = kg_loss + skipgram_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
            else:
                loss = kg_loss
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = init_lr
            # tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.98, staircase=True)
            learning_rate = tf.constant(starter_learning_rate)
            grads_E = tf.reduce_mean(tf.gradients(loss, self.E)[0])
            grads_R = tf.reduce_mean(tf.gradients(loss, self.R)[0])

            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

            ranking_error_l = rank_left_fn_idx(l2_similarity, self.E, self.R, trans, ident_entity, test_inpr, test_inpo)
            ranking_error_r = rank_right_fn_idx(l2_similarity, self.E, self.R, trans, ident_entity, test_inpl, test_inpo)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            g_Es = 0
            g_Rs = 0
            eval_step_size = 10
            best_hits = -np.inf
            best_rank = np.inf
            mean_rank_list = []
            hits_10_list = []
            g_E_list = []
            g_R_list = []
            loss_list = []
            if store_embeddings:
                entity_embs = []
                relation_embs = []
            for b in range(1, num_steps + 1):
                batch_pos, batch_neg = tg.next()
                test_batch_pos, _ = test_tg.next()
                batch_x, batch_y = sg.next()
                batch_y = np.array(batch_y).reshape((self.batch_size_sg, 1))
                session.run([normalize_E, normalize_R, normalize_W])
                #session.run([normalize_W])
                # calculate valid indices for scoring
                feed_dict = {inpl: batch_pos[1, :], inpr: batch_pos[0, :], inpo: batch_pos[2, :],
                             inpln: batch_neg[1, :], inprn: batch_neg[0, :], inpon: batch_neg[2, :],
                             train_inputs: batch_x, train_labels: batch_y,
                             global_step : b
                             }
                _, l, g_E, g_R = session.run([optimizer, loss, grads_E, grads_R], feed_dict=feed_dict)
                print l, l.shape
                average_loss += l
                if b % eval_step_size == 0:
                    feed_dict = {test_inpl: test_batch_pos[1, :], test_inpo: test_batch_pos[2, :],
                                 test_inpr: test_batch_pos[0, :]}
                    scores_l, scores_r = session.run([ranking_error_l, ranking_error_r], feed_dict=feed_dict)

                    errl = []
                    errr = []
                    for i, (l, o, r) in enumerate(
                            zip(test_batch_pos[1, :], test_batch_pos[2, :], test_batch_pos[0, :])):
                        # find those triples that have <*,o,r> and * != l
                        rmv_idx_l = [l_rmv for (l_rmv, rel, rhs) in test_tg.all_triples if
                                     rel == o and r == rel and l_rmv != l]
                        # *l* is the correct index
                        scores_l[i, rmv_idx_l] = -np.inf
                        errl += [np.argsort(np.argsort(-scores_l[i, :]))[l] + 1]
                        # since index start at 0, best possible value is 1
                        rmv_idx_r = [r_rmv for (lhs, rel, r_rmv) in test_tg.all_triples if
                                     rel == o and lhs == l and r_rmv != r]
                        # *l* is the correct index
                        scores_r[i, rmv_idx_r] = -np.inf
                        errr += [np.argsort(np.argsort(-scores_r[i, :]))[r] + 1]

                    hits_10 = np.mean(np.asarray(errl + errr) <= 10) * 100
                    mean_rank = np.mean(np.asarray(errl + errr))
                    print "Hits10: ", hits_10
                    print "MeanRank: ", mean_rank

                    mean_rank_list.append(mean_rank)
                    hits_10_list.append(hits_10)

                    if best_hits < hits_10:
                        best_hits = hits_10
                    if best_rank > mean_rank:
                        best_rank = mean_rank

                    if b > 0:
                        average_loss = average_loss / eval_step_size
                        g_Es /= eval_step_size
                        g_Rs /= eval_step_size
                    loss_list.append(average_loss)
                    g_E_list.append(g_Es)
                    g_R_list.append(g_Rs)

                    if store_embeddings:
                        entity_embs.append(session.run(self.E))
                        relation_embs.append(session.run(self.R))

                    # The average loss is an estimate of the loss over the last eval_step_size batches.
                    print('Average loss at step %d: %f' % (b, average_loss))
                    average_loss = 0
                    g_Es = 0
                    g_Rs = 0
                if not store_embeddings:
                    entity_embs = [session.run(self.E)]
                    relation_embs = [session.run(self.R)]
            return entity_embs, relation_embs, best_hits, best_rank, mean_rank_list, hits_10_list, g_E_list, g_R_list, loss_list