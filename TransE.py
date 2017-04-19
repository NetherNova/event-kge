import tensorflow as tf
import numpy as np
from model import dot_similarity, dot, max_margin, skipgram_loss, lstm_loss, concat_window_loss


class TransE(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, leftop, rightop, fnsim, sub_prop_constr=None, init_lr=1.0, event_layer="Skipgram",
                 lambd=None, subclass_constr=None, num_sequences=None):
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
        self.sub_prop_constr = sub_prop_constr
        self.init_lr = init_lr
        self.event_layer = event_layer
        self.subclass_constr = subclass_constr
        self.num_sequences = num_sequences

    def rank_left_idx(self, test_inpr, test_inpo, r_embs, ent_embs, cache=True):
        lhs = ent_embs
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        rhs = ent_embs[test_inpr]
        unique_lhs = lhs[:, np.newaxis] + unique_rell   # TODO: move inside loop for less memory consumption
        results = np.zeros((len(test_inpr), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            rhs_inds = np.argwhere(test_inpo == i)[:,0]
            tmp_lhs = unique_lhs[:, r, :]
            results[rhs_inds] = -np.square(tmp_lhs[:, np.newaxis] - rhs[rhs_inds]).sum(axis=2).transpose()
        return results

    def rank_right_idx(self, test_inpl, test_inpo, r_embs, ent_embs, cache=True):
        rhs = ent_embs
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        unique_rhs = unique_rell - rhs[:, np.newaxis]
        lhs = ent_embs[test_inpl]  # [num_test, d]
        results = np.zeros((len(test_inpl), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            lhs_inds = np.argwhere(test_inpo == i)[:, 0]
            tmp_rhs = unique_rhs[:, r, :]
            results[lhs_inds] = -np.square(lhs[lhs_inds] + tmp_rhs[:,np.newaxis]).sum(axis=2).transpose()
        return results

    def create_graph(self):
        print('Building Model')
        # Translation Model initialisation
        w_bound = np.sqrt(6. / self.embedding_size)
        self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound), name="E")
        self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound), name="R")

        self.normalize_E = self.E.assign(tf.nn.l2_normalize(self.E, 1))
        self.normalize_R = self.R.assign(tf.nn.l2_normalize(self.R, 1))

        self.inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
        self.inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
        self.inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

        self.inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhsn")
        self.inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhsn")
        self.inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="relln")

        self.test_inpr = tf.placeholder(tf.int32, [None], name="test_rhs")
        self.test_inpl = tf.placeholder(tf.int32, [None], name="test_lhs")
        self.test_inpo = tf.placeholder(tf.int32, [None], name="test_rell")

        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        rell = tf.nn.embedding_lookup(self.R, self.inpo)
        relr = tf.nn.embedding_lookup(self.R, self.inpo)

        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        relln = tf.nn.embedding_lookup(self.R, self.inpon)
        relrn = tf.nn.embedding_lookup(self.R, self.inpon)

        if self.fnsim == dot_similarity:
            simi = tf.diag_part(self.fnsim(self.leftop(lhs, rell), tf.transpose(self.rightop(rhs, relr)),
                                           broadcast=False))
            simin = tf.diag_part(self.fnsim(self.leftop(lhsn, relln), tf.transpose(self.rightop(rhsn, relrn)),
                                            broadcast=False))
        else:
            simi = self.fnsim(self.leftop(lhs, rell), self.rightop(rhs, relr), broadcast=False)
            simin = self.fnsim(self.leftop(lhsn, relln), self.rightop(rhsn, relrn), broadcast=False)

        kg_loss = max_margin(simi, simin)

        if self.sub_prop_constr:
            sub_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sub"])
            sup_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sup"])
            kg_loss += tf.reduce_sum(dot(sub_relations, sup_relations) - 1)

        if len(self.subclass_constr) > 0:
            subclass_types = tf.nn.embedding_lookup(self.E, self.subclass_constr[:,0])
            supclass_types = tf.nn.embedding_lookup(self.E, self.subclass_constr[:,1])
            kg_loss += tf.maximum(0., 1 - tf.reduce_sum(dot(subclass_types, supclass_types)))

        self.loss = kg_loss

        mu = tf.constant(0.5)

        if self.event_layer == "Skipgram":
            # Skipgram Model
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            sg_embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size,
                                    self.train_labels)
            self.loss += mu * sg_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
        elif self.event_layer == "LSTM":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 10]) # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = lstm_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size, self.train_labels)
            self.loss += mu * concat_loss
        elif self.event_layer == "Concat":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 10])  # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 2])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = concat_window_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size,
                                             self.train_labels, self.num_sequences)
            self.loss += mu * concat_loss
        else:
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.init_lr
        # tf.train.exponential_decay(starter_learning_rate, self.global_step, 10, 0.98, staircase=True)
        learning_rate = tf.constant(starter_learning_rate)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

    def assign_initial(self, init_embeddings):
        return self.E.assign(init_embeddings)

    def post_ops(self):
        return [self.normalize_E, self.normalize_R]

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        return [self.E, self.R]