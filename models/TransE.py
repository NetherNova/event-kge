import tensorflow as tf
import numpy as np
from models.model import dot_similarity, dot, max_margin, skipgram_loss, cnn_loss, concat_window_loss, rnn_loss, trans, ident_entity
from event_models.LinearEventModel import Skipgram


class TransE(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, fnsim, init_lr=1.0, event_layer=None):
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
        self.leftop = trans
        self.rightop = ident_entity
        self.fnsim = fnsim
        self.init_lr = init_lr
        self.event_layer = event_layer

    def rank_left_idx(self, test_inpr, test_inpo, r_embs, ent_embs):
        lhs = ent_embs
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        rhs = ent_embs[test_inpr]
        unique_lhs = lhs[:, np.newaxis] + unique_rell
        results = np.zeros((len(test_inpr), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            rhs_inds = np.argwhere(test_inpo == i)[:,0]
            tmp_lhs = unique_lhs[:, r, :]
            results[rhs_inds] = -np.square(tmp_lhs[:, np.newaxis] - rhs[rhs_inds]).sum(axis=2).transpose()
        return results

    def rank_right_idx(self, test_inpl, test_inpo, r_embs, ent_embs):
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

        self.inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
        self.inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
        self.inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

        self.inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhsn")
        self.inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhsn")
        self.inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="relln")

        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        rell = tf.nn.embedding_lookup(self.R, self.inpo)

        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        relln = tf.nn.embedding_lookup(self.R, self.inpon)

        if self.event_layer is not None:
            self.event_layer.create_graph()
            if not self.event_layer.shared:
                # self.a = tf.Variable(tf.random_uniform([self.embedding_size], minval=-w_bound,
                #                                    maxval=w_bound), name="a")
                # self.b = tf.Variable(tf.random_uniform([self.embedding_size], minval=-w_bound,
                #                                        maxval=w_bound), name="b")
                # lhs = tf.multiply(self.a, lhs) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                #                                                                             self.inpl))
                # rhs = tf.multiply(self.a, rhs) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                #                                                                             self.inpr))
                #
                # lhsn = tf.multiply(self.a, lhsn) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                #                                                                               self.inpln))
                # rhsn = tf.multiply(self.a, rhsn) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                #                                                                               self.inprn))

                # Option 2
                self.a = tf.Variable(tf.random_uniform([self.num_relations, self.embedding_size], minval=-w_bound,
                                                       maxval=w_bound), name="a")
                self.b = tf.Variable(tf.random_uniform([self.num_relations, self.embedding_size], minval=-w_bound,
                                                       maxval=w_bound), name="b")

                a_r = tf.nn.embedding_lookup(self.a, self.inpo)
                b_r = tf.nn.embedding_lookup(self.b, self.inpo)

                lhs = tf.multiply(a_r, lhs) + tf.multiply(b_r, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                                self.inpl))
                rhs = tf.multiply(a_r, rhs) + tf.multiply(b_r, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                                self.inpr))

                lhsn = tf.multiply(a_r, lhsn) + tf.multiply(b_r, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                                  self.inpln))
                rhsn = tf.multiply(a_r, rhsn) + tf.multiply(b_r, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                                  self.inprn))
                # more flexible connections

        if self.fnsim == dot_similarity:
            simi = tf.diag_part(self.fnsim(self.leftop(lhs, rell), tf.transpose(self.rightop(rhs, rell)),
                                           broadcast=False))
            simin = tf.diag_part(self.fnsim(self.leftop(lhsn, rell), tf.transpose(self.rightop(rhsn, rell)),
                                            broadcast=False))
        else:
            simi = self.fnsim(self.leftop(lhs, rell), self.rightop(rhs, rell), broadcast=False)
            simin = self.fnsim(self.leftop(lhsn, relln), self.rightop(rhsn, relln), broadcast=False)

        kg_loss = max_margin(simi, simin)
        self.loss = kg_loss

        if self.event_layer is not None:
            if type(self.event_layer) == Skipgram:
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            else:
                self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, None])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            if not self.event_layer.shared:
                self.loss += self.event_layer.alpha * self.event_layer.loss(self.num_sampled, self.train_labels,
                                                                            self.train_inputs, embeddings=None)
            else:
                self.loss += self.event_layer.alpha * self.event_layer.loss(self.num_sampled, self.train_labels,
                                                                            self.train_inputs, embeddings=self.E)

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.init_lr

        learning_rate = tf.constant(starter_learning_rate)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

    def assign_initial(self, init_embeddings):
        if self.event_layer and not self.event_layer.shared:
            return self.event_layer.V.assign(init_embeddings)
        else:
            return self.E.assign(init_embeddings)

    def post_ops(self):
        return [self.normalize_E]

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        vars = [self.E, self.R]
        if self.event_layer is not None:
            vars += self.event_layer.variables()
            if not self.event_layer.shared:
                vars += [self.a, self.b]
        return vars

    def scores(self, session, inpl, inpr, inpo):
        # need to get embeddings out of tf into python numpy
        r_embs, embs = session.run([self.R, self.E], feed_dict={})
        if self.event_layer is not None and not self.event_layer.shared:
            a, b, v_embs = session.run([self.a, self.b, self.event_layer.V], feed_dict={})
            embs = np.multiply(embs, a) + np.multiply(v_embs, b)
        scores_l = self.rank_left_idx(inpr, inpo, r_embs, embs)
        scores_r = self.rank_right_idx(inpl, inpo, r_embs, embs)
        return scores_l, scores_r