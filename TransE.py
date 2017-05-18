import tensorflow as tf
import numpy as np
from model import dot_similarity, dot, max_margin, skipgram_loss, lstm_loss, concat_window_loss, rnn_loss


class TransE(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, leftop, rightop, fnsim, init_lr=1.0, event_layer="Skipgram",
                 lambd=None, subclass_constr=None, num_sequences=None, num_events=None, alpha=1.0):
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
        self.init_lr = init_lr
        self.event_layer = event_layer
        self.subclass_constr = subclass_constr
        self.num_sequences = num_sequences
        self.num_events = num_events
        self.len_hierarchy = 7
        self.lambd = lambd
        self.alpha = alpha

    def rank_left_idx_new(self, test_inpr, test_inpo, r_embs, ent_embs, w, test_inpc):
        lhs = ent_embs
        results = np.zeros((len(test_inpr), ent_embs.shape[0]))
        for i, r in enumerate(test_inpo):
            rell = r_embs[r]
            # lhs_ext = lhs[:, np.newaxis] + ((test_inpc[i] + 1) * w * rell)  # test_inpc[i] + w * rell
            lhs_ext = lhs[:, np.newaxis] + ((test_inpc[i] * w + 1) * rell)
            rhs = ent_embs[test_inpr[i]]
            results[i] = -np.square(lhs_ext - rhs).sum(axis=2).transpose()
        return results

    def rank_right_idx_new(self, test_inpl, test_inpo, r_embs, ent_embs, w, test_inpc):
        rhs = ent_embs
        results = np.zeros((len(test_inpl), ent_embs.shape[0]))
        for i, r in enumerate(test_inpo):
            rell = r_embs[r]
            lhs = ent_embs[test_inpl[i]]

            if test_inpc[i] != 0:
                max_score = -np.inf
                for j in [1,2,3,4]:
                    lhs = lhs + (j * w + 1) * rell
                    # lhs = lhs + ((j + 1) * w) * rell
                    scores = -np.square(lhs - rhs[:, np.newaxis]).sum(axis=2).transpose()
                    if np.max(scores) > max_score:
                        results[i] = scores
                        max_score = np.max(scores)
            else:
                lhs = lhs + 1 * rell
                results[i] = -np.square(lhs - rhs[:, np.newaxis]).sum(axis=2).transpose()
        return results

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
        # type vector w
        self.w = tf.Variable(tf.random_uniform((1,), minval=0.01,
                                               maxval=1.0), name="w", dtype=tf.float32)
        # constant range of hierarchy depth
        self.C = tf.constant(range(1, self.len_hierarchy), dtype=tf.int32)

        self.normalize_E = self.E.assign(tf.nn.l2_normalize(self.E, 1))
        self.normalize_R = self.R.assign(tf.nn.l2_normalize(self.R, 1))

        self.inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
        self.inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
        self.inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")
        self.inpc = tf.placeholder(tf.float32, [self.batch_size_kg], name="c")

        self.inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhsn")
        self.inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhsn")
        self.inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="relln")
        self.inpcn = tf.placeholder(tf.float32, [self.batch_size_kg], name="cn")

        self.test_inpr = tf.placeholder(tf.int32, [None], name="test_rhs")
        self.test_inpl = tf.placeholder(tf.int32, [None], name="test_lhs")
        self.test_inpo = tf.placeholder(tf.int32, [None], name="test_rell")
        self.test_inpc = tf.placeholder(tf.float32, [None], name="test_c")

        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        if len(self.subclass_constr) > 0:
            # rell = tf.mul(tf.expand_dims((self.inpc + tf.ones_like(self.inpc)) * self.w, 1), tf.nn.embedding_lookup(self.R, self.inpo))
            rell = tf.mul(tf.expand_dims((self.inpc * self.w + tf.ones_like(self.inpc)), 1), tf.nn.embedding_lookup(self.R, self.inpo))
        else:
            rell = tf.nn.embedding_lookup(self.R, self.inpo)
        #relr = tf.nn.embedding_lookup(self.R, self.inpo)

        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        if len(self.subclass_constr) > 0:
            # relln = tf.mul(tf.expand_dims((self.inpcn + tf.ones_like(self.inpcn)) * self.w, 1), tf.nn.embedding_lookup(self.R, self.inpon))
            relln = tf.mul(tf.expand_dims((self.inpcn * self.w + tf.ones_like(self.inpcn)), 1), tf.nn.embedding_lookup(self.R, self.inpon))
        else:
            relln = tf.nn.embedding_lookup(self.R, self.inpon)

        if self.fnsim == dot_similarity:
            simi = tf.diag_part(self.fnsim(self.leftop(lhs, rell), tf.transpose(self.rightop(rhs, rell)),
                                           broadcast=False))
            simin = tf.diag_part(self.fnsim(self.leftop(lhsn, rell), tf.transpose(self.rightop(rhsn, rell)),
                                            broadcast=False))
        else:
            simi = self.fnsim(self.leftop(lhs, rell), self.rightop(rhs, rell), broadcast=False)
            simin = self.fnsim(self.leftop(lhsn, relln), self.rightop(rhsn, relln), broadcast=False)

        kg_loss = max_margin(simi, simin)

        # constraint on distance of subclasses
        if len(self.subclass_constr) > 0:
            subclass_types = tf.nn.embedding_lookup(self.E, self.subclass_constr[:, 0])
            supclass_types = tf.nn.embedding_lookup(self.E, self.subclass_constr[:, 1])
            kg_loss += self.lambd * tf.abs(tf.nn.l2_loss(subclass_types - supclass_types) - self.w)

        """
        if len(self.subclass_constr) > 0:
            subclass_types = tf.nn.embedding_lookup(self.E, self.subclass_constr[:,0])
            supclass_types = tf.nn.embedding_lookup(self.E, self.subclass_constr[:,1])
            kg_loss += tf.maximum(0., 1 - tf.reduce_sum(dot(subclass_types, supclass_types)))
        """
        self.loss = kg_loss

        if self.event_layer == "Skipgram":
            # Skipgram Model
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            sg_embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size,
                                    self.train_labels)
            self.loss += self.alpha * sg_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
        elif self.event_layer == "LSTM":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, self.num_events]) # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = lstm_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size, self.train_labels)
            self.loss += self.alpha * concat_loss
        elif self.event_layer == "RNN":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, self.num_events]) # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = rnn_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size, self.train_labels)
            self.loss += self.alpha * concat_loss
        elif self.event_layer == "Concat":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, self.num_events])  # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 2])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = concat_window_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size,
                                             self.train_labels, self.num_sequences)
            self.loss += self.alpha * concat_loss
        else:
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.init_lr

        learning_rate = tf.constant(starter_learning_rate)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

    def assign_initial(self, init_embeddings):
        return self.E.assign(init_embeddings)

    def post_ops(self):
        return [self.normalize_E, self.normalize_R]

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        return [self.E, self.R, self.w]