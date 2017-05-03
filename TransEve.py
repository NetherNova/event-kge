import tensorflow as tf
import numpy as np
from model import dot, max_margin, dot_similarity, skipgram_loss, lstm_loss, concat_window_loss


class TransEve(object):
    def __init__(self, num_entities, num_relations, embedding_size, seq_embeddings_size, batch_size_kg, batch_size_sg,
                 num_sampled, vocab_size, leftop, rightop, fnsim, zero_elements, init_lr=1.0,
                 event_layer="Skipgram", lambd=None, subclass_constr=None, num_sequences=None, num_events=None):
        """
        TransE plus linear transformation of sequential embeddings
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
        self.seq_embeddings_size = seq_embeddings_size
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.batch_size_kg = batch_size_kg
        self.batch_size_sg = batch_size_sg
        self.leftop = leftop
        self.rightop = rightop
        self.fnsim = fnsim
        self.zero_elements = zero_elements
        self.init_lr = init_lr
        self.event_layer = event_layer
        self.lambd = lambd
        self.subclass_constr = subclass_constr
        self.num_sequences = num_sequences
        self.num_events = num_events

    def rank_left_idx(self, test_inpr, test_inpo, r_embs, ent_embs, w_embs, v_embs):
        lhs = ent_embs
        rhs = ent_embs[test_inpr]
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        unique_wr = w_embs[unique_inpo]
        results = np.zeros((len(test_inpr), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            rhs_inds = np.argwhere(test_inpo == i)[:, 0]
            proj_lhs = lhs + v_embs.dot(unique_wr[r].transpose())
            proj_rhs = rhs[rhs_inds] + v_embs[rhs_inds].dot(unique_wr[r].transpose())
            lhs_tmp = (proj_lhs + unique_rell[r])
            results[rhs_inds] = -np.square(lhs_tmp[:, np.newaxis] - proj_rhs).sum(axis=2).transpose()
        return results

    def rank_right_idx(self, test_inpl, test_inpo, r_embs, ent_embs, w_embs, v_embs):
        rhs = ent_embs
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        lhs = ent_embs[test_inpl]
        unique_wr = w_embs[unique_inpo]
        results = np.zeros((len(test_inpl), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            lhs_inds = np.argwhere(test_inpo == i)[:, 0]
            proj_lhs = lhs[lhs_inds] + v_embs[lhs_inds].dot(unique_wr[r].transpose())
            proj_rhs = rhs + v_embs.dot(unique_wr[r].transpose())
            lhs_tmp = (proj_lhs + unique_rell[r])
            results[lhs_inds] = -np.square(lhs_tmp - proj_rhs[:, np.newaxis]).sum(axis=2).transpose()
        return results

    def create_graph(self):
        print('Building Model')
        # Translation Model initialisation
        w_bound = np.sqrt(6. / self.embedding_size)
        self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound))
        self.V = tf.Variable(tf.random_uniform((self.num_entities, self.seq_embeddings_size), minval=-w_bound,
                                               maxval=w_bound), trainable=False)
        # TODO: set V entries to 0-vector for unused ones
        self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound))
        self.W = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size, self.seq_embeddings_size),
                                               minval=-w_bound,
                                               maxval=w_bound))
        # TODO: divide in to two matrices Wr_t and Wr_h (then we do not have to modife r)
        # i.e. decide if head or tail entity are influced by sequential representation in this relation

        self.normalize_E = self.E.assign(tf.nn.l2_normalize(self.E, 1))
        # TODO: do not normalize V, but <v, wr> as below
        # normalize_V = self.V.assign(tf.nn.l2_normalize(self.V, 1))
        self.normalize_R = self.R.assign(tf.nn.l2_normalize(self.R, 1))

        self.setzero_V = tf.scatter_update(self.V, self.zero_elements, tf.zeros((self.num_entities - self.vocab_size,
                                                                                 self.seq_embeddings_size)))

        self.inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
        self.inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
        self.inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

        self.inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhsn")
        self.inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhsn")
        self.inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="relln")

        self.test_inpr = tf.placeholder(tf.int32, [None], name="test_rhs")
        self.test_inpl = tf.placeholder(tf.int32, [None], name="test_lhs")
        self.test_inpo = tf.placeholder(tf.int32, [None], name="test_rell")

        wr = tf.nn.embedding_lookup(self.W, self.inpo)

        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        v_lhs = tf.expand_dims(tf.nn.embedding_lookup(self.V, self.inpl), 1)
        lhs = lhs + tf.reduce_sum(tf.mul(v_lhs, wr), 2)

        rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        v_rhs = tf.expand_dims(tf.nn.embedding_lookup(self.V, self.inpr), 1)
        rhs = rhs + tf.reduce_sum(tf.mul(v_rhs, wr), 2)

        rell = tf.nn.embedding_lookup(self.R, self.inpo)
        relr = tf.nn.embedding_lookup(self.R, self.inpo)

        wrn = tf.nn.embedding_lookup(self.W, self.inpon)

        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        v_lhsn = tf.expand_dims(tf.nn.embedding_lookup(self.V, self.inpln), 1)
        lhsn = lhsn + tf.reduce_sum(tf.mul(v_lhsn, wrn), 2)

        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        v_rhsn = tf.expand_dims(tf.nn.embedding_lookup(self.V, self.inprn), 1)
        rhsn = rhsn + tf.reduce_sum(tf.mul(v_rhsn, wrn), 2)

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

        reg_l = tf.reduce_sum(tf.mul(v_lhs, wr), 2)
        reg_r = tf.reduce_sum(tf.mul(v_rhs, wr), 2)

        #reg_l = tf.maximum(0., tf.reduce_sum(tf.reduce_sum(reg_l ** 2, axis=1) - 1))
        #reg_r = tf.maximum(0., tf.reduce_sum(tf.reduce_sum(reg_r ** 2, axis=1) - 1))

        kg_loss = max_margin(simi, simin) + self.lambd * (tf.nn.l2_loss(reg_l) + tf.nn.l2_loss(reg_r))

        self.loss = kg_loss

        mu = tf.constant(1.0)

        if len(self.subclass_constr) > 0:
            subclass_types = tf.nn.embedding_lookup(self.E, self.subclass_constr[:,0])
            supclass_types = tf.nn.embedding_lookup(self.E, self.subclass_constr[:,1])
            self.loss += tf.maximum(0., 1 - tf.reduce_sum(dot(subclass_types, supclass_types)))

        if self.event_layer == "Skipgram":
            # Skipgram Model
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            sg_embed = tf.nn.embedding_lookup(self.V, self.train_inputs)
            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size,
                                    self.train_labels)
            self.loss += mu * sg_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
        elif self.event_layer == "LSTM":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, self.num_events]) # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            embed = tf.nn.embedding_lookup(self.V, self.train_inputs)
            concat_loss = lstm_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size, self.train_labels)
            self.loss += mu * concat_loss
        elif self.event_layer == "Concat":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, self.num_events])  # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 2])
            embed = tf.nn.embedding_lookup(self.V, self.train_inputs)
            concat_loss = concat_window_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size,
                                             self.train_labels, self.num_sequences)
            self.loss += mu * concat_loss
        else:
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.init_lr
        learning_rate = tf.constant(starter_learning_rate)

        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

        self.ranking_test_inpo = tf.nn.embedding_lookup(self.R, self.test_inpo)
        self.ranking_test_inpw = tf.nn.embedding_lookup(self.W, self.test_inpo)

    def post_ops(self):
        return [self.normalize_E, self.normalize_R, self.setzero_V]

    def train(self):
        return [self.optimizer, self.loss]

    def assign_initial(self, init_embeddings):
        return self.V.assign(init_embeddings)

    def variables(self):
        return [self.E, self.R, self.W, self.V]