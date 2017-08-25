import tensorflow as tf
import numpy as np
from models.model import dot_similarity, dot, max_margin, skipgram_loss, cnn_loss, concat_window_loss, rnn_loss, trans, ident_entity


class ProjE(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, fnsim, init_lr=1.0, event_layer="Skipgram", num_events=None, alpha=1.0):
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
        self.num_events = num_events
        self.alpha = alpha

    def rank_left_idx(self, test_inpr, test_inpo, r_embs, ent_embs, Dr, De, bc, bp):
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

    def rank_right_idx(self, test_inpl, test_inpo, r_embs, ent_embs, Dr, De, bc, bp):
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

        self.bias_p = tf.Variable(tf.random_uniform(self.num_entities), minval=-w_bound,
                                               maxval=w_bound), name="bp")

        self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound), name="R")

        self.Dr = tf.Variable(tf.random_uniform(self.embedding_size), minval=-w_bound,
                                               maxval=w_bound), name="Dr")

        self.De = tf.Variable(tf.random_uniform(self.embedding_size), minval=-w_bound,
                                               maxval=w_bound), name="De")

        self.bias_c = tf.Variable(tf.random_uniform(self.embedding_size), minval=-w_bound,
                                               maxval=w_bound), name="bc")

        self.normalize_E = self.E.assign(tf.nn.l2_normalize(self.E, 1))

        self.inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
        self.inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
        self.inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

        self.inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhsn")
        self.inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhsn")
        self.inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="relln")

        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        #rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        rell = tf.nn.embedding_lookup(self.R, self.inpo)
        #relr = tf.nn.embedding_lookup(self.R, self.inpo)

        #lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        #rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        #relln = tf.nn.embedding_lookup(self.R, self.inpon)

        # predict lhs + rell

        h_e_r = tf.tanh(tf.mul(lhs, self.De) + tf.mul(rell, self.Dr) + self.bias_c)

        kg_loss1 = tf.reduce_mean(
                            tf.nn.nce_loss(
                            weights=self.E,
                           biases=self.bias_p,
                           labels=self.inpr,
                           inputs=h_e_r,
                           num_sampled= int(self.num_entities * 0.25),
                           num_classes=self.num_entities,
                           remove_accidental_hits=True))

        # predict rhs + rell

        h_e_r2 = h_e_r = tf.tanh(tf.mul(lhs, self.De) + tf.mul(rell, self.Dr) + self.bias_c)

        kg_loss2 = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.E,
                biases=self.bias_p,
                labels=self.inpl,
                inputs=h_e_r2,
                num_sampled=int(self.num_entities * 0.25),
                num_classes=self.num_entities,
                remove_accidental_hits=True))

        self.loss = kg_loss1 + kg_loss2

        if self.event_layer == "Skipgram":
            # Skipgram Model
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            sg_embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size,
                                    self.train_labels)
            self.loss += self.alpha * sg_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
        elif self.event_layer == "CNN":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, self.num_events]) # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = cnn_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size, self.train_labels)
            self.loss += self.alpha * concat_loss
        elif self.event_layer == "RNN":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, self.num_events]) # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = rnn_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size, self.train_labels)
            self.loss += self.alpha * concat_loss
        elif self.event_layer == "Concat":
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg, self.num_events])  # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = concat_window_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size,
                                             self.train_labels)
            self.loss += self.alpha * concat_loss
        else:
            # dummy inputs
            self.train_inputs = tf.placeholder(tf.int32, shape=[None])
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.init_lr

        learning_rate = tf.constant(starter_learning_rate)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

    def assign_initial(self, init_embeddings):
        return self.E.assign(init_embeddings)

    def post_ops(self):
        return [self.normalize_E]

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        return [self.E, self.R]