import tensorflow as tf
import numpy as np
from model import dot, trans, ident_entity, max_margin, rank_right_fn_idx, rank_left_fn_idx, skipgram_loss, \
    rescal_similarity, lstm_loss, concat_window_loss
from scipy.special import expit


class RESCAL(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, sub_prop_constr=None, init_lr=1.0, event_layer="Skipgram", lambd=None,
                 subclass_constr=None, num_sequences=None, num_events=None):
        """
        RESCAL with max-margin loss (not Alternating least-squares)
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
        self.sub_prop_constr = sub_prop_constr
        self.init_lr = init_lr
        self.lambd = lambd
        self.event_layer = event_layer
        self.subclass_constr = subclass_constr
        self.num_sequences = num_sequences
        self.num_events = num_events

    def rank_left_idx(self, test_inpr, test_inpo, r_embs, ent_embs, cache=True):
        lhs = ent_embs
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        rhs = ent_embs[test_inpr]
        results = np.zeros((len(test_inpr), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            unique_lhs_tmp = lhs.dot(unique_rell[r,:,:].transpose())
            rhs_inds = np.argwhere(test_inpo == i)[:,0]
            results[rhs_inds] = expit(rhs[rhs_inds].dot(unique_lhs_tmp.transpose()))
        return results

    def rank_right_idx(self, test_inpl, test_inpo, r_embs, ent_embs, cache=True):
        rhs = ent_embs
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        lhs = ent_embs[test_inpl]  # [num_test, d]
        results = np.zeros((len(test_inpl), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            lhs_inds = np.argwhere(test_inpo == i)[:, 0]
            lhs_tmp = lhs[lhs_inds].dot(unique_rell[r,:,:].transpose())
            results[lhs_inds] = expit(lhs_tmp.dot(rhs.transpose()))
        return results

    def create_graph(self):
        print('Building Model')
        # Translation Model initialisation
        self.w_bound = np.sqrt(6. / self.embedding_size)
        self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-self.w_bound,
                                               maxval=self.w_bound), name="E")
        self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size, self.embedding_size),
                                               minval=-self.w_bound, maxval=self.w_bound), name="R")

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

        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        relln = tf.nn.embedding_lookup(self.R, self.inpon)

        # RESCAL with ranking loss
        lhs = tf.expand_dims(lhs, 1) # [batch, 1, d] # zeile mal zeile
        lhs = tf.reduce_sum(tf.mul(lhs, rell), 2)

        lhsn = tf.expand_dims(lhsn, 1)
        lhsn = tf.reduce_sum(tf.mul(lhsn, relln), 2)

        simi = tf.nn.sigmoid(tf.reduce_sum(tf.mul(lhs, rhs), 1))
        simin = tf.nn.sigmoid(tf.reduce_sum(tf.mul(lhsn, rhsn), 1))
        kg_loss = max_margin(simi, simin) + self.lambd * (tf.nn.l2_loss(self.E) + tf.nn.l2_loss(self.R))

        self.loss = kg_loss

        if self.sub_prop_constr:
            sub_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sub"])
            sup_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sup"])
            self.loss += tf.reduce_sum(dot(sub_relations, sup_relations) - 1)

        mu = tf.constant(1.0)

        if self.event_layer == "Skipgram":
            # Skipgram Model
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            sg_embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size,
                                    self.train_labels)
            self.loss += mu * sg_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
        elif self.event_layer == "LSTM":
            self.train_inputs = tf.placeholder(tf.int32,
                                               shape=[self.batch_size_sg, self.num_events])  # TODO: skip window size
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])
            embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
            concat_loss = lstm_loss(self.vocab_size, self.num_sampled, embed, self.embedding_size, self.train_labels)
            self.loss += mu * concat_loss
        elif self.event_layer == "Concat":
            self.train_inputs = tf.placeholder(tf.int32,
                                               shape=[self.batch_size_sg, self.num_events])  # TODO: skip window size
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
        # tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.98, staircase=True)
        learning_rate = tf.constant(starter_learning_rate)

        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

        self.ranking_error_l = rank_left_fn_idx(rescal_similarity, self.E, self.R, trans, ident_entity, self.test_inpr,
                                                self.test_inpo)
        self.ranking_error_r = rank_right_fn_idx(rescal_similarity, self.E, self.R, trans, ident_entity, self.test_inpl,
                                                 self.test_inpo)

    def assign_initial(self, init_embeddings):
        return self.E.assign(init_embeddings)

    def post_ops(self):
        return []

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        return [self.E, self.R]