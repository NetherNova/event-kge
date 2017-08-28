import tensorflow as tf
import numpy as np
from models.model import max_margin
from scipy.special import expit
from event_models.Skipgram import Skipgram


class RESCAL(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, init_lr=1.0, event_layer=None, lambd=None):
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
        self.init_lr = init_lr
        self.lambd = lambd
        self.event_layer = event_layer

    def rank_left_idx(self, test_inpr, test_inpo, r_embs, ent_embs):
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

    def rank_right_idx(self, test_inpl, test_inpo, r_embs, ent_embs):
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

        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        rell = tf.nn.embedding_lookup(self.R, self.inpo)

        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        relln = tf.nn.embedding_lookup(self.R, self.inpon)

        # RESCAL with ranking loss
        lhs = tf.expand_dims(lhs, 1) # [batch, 1, d] # zeile mal zeile
        lhs = tf.reduce_sum(tf.multiply(lhs, rell), 2) # [batch, 1, d] * [batch, d, d] ==> [batch, d]

        lhsn = tf.expand_dims(lhsn, 1)
        lhsn = tf.reduce_sum(tf.multiply(lhsn, relln), 2)

        if self.event_layer is not None:
            self.event_layer.create_graph()
            if not self.event_layer.shared:
                self.a = tf.Variable(tf.random_uniform([self.embedding_size], minval=-self.w_bound,
                                                   maxval=self.w_bound), name="a")
                self.b = tf.Variable(tf.random_uniform([self.embedding_size], minval=-self.w_bound,
                                                       maxval=self.w_bound), name="b")
                lhs = tf.multiply(self.a, lhs) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                            self.inpl))
                rhs = tf.multiply(self.a, rhs) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                            self.inpr))

                lhsn = tf.multiply(self.a, lhsn) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                              self.inpln))
                rhsn = tf.multiply(self.a, rhsn) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                              self.inprn))

        simi = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(lhs, rhs), 1)) # [batch, d] * [batch, d]
        simin = tf.nn.sigmoid(tf.reduce_sum(tf.multiply(lhsn, rhsn), 1))

        kg_loss = max_margin(simi, simin) + self.lambd * (tf.nn.l2_loss(self.E) + tf.nn.l2_loss(self.R))

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
        else:
            # Dummy Inputs
            self.train_inputs = tf.placeholder(tf.int32, shape=[None])
            self.train_labels = tf.placeholder(tf.int32, shape=[None, 1])

        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.init_lr
        # tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.98, staircase=True)
        learning_rate = tf.constant(starter_learning_rate)

        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

    def assign_initial(self, init_embeddings):
        return self.E.assign(init_embeddings)

    def post_ops(self):
        return []

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        vars = [self.E, self.R]
        if self.event_layer is not None:
            vars += self.event_layer.variables()
            if not self.event_layer.shared:
                vars += vars + [self.a, self.b]
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