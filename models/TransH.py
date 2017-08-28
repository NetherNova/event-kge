import tensorflow as tf
import numpy as np
from models.model import l2_similarity, dot, trans, ident_entity, max_margin, skipgram_loss, ranking_error_triples
from event_models.Skipgram import Skipgram


class TransH(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, init_lr=1.0, skipgram=True, lambd=None, alpha=1.0):
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
        self.init_lr = init_lr
        self.skipgram = skipgram
        self.lambd = lambd
        self.alpha = alpha

    def rank_left_idx(self, test_inpr, test_inpo, r_embs, ent_embs, w_embs):
        lhs = ent_embs # [num_entities, d]
        rhs = ent_embs[test_inpr]  # [num_test, d]
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        unique_wr = w_embs[unique_inpo]
        results = np.zeros((len(test_inpr), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            rhs_inds = np.argwhere(test_inpo == i)[:, 0]
            proj_lhs = lhs - lhs.dot(unique_wr[r].transpose())[:, np.newaxis] * unique_wr[r]
            proj_rhs = rhs[rhs_inds] - rhs[rhs_inds].dot(unique_wr[r].transpose())[:, np.newaxis] * unique_wr[r]
            lhs_tmp = (proj_lhs + unique_rell[r])
            results[rhs_inds] = -np.square(lhs_tmp[:, np.newaxis] - proj_rhs).sum(axis=2).transpose()
        return results

    def rank_right_idx(self, test_inpl, test_inpo, r_embs, ent_embs, w_embs):
        rhs = ent_embs  # [num_entities, d]
        lhs = ent_embs[test_inpl]  # [num_test, d]
        unique_inpo = np.unique(test_inpo)
        unique_rell = r_embs[unique_inpo]
        unique_wr = w_embs[unique_inpo]
        results = np.zeros((len(test_inpl), ent_embs.shape[0]))
        for r, i in enumerate(unique_inpo):
            lhs_inds = np.argwhere(test_inpo == i)[:, 0]
            proj_lhs = lhs[lhs_inds] - lhs[lhs_inds].dot(unique_wr[r].transpose())[:, np.newaxis] * unique_wr[r]
            proj_rhs = rhs - rhs.dot(unique_wr[r].transpose())[:, np.newaxis] * unique_wr[r]
            lhs_tmp = (proj_lhs + unique_rell[r])
            results[lhs_inds] = -np.square(lhs_tmp - proj_rhs[:, np.newaxis]).sum(axis=2).transpose()
        return results

    def create_graph(self):
        print('Building Model')
        # Translation Model initialisation
        w_bound = np.sqrt(6. / self.embedding_size)
        self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound))
        self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound))

        self.W = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound))

        self.normalize_W = self.W.assign(tf.nn.l2_normalize(self.W, 1))

        self.inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
        self.inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
        self.inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

        self.inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhsn")
        self.inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhsn")
        self.inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="relln")

        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        rell = tf.nn.embedding_lookup(self.R, self.inpo)
        wr = tf.nn.embedding_lookup(self.W, self.inpo)

        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        relln = tf.nn.embedding_lookup(self.R, self.inpon)
        wrn = tf.nn.embedding_lookup(self.W, self.inpon)

        if self.event_layer is not None:
            self.event_layer.create_graph()
            if not self.event_layer.shared:
                self.a = tf.Variable(tf.random_uniform([self.embedding_size], minval=-w_bound,
                                                   maxval=w_bound), name="a")
                self.b = tf.Variable(tf.random_uniform([self.embedding_size], minval=-w_bound,
                                                       maxval=w_bound), name="b")
                lhs = tf.multiply(self.a, lhs) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                            self.inpl))
                rhs = tf.multiply(self.a, rhs) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                            self.inpr))

                lhsn = tf.multiply(self.a, lhsn) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                              self.inpln))
                rhsn = tf.multiply(self.a, rhsn) + tf.multiply(self.b, tf.nn.embedding_lookup(self.event_layer.V,
                                                                                              self.inprn))

        lhs_proj = lhs - dot(lhs, wr) * wr  # dot and elementwise mul => projection
        rhs_proj = rhs - dot(rhs, wr) * wr

        lhs_proj_n = lhsn - dot(lhsn, wrn) * wrn
        rhs_proj_n = rhsn - dot(rhsn, wrn) * wrn

        simi = l2_similarity(trans(lhs_proj, rell), ident_entity(rhs_proj, rell))
        simin = l2_similarity(trans(lhs_proj_n, relln), ident_entity(rhs_proj_n, relln))

        # TransH Loss
        epsilon = tf.constant(0.0001)
        reg1 = tf.maximum(0., tf.reduce_sum(tf.sqrt(tf.reduce_sum(self.E ** 2, axis=1)) - 1))
        reg2_z = dot(self.W, self.R) ** 2
        reg2_n = tf.expand_dims(tf.sqrt(tf.reduce_sum(self.R ** 2, axis=1)), 1)
        reg2 = tf.reduce_sum(tf.maximum(0., (reg2_z / reg2_n) - epsilon))

        kg_loss = max_margin(simi, simin) + self.lambd * (reg1 + reg2)

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
        learning_rate = tf.constant(starter_learning_rate)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)


    def assign_initial(self, init_embeddings):
        return self.E.assign(init_embeddings)

    def post_ops(self):
        return [self.normalize_W]

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        vars = [self.E, self.R, self.W]
        if self.event_layer is not None:
            vars += self.event_layer.variables()
            if not self.event_layer.shared:
                vars += [self.a, self.b]
        return vars

    def scores(self, session, inpl, inpr, inpo):
        r_embs, embs, w_embs = session.run([self.R, self.E, self.W], feed_dict={})
        if self.event_layer is not None and not self.event_layer.shared:
            a, b, v_embs = session.run([self.a, self.b, self.event_layer.V], feed_dict={})
            embs = np.multiply(embs, a) + np.multiply(v_embs, b)
        scores_l = self.rank_left_idx(inpr, inpo, r_embs, embs, w_embs)
        scores_r = self.rank_right_idx(inpl, inpo, r_embs, embs, w_embs)
        return scores_l, scores_r
