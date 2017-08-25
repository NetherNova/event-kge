import tensorflow as tf
import numpy as np
from models.model import l2_similarity, dot, trans, ident_entity, max_margin, skipgram_loss, ranking_error_triples


class TransEveR(object):
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
        # e = |h + r - t|
        # cross-entropy(y, softmax(W e))
        #
        # TEKE h = A cooc + h
        # r = B cooc + r --> we dont have that since there are no triples between two events!!

        # SSP
        # project loss vector e to "semantic subspace" s
        #

        # learn two representations of event vectors "from" and "to"
        # [e1, e2, e3]
        # a) prediction pairs (e1 -> e2), (e1 -> e3) "from the cause predict all effects" We1
        # b) prediction pairs (e3 -> e1), (e2 -> e1) "from the effect, predict all causes" We2
        # two-way LSTM (back and forth)


        # combine
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

        self.test_inpr = tf.placeholder(tf.int32, [None], name="test_rhs")
        self.test_inpl = tf.placeholder(tf.int32, [None], name="test_lhs")
        self.test_inpo = tf.placeholder(tf.int32, [None], name="test_rell")

        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        rell = tf.nn.embedding_lookup(self.R, self.inpo)
        wr = tf.nn.embedding_lookup(self.W, self.inpo)

        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        relln = tf.nn.embedding_lookup(self.R, self.inpon)
        wrn = tf.nn.embedding_lookup(self.W, self.inpon)

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

        # Skipgram Model
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

        sg_embed = tf.nn.embedding_lookup(self.E, self.train_inputs)

        if self.skipgram:
            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size, self.train_labels)
            self.loss = kg_loss + self.alpha * sg_loss
        else:
            self.loss = kg_loss
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.init_lr
        learning_rate = tf.constant(starter_learning_rate)
        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

        self.ranking_test_inpo = tf.nn.embedding_lookup(self.R, self.test_inpo)
        self.ranking_test_inpw = tf.nn.embedding_lookup(self.W, self.test_inpo)

    def assign_initial(self, init_embeddings):
        return self.E.assign(init_embeddings)

    def post_ops(self):
        return [self.normalize_W]

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        return [self.E, self.R, self.W]
