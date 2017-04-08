import tensorflow as tf
import numpy as np
from model import l2_similarity, dot, trans, ident_entity, max_margin, dot_similarity, skipgram_loss, ranking_error_triples
import pickle


class TransESeq(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, leftop, rightop, fnsim, sub_prop_constr=None,
                 init_lr=1.0, skipgram=True, lambd=None):
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
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.batch_size_kg = batch_size_kg
        self.batch_size_sg = batch_size_sg
        self.leftop = leftop
        self.rightop = rightop
        self.fnsim = fnsim
        self.sub_prop_constr = sub_prop_constr
        self.init_lr = init_lr
        self.skipgram = skipgram
        self.lambd = lambd

    def rank_left_idx(self, test_inpr, test_o, test_w, ent_embs, v_embs):
        lhs = ent_embs
        rell = test_o
        rhs = ent_embs[test_inpr]
        wr = test_w
        result = np.zeros((rhs.shape[0], lhs.shape[0]))
        for i in xrange(rhs.shape[0]):
            proj_rhs = rhs[i] + np.dot(v_embs[i], np.transpose(wr[i]))
            #proj_rhs = (rhs[i] + np.dot(v_embs[i], np.transpose(wr[i])) * wr[i]) / 2
            for j in xrange(lhs.shape[0]):
                proj_lhs = lhs[j] + np.dot(v_embs[j], np.transpose(wr[i]))
                #proj_lhs = (lhs[j] + np.dot(v_embs[j], np.transpose(wr[i])) * wr[i]) / 2
                temp_diff = (proj_lhs + rell[i]) - proj_rhs
                result[i][j] = -np.sqrt(np.sum(temp_diff ** 2))
        return result

    def rank_right_idx(self, test_inpl, test_o, test_w, ent_embs, v_embs):
        rhs = ent_embs
        rell = test_o
        lhs = ent_embs[test_inpl]
        wr = test_w
        result = np.zeros((lhs.shape[0], rhs.shape[0]))
        for i in xrange(lhs.shape[0]):
            proj_lhs = lhs[i] + np.dot(v_embs[i], np.transpose(wr[i]))
            #proj_lhs = (lhs[i] + np.dot(v_embs[i], np.transpose(wr[i])) * wr[i]) / 2
            proj_lhs = proj_lhs + rell[i]
            for j in xrange(rhs.shape[0]):
                proj_rhs = rhs[j] + np.dot(v_embs[j], np.transpose(wr[i]))
                #proj_rhs = (rhs[j] + np.dot(v_embs[j], np.transpose(wr[i])) * wr[i]) / 2
                temp_diff = proj_lhs - proj_rhs
                result[i][j] = -np.sqrt(np.sum(temp_diff ** 2))
        return result

    def create_graph(self):
        print('Building Model')
        # Translation Model initialisation
        w_bound = np.sqrt(6. / self.embedding_size)
        self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound))
        self.V = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound))
        # TODO: set V entries to 0-vector for unused ones
        self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                               maxval=w_bound))
        self.W = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size, self.embedding_size),
                                               minval=-w_bound,
                                               maxval=w_bound))
        # TODO: divide in to two matrices Wr_t and Wr_h (then we do not have to modife r)
        # i.e. decide if head or tail entity are influced by sequential representation in this relation

        self.normalize_E = self.E.assign(tf.nn.l2_normalize(self.E, 1))
        # TODO: do not normalize V, but <v, wr> as below
        # normalize_V = self.V.assign(tf.nn.l2_normalize(self.V, 1))
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

        wr = tf.nn.embedding_lookup(self.W, self.inpo)
        lhs = tf.nn.embedding_lookup(self.E, self.inpl)
        v_lhs = tf.expand_dims(tf.nn.embedding_lookup(self.V, self.inpl), 1)
        #v_lhs = tf.nn.embedding_lookup(self.V, inpl)
        lhs = lhs + tf.reduce_sum(tf.mul(v_lhs, wr), 2)
        #lhs = (lhs + dot(v_lhs, wr) * wr) / 2

        rhs = tf.nn.embedding_lookup(self.E, self.inpr)
        v_rhs = tf.expand_dims(tf.nn.embedding_lookup(self.V, self.inpr), 1)
        #v_rhs = tf.nn.embedding_lookup(self.V, inpr)
        rhs = rhs + tf.reduce_sum(tf.mul(v_rhs, wr), 2)
        #rhs = (rhs + dot(v_rhs, wr) * wr) / 2

        rell = tf.nn.embedding_lookup(self.R, self.inpo)
        relr = tf.nn.embedding_lookup(self.R, self.inpo)

        wrn = tf.nn.embedding_lookup(self.W, self.inpon)
        lhsn = tf.nn.embedding_lookup(self.E, self.inpln)
        v_lhsn = tf.expand_dims(tf.nn.embedding_lookup(self.V, self.inpln), 1)
        #v_lhsn = tf.nn.embedding_lookup(self.V, inpln)
        lhsn = lhsn + tf.reduce_sum(tf.mul(v_lhsn, wrn), 2)
        #lhsn = (lhsn + dot(v_lhsn, wrn) * wrn) / 2

        rhsn = tf.nn.embedding_lookup(self.E, self.inprn)
        v_rhsn = tf.expand_dims(tf.nn.embedding_lookup(self.V, self.inprn), 1)
        #v_rhsn = tf.nn.embedding_lookup(self.V, inprn)
        rhsn = rhsn + tf.reduce_sum(tf.mul(v_rhsn, wrn), 2)
        #rhsn = (rhsn + dot(v_rhsn, wrn) * wrn) / 2

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
        reg_l = tf.maximum(0., tf.reduce_sum(tf.sqrt(tf.reduce_sum(reg_l ** 2, axis=1)) - 1))
        reg_r = tf.maximum(0., tf.reduce_sum(tf.sqrt(tf.reduce_sum(reg_r ** 2, axis=1)) - 1))

        kg_loss = max_margin(simi, simin) + self.lambd * (reg_l + reg_r)

        if self.sub_prop_constr:
            sub_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sub"])
            sup_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sup"])
            kg_loss += tf.reduce_sum(dot(sub_relations, sup_relations) - 1)

        # Skipgram Model
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

        # In this model we put the extra embeddings into skipgram, not E
        sg_embed = tf.nn.embedding_lookup(self.V, self.train_inputs)

        sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size, self.train_labels)

        if self.skipgram:
            self.loss = kg_loss + sg_loss
        else:
            self.loss = kg_loss
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.init_lr
        learning_rate = tf.constant(starter_learning_rate)

        self.optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(self.loss)

        self.ranking_test_inpo = tf.nn.embedding_lookup(self.R, self.test_inpo)
        self.ranking_test_inpw = tf.nn.embedding_lookup(self.W, self.test_inpo)

    def post_ops(self):
        return [self.normalize_E, self.normalize_R]

    def train(self):
        return [self.optimizer, self.loss]

    def assign_initial(self, init_embeddings):
        return self.V.assign(init_embeddings)

    def variables(self):
        return [self.E, self.R, self.W, self.V]