import tensorflow as tf
import numpy as np
from model import l2_similarity, dot, trans, ident_entity, max_margin, dot_similarity, skipgram_loss, rank_triples_left_right
import pickle


class TransESeq(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, leftop, rightop, fnsim, supp_event_embeddings=None, sub_prop_constr=None):
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
        self.supp_event_embeddings = supp_event_embeddings
        self.sub_prop_constr = sub_prop_constr

    def rank_left_idx(self, test_inpr, test_o, test_w, ent_embs, v_embs):
        lhs = ent_embs  # [num_entities, d]
        rell = test_o  # [num_test, d]
        rhs = ent_embs[test_inpr]  # [num_test, d]
        wr = test_w
        result = np.zeros((rhs.shape[0], lhs.shape[0]))
        for i in xrange(rhs.shape[0]):
            proj_rhs = rhs[i] - np.dot(rhs[i], np.transpose(wr[i])) * wr[i]
            for j in xrange(lhs.shape[0]):
                proj_lhs = lhs[j] - np.dot(lhs[j], np.transpose(wr[i])) * wr[i]
                temp_diff = (proj_lhs + rell[i]) - proj_rhs
                result[i][j] = -np.sqrt(np.sum(temp_diff ** 2))
        return result

    def rank_right_idx(self, test_inpl, test_o, test_w, ent_embs, v_embs):
        rhs = ent_embs  # [num_entities, d]
        rell = test_o  # [num_test, d]
        lhs = ent_embs[test_inpl]  # [num_test, d]
        wr = test_w
        result = np.zeros((lhs.shape[0], rhs.shape[0]))
        for i in xrange(lhs.shape[0]):
            proj_lhs = lhs[i] - np.dot(lhs[i], np.transpose(wr[i])) * wr[i]
            proj_lhs = proj_lhs + rell[i]
            for j in xrange(rhs.shape[0]):
                proj_rhs = rhs[j] - np.dot(rhs[j], np.transpose(wr[i])) * wr[i]
                temp_diff = proj_lhs - proj_rhs
                result[i][j] = -np.sqrt(np.sum(temp_diff ** 2))
        return result

    def run(self, tg, sg, test_tg, test_size, num_steps, init_lr=1.0, skipgram=True, store_embeddings=False, lambd=None):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            # Translation Model initialisation
            w_bound = np.sqrt(6. / self.embedding_size)
            self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))
            self.V = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))
            self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))
            self.W = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))

            normalize_E = self.E.assign(tf.nn.l2_normalize(self.E, 1))
            normalize_V = self.V.assign(tf.nn.l2_normalize(self.V, 1))
            normalize_R = self.R.assign(tf.nn.l2_normalize(self.R, 1))

            inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            test_inpr = tf.placeholder(tf.int32, [test_size], name="test_rhs")
            test_inpl = tf.placeholder(tf.int32, [test_size], name="test_lhs")
            test_inpo = tf.placeholder(tf.int32, [test_size], name="test_rell")

            wr = tf.nn.embedding_lookup(self.W, inpo)
            lhs = tf.nn.embedding_lookup(self.E, inpl)
            lhs = lhs + dot(tf.nn.embedding_lookup(self.V, inpl), wr) * wr
            rhs = tf.nn.embedding_lookup(self.E, inpr)
            rhs = rhs + dot(tf.nn.embedding_lookup(self.V, inpr), wr) * wr
            rell = tf.nn.embedding_lookup(self.R, inpo)
            relr = tf.nn.embedding_lookup(self.R, inpo)

            wrn = tf.nn.embedding_lookup(self.W, inpon)
            lhsn = tf.nn.embedding_lookup(self.E, inpln)
            lhs = lhs + dot(tf.nn.embedding_lookup(self.V, inpl), wrn) * wrn
            rhsn = tf.nn.embedding_lookup(self.E, inprn)
            rhsn = rhsn + dot(tf.nn.embedding_lookup(self.V, inprn), wrn) * wrn
            relln = tf.nn.embedding_lookup(self.R, inpon)
            relrn = tf.nn.embedding_lookup(self.R, inpon)

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

            # Skipgram Model
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

            sg_embed = tf.nn.embedding_lookup(self.V, train_inputs)

            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size, train_labels)

            if skipgram:
                loss = kg_loss + sg_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
            else:
                loss = kg_loss
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = init_lr
            # tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.98, staircase=True)
            learning_rate = tf.constant(starter_learning_rate)
            # grads_E = tf.reduce_mean(tf.gradients(loss, self.E)[0])
            # grads_R = tf.reduce_mean(tf.gradients(loss, self.R)[0])

            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

            ranking_error_l = self.rank_left_idx(test_inpr, test_inpo)
            ranking_error_r = self.rank_right_idx(test_inpl, test_inpo)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            eval_step_size = 10
            best_hits = -np.inf
            best_rank = np.inf
            mean_rank_list = []
            hits_10_list = []
            loss_list = []

            # Initialize some / event entities with supplied embeddings
            if self.supp_event_embeddings:
                w_bound = np.sqrt(6. / self.embedding_size)
                initE = np.random.uniform((len(self.vocab_size), self.embedding_size), minval=-w_bound, maxval=w_bound)
                print("Load supplied embeddings")
                with open(self.self.supp_event_embeddings, "rb") as f:
                    supplied_embeddings = pickle.load(f)
                    supplied_dict = supplied_embeddings.get_dictionary()
                    for word, id in supplied_dict.iteritems():
                        initE[id] = supplied_embeddings.get_embeddings()[id]
                session.run(self.E.assign(initE))

            if store_embeddings:
                entity_embs = []
                relation_embs = []
            for b in range(1, num_steps + 1):
                batch_pos, batch_neg = tg.next()
                test_batch_pos, _ = test_tg.next()
                batch_x, batch_y = sg.next()
                batch_y = np.array(batch_y).reshape((self.batch_size_sg, 1))
                session.run([normalize_E, normalize_R, normalize_V])
                # calculate valid indices for scoring
                feed_dict = {inpl: batch_pos[1, :], inpr: batch_pos[0, :], inpo: batch_pos[2, :],
                             inpln: batch_neg[1, :], inprn: batch_neg[0, :], inpon: batch_neg[2, :],
                             train_inputs: batch_x, train_labels: batch_y,
                             global_step : b
                             }
                _, l = session.run([optimizer, loss], feed_dict=feed_dict)

                average_loss += l
                if b % eval_step_size == 0:
                    feed_dict = {test_inpl: test_batch_pos[1, :], test_inpo: test_batch_pos[2, :],
                                 test_inpr: test_batch_pos[0, :]}
                    scores_l, scores_r = session.run([ranking_error_l, ranking_error_r], feed_dict=feed_dict)

                    errl, errr = rank_triples_left_right(test_tg, scores_l, scores_r, test_batch_pos[1, :],
                                             test_batch_pos[2, :], test_batch_pos[0, :])

                    hits_10 = np.mean(np.asarray(errl + errr) <= 10) * 100
                    mean_rank = np.mean(np.asarray(errl + errr))
                    mean_rank_list.append(mean_rank)
                    hits_10_list.append(hits_10)

                    if best_hits < hits_10:
                        best_hits = hits_10
                    if best_rank > mean_rank:
                        best_rank = mean_rank

                    if b > 0:
                        average_loss = average_loss / eval_step_size
                    loss_list.append(average_loss)

                    if store_embeddings:
                        entity_embs.append(session.run(self.E))
                        relation_embs.append(session.run(self.R))

                    # The average loss is an estimate of the loss over the last eval_step_size batches.
                    print('Average loss at step %d: %f' % (b, average_loss))
                    print "Hits10: ", hits_10
                    print "MeanRank: ", mean_rank
                    average_loss = 0
                if not store_embeddings:
                    entity_embs = [session.run(self.E)]
                    relation_embs = [session.run(self.R)]
            return entity_embs, relation_embs, best_hits, best_rank, mean_rank_list, hits_10_list, loss_list