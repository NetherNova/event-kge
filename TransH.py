import tensorflow as tf
import numpy as np
from model import l2_similarity, dot, trans, ident_entity, max_margin, skipgram_loss, rank_triples_left_right
import pickle


class TransH(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, supp_event_embeddings=None, sub_prop_constr=None):
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
        self.supp_event_embeddings = supp_event_embeddings
        self.sub_prop_constr = sub_prop_constr

    def rank_left_idx(self, test_inpr, test_o, test_w, ent_embs):
        lhs = ent_embs # [num_entities, d]
        rell = test_o  # [num_test, d]
        rhs = ent_embs[test_inpr]  # [num_test, d]
        wr = test_w
        result = np.zeros((rhs.shape[0], lhs.shape[0]))
        for i in xrange(rhs.shape[0]):
            proj_rhs = rhs[i] - np.dot(rhs[i], np.transpose(wr[i])) * wr[i]
            for j in xrange(lhs.shape[0]):
                proj_lhs = lhs[j] - np.dot(lhs[j], np.transpose(wr[i])) * wr[i]
                temp_diff = (proj_lhs + rell[i]) - proj_rhs
                result[i][j] = -np.sqrt(np.sum(temp_diff**2))
        return result

    def rank_right_idx(self, test_inpl, test_o, test_w, ent_embs):
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
                result[i][j] = -np.sqrt(np.sum(temp_diff**2))
        return result

    def run(self, tg, sg, test_tg, test_size, num_steps, init_lr=1.0, skipgram=True, store_embeddings=False, lambd=0.05):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            # Translation Model initialisation
            w_bound = np.sqrt(6. / self.embedding_size)
            self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))
            self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))

            self.W = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound,
                                                   maxval=w_bound))

            normalize_W = self.W.assign(tf.nn.l2_normalize(self.W, 1))

            inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhsn")
            inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhsn")
            inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="relln")

            test_inpr = tf.placeholder(tf.int32, [test_size], name="test_rhs")
            test_inpl = tf.placeholder(tf.int32, [test_size], name="test_lhs")
            test_inpo = tf.placeholder(tf.int32, [test_size], name="test_rell")

            lhs = tf.nn.embedding_lookup(self.E, inpl)
            rhs = tf.nn.embedding_lookup(self.E, inpr)
            rell = tf.nn.embedding_lookup(self.R, inpo)
            wr = tf.nn.embedding_lookup(self.W, inpo)

            lhsn = tf.nn.embedding_lookup(self.E, inpln)
            rhsn = tf.nn.embedding_lookup(self.E, inprn)
            relln = tf.nn.embedding_lookup(self.R, inpon)
            wrn = tf.nn.embedding_lookup(self.W, inpon)

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
            reg2_n =  tf.expand_dims(tf.sqrt(tf.reduce_sum(self.R ** 2, axis=1)), 1)
            reg2 = tf.reduce_sum(tf.maximum(0., (reg2_z / reg2_n) - epsilon))

            kg_loss = max_margin(simi, simin) + lambd * (reg1 + reg2)

            if self.sub_prop_constr:
                sub_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sub"])
                sup_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sup"])
                kg_loss += tf.reduce_sum(dot(sub_relations, sup_relations) - 1)

            # Skipgram Model
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

            sg_embed = tf.nn.embedding_lookup(self.E, train_inputs)

            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size, train_labels)

            if skipgram:
                loss = kg_loss + sg_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
            else:
                loss = kg_loss
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = init_lr
            # tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.98, staircase=True)
            learning_rate = tf.constant(starter_learning_rate)

            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

            ranking_test_inpo = tf.nn.embedding_lookup(self.R, test_inpo)
            ranking_test_inpw = tf.nn.embedding_lookup(self.W, test_inpo)

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
                print("Load supplied embeddings...")
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
                session.run([normalize_W])
                feed_dict = {inpl: batch_pos[1, :], inpr: batch_pos[0, :], inpo: batch_pos[2, :],
                             inpln: batch_neg[1, :], inprn: batch_neg[0, :], inpon: batch_neg[2, :],
                             train_inputs: batch_x, train_labels: batch_y,
                             global_step : b
                             }
                _, l, = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if b % eval_step_size == 0:
                    feed_dict = {test_inpl: test_batch_pos[1, :], test_inpo: test_batch_pos[2, :],
                                 test_inpr: test_batch_pos[0, :]}
                    test_o, test_w, embs = session.run([ranking_test_inpo, ranking_test_inpw, self.E],
                                                     feed_dict=feed_dict)

                    scores_l = self.rank_left_idx(test_batch_pos[0, :], test_o, test_w, embs)
                    scores_r = self.rank_right_idx( test_batch_pos[1, :], test_o, test_w, embs)

                    errl, errr = rank_triples_left_right(test_tg, scores_l, scores_r, test_batch_pos[1, :],
                                                         test_batch_pos[2, :], test_batch_pos[0, :])

                    hits_10 = np.mean(np.asarray(errl + errr) <= 10) * 100
                    mean_rank = np.mean(np.asarray(errl + errr))
                    print "Hits10: ", hits_10
                    print "MeanRank: ", mean_rank

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
                    average_loss = 0

                if not store_embeddings:
                    entity_embs = [session.run(self.E)]
                    relation_embs = [session.run(self.R)]
            return entity_embs, relation_embs, best_hits, best_rank, mean_rank_list, hits_10_list, loss_list