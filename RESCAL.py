import tensorflow as tf
import numpy as np
from model import dot, trans, ident_entity, max_margin, rank_right_fn_idx, rank_left_fn_idx, rescal_similarity, skipgram_loss, rank_triples_left_right
import pickle


class RESCAL(object):
    def __init__(self, num_entities, num_relations, embedding_size, batch_size_kg, batch_size_sg, num_sampled,
                 vocab_size, supp_event_embeddings=None, sub_prop_constr=None):
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
        self.supp_event_embeddings = supp_event_embeddings
        self.sub_prop_constr = sub_prop_constr

    def run(self, tg, sg, test_tg, test_size, num_steps, init_lr=1.0, skipgram=True, store_embeddings=False, lambd=0.05):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            # Translation Model initialisation
            self.w_bound = np.sqrt(6. / self.embedding_size)
            self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-self.w_bound,
                                                   maxval=self.w_bound))
            self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size, self.embedding_size),
                                                   minval=-self.w_bound, maxval=self.w_bound))

            inpr = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpl = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpo = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            inprn = tf.placeholder(tf.int32, [self.batch_size_kg], name="rhs")
            inpln = tf.placeholder(tf.int32, [self.batch_size_kg], name="lhs")
            inpon = tf.placeholder(tf.int32, [self.batch_size_kg], name="rell")

            test_inpr = tf.placeholder(tf.int32, [test_size], name="test_rhs")
            test_inpl = tf.placeholder(tf.int32, [test_size], name="test_lhs")
            test_inpo = tf.placeholder(tf.int32, [test_size], name="test_rell")

            lhs = tf.nn.embedding_lookup(self.E, inpl)
            rhs = tf.nn.embedding_lookup(self.E, inpr)
            rell = tf.nn.embedding_lookup(self.R, inpo)

            lhsn = tf.nn.embedding_lookup(self.E, inpln)
            rhsn = tf.nn.embedding_lookup(self.E, inprn)
            relln = tf.nn.embedding_lookup(self.R, inpon)

            # RESCAL with ranking loss
            lhs = tf.expand_dims(lhs, 1) # [batch, 1, d] # zeile mal zeile
            lhs = tf.reduce_sum(tf.mul(lhs, rell), 2)

            lhsn = tf.expand_dims(lhsn, 1)
            lhsn = tf.reduce_sum(tf.mul(lhsn, relln), 2)

            simi = tf.nn.sigmoid(tf.reduce_sum(tf.mul(lhs, rhs), 1))
            simin = tf.nn.sigmoid(tf.reduce_sum(tf.mul(lhsn, rhsn), 1))
            kg_loss = max_margin(simi, simin) #+ lambd * (tf.nn.l2_loss(self.E) + tf.nn.l2_loss(self.R))

            if self.sub_prop_constr:
                sub_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sub"])
                sup_relations = tf.nn.embedding_lookup(self.R, self.sub_prop_constr["sup"])
                kg_loss += tf.reduce_sum(dot(sub_relations, sup_relations) - 1)

            # TODO: add possibility to switch on transitivity constraint
            # [(e1, r_trans, e2) - (e2, r_trans, e3)] - (e1, r_trans, e3)
            # transitive closure need to be calculated incrementally...do in preprocessing

            # Skipgram Model
            train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size_sg])
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size_sg, 1])

            sg_embed = tf.nn.embedding_lookup(self.E, train_inputs)

            sg_loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size, train_labels)

            if skipgram:
                loss = kg_loss + sg_loss
            else:
                loss = kg_loss
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = init_lr
            # tf.train.exponential_decay(starter_learning_rate, global_step, 10, 0.98, staircase=True)
            learning_rate = tf.constant(starter_learning_rate)

            optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

            ranking_error_l = rank_left_fn_idx(rescal_similarity, self.E, self.R, trans, ident_entity, test_inpr, test_inpo)
            ranking_error_r = rank_right_fn_idx(rescal_similarity, self.E, self.R, trans, ident_entity, test_inpl,
                                                test_inpo)

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
                initE = np.random.uniform((len(self.vocab_size), self.embedding_size),
                                          minval=-self.w_bound, maxval=self.w_bound)
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
                # calculate valid indices for scoring
                feed_dict = {inpl: batch_pos[1, :], inpr: batch_pos[0, :], inpo: batch_pos[2, :],
                             inpln: batch_neg[1, :], inprn: batch_neg[0, :], inpon: batch_neg[2, :],
                             train_inputs: batch_x, train_labels: batch_y,
                             global_step: b
                             }
                _, l, = session.run([optimizer, loss], feed_dict=feed_dict)
                average_loss += l
                if b % eval_step_size == 0:
                    feed_dict = {test_inpl: test_batch_pos[1, :], test_inpo: test_batch_pos[2, :],
                                 test_inpr: test_batch_pos[0, :]}
                    scores_l, scores_r = session.run([ranking_error_l, ranking_error_r], feed_dict=feed_dict)

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
                    relation_embs = [session.run(tf.reshape(self.R, [self.num_relations, self.embedding_size ** 2]))]
            return entity_embs, relation_embs, best_hits, best_rank, mean_rank_list, hits_10_list, loss_list