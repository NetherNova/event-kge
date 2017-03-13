import tensorflow as tf
import numpy as np
import pandas as pd
import time
import datetime
import pickle
from rdflib import ConjunctiveGraph
from etl import update_ontology
from sklearn.manifold import TSNE
import csv


def dot_similarity(x, y):
    return tf.matmul(x, tf.transpose(y))


def l2_similarity(x, y):
    return -tf.nn.l2_loss(x - y)


def l1_similarity(x, y):
    return - tf.reduce_sum(tf.abs(x - y))


def trans(x, y):
    return x+y


def ident_entity(x, y):
    return x


def max_margin(pos, neg, marge=1.0):
    cost = 1. - pos + neg
    return tf.reduce_mean(tf.maximum(0., cost)) # sum oder mean?


def rank_left_fn_idx(simfn, embeddings_ent, embeddings_rel, leftop, rightop, inpr, inpo):
    """
    compute similarity score of all 'left' entities given 'right' and 'rel' members
    :param simfn:
    :param embeddings:
    :param leftop:
    :param rightop:
    :return:
    """
    lhs = embeddings_ent
    rell = tf.nn.embedding_lookup(embeddings_rel, inpo)
    relr = tf.nn.embedding_lookup(embeddings_rel, inpo)
    rhs = tf.nn.embedding_lookup(embeddings_ent, inpr)
    # [1377, 32, 64], [64,32]
    expanded_lhs = tf.expand_dims(lhs, 1)
    batch_lhs = tf.transpose(leftop(expanded_lhs, rell), [1, 2, 0])
    batch_rhs = tf.expand_dims(rightop(rhs, relr), 1)
    # only dot_sim support
    simi = tf.squeeze(tf.batch_matmul(batch_rhs, batch_lhs), 1) # [32, 1, 64], [32, 1377, 64]
    #simi = simfn(tf.reshape(leftop(tf.expand_dims(lhs, 1), rell), [1377*32, 64]), ) # [1377*32, 32]
    # getting *batch_size* rank lists for all entities [all_entities, batch_size] similarity
    return simi


def rank_right_fn_idx(simfn, embeddings_ent, embeddings_rel, leftop, rightop, inpl, inpo):
    """
    compute similarity score of all 'right' entities given 'left' and 'rel' members
    :param simfn:
    :param embeddings:
    :param leftop:
    :param rightop:
    :return:
    """
    rhs = embeddings_ent
    rell = tf.nn.embedding_lookup(embeddings_rel, inpo)
    relr = tf.nn.embedding_lookup(embeddings_rel, inpo)
    lhs = tf.nn.embedding_lookup(embeddings_ent, inpl)
    simi = simfn(leftop(lhs, rell), rightop(rhs, relr)) # [32, 64], [64, 1377]
    return simi


class SkipgramBatchGenerator(object):
    def __init__(self, sequences, num_skips, batch_size):
        """

        :param sequences: list of lists of event entities
        :param num_skips:  window left and right of target
        :param batch_size:
        """
        self.sequences = sequences
        self.sequence_index = 0
        self.num_skips = num_skips
        self.event_index = num_skips
        self.batch_size = batch_size
        self.prepare_target_skips()

    def prepare_target_skips(self):
        self.data_index = 0
        self.data = []
        for seq in self.sequences:
            for target_ind in range(self.num_skips, len(seq) - self.num_skips):
                for i in range(-self.num_skips, self.num_skips+1):
                    if i == 0:
                        #avoid the target_ind itself
                        continue
                    self.data.append( (seq[target_ind], seq[target_ind + i]) )

    def next(self):
        batch_x = []
        batch_y = []
        for b in range(self.batch_size):
            self.data_index = self.data_index % len(self.data)
            batch_x.append(self.data[self.data_index][0])
            batch_y.append(self.data[self.data_index][1])
            self.data_index += 1
        return batch_x, batch_y



class TripleBatchGenerator(object):
    def __init__(self, triples, entity_dictionary, relation_dictionary, num_neg_samples, batch_size):
        self.all_triples = []
        self.batch_index = 0
        self.batch_size = batch_size
        self.num_neg_samples = num_neg_samples
        self.entity_dictionary = entity_dictionary
        self.relation_dictionary = relation_dictionary

        for (s,p,o) in triples:
            if s not in self.entity_dictionary:
                continue
            if o not in self.entity_dictionary:
                continue
            if p not in self.relation_dictionary:
                continue
            s_ind = self.entity_dictionary[s]
            p_ind = self.relation_dictionary[p]
            o_ind = self.entity_dictionary[o]
            self.all_triples.append((s_ind, p_ind, o_ind))

    def next(self):
        # return lists of entity and reltaion indices
        inpr = []
        inpl = []
        inpo = []

        inprn = []
        inpln = []
        inpon = []

        for b in range(self.batch_size // 2):
            if self.batch_index >= len(self.all_triples):
                self.batch_index = 0
            current_triple = self.all_triples[self.batch_index]
            inpl.append(current_triple[0])
            inpr.append(current_triple[2])
            inpo.append(current_triple[1])
            # Append current triple twice
            inpl.append(current_triple[0])
            inpr.append(current_triple[2])
            inpo.append(current_triple[1])

            rn, ln, on = self.sample_negative(current_triple, True)
            inpln.append(ln)
            inprn.append(rn)
            inpon.append(on)
            # repeat
            rn, ln, on = self.sample_negative(current_triple, False)
            inpln.append(ln)
            inprn.append(rn)
            inpon.append(on)
            self.batch_index += 1
        return np.array([inpr, inpl, inpo]), np.array([inprn, inpln, inpon])

    def sample_negative(self, (s_ind,p_ind,o_ind), relation=True):
        if relation:
            sample_set = [rel for rel in self.relation_dictionary.values() if rel != p_ind]
            p_ind = np.random.choice(sample_set, 1)[0]
        else:
            sample_set = [ent for ent in self.entity_dictionary.values() if ent != o_ind]
            o_ind = np.random.choice(sample_set, 1)[0]
        return (o_ind, s_ind, p_ind)


class TranslationEmbeddings(object):
    """
    Implements triplet scoring from negative sampling
    """
    def __init__(self, num_entities, num_relations, embedding_size, batch_size, vocab_size, leftop, rightop, fnsim):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_sampled = 5
        self.batch_size = batch_size
        self.leftop = leftop
        self.rightop = rightop
        self.fnsim = fnsim

    def normalize_embeddings(self):
        self.E = self.normalize(self.E)
        self.R = self.normalize(self.R)
        return self.E, self.R

    def normalize(self, W):
        return W / tf.expand_dims(tf.sqrt(tf.reduce_sum(W ** 2, axis=1)), 1)

    def run(self, tg, sg, reverse_dictionary, data, test_tg):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            # Translation Model
            w_bound = np.sqrt(6. / self.embedding_size)
            self.E = tf.Variable(tf.random_uniform((self.num_entities, self.embedding_size), minval=-w_bound, maxval=w_bound))
            self.R = tf.Variable(tf.random_uniform((self.num_relations, self.embedding_size), minval=-w_bound, maxval=w_bound))

            inpr = tf.placeholder(tf.int32, [self.batch_size], name="rhs")
            inpl = tf.placeholder(tf.int32, [self.batch_size], name="lhs")
            inpo = tf.placeholder(tf.int32, [self.batch_size], name="rell")

            inprn = tf.placeholder(tf.int32, [self.batch_size], name="rhs")
            inpln = tf.placeholder(tf.int32, [self.batch_size], name="lhs")
            inpon = tf.placeholder(tf.int32, [self.batch_size], name="rell")

            test_inpr = tf.placeholder(tf.int32, [100], name="test_rhs")
            test_inpl = tf.placeholder(tf.int32, [100], name="test_lhs")
            test_inpo = tf.placeholder(tf.int32, [100], name="test_ell")

            lhs = tf.nn.embedding_lookup(self.E, inpl)
            rhs = tf.nn.embedding_lookup(self.E, inpr)
            rell = tf.nn.embedding_lookup(self.R, inpo)
            relr = tf.nn.embedding_lookup(self.R, inpo)

            lhsn = tf.nn.embedding_lookup(self.E, inpln)
            rhsn = tf.nn.embedding_lookup(self.E, inprn)
            relln = tf.nn.embedding_lookup(self.R, inpon)
            relrn = tf.nn.embedding_lookup(self.R, inpon)

            simi = self.fnsim(self.leftop(lhs, rell), self.rightop(rhs, relr))
            simin = self.fnsim(self.leftop(lhsn, relln), self.rightop(rhsn, relrn))
            kg_loss = max_margin(simi, simin)

            # Skipgram Model
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            #valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            #embeddings = tf.Variable(
            #    tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(self.E, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal([self.vocab_size, embedding_size],
                                    stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
            nce_biases = tf.Variable(tf.zeros([self.vocab_size]))
            skipgram_loss = loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocab_size,
                               remove_accidental_hits=True))

            loss = kg_loss + skipgram_loss  # max-margin loss + sigmoid_cross_entropy_loss for sampled values
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

            ranking_error_l = rank_left_fn_idx(self.fnsim, self.E, self.R, leftop, rightop, test_inpr, test_inpo)
            ranking_error_r = rank_right_fn_idx(self.fnsim, self.E, self.R, leftop, rightop, test_inpl, test_inpo)

            # anderer Ansatz: Trainiere erst skipgram embeddings --> dann faktorisiere kg mit vortrainierten
            # varianten-spezigische embeddings --> trainiere auf datensatz mit
            # neural language model --> "topic"-spezifische Embeddings
            # Topic-spezigische KG Embeddings?

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            with open("low_dim_embeddings.csv", "wb") as csvfile:
                writer = csv.writer(csvfile)
                for b in range(200):
                    batch_pos, batch_neg = tg.next()
                    test_batch_pos, _ = test_tg.next()
                    batch_x, batch_y = sg.next()
                    batch_y = np.array(batch_y).reshape((batch_size, 1))
                    session.run(self.normalize_embeddings())
                    # calculate valid indices for scoring
                    feed_dict = {test_inpl: test_batch_pos[1, :], test_inpo: test_batch_pos[2, :], test_inpr: test_batch_pos[0, :]}
                    scores_l, scores_r = session.run([ranking_error_l, ranking_error_r], feed_dict=feed_dict)
                    errl = []
                    errr = []

                    np.savetxt("scores_l.txt", scores_l)
                    np.savetxt("scores_r.txt", scores_r)

                    for i, (l, o, r) in enumerate(zip(test_batch_pos[0, :], test_batch_pos[2, :], test_batch_pos[1, :])):
                        # find those triples that have <*,o,r> and * != l
                        rmv_idx_l = [l_rmv for (l_rmv, rel, rhs) in test_tg.all_triples if
                                     rel == o and r == rel and l_rmv != l]
                        # *l* is the correct index
                        scores_l[i, rmv_idx_l] = -np.inf
                        print "Should be: " + str(l) + " is: " + str(np.argmax(scores_l[i, :]))
                        errl += [np.argsort(np.argsort(-scores_l[i, :]))[l] + 1]
                        # since index start at 0, best possible value is 1

                        rmv_idx_r = [r_rmv for (lhs, rel, r_rmv) in test_tg.all_triples if
                                     rel == o and lhs == l and r_rmv != r]
                        # *l* is the correct index
                        scores_r[i, rmv_idx_r] = -np.inf
                        print "Should be: " + str(r) + " is: " + str(np.argmax(scores_r[i, :]))
                        errr += [np.argsort(np.argsort(-scores_r[i, :]))[r] + 1]
                        # since index start at 0, best possible value is 1

                    print errl, errr

                    feed_dict = {inpl: batch_pos[1, :], inpr: batch_pos[0, :], inpo: batch_pos[2, :],
                                 inpln: batch_neg[1, :] , inprn: batch_neg[0, :], inpon: batch_neg[2, :],
                                 train_inputs:  batch_x, train_labels: batch_y
                    }
                    _, l = (0, 1) #session.run([optimizer, loss], feed_dict=feed_dict)
                    session.run(self.normalize_embeddings())
                    average_loss += l
                    if b % 20 == 0:
                        if b > 0:
                            average_loss = average_loss / 20
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at step %d: %f' % (b, average_loss))
                        average_loss = 0
                        embs = session.run(self.E)
                        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                        low_dim_embs = tsne.fit_transform(embs)
                        for i in range(low_dim_embs.shape[0]):
                            if i not in reverse_dictionary:
                                continue
                            module = data[data["Meldetext"] == reverse_dictionary[i]]["Module"]
                            module = np.unique(module)[0]
                            writer.writerow([str(b), module, reverse_dictionary[i], low_dim_embs[i][0], low_dim_embs[i][1]])
                return session.run(self.E), session.run(self.R)

    def ranking(self):
        #TODO: get top ranks for lhs, re -> rhs
        pass


#ontology (uri, r_uri, uri) ... uri -> id, r_uri -> [0...r]
#sequences (id, id, id) ... reverse[id] -> entitiy
embedding_size = 64
batch_size = 32
fnsim = dot_similarity
leftop = trans
rightop = ident_entity

g = ConjunctiveGraph()
g.load("./test_data/amberg_inferred.xml")

from etl import update_ontology, unique_msgs, unique_mods, unique_fes, unique_vars, merged

g, uri_to_id = update_ontology(g, unique_msgs,unique_mods, unique_fes, unique_vars, merged)
ent_dict = uri_to_id
rel_dict = {}
for t in g.triples((None, None, None)):
    if t[0] not in ent_dict:
        ent_dict.setdefault(t[0], len(ent_dict))
    if t[1] not in ent_dict:
        ent_dict.setdefault(t[2], len(ent_dict))
    rel_dict.setdefault(t[1], len(rel_dict))

test_size = 100
test_indices = np.random.randint(0, len(g), test_size)

g_test = ConjunctiveGraph()
for i, (s,p,o) in enumerate(g.triples((None, None, None))):
    if i in test_indices:
        g_test.add((s,p,o))
        g.remove((s,p,o))


def prepare_data(ontology, unique_msgs, unique_fes, unique_vars, messages_to_fe):
    #unique_msgs [0 - num_msgs]
    #ontology_entities 1) align 2) for additional entities start at num_msgs + 1
    #TODO: evtl. ontology anpassen, dass sich mehrere Sachen aendern von Var1 auf Var2
    #TODO: als test, neue events vorhersagen aus welchem FE sie kommen
    ent_dict = {}
    rel_dict = {}
    for t in g.triples((None, None, None)):
        ent_dict.setdefault(t[0], len(ent_dict))
        ent_dict.setdefault(t[2], len(ent_dict))
        rel_dict.setdefault(t[1], len(rel_dict))


num_entities = len(ent_dict)
num_relations = len(rel_dict)

tg = TripleBatchGenerator(g, ent_dict, rel_dict, 1, batch_size)
test_tg = TripleBatchGenerator(g_test, ent_dict, rel_dict, 1, test_size)
sequences = [seq.split(' ') for seq in pickle.load(open("./test_data/train_sequences.pickle", "rb"))]
sg = SkipgramBatchGenerator(sequences, 2, batch_size)

reverse_dictionary = dict(zip(unique_msgs.values(), unique_msgs.keys()))

model = TranslationEmbeddings(num_entities, num_relations, embedding_size, batch_size, len(unique_msgs), leftop, rightop, fnsim)
embs, r_embs = model.run(tg, sg, reverse_dictionary, merged, test_tg)
print embs.shape
print np.linalg.norm(embs[0,:])

r = r_embs[rel_dict['http://www.siemens.com/ontology/demonstrator#occursOn']]

#TODO: as labels emit modules for events too --> show in plots clusters of modules
#TODO: prediction tests machen auf modul struktur
