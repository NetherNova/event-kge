import tensorflow as tf
import numpy as np
import pandas as pd
import time
import datetime
import pickle
from rdflib import ConjunctiveGraph


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
    def __init__(self, num_entities, num_relations, embedding_size, batch_size, leftop, rightop, fnsim):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_size = embedding_size
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

    def run(self, tg):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            # Translation Model
            self.E = tf.Variable(tf.truncated_normal((self.num_entities, self.embedding_size)))
            self.R = tf.Variable(tf.truncated_normal((self.num_relations, self.embedding_size)))

            inpr = tf.placeholder(tf.int32, [self.batch_size], name="lhs")
            inpl = tf.placeholder(tf.int32, [self.batch_size], name="rhs")
            inpo = tf.placeholder(tf.int32, [self.batch_size], name="rell")

            inprn = tf.placeholder(tf.int32, [self.batch_size], name="lhs")
            inpln = tf.placeholder(tf.int32, [self.batch_size], name="rhs")
            inpon = tf.placeholder(tf.int32, [self.batch_size], name="rell")

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
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / tf.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            skipgram_loss = loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=train_labels,
                               inputs=embed,
                               num_sampled=num_sampled,
                               num_classes=vocabulary_size))

            loss = kg_loss + skipgram_loss
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        with tf.Session(graph=graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            for b in range(100):
                batch_pos, batch_neg = tg.next()
                feed_dict = {inpl: batch_pos[1, :], inpr: batch_pos[0, :], inpo: batch_pos[2, :],
                             inpln: batch_neg[1, :] , inprn: batch_neg[0, :], inpon: batch_neg[2, :]
                }
                _, l = session.run(
                    [optimizer, loss], feed_dict=feed_dict)
                session.run(self.normalize_embeddings())
                average_loss += l
                if b % 20 == 0:
                    if b > 0:
                        average_loss = average_loss / 20
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (b, average_loss))
                    average_loss = 0
            return session.run(self.E)

    def ranking(self):
        #TODO: get top ranks for lhs, re -> rhs
        pass


embedding_size = 1024
batch_size = 32
fnsim = dot_similarity
leftop = trans
rightop = ident_entity

g = ConjunctiveGraph()
g.load("amberg_individuals.xml")
ent_dict = {}
rel_dict = {}
for t in g.triples((None, None, None)):
    ent_dict.setdefault(t[0], len(ent_dict))
    ent_dict.setdefault(t[2], len(ent_dict))
    rel_dict.setdefault(t[1], len(rel_dict))

def prepare_data(ontology, unique_msgs, unique_fes, unique_vars, messages_to_fe, fe_to_fe_uri):
    #unique_msgs [0 - num_msgs]
    #ontology_entities 1) align 2) for additional entities start at num_msgs + 1
    #TODO: evtl. ontology anpassen, dass sich mehrere Sachen aendern von Var1 auf Var2
    #TODO: als test, neue events vorhersagen aus welchem FE sie kommen
    p



num_entities = len(ent_dict)
num_relations = len(rel_dict)
tg = TripleBatchGenerator(g, ent_dict, rel_dict, 1, batch_size)
model = TranslationEmbeddings(num_entities, num_relations, embedding_size, batch_size, leftop, rightop, fnsim)
embs=model.run(tg)
print embs.shape
print np.linalg.norm(embs[0,:])
