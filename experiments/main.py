from models.RESCAL import RESCAL
from models.TransE import TransE
from models.TransH import TransH
from models.model import ranking_error_triples, bernoulli_probs, trans, ident_entity, l2_similarity
from prep.batch_generators import SkipgramBatchGenerator, TripleBatchGenerator, PredictiveEventBatchGenerator
from prep.etl import embs_to_df
from prep.preprocessing import PreProcessor

import tensorflow as tf
import numpy as np
from rdflib import ConjunctiveGraph, RDF
import matplotlib.pyplot as plt

rnd = np.random.RandomState(42)


class TranslationModels:
    Trans_E, Trans_H, RESCAL, Trans_Eve = range(4)

    @staticmethod
    def get_model_name(event_layer, num):
        name = None
        if num == TranslationModels.Trans_E:
            name = "TransE"
        elif num == TranslationModels.Trans_H:
            name = "TransH"
        elif num == TranslationModels.Trans_Eve:
            name = "TranesESq"
        else:
            name = "RESCAL"
        if event_layer is not None:
            name += "-" + event_layer
        return name


def slice_ontology(ontology, valid_proportion, test_proportion, zero_shot_entities=[]):
    """
    Slice ontology into two splits (train, test), with test *proportion*
    Work with copy of original ontology (do not modify)
    :param ontology:
    :param proportion: percentage to be sliced out
    :return:
    """
    ont_valid = ConjunctiveGraph()
    ont_test = ConjunctiveGraph()
    ont_train = ConjunctiveGraph()
    valid_size = int(np.floor(valid_proportion * len(ontology)))
    test_size = int(np.floor(test_proportion * len(ontology)))
    # add all zero_shot entities to test set and remove from overall ontology
    if len(zero_shot_entities) > 0:
        remove_triples = []
        for zero_shot_uri in zero_shot_entities:
            for p, o in ontology.predicate_objects(zero_shot_uri):
                tmp_triple = (zero_shot_uri, p, o)
                ont_test.add(tmp_triple)
                remove_triples.append(tmp_triple)
            for s, p in ontology.subject_predicates(zero_shot_uri):
                tmp_triple = (s, p, zero_shot_uri)
                ont_test.add(tmp_triple)
                remove_triples.append(tmp_triple)
    if len(ont_test) > test_size:
        print "More zero shot triples than test proportion"
        return
    # remaining test size
    test_size = test_size - len(ont_test)
    # random splits
    slice_indices = rnd.choice(range(0, len(ontology)), valid_size + test_size, replace=False)
    valid_indices = slice_indices[:valid_size]
    test_indices = slice_indices[valid_size:]
    for i, (s, p, o) in enumerate(sorted(ontology.triples((None, None, None)))):
        if i in valid_indices:
            ont_valid.add((s, p, o))
        elif i in test_indices:
            ont_test.add((s, p, o))
        else:
            ont_train.add((s, p, o))
    return ont_train, ont_valid, ont_test


class EKLBaseExperiment(object):
    def __init__(self, model_type, kg_path, entity_embedding_size, num_epochs, batch_size, learning_rate, valid_size):
        self.model_type = model_type
        self.kg_path = kg_path
        self.entity_embeding_size = entity_embedding_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.valid_size = valid_size
        self.preprocessor = PreProcessor(self.kg_path)
        # default settings for TransE
        self.fnsim = l2_similarity
        self.leftop = trans
        self.rightop = ident_entity

    def run(self, excluded_rels):
        self.preprocessor.load_knowledge_graph(exclude_rels=excluded_rels)
        g = self.preprocessor.get_kg()
        print "Read %d number of triples" % len(g)

        zero_shot_entities = []
        # Train dev test splitting
        test_size = 0.0
        g_train, g_valid, g_test = slice_ontology(g, self.valid_size, test_size, zero_shot_entities)

        train_size = len(g_train)
        valid_size = len(g_valid)
        test_size = len(g_test)
        print "Train size: ", train_size
        print "Valid size: ", valid_size
        print "Test size: ", test_size

        num_entities = len(self.preprocessor.get_ent_dict())
        num_relations = len(self.preprocessor.get_rel_dict())
        print "Num entities:", num_entities
        print "Num relations:", num_relations

        bern_probs = bernoulli_probs(g, self.preprocessor.get_rel_dict())

        batch_size_sg = 0
        num_sampled = 0
        vocab_size = 0
        event_layer = None

        num_steps = int(np.floor((1.0 * train_size / self.batch_size) * self.num_epochs))
        self.eval_step_size = int(np.floor(num_steps / 10))

        embs = None

        train_tg = TripleBatchGenerator(g_train, self.preprocessor.get_ent_dict(), self.preprocessor.get_rel_dict(), 2, rnd, bern_probs=bern_probs)
        valid_tg = TripleBatchGenerator(g_valid, self.preprocessor.get_ent_dict(), self.preprocessor.get_rel_dict(), 1, rnd, sample_negative=False)

        if self.model_type == TranslationModels.Trans_E:
            param_list = [num_entities, num_relations, self.entity_embeding_size, self.batch_size, batch_size_sg,
                          num_sampled, vocab_size, self.leftop, self.rightop, self.fnsim, self.learning_rate,
                          event_layer]
            model = TransE(*param_list)
        elif self.model_type == TranslationModels.Trans_H:
            param_list = [num_entities, num_relations, self.entity_embeding_size, self.batch_size, batch_size_sg,
                          num_sampled, vocab_size, self.learning_rate, event_layer]
            model = TransH(*param_list)
        elif self.model_type == TranslationModels.RESCAL:
            param_list = [num_entities, num_relations, self.entity_embeding_size, self.batch_size, batch_size_sg,
                          num_sampled, vocab_size, self.learning_rate, event_layer]
            model = RESCAL(*param_list)

        # tf.set_random_seed(23)
        with tf.Session() as session:
            model.create_graph()
            tf.global_variables_initializer().run()
            print('Initialized graph')

            average_loss = 0
            best_hits_local = -np.inf
            best_rank_local = np.inf
            best_entity_embeddings = None
            self.mean_rank_list = []
            self.hits_10_list = []
            self.loss_list = []

            filter_triples = valid_tg.all_triples

            for b in range(1, num_steps + 1):
                batch_pos, batch_neg = train_tg.next(self.batch_size)
                valid_batch_pos, _ = valid_tg.next(valid_size)
                # Event batches

                # calculate valid indices for scoring
                feed_dict = {
                    model.inpl: batch_pos[1, :], model.inpr: batch_pos[0, :], model.inpo: batch_pos[2, :],
                    model.inpln: batch_neg[1, :], model.inprn: batch_neg[0, :], model.inpon: batch_neg[2, :],
                    model.global_step: b
                }
                # One train step in mini-batch
                _, l = session.run(model.train(), feed_dict=feed_dict)
                average_loss += l
                # Run post-ops: regularization etc.
                session.run(model.post_ops())
                if b % self.eval_step_size == 0:
                    if valid_size > 0:
                        valid_inpl = valid_batch_pos[1, :]
                        valid_inpr = valid_batch_pos[0, :]
                        valid_inpo = valid_batch_pos[2, :]
                        if self.model_type == TranslationModels.Trans_H:
                            r_embs, embs, w_embs = session.run([model.R, model.E, model.W], feed_dict=feed_dict)
                            scores_l = model.rank_left_idx(valid_inpr, valid_inpo, r_embs, embs, w_embs)
                            scores_r = model.rank_right_idx(valid_inpl, valid_inpo, r_embs, embs, w_embs)
                        else:
                            r_embs, embs = session.run([model.R, model.E], feed_dict={})
                            scores_l = model.rank_left_idx(valid_inpr, valid_inpo, r_embs, embs)
                            scores_r = model.rank_right_idx(valid_inpl, valid_inpo, r_embs, embs)

                        errl, errr = ranking_error_triples(filter_triples, scores_l, scores_r, valid_inpl,
                                                           valid_inpo, valid_inpr)
                        hits_10 = np.mean(np.asarray(errl + errr) <= 10) * 100
                        mean_rank = np.mean(np.asarray(errl + errr))
                        self.mean_rank_list.append(mean_rank)
                        self.hits_10_list.append(hits_10)
                        print "\t Validation Hits10: ", hits_10
                        print "\t Validation MeanRank: ", mean_rank

                        if mean_rank < best_rank_local:
                            # store best embeddings w.r.t. mean rank
                            best_entity_embeddings = embs
                    else:
                        # only entity embeddings without evaluation (always the latest embeddings)
                        embs = session.run([model.E])[0]
                        best_entity_embeddings = embs
                    if b > 0:
                        average_loss = average_loss / self.eval_step_size
                    self.loss_list.append(average_loss)
                    print('Average loss at step %d: %10.2f' % (b, average_loss))
                    average_loss = 0
            # in case no eval step has been reached
            if embs is None:
                embs = session.run([model.E])[0]

        ent_dict = self.preprocessor.get_ent_dict()
        reverse_entity_dictionary = dict(zip(ent_dict.values(), ent_dict.keys()))
        df_embs = embs_to_df(embs, reverse_entity_dictionary)
        #df_embs.to_csv(path_to_store_embeddings + "entity_embeddings" + '_last_cleaned' + ".csv", sep=',',
        #               encoding='utf-8')
        return df_embs

    def plot_model_summary(self):
        x = np.array(range(len(self.loss_list))) * self.eval_step_size
        if self.valid_size > 0:
            plt.subplot(311)
            plt.plot(x, self.loss_list)
            plt.ylabel('loss')
            plt.title('Knowledge Graph Embeddings')
            plt.grid(True)

            plt.subplot(312)
            plt.plot(x, self.mean_rank_list)
            plt.ylabel('MR')
            plt.grid(True)

            plt.subplot(313)
            plt.plot(x, self.hits_10_list)
            plt.ylabel('hits@10')
            plt.xlabel('batch')
            plt.grid(True)
        else:
            plt.plot(x, self.loss_list)
            plt.title('Knowledge Graph Embeddings')
            plt.ylabel('loss')
            plt.xlabel('batch')
        plt.show()

if __name__ == '__main__':
    base_path = "../routing_data/"
    path_to_store_model = base_path + "Embeddings/"
    path_to_kg = base_path + "Ontology/PPR_individuals.rdf"

    excluded_rels = ['http://www.siemens.com/knowledge_graph/ppr#resourceName',
                     'http://www.siemens.com/knowledge_graph/ppr#shortText',
                     'http://www.siemens.com/knowledge_graph/ppr#compVariant',
                     'http://www.siemens.com/knowledge_graph/ppr#compVersion',
                     'http://www.siemens.com/knowledge_graph/ppr#personTime_min',
                     'http://www.siemens.com/knowledge_graph/ppr#machiningTime_min',
                     'http://www.siemens.com/knowledge_graph/ppr#setupTime_min',
                     'http://siemens.com/knowledge_graph/industrial_upper_ontology#hasPart',
                     'http://siemens.com/knowledge_graph/cyber_physical_systems/industrial_cps#consistsOf',
                     RDF.type]

    embedding_size = 40
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.05
    valid_proportion = 0.01

    exp = EKLBaseExperiment(TranslationModels.Trans_E, path_to_kg, embedding_size, num_epochs, batch_size, learning_rate,
                            valid_proportion)
    embs = exp.run(excluded_rels)
    exp.plot_model_summary()

