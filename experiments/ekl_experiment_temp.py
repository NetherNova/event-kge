##################################################################
# Experiments for event-enhanced knowledge graph embeddings
#
# How to run:
#
# To train the embeddings for a given knowledge graph and event dataset
# python ekl_experiment.py --dir 'path/to/dir' --...

# Up to now there is no flag to switch to GPU support, but this should be
# easy to change when needed
#
# Requirements:
#
# - Python 2.7
# - Tensorflow 0.12.1
# - numpy 1.12
# - rdflib 4.1.2
# - pandas

import csv
import itertools
import matplotlib.pyplot as plt
import pickle
import sys
from rdflib import ConjunctiveGraph, RDF, URIRef

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE

from models.RESCAL import RESCAL
from models.TEKE import TEKE
from models.TransE import TransE
from models.TransH import TransH
from models.model import ranking_error_triples
from models.model import l2_similarity
from models.pre_training import EmbeddingPreTrainer, TEKEPreparation
from prep.batch_generators import SkipgramBatchGenerator, TripleBatchGenerator, PredictiveEventBatchGenerator
from prep.etl import embs_to_df, prepare_sequences, message_index
from prep.preprocessing import PreProcessor




def get_kg_statistics(g):
    classes = set(g.objects(None, RDF.type))
    for c in classes:
        instances = set(g.subjects(RDF.type, c))
        print("Class: {0}: {1}".format(c, len(instances)))
        out_num = 0
        in_num = 0
        for i in instances:
            outgoing = list(g.predicate_objects(i))
            incoming = list(g.subject_predicates(i))
            out_num += len(outgoing)
            in_num += len(incoming)
        print("Out: {0} ".format(1.0*out_num / len(instances)))
        print("In: {0}".format(1.0*in_num / len(instances)))


class Parameters(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def cross_parameter_eval(param_dict):
    keys = param_dict.keys()
    return [dict(zip(keys, k)) for k in itertools.product(*param_dict.values())]


def slice_ontology(ontology, valid_proportion, test_proportion, zero_shot_triples=[]):
    """
    Slice ontology into two splits (train, test), with test *proportion*
    Work with copy of original ontology (do not modify)
    :param ontology:
    :param valid_proportion: percentage to be sliced out
    :param test_proportion
    :return:
    """
    ont_valid = ConjunctiveGraph()
    ont_test = ConjunctiveGraph()
    ont_train = ConjunctiveGraph()
    valid_size = int(np.floor(valid_proportion * len(ontology)))
    # TODO: only correct if event entities occur in two triples?
    test_size = int(np.floor(test_proportion * len(ontology)))
    # add all zero_shot entities to test set and remove from overall ontology
    if len(zero_shot_triples) > 0:
        remove_triples = []
        for zero_shot_triple in zero_shot_triples:
            ont_test.add(zero_shot_triple)
            remove_triples.append(zero_shot_triple)
        for s,p,o in remove_triples:
            ontology.remove((s,p,o))
    n_test = len(ont_test)
    if n_test > test_size:
        print("More zero shot triples than test proportion")
        sys.exit(0)
    # remaining test size
    test_size = test_size - n_test
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


def plot_embeddings(embs, reverse_dictionary):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embs)
    df = pd.DataFrame(low_dim_embs, columns=['x1', 'x2'])
    sns.lmplot('x1', 'x2', data=df, scatter=True, fit_reg=False)

    for i in range(low_dim_embs.shape[0]):
        if i not in reverse_dictionary:
            continue
        x, y = low_dim_embs[i, :]
        plt.annotate(reverse_dictionary[i],
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')


def get_low_dim_embs(embs, reverse_dictionary, dim=2):
    tsne = TSNE(perplexity=30, n_components=dim, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embs)
    df = embs_to_df(low_dim_embs, reverse_dictionary)
    return df


def bernoulli_probs(ontology, relation_dictionary):
    """
    Obtain bernoulli probabilities for each relation
    :param ontology:
    :param relation_dictionary:
    :return:
    """
    probs = dict()
    relations = set(ontology.predicates(None, None))
    for r in relations:
        heads = set(ontology.subjects(r, None))
        tph = 0
        for h in heads:
            tails = set(ontology.objects(h, r))
            tph += len(tails)
        tph = tph / (1.0 * len(heads))

        tails = set(ontology.objects(None, r))
        hpt = 0
        for t in tails:
            heads = set(ontology.subjects(r, t))
            hpt += len(heads)
        hpt = hpt / (1.0 * len(tails))
        probs[relation_dictionary[unicode(r)]] = tph / (tph + hpt)
    return probs


def evaluate_on_test(model_type, parameter_list, test_tg, saved_model_path):
    tf.reset_default_graph()
    with tf.Session() as session:
        # Need to instantiate model again
        print(parameter_list)
        if model_type == TranslationModels.Trans_E:
            model = TransE(*parameter_list)
        elif model_type == TranslationModels.Trans_H:
            model = TransH(*parameter_list)
        elif model_type == TranslationModels.RESCAL:
            model = RESCAL(*parameter_list)

        model.create_graph()
        saver = tf.train.Saver(model.variables())
        saver.restore(session, saved_model_path)
        test_batch_pos, _ = test_tg.next(test_size)
        filter_triples = test_tg.all_triples

        test_inpl = test_batch_pos[1, :]
        test_inpr = test_batch_pos[0, :]
        test_inpo = test_batch_pos[2, :]
        if model_type == TranslationModels.Trans_H:
            r_embs, embs, w_embs = session.run([model.R, model.E, model.W], feed_dict=feed_dict)
            scores_l = model.rank_left_idx(test_inpr, test_inpo, r_embs, embs, w_embs)
            scores_r = model.rank_right_idx(test_inpl, test_inpo, r_embs, embs, w_embs)
        else:
            r_embs, embs = session.run([model.R, model.E], feed_dict={})
            scores_l = model.rank_left_idx(test_inpr, test_inpo, r_embs, embs)
            scores_r = model.rank_right_idx(test_inpl, test_inpo, r_embs, embs)

        errl, errr = ranking_error_triples(filter_triples, scores_l, scores_r, test_inpl, test_inpo, test_inpr)

    # END OF SESSION
    results = []
    err_arr = np.asarray(errl + errr)
    hits_10 = np.mean(err_arr <= 10) * 100
    hits_3 = np.mean(err_arr <= 3) * 100
    hits_1 = np.mean(err_arr <= 1) * 100
    mean_rank = np.mean(err_arr)
    mrr = np.mean(1.0 / err_arr)
    print("Test Hits10: ", hits_10)
    print("Test Hits3: ", hits_3)
    print("Test Hits1: ", hits_1)
    print("Test MRR: ", mrr)
    print("Test MeanRank: ", mean_rank)

    results.append(mean_rank)
    results.append(mrr)
    results.append(hits_10)
    results.append(hits_3)
    results.append(hits_1)

    relation_results = dict()
    for i in np.unique(test_inpo):
        indices = np.argwhere(np.array(test_inpo) == i)
        err_arr = np.concatenate([np.array(errl)[indices], np.array(errr)[indices]])
        hits_10 = np.mean(err_arr <= 10) * 100
        hits_3 = np.mean(err_arr <= 3) * 100
        hits_1 = np.mean(err_arr <= 1) * 100
        mean_rank = np.mean(err_arr)
        mrr = np.mean(1.0 / err_arr)

        relation_results[reverse_relation_dictionary[i]] = {'MeanRank': mean_rank, 'MRR' : mrr, 'Hits@10' : hits_10,
                                                            'Hits@3': hits_3, 'Hits@1': hits_1}
    for k, v in relation_results.iteritems():
        print(k, v)
    return results, relation_results


def get_zero_shot_scenario(g, type_uri, links, percent):
    """
    Get triples about entities of RDF:type *type_uri* with predicate *link*
    :param type_uri:
    :param link:
    :param percent:
    :return:
    """
    subs = list(g.subjects(RDF.type, type_uri))
    triples = set()
    for s in subs:
        for link in links:
            # check for both sides of link
            for s,p,o in set(g.triples((s, link, None))).union(set(g.triples((None, link, s)))):
                triples.add((s,p,o))
    triples = list(triples)
    n = len(triples)
    n_reduced = int(np.floor(n * percent))
    print("Found {0} possible zero shot triples -- using {1}".format(n, n_reduced))
    indices = rnd.choice(range(n), n_reduced)
    kg_prop = n_reduced / (len(g) * 1.0)
    return [(URIRef(s), URIRef(p), URIRef(o)) for (s,p,o) in np.array(triples)[indices]], kg_prop


class TranslationModels:
    Trans_E, Trans_H, RESCAL, TEKE = range(4)

    @staticmethod
    def get_model_name(event_layer, num):
        name = None
        if num == TranslationModels.Trans_E:
            name = "TransE"
        elif num == TranslationModels.Trans_H:
            name = "TransH"
        elif num == TranslationModels.TEKE:
            name = "TEKE"
        else:
            name = "RESCAL"
        if event_layer is not None:
            name += "-" + event_layer
        return name


if __name__ == '__main__':
    for seq_data_ in [0.1, 0.25, 0.5, 0.75, 1.0]:
        rnd = np.random.RandomState(42)
        ####### PATH PARAMETERS ########
        base_path = "../clones/"
        path_to_store_model = base_path + "Embeddings/"
        path_to_events = base_path + "Sequences/"
        path_to_kg = base_path + "Ontology/amberg_clone.rdf"
        path_to_store_sequences = base_path + "Sequences/"
        path_to_store_embeddings = base_path + "Embeddings/"
        traffic_data = False
        routing_data = False
        path_to_sequence = base_path + 'Sequences/sequence.txt'
        num_sequences = None
        pre_train = True
        supp_event_embeddings = base_path + "Embeddings/supplied_embeddings.pickle"

        preprocessor = PreProcessor(path_to_kg)

        if routing_data:
            exclude_rels = ['http://www.siemens.com/knowledge_graph/ppr#resourceName',
                            'http://www.siemens.com/knowledge_graph/ppr#shortText',
                            'http://www.siemens.com/knowledge_graph/ppr#compVariant',
                            'http://www.siemens.com/knowledge_graph/ppr#compVersion',
                            'http://www.siemens.com/knowledge_graph/ppr#personTime_min',
                            'http://www.siemens.com/knowledge_graph/ppr#machiningTime_min',
                            'http://www.siemens.com/knowledge_graph/ppr#setupTime_min',
                            'http://siemens.com/knowledge_graph/industrial_upper_ontology#hasPart',
                            RDF.type,
                            'http://siemens.com/knowledge_graph/cyber_physical_systems/industrial_cps#consistsOf']
            preprocessor = PreProcessor(path_to_kg)
            preprocessor.load_unique_msgs_from_txt(base_path + 'unique_msgs.txt')
            amberg_params = None
        elif traffic_data:
            exclude_rels = []
            preprocessor = PreProcessor(path_to_kg)
            preprocessor.load_unique_msgs_from_txt(base_path + 'unique_msgs.txt')
            amberg_params = None
        else:
            exclude_rels = ['http://www.siemens.com/ontology/demonstrator#tagAlias']
            max_events = None
            max_seq = None
            # sequence window size in minutes
            window_size = 0.3
            amberg_params = (path_to_events, max_events)

        preprocessor.load_knowledge_graph(format='xml', exclude_rels=exclude_rels, amberg_params=amberg_params)
        vocab_size = preprocessor.get_vocab_size()
        unique_msgs = preprocessor.get_unique_msgs()
        ent_dict = preprocessor.get_ent_dict()
        rel_dict = preprocessor.get_rel_dict()
        g = preprocessor.get_kg()

        print("Read {0} number of triples".format(len(g)))
        get_kg_statistics(g)

        zero_shot_prop = 0.3
        zero_shot_entity =  URIRef('http://www.siemens.com/ontology/demonstrator#Event') #URIRef('http://purl.oclc.org/NET/ssnx/ssn#Device')
        zero_shot_relation = [URIRef(RDF.type), URIRef('http://www.siemens.com/ontology/demonstrator#occursOn')] # URIRef('http://www.loa-cnr.it/ontologies/DUL.owl#follows') # URIRef('http://www.siemens.com/ontology/demonstrator#involvedEquipment') URIRef('http://www.loa-cnr.it/ontologies/DUL.owl#hasPart')
        zero_shot_triples, kg_prop = get_zero_shot_scenario(g, zero_shot_entity, zero_shot_relation, zero_shot_prop)
        # zero_shot_triples = []

        #TODO: RNN 0.1, 0.3, 0.5

        ######### Model selection ##########
        model_type = TranslationModels.Trans_E
        bernoulli = True
        # "Skipgram", "Concat", "RNN"
        event_layer = 'Concat'
        store_embeddings = False

        ######### Hyper-Parameters #########
        param_dict = {}
        param_dict['embedding_size'] = [100]
        #param_dict['seq_data_size'] = [1.0]
        param_dict['batch_size'] = [32]     # [32, 64, 128]
        param_dict['learning_rate'] = [0.05]     # [0.5, 0.8, 1.0]
        param_dict['lambd'] = [0.001]     # regularizer (RESCAL)
        param_dict['alpha'] = [1.0]     # event embedding weighting
        eval_step_size = 1000
        num_epochs = 100
        test_proportion = kg_prop
        validation_proportion = 0.1 # 0.1
        fnsim = l2_similarity

        # Train dev test splitting
        g_train, g_valid, g_test = slice_ontology(g, validation_proportion, test_proportion, zero_shot_triples)

        train_size = len(g_train)
        valid_size = len(g_valid)
        test_size = len(g_test)
        print("Train size: ", train_size)
        print("Valid size: ", valid_size)
        print("Test size: ", test_size)

        # SKIP Parameters
        if event_layer is not None:
            param_dict['num_skips'] = [1]   # [2, 4]
            param_dict['num_sampled'] = [7]     # [5, 9]
            # param_dict['batch_size_sg'] = [2]     # [128, 512]
            pre_train_steps = 10000
            if routing_data or traffic_data:
                sequences = preprocessor.prepare_sequences(path_to_sequence)
            else:
                merged = preprocessor.get_merged()
                sequences = prepare_sequences(merged, message_index,
                                                  unique_msgs, window_size, max_seq, g_train)
            num_sequences = len(sequences)
            sequences = sequences[:int(num_sequences * seq_data_)]

        num_entities = len(ent_dict)
        num_relations = len(rel_dict)
        print("Num entities:", num_entities)
        print("Num relations:", num_relations)
        print("Event entity percentage: {0} prct".format(100.0 * vocab_size / num_entities))

        if bernoulli:
            bern_probs = bernoulli_probs(g, rel_dict)

        # free some memory
        g = None
        model_name = TranslationModels.get_model_name(event_layer, model_type)
        overall_best_performance = np.inf
        best_param_list = []

        train_tg = TripleBatchGenerator(g_train, ent_dict, rel_dict, 2, rnd, bern_probs=bern_probs)
        valid_tg = TripleBatchGenerator(g_valid, ent_dict, rel_dict, 1, rnd, sample_negative=False)
        test_tg = TripleBatchGenerator(g_test, ent_dict, rel_dict, 1, rnd, sample_negative=False)

        print(test_tg.next(5))

        # Loop trough all hyper-paramter combinations
        param_combs = cross_parameter_eval(param_dict)
        for comb_num, tmp_param_dict in enumerate(param_combs):
            params = Parameters(**tmp_param_dict)
            num_steps = (train_size / params.batch_size) * num_epochs

            print("Progress: {0} prct".format(int((100.0 * comb_num) / len(param_combs))))
            print("Embedding size: ", params.embedding_size)
            print("Batch size: ", params.batch_size)

            filter_triples = valid_tg.all_triples

            if event_layer:
                batch_size_sg = (num_sequences * num_epochs) / num_steps
                print("Batch size sg:", batch_size_sg)
                num_skips = params.num_skips
                num_sampled = params.num_sampled
                if pre_train:
                    pre_trainer = EmbeddingPreTrainer(unique_msgs, SkipgramBatchGenerator(sequences, num_skips, rnd),
                                                      params.embedding_size, vocab_size, num_sampled, batch_size_sg,
                                                      supp_event_embeddings)
                    pre_trainer.train(pre_train_steps)
                    pre_trainer.save()
                if event_layer == "Skipgram":
                    sg = SkipgramBatchGenerator(sequences, num_skips, rnd)
                else:
                    sg = PredictiveEventBatchGenerator(sequences, num_skips, rnd)
            else:
                num_sampled = 0
                batch_size_sg = 0
                num_skips = 0
                sequences = []
                # dummy batch generator for empty sequence TODO: can we get rig of this?
                sg = SkipgramBatchGenerator(sequences, num_skips, rnd)

            # Model Selection
            if model_type == TranslationModels.Trans_E:
                param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                              batch_size_sg, num_sampled, vocab_size, fnsim, params.learning_rate,
                              event_layer, num_skips, params.alpha]
                model = TransE(*param_list)
            elif model_type == TranslationModels.Trans_H:
                param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                              batch_size_sg, num_sampled, vocab_size, params.learning_rate, event_layer,
                              params.lambd, num_skips, params.alpha]
                model = TransH(*param_list)
            elif model_type == TranslationModels.RESCAL:
                param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                              batch_size_sg, num_sampled, vocab_size, params.learning_rate, event_layer,
                              params.lambd, num_skips, params.alpha]
                model = RESCAL(*param_list)
            elif model_type == TranslationModels.TEKE:
                pre_trainer = EmbeddingPreTrainer(unique_msgs, SkipgramBatchGenerator(sequences, num_skips, rnd),
                                                  params.embedding_size, vocab_size, num_sampled, batch_size_sg,
                                                  supp_event_embeddings)
                pre_trainer.train(5000)
                pre_trainer.save()
                w_bound = np.sqrt(6. / params.embedding_size)
                initE = rnd.uniform(-w_bound, w_bound, (num_entities, params.embedding_size))
                print("Loading supplied embeddings...")
                with open(supp_event_embeddings, "rb") as f:
                    supplied_embeddings = pickle.load(f)
                    supplied_dict = supplied_embeddings.get_dictionary()
                    for event_id, emb_id in supplied_dict.iteritems():
                        if event_id in unique_msgs:
                            new_id = unique_msgs[event_id]
                            initE[new_id] = supplied_embeddings.get_embeddings()[emb_id]
                tk = TEKEPreparation(sequences, initE, num_entities)
                param_list = [num_entities, num_relations, params.embedding_size, params.batch_size, fnsim]
                model = TEKE(*param_list)

            # Build tensorflow computation graph
            tf.reset_default_graph()
            # tf.set_random_seed(23)
            with tf.Session() as session:
                model.create_graph()
                saver = tf.train.Saver(model.variables())
                tf.global_variables_initializer().run()
                print('Initialized graph')

                average_loss = 0
                mean_rank_list = []
                hits_10_list = []
                loss_list = []

                # Initialize some / event entities with supplied embeddings
                if supp_event_embeddings and model_type != TranslationModels.TEKE:
                    w_bound = np.sqrt(6. / params.embedding_size)
                    initE = rnd.uniform(-w_bound, w_bound, (num_entities, params.embedding_size))
                    print("Loading supplied embeddings...")
                    with open(supp_event_embeddings, "rb") as f:
                        supplied_embeddings = pickle.load(f)
                        supplied_dict = supplied_embeddings.get_dictionary()
                        for event_id, emb_id in supplied_dict.iteritems():
                            if event_id in unique_msgs:
                                new_id = unique_msgs[event_id]
                                initE[new_id] = supplied_embeddings.get_embeddings()[emb_id]
                    session.run(model.assign_initial(initE))

                if store_embeddings:
                    entity_embs = []
                    relation_embs = []

                # Steps loop
                for b in range(1, num_steps + 1):
                    batch_pos, batch_neg = train_tg.next(params.batch_size)
                    valid_batch_pos, _ = valid_tg.next(valid_size)
                    # Event batches
                    batch_x, batch_y = sg.next(batch_size_sg)
                    batch_y = np.array(batch_y).reshape((batch_size_sg, 1))

                    if model_type == TranslationModels.TEKE:
                        n_x_h, n_x_t, n_x_hn, n_x_tn = tk.get_pointwise_batch(batch_pos, batch_neg)
                        xy, xy_n = tk.get_pairwise_batch(batch_pos, batch_neg)
                        feed_dict = {
                            model.inpl: batch_pos[1, :], model.inpr: batch_pos[0, :], model.inpo: batch_pos[2, :],
                            model.inpln: batch_neg[1, :], model.inprn: batch_neg[0, :], model.inpon: batch_neg[2, :],
                            model.train_inputs: batch_x, model.train_labels: batch_y,
                            model.n_x_h : n_x_h, model.n_x_t : n_x_t, model.n_x_y : xy,
                            model.n_x_hn: n_x_hn, model.n_x_tn: n_x_tn, model.n_x_yn: xy_n,
                            model.global_step: b
                        }
                    else:
                    # calculate valid indices for scoring
                        feed_dict = {
                            model.inpl: batch_pos[1, :], model.inpr: batch_pos[0, :], model.inpo: batch_pos[2, :],
                            model.inpln: batch_neg[1, :], model.inprn: batch_neg[0, :], model.inpon: batch_neg[2, :],
                            model.train_inputs: batch_x, model.train_labels: batch_y,
                            model.global_step: b
                        }
                    # One train step in mini-batch
                    _, l = session.run(model.train(), feed_dict=feed_dict)
                    average_loss += l
                    # Run post-ops: regularization etc.
                    session.run(model.post_ops())
                    # Evaluate on validation set
                    if b % eval_step_size == 0:
                        valid_inpl = valid_batch_pos[1, :]
                        valid_inpr = valid_batch_pos[0, :]
                        valid_inpo = valid_batch_pos[2, :]
                        if model_type == TranslationModels.Trans_H:
                            r_embs, embs, w_embs = session.run([model.R, model.E, model.W], feed_dict=feed_dict)
                            scores_l = model.rank_left_idx(valid_inpr, valid_inpo, r_embs, embs, w_embs)
                            scores_r = model.rank_right_idx(valid_inpl, valid_inpo, r_embs, embs, w_embs)
                        elif model_type == TranslationModels.TEKE:
                            n_h_test = tk.get_pointwise(valid_inpl)
                            entities_all = tk.get_pointwise()
                            n_t_test = tk.get_pointwise(valid_inpr)
                            ht_test_all = None
                            ht_all_test = None

                            r_embs, embs, A, B = session.run([model.R, model.E, model.A, model.B], feed_dict={})
                            scores_l = model.rank_left_idx(valid_inpr, valid_inpo, r_embs, embs, A, B, entities_all, n_t_test, ht_all_test)
                            scores_r = model.rank_right_idx(valid_inpl, valid_inpo, r_embs, embs, A, B, n_h_test, entities_all, ht_test_all)
                        else:
                            r_embs, embs = session.run([model.R, model.E], feed_dict={})
                            scores_l = model.rank_left_idx(valid_inpr, valid_inpo, r_embs, embs)
                            scores_r = model.rank_right_idx(valid_inpl, valid_inpo, r_embs, embs)

                        errl, errr = ranking_error_triples(filter_triples, scores_l, scores_r, valid_inpl,
                                                           valid_inpo, valid_inpr)
                        hits_10 = np.mean(np.asarray(errl + errr) <= 10) * 100
                        mean_rank = np.mean(np.asarray(errl + errr))
                        mean_rank_list.append(mean_rank)
                        hits_10_list.append(hits_10)

                        if b > 0:
                            average_loss = average_loss / eval_step_size
                            loss_list.append(average_loss)

                        if store_embeddings:
                            entity_embs.append(session.run(model.E))
                            relation_embs.append(session.run(model.R))

                        # The average loss is an estimate of the loss over the last eval_step_size batches.
                        print('Average loss at step {0}: {1}'.format(b, average_loss))
                        print("\t Validation Hits10: ", hits_10)
                        print("\t Validation MeanRank: ", mean_rank)
                        average_loss = 0

                        if overall_best_performance > mean_rank:
                            overall_best_performance = mean_rank
                            print("Saving overall best model with MeanRank: {0} and hits {1}".format(mean_rank, hits_10))
                            save_path_global = saver.save(session, path_to_store_model + 'tf_model')
                            best_param_list = param_list

                reverse_entity_dictionary = dict(zip(ent_dict.values(), ent_dict.keys()))
                reverse_relation_dictionary = dict(zip(rel_dict.values(), rel_dict.keys()))

                # save embeddings to disk
                if store_embeddings:
                    for i in range(len(entity_embs)):
                        if i % 50 == 0:
                            df_embs = get_low_dim_embs(entity_embs[i], reverse_entity_dictionary)
                            df_embs.to_csv(path_to_store_embeddings + "entity_embeddings_low" + str(i) + ".csv", sep=',',
                                           encoding='utf-8')

                            df_r_embs = get_low_dim_embs(relation_embs[i], reverse_relation_dictionary)
                            df_r_embs.to_csv(path_to_store_embeddings + "relation_embeddings" + str(i) + ".csv", sep=',',
                                             encoding='utf-8')

                    # TODO: only of best model (not last)
                    df_embs = embs_to_df(entity_embs[len(entity_embs)-1], reverse_entity_dictionary)
                    df_embs.to_csv(path_to_store_embeddings + "entity_embeddings" + '_last_cleaned' + ".csv", sep=',',
                                       encoding='utf-8')

        # Reset graph, load best model and apply to test data set
        with open(base_path + 'evaluation_parameters_' + model_name + '_' + str(seq_data_) +
                          '_best.csv', "wb") as eval_file:
            writer = csv.writer(eval_file)
            results, relation_results = evaluate_on_test(model_type, best_param_list, test_tg, save_path_global)
            writer.writerow (
                    ["relation", "embedding_size", "batch_size", "learning_rate", "num_skips", "num_sampled",
                     "batch_size_sg", "mean_rank", "mrr", "hits_top_10", "hits_top_3", "hits_top_1"]
            )
            writer.writerow(
                ['all', params.embedding_size, params.batch_size, params.learning_rate, num_skips, num_sampled,
                 batch_size_sg, results[0], results[1], results[2], results[3], results[4]]
            )
            for rel in relation_results:
                writer.writerow (
                    [rel, params.embedding_size, params.batch_size, params.learning_rate, num_skips, num_sampled,
                     batch_size_sg, relation_results[rel]['MeanRank'], relation_results[rel]['MRR'],
                     relation_results[rel]['Hits@10'], relation_results[rel]['Hits@3'], relation_results[rel]['Hits@1']]
            )