import itertools
import matplotlib.pyplot as plt
import sys
from rdflib import ConjunctiveGraph, RDF, URIRef

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE
from prep.etl import embs_to_df
from models.TransE import TransE
from models.RESCAL import RESCAL
from models.TEKE import TEKE
from models.TransH import TransH
from models.model import ranking_error_triples


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
            name += "-" + event_layer.name()
        return name


class Parameters(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def cross_parameter_eval(param_dict):
    keys = param_dict.keys()
    return [dict(zip(keys, k)) for k in itertools.product(*param_dict.values())]


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


def slice_ontology(rnd, ontology, valid_proportion, test_proportion, zero_shot_triples=[]):
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


def get_zero_shot_scenario(rnd, g, type_uri, link, percent):
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


def evaluate_on_test(model_type, parameter_list, test_tg, saved_model_path, test_size, reverse_relation_dictionary):
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
        elif model_type == TranslationModels.TEKE:
            model = TEKE(*parameter_list)

        model.create_graph()
        saver = tf.train.Saver(model.variables())
        saver.restore(session, saved_model_path)
        test_batch_pos, _ = test_tg.next(test_size)
        filter_triples = test_tg.all_triples

        test_inpl = test_batch_pos[1, :]
        test_inpr = test_batch_pos[0, :]
        test_inpo = test_batch_pos[2, :]
        scores_l, scores_r = model.scores(session, test_inpl, test_inpr, test_inpo)
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
    for k, v in relation_results.items():
        print(k, v)
    return results, relation_results