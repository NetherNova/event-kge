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


import argparse
import csv
import itertools
import matplotlib.pyplot as plt
import pickle
from rdflib import ConjunctiveGraph, RDF, RDFS

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE

from models.RESCAL import RESCAL
from models.TransE import TransE
from models.TransH import TransH
from models.model import ranking_error_triples
from models.model import trans, ident_entity, l2_similarity
from models.pre_training import EmbeddingPreTrainer
from prep.batch_generators import SkipgramBatchGenerator, TripleBatchGenerator, PredictiveEventBatchGenerator
from prep.etl import embs_to_df, prepare_sequences, message_index
from prep.preprocessing import PreProcessor

rnd = np.random.RandomState(42)

def write_results(model_name, model_type, combination_num,)
    with open('evaluation_' + model_name + '.csv'):


class Parameters(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def cross_parameter_eval(param_dict):
    keys = param_dict.keys()
    return [dict(zip(keys, k)) for k in itertools.product(*param_dict.values())]


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
    # TODO: only correct if event entities occur in two triples?
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
        print parameter_list
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
    print "Test Hits10: ", hits_10
    print "Test Hits3: ", hits_3
    print "Test Hits1: ", hits_1
    print "Test MRR: ", mrr
    print "Test MeanRank: ", mean_rank

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
        print k, v

    return results, relation_results


class TranslationModels:
    Trans_E, Trans_H, RESCAL = range(4)

    @staticmethod
    def get_model_name(event_layer, num):
        name = None
        if num == TranslationModels.Trans_E:
            name = "TransE"
        elif num == TranslationModels.Trans_H:
            name = "TransH"
        else:
            name = "RESCAL"
        if event_layer is not None:
            name += "-" + event_layer
        return name

# PATH PARAMETERS
base_path = "./routing_data/"
path_to_store_model = base_path + "Embeddings/"
path_to_events = base_path + "Sequences/" # TODO: should be optional if no skipgram stuff
path_to_schema = base_path + "Ontology/PPR_individuals.rdf" # TODO: also optional if no schema present
path_to_kg = base_path + "Ontology/PPR_individuals.rdf" # "Ontology/amberg_inferred_v2.xml" # "Ontology/players.nt" # "Ontology/amberg_inferred.xml"
path_to_store_sequences = base_path + "Sequences/"
path_to_store_embeddings = base_path + "Embeddings/"
sequence_file_name = "train_sequences"
traffic_data = False
sim_data = True
path_to_sequence = base_path + 'Sequences/sequence.txt' # "Sequences/sequence.txt"
num_sequences = None
pre_train = False
supp_event_embeddings = None    # base_path + Embeddings/supplied_embeddings_60.pickle"
cross_eval_single = True

preprocessor = PreProcessor(path_to_kg)

if sim_data:
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
    max_events = 5000
    max_seq = 5000
    # sequence window size in minutes
    window_size = 3
    amberg_params = (path_to_events, max_events)

preprocessor.load_knowledge_graph(format='xml', exclude_rels=exclude_rels, amberg_params=amberg_params)
zero_shot_entities = [] # zero_shot = [URIRefs(), ]
event_prct_in_kg = 0.1
vocab_size = preprocessor.get_vocab_size()
ent_dict = preprocessor.get_ent_dict()
rel_dict = preprocessor.get_rel_dict()
g = preprocessor.get_kg()
print "Read %d number of triples" % len(g)
event_num_in_kg = np.floor(vocab_size * event_prct_in_kg)

######### Model selection ##########
model_type = TranslationModels.Trans_E
bernoulli = True
# "Skipgram", "Concat", "RNN"
event_layer = None
store_embeddings = True

######### Hyper-Parameters #########
param_dict = {}
param_dict['embedding_size'] = [80]
param_dict['seq_data_size'] = [1.0]
param_dict['batch_size'] = [32]     # [32, 64, 128]
param_dict['learning_rate'] = [0.05]     # [0.5, 0.8, 1.0]
param_dict['lambd'] = [1.0]     # [0.5, 0.1, 0.05]
param_dict['alpha'] = [1.0]     # [0.1, 0.5, 1.0]
eval_step_size = 1000
num_epochs = 50
test_proportion = 0.01 # 0.2
validation_proportion = 0.01 # 0.1
fnsim = l2_similarity
leftop = trans
rightop = ident_entity

# Train dev test splitting
g_train, g_valid, g_test = slice_ontology(g, validation_proportion, test_proportion, zero_shot_entities)

train_size = len(g_train)
valid_size = len(g_valid)
test_size = len(g_test)
print "Train size: ", train_size
print "Valid size: ", valid_size
print "Test size: ", test_size

# SKIP Parameters
if event_layer is not None:
    param_dict['num_skips'] = [1]   # [2, 4]
    param_dict['num_sampled'] = [10]     # [5, 9]
    param_dict['batch_size_sg'] = [32]     # [128, 512]
    pre_train_steps = 5000
    if sim_data or traffic_data:
        num_sequences = preprocessor.prepare_sequences(path_to_sequence, path_to_store_sequences + sequence_file_name)
    else:
        num_sequences = prepare_sequences(merged, path_to_store_sequences + sequence_file_name, message_index,
                                          unique_msgs, window_size, max_seq, g_train)
        merged = None   # Free some memory

num_entities = len(ent_dict)
num_relations = len(rel_dict)
print "Num entities:", num_entities
print "Num relations:", num_relations
print "Event entity percentage: %3.2f prct" %(100.0 * vocab_size / num_entities)

if bernoulli:
    bern_probs = bernoulli_probs(g, rel_dict)

# free some memory
g = None
model_name = TranslationModels.get_model_name(event_layer, model_type)
overall_best_performance = np.inf
best_param_list = []

if event_layer:
    sequences = pickle.load(open(path_to_store_sequences + sequence_file_name + ".pickle", "rb"))

train_tg = TripleBatchGenerator(g_train, ent_dict, rel_dict, 2, rnd, bern_probs=bern_probs)
valid_tg = TripleBatchGenerator(g_valid, ent_dict, rel_dict, 1, rnd, sample_negative=False)
test_tg = TripleBatchGenerator(g_test, ent_dict, rel_dict, 1, rnd, sample_negative=False)

print train_tg.next(5)

# Loop trough all hyper-paramter combinations
param_combs = cross_parameter_eval(param_dict)
for comb_num, tmp_param_dict in enumerate(param_combs):
    params = Parameters(**tmp_param_dict)
    num_steps = (train_size / params.batch_size) * num_epochs
    print "Progress: %d prct" %(int((100.0 * comb_num) / len(param_combs)))
    print "Embedding size: ", params.embedding_size
    print "Batch size: ", params.batch_size

    filter_triples = valid_tg.all_triples
    local_best_param_list = []

    if event_layer:
        batch_size_sg = params.batch_size_sg
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
        elif event_layer in ["LSTM", "RNN"]:
            sg = PredictiveEventBatchGenerator(sequences, num_skips, rnd)
        elif event_layer == "Concat":
            sg = PredictiveEventBatchGenerator(sequences, num_skips, rnd)
    else:
        num_sampled = 1
        batch_size_sg = 0
        num_skips = 0
        sequences = []
        # dummy batch generator for empty sequence TODO: can we get rig of this?
        sg = SkipgramBatchGenerator(sequences, num_skips, rnd)

    # Model Selection
    if model_type == TranslationModels.Trans_E:
        param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                      batch_size_sg, num_sampled, vocab_size, leftop, rightop, fnsim, params.learning_rate,
                      event_layer, params.lambd, num_sequences, num_skips, params.alpha]
        model = TransE(*param_list)
    elif model_type == TranslationModels.Trans_H:
        param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                      batch_size_sg, num_sampled, vocab_size, params.learning_rate, event_layer,
                      params.lambd, num_sequences, num_skips]
        model = TransH(*param_list)
    elif model_type == TranslationModels.RESCAL:
        param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                      batch_size_sg, num_sampled, vocab_size, params.learning_rate, event_layer,
                      params.lambd, num_sequences, num_skips]
        model = RESCAL(*param_list)

    # Build tensorflow computation graph
    tf.reset_default_graph()
    # tf.set_random_seed(23)
    with tf.Session() as session:
        model.create_graph()
        saver = tf.train.Saver(model.variables())
        tf.global_variables_initializer().run()
        print('Initialized graph')

        average_loss = 0
        best_hits_local = -np.inf
        best_rank_local = np.inf
        mean_rank_list = []
        hits_10_list = []
        loss_list = []

        # Initialize some / event entities with supplied embeddings
        if supp_event_embeddings:
            w_bound = np.sqrt(6. / params.embedding_size)
            initE = rnd.uniform(-w_bound, w_bound, (num_entities, params.embedding_size))
            print("Loading supplied embeddings...")
            with open(supp_event_embeddings, "rb") as f:
                supplied_dict = supplied_embeddings.get_dictionary()
                for event_id, emb_id in supplied_dict.iteritems():
                    if event_id in unique_msgs:
                        new_id = unique_msgs[event_id]
                        initE[new_id] = supplied_embeddings.get_embeddings()[emb_id]
                        # TODO: assign V for TransESq
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
            if event_layer == "Concat":
                batch_y = np.array(batch_y)
            else:
                batch_y = np.array(batch_y).reshape((batch_size_sg, 1))
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
                print('Average loss at step %d: %10.2f' % (b, average_loss))
                print "\t Validation Hits10: ", hits_10
                print "\t Validation MeanRank: ", mean_rank
                average_loss = 0

                if best_hits_local < hits_10:
                    best_hits_local = hits_10
                if best_rank_local > mean_rank:
                    best_rank_local = mean_rank
                    print "Saving locally best model with MeanRank: %5.2f and hits %3.2f" % (mean_rank, hits_10)
                    save_path_local = saver.save(session, path_to_store_model + 'tf_local_model')
                    local_best_param_list = param_list

                if overall_best_performance > mean_rank:
                    overall_best_performance = mean_rank
                    print "Saving overall best model with MeanRank: %5.2f and hits %3.2f" % (mean_rank, hits_10)
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
        # Evaluation on Test Set #
        print "Best validation hits10 local", best_hits_local
        print "Best validation MeanRank local", best_rank_local

# Reset graph, load best model and apply to test data set
with open(base_path + 'evaluation_parameters_10pct_' + model_name + '_best.csv', "wb") as eval_file:
    writer = csv.writer(eval_file)
    results, relation_results = evaluate_on_test(model_type, best_param_list, test_tg, save_path_global)
    if comb_num == 0:
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


if __name__ == '__main__':
    desc = "Event-enhanced Learning for Knowledge Graph Completion (EKL)"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--kg", required=True)