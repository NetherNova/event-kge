##################################################################
# Experiments for event-enhanced knowledge graph embeddings
#
# How to run:
#
# To train the embeddings for a given knowledge graph and event dataset
# put the path to the kg *path_to_kg* (optional sequence dataset *path_to_sequence*)
# fiddle with the parameter settings, then run:
# python ekl_experiment.py

# Up to now there is no flag to switch to GPU support, but this should be
# easy to change when needed
#
# Requirements:
#
# - Python 2.7
# - Tensorflow 1.10
# - numpy 1.12
# - rdflib 4.1.2
# - pandas 0.19

from __future__ import print_function

import csv
import itertools
import matplotlib.pyplot as plt
import pickle
import sys
from rdflib import ConjunctiveGraph, RDF, RDFS, URIRef

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.manifold import TSNE

from models.RESCAL import RESCAL
from models.TEKE import TEKE
from models.TransE import TransE
from models.TransH import TransH
from models.model import l2_similarity, ranking_error_triples, bernoulli_probs
from models.pre_training import EmbeddingPreTrainer, TEKEPreparation

from event_models.LinearEventModel import Skipgram, ConcatenationFull, ConcatenationCause, Average
from event_models.Autoencoder import ConvolutionalAutoEncoder, LSTMAutoencoder

from prep.batch_generators import SkipgramBatchGenerator, TripleBatchGenerator, PredictiveEventBatchGenerator, \
    FuturePredictiveBatchGenerator, AutoEncoderBatchGenerator
from prep.etl import embs_to_df, prepare_sequences, message_index
from prep.preprocessing import PreProcessor
from experiments.experiment_helper import slice_ontology, get_kg_statistics, get_low_dim_embs, get_zero_shot_scenario, \
    cross_parameter_eval, TranslationModels, embs_to_df, Parameters, evaluate_on_test

import os
from django.conf import settings
from time import time
from utils import url_parse

rnd = np.random.RandomState(42)
status = ""


def get_status():
    return status


def init(args):
    global path_to_kg, path_to_sequence, path_to_uri_map, modeltype, eventlayer, storeembeddings, embedding_size, seq_data_size, \
        batch_size, learning_rate, lambd, num_epochs, test_proportion, validation_proportion, preprocessor, traffic_data, \
        base_path, path_to_store_model, amberg_params, exclude_rels, path_to_store_embeddings

    path_to_kg = args.get('path_to_kg')
    # path_to_sequence = args.get('path_to_sequence')
    path_to_uri_map = args.get('path_to_uri_map')
    modeltype = args.get('model_type')
    # eventlayer = args.get('event_layer')
    # storeembeddings = args.get('store_embeddings')
    embedding_size = int(args.get('embedding_size'))
    # seq_data_size = int(args.get('seq_data_size'))
    batch_size = int(args.get('batch_size'))
    learning_rate = float(args.get('learning_rate'))
    lambd = float(args.get('lambd'))
    num_epochs = int(args.get('num_epochs'))
    test_proportion = float(args.get('test_proportion'))
    validation_proportion = float(args.get('validation_proportion'))

    ####### PATH PARAMETERS ########
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sim_data/")  # "./traffic_data/"
    path_to_store_model = base_path + "Embeddings/"
    path_to_events = base_path + "Sequences/"
    # path_to_kg = base_path + "Ontology/test_2.xml"  # "Ontology/traffic_individuals.xml" #
    path_to_store_sequences = base_path + "Sequences/"
    path_to_store_embeddings = os.path.join(os.path.join(settings.BASE_DIR, "static"),
                                            "data/")  # path_to_store_embeddings = base_path + "Embeddings/"
    traffic_data = False
    # path_to_sequence = base_path + 'Sequences/sequence_2.txt'
    num_sequences = None
    pre_train = False
    supp_event_embeddings = None  # base_path + "Embeddings/supplied_embeddings.pickle"
    preprocessor = PreProcessor(path_to_kg)
    tk = None
    bern_probs = None

    if traffic_data:
        max_events = 10000
        exclude_rels = [RDFS.comment]
        preprocessor = PreProcessor(path_to_kg)
        excluded_events = preprocessor.load_unique_msgs_from_txt(path_to_uri_map,
                                                                 max_events=max_events)  # 'unique_msgs.txt'
        excluded_events = [URIRef(('http://www.siemens.com/citypulse#' + e)) for e in excluded_events]
        amberg_params = None
    else:
        exclude_rels = ['http://www.siemens.com/ontology/demonstrator#tagAlias']
        max_events = None
        max_seq = None
        preprocessor.load_unique_msgs_from_txt(path_to_uri_map)  # 'unique_msgs2.txt'
        # sequence window size in minutes
        window_size = 0.5
        amberg_params = None
        # amberg_params = (path_to_events, max_events)


def start_experiment():
    global entity_embs, reverse_entity_dictionary, relation_embs, reverse_relation_dictionary, num_sequences, g, ent_dict, rel_dict, status

    status = "Loading Knowledge Graph..."
    preprocessor.load_knowledge_graph(format='xml', exclude_rels=exclude_rels, amberg_params=amberg_params)
    vocab_size = preprocessor.get_vocab_size()
    unique_msgs = preprocessor.get_unique_msgs()
    ent_dict = preprocessor.get_ent_dict()
    rel_dict = preprocessor.get_rel_dict()
    g = preprocessor.get_kg()

    print("Read {0} number of triples".format(len(g)))
    status = "Knowledge Graph loaded..."
    # get_kg_statistics(g)

    zero_shot_triples = []

    ######### Model selection ##########
    model_type = getattr(TranslationModels, modeltype)  # TranslationModels.Trans_E  # TEKE, RESCAL
    bernoulli = True
    event_layer = None  # globals()[eventlayer]  # Skipgram
    store_embeddings = True  # storeembeddings

    # pre-train event embeddings
    pre_train = False
    pre_train_embeddings = base_path + "Embeddings/supplied_embeddings"
    pre_train_steps = 10000

    ######### Hyper-Parameters #########
    param_dict = {}
    param_dict['embedding_size'] = [embedding_size]  # [40]
    param_dict['seq_data_size'] = [1.0]  # [seq_data_size]
    param_dict['batch_size'] = [batch_size]  # [32]  # [32, 64, 128]
    param_dict['learning_rate'] = [learning_rate]  # [0.2]  # [0.5, 0.8, 1.0]
    param_dict['lambd'] = [lambd]  # [0.001]  # regularizer (RESCAL)
    param_dict['alpha'] = [0.5]  # event embedding weighting
    eval_step_size = 1000
    # num_epochs = 100
    num_negative_triples = 2
    # test_proportion = 0.2
    # validation_proportion = 0.1  # 0.1
    bernoulli = True
    fnsim = l2_similarity

    # Train dev test splitting
    g_train, g_valid, g_test = slice_ontology(rnd, g, validation_proportion, test_proportion, zero_shot_triples)

    train_size = len(g_train)
    valid_size = len(g_valid)
    test_size = len(g_test)
    print("Train size: ", train_size)
    print("Valid size: ", valid_size)
    print("Test size: ", test_size)

    # SKIP Parameters
    if event_layer is not None:
        param_dict['num_skips'] = [2]  # range(5, 9)
        param_dict['num_sampled'] = [7]  # [5, 8]
        shared = True

        if traffic_data:
            sequences = preprocessor.prepare_sequences(path_to_sequence, use_dict=True)
        else:
            # merged = preprocessor.get_merged()
            # sequences = prepare_sequences(merged, message_index,
            #                                  unique_msgs, window_size, max_seq, g_train)
            sequences = preprocessor.prepare_sequences(path_to_sequence, use_dict=False)
        num_sequences = len(sequences)

    num_entities = len(ent_dict)
    num_relations = len(rel_dict)
    print("Num entities:", num_entities)
    print("Num relations:", num_relations)
    print("Event entity percentage: {0} prct".format(100.0 * vocab_size / num_entities))

    if bernoulli:
        bern_probs = bernoulli_probs(g, rel_dict)

    # free some memory
    # g = None  # Note: g is required to exclude existing links from link predictions, hence do not set to None
    model_name = TranslationModels.get_model_name(event_layer, model_type)
    overall_best_performance = np.inf
    best_param_list = []

    train_tg = TripleBatchGenerator(g_train, ent_dict, rel_dict, num_negative_triples, rnd, bern_probs=bern_probs)
    valid_tg = TripleBatchGenerator(g_valid, ent_dict, rel_dict, 1, rnd, sample_negative=False)
    test_tg = TripleBatchGenerator(g_test, ent_dict, rel_dict, 1, rnd, sample_negative=False)

    print(test_tg.next(5))

    # Loop trough all hyper-paramter combinations
    param_combs = cross_parameter_eval(param_dict)
    for comb_num, tmp_param_dict in enumerate(param_combs):
        params = Parameters(**tmp_param_dict)
        num_steps = int((train_size / params.batch_size) * num_epochs)

        print("Progress: {0} prct".format(int((100.0 * comb_num) / len(param_combs))))
        print("Embedding size: ", params.embedding_size)
        print("Batch size: ", params.batch_size)

        filter_triples = valid_tg.all_triples

        if event_layer is not None:
            if traffic_data:
                batch_size_sg = params.batch_size
            else:
                batch_size_sg = params.batch_size
            print("Batch size sg:", batch_size_sg)
            num_skips = params.num_skips
            num_sampled = params.num_sampled
            if event_layer == Skipgram:
                sg = SkipgramBatchGenerator(sequences, num_skips, rnd)
            elif event_layer == ConvolutionalAutoEncoder:
                sg = AutoEncoderBatchGenerator(sequences, num_skips, rnd)
            elif event_layer == ConcatenationFull:
                sg = FuturePredictiveBatchGenerator(sequences, num_skips, rnd)
                num_skips = 2 * num_skips
            elif event_layer == ConcatenationCause:
                sg = PredictiveEventBatchGenerator(sequences, num_skips, rnd)
            event_model = event_layer(num_entities, vocab_size, params.embedding_size, num_skips, shared=shared,
                                      alpha=params.alpha)
        else:
            batch_size_sg = 0
            num_sampled = 0
            event_model = None
            pre_train = False
            num_skips = 0

        # Model Selection
        if model_type == TranslationModels.Trans_E:
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                          batch_size_sg, num_sampled, vocab_size, fnsim, params.learning_rate,
                          event_model]
            model = TransE(*param_list)
        elif model_type == TranslationModels.Trans_H:
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                          batch_size_sg, num_sampled, vocab_size, params.learning_rate, event_model,
                          params.lambd]
            model = TransH(*param_list)
        elif model_type == TranslationModels.RESCAL:
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                          batch_size_sg, num_sampled, vocab_size, params.learning_rate, event_model,
                          params.lambd]
            model = RESCAL(*param_list)
        elif model_type == TranslationModels.TEKE:
            pre_trainer = EmbeddingPreTrainer(unique_msgs, SkipgramBatchGenerator(sequences, num_skips, rnd),
                                              pre_train_embeddings)
            initE = pre_trainer.get(pre_train_steps, params.embedding_size, batch_size_sg, num_sampled, vocab_size,
                                    num_entities)
            tk = TEKEPreparation(sequences, initE, num_entities)
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size, fnsim, tk]
            model = TEKE(*param_list)

        # Build tensorflow computation graph
        tf.reset_default_graph()
        # tf.set_random_seed(23)
        with tf.Session() as session:
            model.create_graph()
            saver = tf.train.Saver(model.variables())
            tf.global_variables_initializer().run()
            print('Initialized graph')
            status = "Initializing knowledge graph..."
            average_loss = 0
            mean_rank_list = []
            hits_10_list = []
            loss_list = []

            # Initialize some / event entities with supplied embeddings
            if pre_train and model_type != TranslationModels.TEKE:
                # TODO: adapt to selected event_model for pre-training
                pre_trainer = EmbeddingPreTrainer(unique_msgs, SkipgramBatchGenerator(sequences, num_skips, rnd),
                                                  pre_train_embeddings)
                initE = pre_trainer.get(pre_train_steps, params.embedding_size, batch_size_sg, num_sampled, vocab_size,
                                        num_entities)
                session.run(model.assign_initial(initE))

            if store_embeddings:
                entity_embs = []
                relation_embs = []

            # Steps loop
            for b in range(1, num_steps + 1):
                # triple batches
                batch_pos, batch_neg = train_tg.next(params.batch_size)
                valid_batch_pos, _ = valid_tg.next(valid_size)

                feed_dict = {
                    model.inpl: batch_pos[1, :], model.inpr: batch_pos[0, :], model.inpo: batch_pos[2, :],
                    model.inpln: batch_neg[1, :], model.inprn: batch_neg[0, :], model.inpon: batch_neg[2, :],
                    model.global_step: b
                }

                if event_model is not None and not model_type == TranslationModels.TEKE:
                    # Event batches
                    batch_x, batch_y = sg.next(batch_size_sg)
                    batch_y = np.array(batch_y).reshape((batch_size_sg, 1))
                    feed_dict[model.train_inputs] = batch_x
                    feed_dict[model.train_labels] = batch_y

                # One train step in mini-batch
                _, l = session.run(model.train(), feed_dict=feed_dict)
                average_loss += l
                # Run post-ops: regularization etc.
                session.run(model.post_ops())
                # Evaluate on validation set
                if b % eval_step_size == 0:
                    # get valid batches for scoring
                    valid_inpl = valid_batch_pos[1, :]
                    valid_inpr = valid_batch_pos[0, :]
                    valid_inpo = valid_batch_pos[2, :]

                    scores_l, scores_r = model.scores(session, valid_inpl, valid_inpr, valid_inpo)
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
                    print("\t Validation Hits10: " + str(hits_10))
                    print("\t Validation MeanRank: " + str(mean_rank))
                    average_loss = 0

                    if overall_best_performance > mean_rank:
                        overall_best_performance = mean_rank
                        print("Saving overall best model with MeanRank: {0} and hits {1}".format(mean_rank, hits_10))
                        status = "Saving overall best model in epoch {0}...".format(b)
                        save_path_global = saver.save(session, path_to_store_model + 'tf_model')
                        best_param_list = param_list

            reverse_entity_dictionary = dict(zip(ent_dict.values(), ent_dict.keys()))
            reverse_relation_dictionary = dict(zip(rel_dict.values(), rel_dict.keys()))

            # save embeddings to disk
            if store_embeddings:
                # Generate only one the first embedding initially..
                # Subsequent embeddings are generated dynamically, when invoked from the Front-end
                for i in range(1):  # range(len(entity_embs))
                    run_iteration(i)

                print("Saving best model embeddings...")
                status = "Saving best model embeddings..."
                # TODO: only of best model (not last)
                df_embs = embs_to_df(entity_embs[len(entity_embs) - 1], reverse_entity_dictionary)
                df_embs.to_csv(path_to_store_embeddings + "entity_embeddings" + '_last_cleaned' + ".csv", sep=',',
                               encoding='utf-8')

    # Reset graph, load best model and apply to test data set
    with open(base_path + 'evaluation_parameters_' + model_name +
                      '_best.csv', "w") as eval_file:
        writer = csv.writer(eval_file)
        results, relation_results = evaluate_on_test(model_type, best_param_list, test_tg, save_path_global, test_size,
                                                     reverse_relation_dictionary)
        writer.writerow(
            ["relation", "embedding_size", "batch_size", "learning_rate", "num_skips", "num_sampled",
             "batch_size_sg", "mean_rank", "mrr", "hits_top_10", "hits_top_3", "hits_top_1"]
        )
        writer.writerow(
            ['all', params.embedding_size, params.batch_size, params.learning_rate, num_skips, num_sampled,
             batch_size_sg, results[0], results[1], results[2], results[3], results[4]]
        )
        for rel in relation_results:
            writer.writerow(
                [rel, params.embedding_size, params.batch_size, params.learning_rate, num_skips, num_sampled,
                 batch_size_sg, relation_results[rel]['MeanRank'], relation_results[rel]['MRR'],
                 relation_results[rel]['Hits@10'], relation_results[rel]['Hits@3'], relation_results[rel]['Hits@1']]
            )
    status = "Done."


def generate_link_predictions(i):
    global status
    # Select Top-k links for each src_entity
    k = 3

    start = time()
    print("Generating link predictions...", entity_embs[i].shape, relation_embs[i].shape)
    status = "Generating link predictions..."
    existing_links = [(ent_dict[url_parse(s)], rel_dict[url_parse(p)], ent_dict[url_parse(o)]) for (s, p, o) in
                               g.triples((None, None, None))]

    # Compute all possible combinations for links between entities based on the relations -> (n_e * n_r * n_e-1) links
    temp = np.sqrt(np.square(entity_embs[i][:, None, None] + relation_embs[i][None, :, None] - entity_embs[i][None, None, :]).sum(axis=3, keepdims=True))

    # Exclude already existing links/known links
    for l in existing_links:
        temp[l] = 0

    # # Note: Self-relations/Loops are not explicitly excluded. It is later verified that these relations do not have
    # # high probability of being formed-hence are automatically excluded from the recommendations

    grid = np.indices((len(ent_dict), len(rel_dict), len(ent_dict)))
    s, p, o = grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)
    ids = list(zip(s, p, o))

    s_wise = temp.reshape(len(ent_dict), -1)
    sort_idx = np.argsort(-s_wise, axis=1)

    topk_predictions = [['_'.join(str(s) for s in ids[i * len(ent_dict) * len(rel_dict) + sort_idx[i][j]]), s_wise[i, sort_idx[i][j]]] for i in range(len(s_wise)) for j in range(k)]
    topk_predictions = np.hstack((topk_predictions, np.tile(np.array([[i] for i in range(k)]), (len(ent_dict), 1))))

    np.savetxt(path_to_store_embeddings + "topk_predictions_" + str(i) + ".csv", topk_predictions,
               header="subject_predicate_object,score,rank", delimiter=',', fmt='%s,%s,%s',
               comments='')

    print("Link predictions saved in", time() - start, "seconds")
    status = "Done."


def get_entity_class(entity_uri):
    if '#' in entity_uri:
        entity_class = entity_uri.split('#')[1].strip('- ')
    else:
        entity_class = entity_uri

    return entity_class.replace('\d+', '')


def run_iteration(i):
    global status

    status = "Computing low dimensional embeddings..."
    # Store the entity & relation embeddings to respective files
    df_embs = get_low_dim_embs(entity_embs[i], reverse_entity_dictionary)

    # Compute the class of the entities based on the uri
    df_embs['class'] = df_embs['uri'].apply(lambda x: get_entity_class(x))
    df_embs['class'] = df_embs['class'].str.replace('\d+', '')

    status = "Saving embeddings..."
    df_embs.to_csv(path_to_store_embeddings + "entity_embeddings_low" + str(i) + ".csv", sep=',', encoding='utf-8')

    df_r_embs = get_low_dim_embs(relation_embs[i], reverse_relation_dictionary)
    df_r_embs.to_csv(path_to_store_embeddings + "relation_embeddings_low" + str(i) + ".csv", sep=',', encoding='utf-8')

    generate_link_predictions(i)
