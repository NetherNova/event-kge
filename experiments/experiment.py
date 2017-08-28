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
import pickle
import numpy as np
import tensorflow as tf

from models.RESCAL import RESCAL
from models.TEKE import TEKE
from models.TransE import TransE
from models.TransH import TransH
from models.model import ranking_error_triples, l2_similarity, bernoulli_probs
from models.pre_training import EmbeddingPreTrainer, TEKEPreparation

from event_models.Skipgram import Skipgram
from event_models.Concatenation import Concatenation

from prep.batch_generators import SkipgramBatchGenerator, TripleBatchGenerator, PredictiveEventBatchGenerator, FuturePredictiveBatchGenerator
from prep.etl import prepare_sequences, message_index
from prep.preprocessing import PreProcessor
from experiments.experiment_helper import slice_ontology, get_kg_statistics, get_low_dim_embs, get_zero_shot_scenario, \
    cross_parameter_eval, TranslationModels, embs_to_df, Parameters, evaluate_on_test


# set fixed random seed
rnd = np.random.RandomState(42)


if __name__ == '__main__':
    ####### PATH PARAMETERS ########
    base_path = "../clones/"
    path_to_store_model = base_path + "Embeddings/"
    path_to_events = base_path + "Sequences/"
    path_to_kg = base_path + "Ontology/amberg_clone.rdf"
    path_to_store_sequences = base_path + "Sequences/"
    path_to_store_embeddings = base_path + "Embeddings/"
    traffic_data = False
    path_to_sequence = base_path + 'Sequences/sequence.txt'
    preprocessor = PreProcessor(path_to_kg)
    tk = None
    bern_probs = None
    num_sequences = None

    if traffic_data:
        exclude_rels = []
        preprocessor = PreProcessor(path_to_kg)
        preprocessor.load_unique_msgs_from_txt(base_path + 'unique_msgs.txt')
        amberg_params = None
    else:
        exclude_rels = ['http://www.siemens.com/ontology/demonstrator#tagAlias']
        max_events = None
        max_seq = None
        # sequence window size in minutes
        window_size = 0.5
        amberg_params = (path_to_events, max_events)

    preprocessor.load_knowledge_graph(format='xml', exclude_rels=exclude_rels, amberg_params=amberg_params)
    vocab_size = preprocessor.get_vocab_size()
    unique_msgs = preprocessor.get_unique_msgs()
    ent_dict = preprocessor.get_ent_dict()
    rel_dict = preprocessor.get_rel_dict()
    g = preprocessor.get_kg()

    print("Read {0} number of triples".format(len(g)))
    get_kg_statistics(g)

    # zero_shot_prop = 0.25
    # zero_shot_entity =  URIRef('http://www.siemens.com/ontology/demonstrator#Event') #URIRef('http://purl.oclc.org/NET/ssnx/ssn#Device')
    # zero_shot_relation = URIRef(RDF.type) # URIRef('http://www.loa-cnr.it/ontologies/DUL.owl#follows') # URIRef('http://www.siemens.com/ontology/demonstrator#involvedEquipment') URIRef('http://www.loa-cnr.it/ontologies/DUL.owl#hasPart')
    # zero_shot_triples, kg_prop = get_zero_shot_scenario(rnd, g, zero_shot_entity, zero_shot_relation, zero_shot_prop)
    zero_shot_triples = []

    ######### Model selection ##########
    model_type = TranslationModels.Trans_E
    # "Skipgram", "Concat", "RNN"
    event_layer = Concatenation
    store_embeddings = False

    ######### Hyper-Parameters #########
    param_dict = {}
    param_dict['embedding_size'] = [100]
    param_dict['seq_data_size'] = [1.0]
    param_dict['batch_size'] = [32]     # [32, 64, 128]
    param_dict['learning_rate'] = [0.3]     # [0.5, 0.8, 1.0]
    param_dict['lambd'] = [0.001]     # regularizer (RESCAL)
    param_dict['alpha'] = [1.0]     # event embedding weighting
    eval_step_size = 1000
    num_epochs = 100
    num_negative_triples = 2
    test_proportion = 0.2
    validation_proportion = 0.1
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
        param_dict['num_skips'] = [2]   # [2, 4]
        param_dict['num_sampled'] = [7]     # [5, 9]
        # param_dict['batch_size_sg'] = [2]     # [128, 512]
        pre_train_steps = 10000
        pre_train = False
        supp_event_embeddings = None  # base_path + "Embeddings/supplied_embeddings.pickle"
        if traffic_data:
            sequences = preprocessor.prepare_sequences(path_to_sequence)
        else:
            merged = preprocessor.get_merged()
            sequences = prepare_sequences(merged, message_index,
                                              unique_msgs, window_size, max_seq, g_train)
        num_sequences = len(sequences)

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

    train_tg = TripleBatchGenerator(g_train, ent_dict, rel_dict, num_negative_triples, rnd, bern_probs=bern_probs)
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
            if event_layer == Skipgram:
                sg = SkipgramBatchGenerator(sequences, num_skips, rnd)
            else:
                sg = PredictiveEventBatchGenerator(sequences, num_skips, rnd)
            event_model = event_layer(num_entities, vocab_size, params.embedding_size, params.num_skips, shared=False,
                                      alpha=params.alpha)
        else:
            num_sampled = 0
            batch_size_sg = 0
            num_skips = 0
            sequences = []
            # dummy batch generator for empty sequence TODO: can we get rid of this?
            sg = SkipgramBatchGenerator(sequences, num_skips, rnd)

        # Model Selection
        if model_type == TranslationModels.Trans_E:
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                          batch_size_sg, num_sampled, vocab_size, fnsim, params.learning_rate,
                          event_model]
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
            pre_trainer.train(10000)
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
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size, fnsim, tk]
            model = TEKE(*param_list)

        # Build tensorflow computation graph
        tf.reset_default_graph()
        # tf.set_random_seed(23)
        with tf.Session() as session:
            model.create_graph()
            saver = tf.train.Saver(model.variables())
            tf.global_variables_initializer().run()
            if event_model is not None and not event_model.shared:
                session.run([event_model.update])
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
                # triple batches
                batch_pos, batch_neg = train_tg.next(params.batch_size)
                valid_batch_pos, _ = valid_tg.next(valid_size)
                # Event batches
                batch_x, batch_y = sg.next(batch_size_sg)
                batch_y = np.array(batch_y).reshape((batch_size_sg, 1))

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
    with open(base_path + 'evaluation_parameters_' + model_name +
                      '_best.csv', "wb") as eval_file:
        writer = csv.writer(eval_file)
        results, relation_results = evaluate_on_test(model_type, best_param_list, test_tg, save_path_global, test_size,
                                                     reverse_relation_dictionary)
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