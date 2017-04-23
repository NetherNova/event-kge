import numpy as np
import pickle
from rdflib import ConjunctiveGraph, RDF, RDFS, OWL, URIRef
from etl import update_ontology, prepare_sequences, message_index, get_merged_dataframe, get_unique_entities, read_ontology, load_text_file
from sklearn.manifold import TSNE
import csv
import itertools
from model import dot_similarity, trans, ident_entity, l2_similarity
from TransE import TransE
from TransESq import TransESeq
from TransH import TransH
from RESCAL import RESCAL
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from model import ranking_error_triples
from pre_training import EmbeddingPreTrainer


rnd = np.random.RandomState(25)


class SkipgramBatchGenerator(object):
    def __init__(self, sequences, num_skips, batch_size):
        """
        center word is target, context should predict center word
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
                        # avoid the target_ind itself
                        continue
                    self.data.append( (seq[target_ind], seq[target_ind + i]) )
        rnd.shuffle(self.data)

    def next(self):
        batch_x = []
        batch_y = []
        for b in range(self.batch_size):
            self.data_index = self.data_index % len(self.data)
            batch_x.append(self.data[self.data_index][0])
            batch_y.append(self.data[self.data_index][1])
            self.data_index += 1
        return batch_x, batch_y


class PredictiveEventBatchGenerator(object):
    def __init__(self, sequences, num_skips, batch_size, include_seq_ids=False):
        """
        center word is target, context should predict center word
        :param sequences: list of lists of event entities
        :param num_skips:  window left and right of target
        :param batch_size:
        """
        self.sequences = sequences
        self.sequence_index = 0
        self.num_skips = num_skips
        self.event_index = num_skips
        self.batch_size = batch_size
        self.include_seq_ids = include_seq_ids
        self.prepare_target_skips()

    def prepare_target_skips(self):
        self.data_index = 0
        self.data = []
        for n, seq in enumerate(self.sequences):
            for target_ind in range(self.num_skips, len(seq)):
                tmp_list = []
                for i in range(-self.num_skips, 0):
                    tmp_list.append(seq[target_ind + i])
                if self.include_seq_ids:
                    self.data.append( (tmp_list, (n, seq[target_ind])) )
                else:
                    self.data.append( (tmp_list, seq[target_ind]) )
        rnd.shuffle(self.data)

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
    def __init__(self, triples, entity_dictionary, relation_dictionary, num_neg_samples, batch_size,
                 sample_negative=True, bern_probs=None):
        self.all_triples = []
        self.batch_index = 0
        self.batch_size = batch_size
        self.num_neg_samples = num_neg_samples
        self.entity_dictionary = entity_dictionary
        self.relation_dictionary = relation_dictionary
        self.sample_negative = sample_negative
        self.bern_probs = bern_probs
        self.ent_array = np.array(self.entity_dictionary.values())
        self.ids_in_ent_dict = dict(zip(self.ent_array, range(0, len(self.ent_array))))

        for (s, p, o) in triples:
            s = unicode(s)
            p = unicode(p)
            o = unicode(o)
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
        if self.sample_negative:
            batch_size_tmp = self.batch_size // self.num_neg_samples
        else:
            batch_size_tmp = self.batch_size

        for b in range(batch_size_tmp):
            if self.batch_index >= len(self.all_triples):
                self.batch_index = 0
            current_triple = self.all_triples[self.batch_index]
            # Append current triple twice
            if self.sample_negative:
                for i in xrange(self.num_neg_samples):
                    inpl.append(current_triple[0])
                    inpr.append(current_triple[2])
                    inpo.append(current_triple[1])
                    rn, ln, on = self.get_negative_sample(current_triple)
                    inpln.append(ln)
                    inprn.append(rn)
                    inpon.append(on)
            else:
                inpl.append(current_triple[0])
                inpr.append(current_triple[2])
                inpo.append(current_triple[1])
            self.batch_index += 1
        return np.array([inpr, inpl, inpo]), np.array([inprn, inpln, inpon])

    def get_negative_sample(self, (s_ind,p_ind,o_ind), left_probability=0.5):
        """
        Uniform sampling (avoiding correct triple from being sampled again)
        :param left_probability:
        :return:
        """
        if self.bern_probs:
            # with (tph / (tph + hpt)) probability we sample a *head*
            left_probability = self.bern_probs[p_ind]
        if rnd.binomial(1, left_probability) > 0:
            mask = np.ones(len(self.ent_array), dtype=bool)
            mask[self.ids_in_ent_dict[s_ind]] = 0
            sample_set = self.ent_array[mask]
            s_ind = rnd.choice(sample_set, 1)[0]
        else:
            mask = np.ones(len(self.ent_array), dtype=bool)
            mask[self.ids_in_ent_dict[o_ind]] = 0
            sample_set = self.ent_array[mask]
            o_ind = rnd.choice(sample_set, 1)[0]
        return o_ind, s_ind, p_ind


class Parameters(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def cross_parameter_eval(param_dict):
    keys = param_dict.keys()
    return [dict(zip(keys, k)) for k in itertools.product(*param_dict.values())]


def slice_ontology(ontology, valid_proportion, test_proportion):
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


def update_entity_relation_dictionary(ontology, ent_dict):
    """
    Given an existing entity dictionary, update it to *ontology*
    :param ontology:
    :param ent_dict: the existing entity dictionary
    :return:
    """
    rel_dict = {}
    ent_counter = 0
    fixed_ids = [id for id in ent_dict.values()]
    for h in ontology.subjects(None, None):
        uni_h = unicode(h)
        if uni_h not in ent_dict:
            while ent_counter in fixed_ids:
                ent_counter += 1
            ent_dict.setdefault(uni_h, ent_counter)
            ent_counter += 1
    for t in ontology.objects(None, None):
        uni_t = unicode(t)
        if uni_t not in ent_dict:
            while ent_counter in fixed_ids:
                ent_counter += 1
            ent_dict.setdefault(uni_t, ent_counter)
            ent_counter += 1
    for r in ontology.predicates(None, None):
        uni_r = unicode(r)
        if uni_r not in rel_dict:
            rel_dict.setdefault(uni_r, len(rel_dict))
    return ent_dict, rel_dict


def parse_axioms(ontology, ent_dict, rel_dict):
    schema_info = defaultdict(dict)
    subclass_info = []
    for s,p,o in ontology.triples((None, None, None)):
        if unicode(s) in rel_dict:
            if p == RDFS.domain:
                schema_info[rel_dict[unicode(s)]]["domain"] = ent_dict[unicode(o)]
            elif p == RDFS.range:
                schema_info[rel_dict[unicode(s)]]["range"] = ent_dict[unicode(o)]
            elif p == RDFS.subPropertyOf:
                schema_info[rel_dict[unicode(s)]]["sub"] = ent_dict[unicode(o)]
                schema_info[rel_dict[unicode(o)]]["sup"] = ent_dict[unicode(s)]
        if unicode(s) in ent_dict and unicode(o) in ent_dict and p == RDFS.subClassOf:
            subclass_info.append((ent_dict[unicode(s)], ent_dict[unicode(o)]))
    return schema_info, np.array(subclass_info)


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
    colnames = ['x' + str(i) for i in range(dim)]
    df = pd.DataFrame(low_dim_embs, columns=colnames)
    df["id"] = [i for i in range(low_dim_embs.shape[0])]
    df["uri"] = [reverse_dictionary[k] for k in reverse_dictionary]
    df = df.set_index("id")
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


class TranslationModels:
    Trans_E, Trans_H, RESCAL, Trans_E_seq = range(4)

    @staticmethod
    def get_model_name(event_layer, num):
        name = None
        if num == TranslationModels.Trans_E:
            name = "TransE"
        elif num == TranslationModels.Trans_H:
            name = "TransH"
        elif num == TranslationModels.Trans_E_seq:
            name = "TranesESq"
        else:
            name = "RESCAL"
        if event_layer is not None:
            name += "-" + event_layer
        return name

# PATH PARAMETERS
base_path = "./test_data/"
path_to_store_model = base_path + "Embeddings/"
path_to_events = base_path + "Sequences/" # TODO: should be optional if no skipgram stuff
path_to_schema = base_path + "Ontology/manufacturing_schema.rdf" # TODO: also optional if no schema present
path_to_kg = base_path + "Ontology/amberg_inferred.xml"   # "./test_data/amberg_inferred.xml"
path_to_store_sequences = base_path + "Sequences/"
path_to_store_embeddings = base_path + "Embeddings/"
sequence_file_name = "train_sequences"
public_data = False
train_path = "./wn18/train_reduced.txt"
valid_path = "./wn18/valid_reduced.txt"
test_path = "./wn18/test_reduced.txt"

num_sequences = None

if not public_data:
    max_events = 5000
    # sequence window size in minutes
    window_size = 3
    merged = get_merged_dataframe(path_to_events, max_events)
    unique_msgs, unique_vars, unique_mods, unique_fes = get_unique_entities(merged)
    # includes relations
    g = read_ontology(path_to_kg)
    print "Read %d number of triples" % len(g)
    g, ent_dict = update_ontology(g, unique_msgs, unique_mods, unique_fes, unique_vars, merged)
    print "After update: %d Number of triples: " % len(g)
    ent_dict, rel_dict = update_entity_relation_dictionary(g, ent_dict)
    g_schema = read_ontology(path_to_schema)
    _, subclass_info = parse_axioms(g_schema, ent_dict, rel_dict)
    # subclass_info = []
    vocab_size = len(unique_msgs)

# Hyper-Parameters
model_type = TranslationModels.Trans_E
bernoulli = True
# "Skipgram", "Concat", "LSTM"
event_layer = "Skipgram"
store_embeddings = False
param_dict = {}
param_dict['embedding_size'] = [60, 100, 140]
param_dict['seq_data_size'] = [1.0]
param_dict['batch_size'] = [64, 128]     # [32, 64, 128]
param_dict['learning_rate'] = [0.5, 1.0]     # [0.5, 0.8, 1.0]
param_dict['lambd'] = [0.0005]
# seq_data_sizes = np.arange(0.1, 1.0, 0.2)
eval_step_size = 100
test_proportion = 0.2
validation_proportion = 0.1
fnsim = l2_similarity
leftop = trans
rightop = ident_entity
supp_event_embeddings = None #"./Embeddings/supplied_embeddings_60.pickle"
pre_train = False
sub_prop_constr = False

# SKIP Parameters
if event_layer is not None:
    param_dict['num_skips'] = [3, 5]   # [2, 4]
    param_dict['num_sampled'] = [10]     # [5, 9]
    param_dict['batch_size_sg'] = [64, 128]     # [128, 512]
    num_sequences = prepare_sequences(merged, path_to_store_sequences + sequence_file_name, message_index, unique_msgs,
                                      window_size)

if public_data:
    print "Loading from text files"
    g_train = load_text_file(train_path)
    g_valid = load_text_file(valid_path)
    g_test = load_text_file(test_path)
    print "Merging graphs"
    g = ConjunctiveGraph()
    for s,p,o in g_train.triples((None, None, None)):
        g.add((s,p,o))
    for s, p, o in g_valid.triples((None, None, None)):
        g.add((s, p, o))
    for s, p, o in g_test.triples((None, None, None)):
        g.add((s, p, o))
    print "Updating graph"
    ent_dict = {}
    ent_dict, rel_dict = update_entity_relation_dictionary(g, ent_dict)
    vocab_size = 0
    subclass_info = []
    unique_msgs = {}
else:
    g_train, g_valid, g_test = slice_ontology(g, validation_proportion, test_proportion)
train_size = len(g_train)
valid_size = len(g_valid)
test_size = len(g_test)
print "Train size: ", train_size
print "Valid size: ", valid_size
print "Test size: ", test_size

num_entities = len(ent_dict)
num_relations = len(rel_dict)
print "Num entities:", num_entities
print "Num relations:", num_relations
print "Event entity percentage: %3.2f prct" %(100.0 * len(unique_msgs) / num_entities)

if bernoulli:
    bern_probs = bernoulli_probs(g, rel_dict)

# free some memory
g = None
model_name = TranslationModels.get_model_name(event_layer, model_type)
overall_best_performance = np.inf
best_param_list = []

with open("evaluation_parameters_10pct_" + model_name + ".csv", "wb") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ["embedding_size", "batch_size", "learning_rate", "num_skips", "num_sampled", "batch_size_sg", "fold",
         "training_step", "mean_rank", "hits_top_10", "loss"])
    # Loop trough all hyper-paramter combinations
    param_combs = cross_parameter_eval(param_dict)
    for comb_num, tmp_param_dict in enumerate(param_combs):
        params = Parameters(**tmp_param_dict)
        num_steps = (2345 / params.batch_size) * 100
        print "Progress: %d prct" %(int((100.0 * comb_num) / len(param_combs)))
        print "Embedding size: ", params.embedding_size
        print "Batch size: ", params.batch_size

        train_tg = TripleBatchGenerator(g_train, ent_dict, rel_dict, 2, params.batch_size, bern_probs=bern_probs)
        valid_tg = TripleBatchGenerator(g_valid, ent_dict, rel_dict, 1, valid_size, sample_negative=False)
        test_tg = TripleBatchGenerator(g_test, ent_dict, rel_dict, 1, test_size, sample_negative=False)

        filter_triples = valid_tg.all_triples

        if event_layer:
            sequences = pickle.load(open(path_to_store_sequences + sequence_file_name + ".pickle", "rb"))
            batch_size_sg = params.batch_size_sg
            num_skips = params.num_skips
            num_sampled = params.num_sampled
            if pre_train:
                pre_trainer = EmbeddingPreTrainer(unique_msgs, SkipgramBatchGenerator(sequences, num_skips, batch_size_sg),
                                                  params.embedding_size, vocab_size, num_sampled, batch_size_sg,
                                                  supp_event_embeddings)
                pre_trainer.train(5000)
                pre_trainer.save()
            if event_layer == "Skipgram":
                sg = SkipgramBatchGenerator(sequences, num_skips, batch_size_sg)
            elif event_layer == "LSTM":
                sg = PredictiveEventBatchGenerator(sequences, num_skips, batch_size_sg)
            elif event_layer == "Concat":
                sg = PredictiveEventBatchGenerator(sequences, num_skips, batch_size_sg, True)
        else:
            num_sampled = 1
            batch_size_sg = 0
            num_skips = 0
            sequences = []
            # dummy batch generator for empty sequence TODO: can we get rig of this?
            sg = SkipgramBatchGenerator(sequences, num_skips, batch_size_sg)

        # Model Selection
        if model_type == TranslationModels.Trans_E:
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                          batch_size_sg, num_sampled, vocab_size, leftop, rightop, fnsim, sub_prop_constr,
                          params.learning_rate, event_layer, params.lambd, subclass_info, num_sequences, num_skips]
            model = TransE(*param_list)
        elif model_type == TranslationModels.Trans_E_seq:
            zero_elements = np.array([i for i in range(num_entities) if i not in unique_msgs.values()])
            param_list = [num_entities, num_relations, params.embedding_size, params.embedding_size, params.batch_size,
                          batch_size_sg, num_sampled, vocab_size, leftop, rightop, fnsim, zero_elements, sub_prop_constr,
                          params.learning_rate, event_layer, params.lambd, subclass_info, num_sequences, num_skips]
            model = TransESeq(*param_list)
        elif model_type == TranslationModels.Trans_H:
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                          batch_size_sg, num_sampled, vocab_size, sub_prop_constr, params.learning_rate, event_layer,
                          params.lambd, subclass_info, num_sequences, num_skips]
            model = TransH(*param_list)
        elif model_type == TranslationModels.RESCAL:
            param_list = [num_entities, num_relations, params.embedding_size, params.batch_size,
                          batch_size_sg, num_sampled, vocab_size, sub_prop_constr, params.learning_rate, event_layer,
                          params.lambd, subclass_info, num_sequences, num_skips]
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
                    supplied_embeddings = pickle.load(f)
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
                batch_pos, batch_neg = train_tg.next()
                valid_batch_pos, _ = valid_tg.next()
                # Event batches
                batch_x, batch_y = sg.next()
                if event_layer == "Concat":
                    batch_y = np.array(batch_y)
                else:
                    batch_y = np.array(batch_y).reshape((batch_size_sg, 1))
                # calculate valid indices for scoring
                feed_dict = {model.inpl: batch_pos[1, :], model.inpr: batch_pos[0, :], model.inpo: batch_pos[2, :],
                             model.inpln: batch_neg[1, :], model.inprn: batch_neg[0, :], model.inpon: batch_neg[2, :],
                             model.train_inputs: batch_x, model.train_labels: batch_y,
                             model.global_step : b
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
                    feed_dict = {model.test_inpl: valid_inpl, model.test_inpo: valid_inpo,
                                 model.test_inpr: valid_inpr}

                    if model_type == TranslationModels.Trans_E_seq:
                        r_embs, embs, w_embs, v_embs = session.run([model.R, model.E, model.W, model.V],
                                                                   feed_dict=feed_dict)
                        scores_l = model.rank_left_idx(valid_inpr, valid_inpo, r_embs, embs, w_embs, v_embs)
                        scores_r = model.rank_right_idx(valid_inpl, valid_inpo, r_embs, embs, w_embs, v_embs)
                    elif model_type == TranslationModels.Trans_H:
                        r_embs, embs, w_embs = session.run([model.R, model.E, model.W], feed_dict=feed_dict)
                        scores_l = model.rank_left_idx(valid_inpr, valid_inpo, r_embs, embs, w_embs)
                        scores_r = model.rank_right_idx(valid_inpl, valid_inpo, r_embs, embs, w_embs)
                    else:
                        r_embs, embs = session.run([model.R, model.E], feed_dict=feed_dict)
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

                    if overall_best_performance > mean_rank:
                        overall_best_performance = mean_rank
                        print "Saving best model with MeanRank: %5.2f and hits %3.2f" %(mean_rank, hits_10)
                        save_path = saver.save(session, path_to_store_model + 'tf_model')
                        best_param_list = param_list

                if not store_embeddings:
                    entity_embs = [session.run(model.E)]
                    relation_embs = [session.run(model.R)]

            reverse_entity_dictionary = dict(zip(ent_dict.values(), ent_dict.keys()))
            if not public_data:
                for msg_name, msg_id in unique_msgs.iteritems():
                    reverse_entity_dictionary[msg_id] = msg_name
            reverse_relation_dictionary = dict(zip(rel_dict.values(), rel_dict.keys()))
            # save embeddings to disk
            if store_embeddings:
                for i in range(len(entity_embs)):
                    df_embs = get_low_dim_embs(entity_embs[i], reverse_entity_dictionary)
                    df_embs.to_csv(path_to_store_embeddings + "entity_embeddings" + str(i) + ".csv", sep=',',
                                    encoding='utf-8')

                    df_r_embs = get_low_dim_embs(relation_embs[i], reverse_relation_dictionary)
                    df_r_embs.to_csv(path_to_store_embeddings + "relation_embeddings" + str(i) + ".csv", sep=',',
                                     encoding='utf-8')

            print "Best validation hits10 local", best_hits_local
            print "Best validation MeanRank local", best_rank_local

            for i in range(len(mean_rank_list)):
                writer.writerow([params.embedding_size, params.batch_size, params.learning_rate, num_skips, num_sampled,
                                 batch_size_sg, i, mean_rank_list[i], hits_10_list[i], loss_list[i]])

################### Test Set Evaluation ######################
# Reset graph, load best model and apply to test data set
tf.reset_default_graph()
with tf.Session() as session:
    # Need to instantiate model again
    print best_param_list
    if model_type == TranslationModels.Trans_E:
        model = TransE(*best_param_list)
    elif model_type == TranslationModels.Trans_E_seq:
        model = TransESeq(*best_param_list)
    elif model_type == TranslationModels.Trans_H:
        model = TransH(*best_param_list)
    elif model_type == TranslationModels.RESCAL:
        model = RESCAL(*best_param_list)

    model.create_graph()
    saver = tf.train.Saver(model.variables())
    saver.restore(session, save_path)
    test_batch_pos, _ = test_tg.next()
    filter_triples = test_tg.all_triples

    test_inpl = test_batch_pos[1, :]
    test_inpr = test_batch_pos[0, :]
    test_inpo = test_batch_pos[2, :]
    feed_dict = {model.test_inpl: test_inpl, model.test_inpo: test_inpo,
                 model.test_inpr: test_inpr}

    if model_type == TranslationModels.Trans_E_seq:
        r_embs, embs, w_embs, v_embs = session.run([model.R, model.E, model.W, model.V],
                                                   feed_dict=feed_dict)
        scores_l = model.rank_left_idx(test_inpr, test_inpo, r_embs, embs, w_embs, v_embs)
        scores_r = model.rank_right_idx(test_inpl, test_inpo, r_embs, embs, w_embs, v_embs)
    elif model_type == TranslationModels.Trans_H:
        r_embs, embs, w_embs = session.run([model.R, model.E, model.W], feed_dict=feed_dict)
        scores_l = model.rank_left_idx(test_inpr, test_inpo, r_embs, embs, w_embs)
        scores_r = model.rank_right_idx(test_inpl, test_inpo, r_embs, embs, w_embs)
    elif model_type == TranslationModels.Trans_E:
        r_embs, embs = session.run([model.R, model.E], feed_dict=feed_dict)
        scores_l = model.rank_left_idx(test_inpr, test_inpo, r_embs, embs)
        scores_r = model.rank_right_idx(test_inpl, test_inpo, r_embs, embs)
    else:
        r_embs, embs = session.run([model.R, model.E], feed_dict=feed_dict)
        scores_l = model.rank_left_idx(test_inpr, test_inpo, r_embs, embs)
        scores_r = model.rank_right_idx(test_inpl, test_inpo, r_embs, embs)

    errl, errr = ranking_error_triples(filter_triples, scores_l, scores_r, test_inpl,
                                       test_inpo, test_inpr)
# END OF SESSION

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

relation_results = dict()
for i in np.unique(test_inpo):
    indices = np.argwhere(np.array(test_inpo) == i)
    err_arr = np.concatenate([np.array(errl)[indices], np.array(errr)[indices]])
    relation_results[reverse_relation_dictionary[i]] = \
        {'Hits@10': np.mean(err_arr <= 10) * 100}
    relation_results[reverse_relation_dictionary[i]]['MRR'] = np.mean(1.0 / err_arr)

for k, v in relation_results.iteritems():
    print k, v
