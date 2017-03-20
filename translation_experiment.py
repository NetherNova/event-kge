import numpy as np
import pickle
from rdflib import ConjunctiveGraph
from etl import update_ontology, unique_msgs, unique_mods, unique_fes, unique_vars, merged, prepare_sequences, message_index
from sklearn.manifold import TSNE
import csv
import itertools
from model import TranslationEmbeddings, dot_similarity, trans, ident_entity


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
                        # avoid the target_ind itself
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
    def __init__(self, triples, entity_dictionary, relation_dictionary, num_neg_samples, batch_size,
                 sample_negative=True):
        self.all_triples = []
        self.batch_index = 0
        self.batch_size = batch_size
        self.num_neg_samples = num_neg_samples
        self.entity_dictionary = entity_dictionary
        self.relation_dictionary = relation_dictionary
        self.sample_negative = sample_negative

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
        if self.sample_negative:
            batch_size_tmp = self.batch_size // 2
        else:
            batch_size_tmp = self.batch_size

        for b in range(batch_size_tmp):
            if self.batch_index >= len(self.all_triples):
                self.batch_index = 0
            current_triple = self.all_triples[self.batch_index]
            inpl.append(current_triple[0])
            inpr.append(current_triple[2])
            inpo.append(current_triple[1])
            # Append current triple twice
            if self.sample_negative:
                inpl.append(current_triple[0])
                inpr.append(current_triple[2])
                inpo.append(current_triple[1])

                rn, ln, on = self.get_negative_sample(current_triple, True)
                inpln.append(ln)
                inprn.append(rn)
                inpon.append(on)
                # repeat
                rn, ln, on = self.get_negative_sample(current_triple, False)
                inpln.append(ln)
                inprn.append(rn)
                inpon.append(on)
            self.batch_index += 1
        return np.array([inpr, inpl, inpo]), np.array([inprn, inpln, inpon])

    def get_negative_sample(self, (s_ind,p_ind,o_ind), left=True):
        if left:
            sample_set = [ent for ent in self.entity_dictionary.values() if ent != s_ind]
            s_ind = np.random.choice(sample_set, 1)[0]
        else:
            sample_set = [ent for ent in self.entity_dictionary.values() if ent != o_ind]
            o_ind = np.random.choice(sample_set, 1)[0]
        return o_ind, s_ind, p_ind


class Parameters(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def cross_parameter_eval(param_dict):
    keys = param_dict.keys()
    num_params = len(keys)
    value_lists = param_dict.values()
    return (dict(zip(keys, k)) for k in itertools.product(*param_dict.values()))


def slice_ontology(ontology, proportion):
    ont_slice = ConjunctiveGraph()
    reduced_size = int(np.floor(proportion * len(ontology)))
    remove_indices = np.random.randint(0, len(ontology), len(ontology) - reduced_size)
    for i, (s, p, o) in enumerate(ontology.triples((None, None, None))):
        if i in remove_indices:
            ontology.remove((s, p, o))
            ont_slice.add((s, p, o))
    return reduced_size, ont_slice


def plot_embeddings(embs, reverse_dictionary, data):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(embs)
    for i in range(low_dim_embs.shape[0]):
        if i not in reverse_dictionary:
            continue
        module = data[data["Meldetext"] == reverse_dictionary[i]]["Module"]

# Hyper-Parameters
param_dict = {}
param_dict['embedding_size'] = [60, 100, 140, 180]
param_dict['seq_data_size'] = [1.0]
param_dict['batch_size'] = [32, 64, 128]
param_dict['learning_rate'] = [0.5, 0.8, 1.0]
# seq_data_sizes = np.arange(0.1, 1.0, 0.2)
n_folds = 4
test_proportion = 0.03
fnsim = dot_similarity
leftop = trans
rightop = ident_entity

# SKIP
skipgram = True
if skipgram:
    param_dict['num_skips'] = [2, 4]
    param_dict['num_sampled'] = [5, 9]
    param_dict['batch_size_sg'] = [128, 512]
    prepare_sequences(merged, "train_sequences", message_index, classification_event=None)


with open("evaluation_kg_skipgram_parameters_3pct" ".csv", "wb") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ["embedding_size", "batch_size", "learning_rate", "num_skips", "num_sampled", "batch_size_sg", "fold",
         "training_step", "mean_rank", "hits_top_10", "grad_E", "grad_R", "loss"])
    for tmp_param_dict in cross_parameter_eval(param_dict):
        params = Parameters(**tmp_param_dict)
        print "Embedding size: ", params.embedding_size
        print "Batch size: ", params.batch_size
        print "Learning rate: ", params.learning_rate
        for fold in xrange(n_folds):
            print "Fold: ", fold
            # TODO: second (transformed) version of ontology
            g = ConjunctiveGraph()
            g.load("./test_data/amberg_inferred.xml")

            g, uri_to_id = update_ontology(g, unique_msgs, unique_mods, unique_fes, unique_vars, merged)
            ent_dict = uri_to_id
            rel_dict = {}
            for t in g.triples((None, None, None)):
                if t[0] not in ent_dict:
                    ent_dict.setdefault(t[0], len(ent_dict))
                if t[1] not in ent_dict:
                    ent_dict.setdefault(t[2], len(ent_dict))
                rel_dict.setdefault(t[1], len(rel_dict))

            train_size = slice_ontology(g, params.seq_data_size)
            print "Train size: ", train_size
            # randomly reduce g
            test_size, g_test = slice_ontology(g, test_proportion)
            print "Test size: ", test_size

            num_entities = len(ent_dict)
            num_relations = len(rel_dict)
            tg = TripleBatchGenerator(g, ent_dict, rel_dict, 1, params.batch_size)
            test_tg = TripleBatchGenerator(g_test, ent_dict, rel_dict, 1, test_size, sample_negative=False)
            sequences = [seq.split(' ') for seq in pickle.load(open("./test_data/train_sequences.pickle", "rb"))]
            # sequences = sequences[: int(np.floor(len(sequences) *  0.5))]
            if skipgram:
                batch_size_sg = params.batch_size_sg
                num_skips = params.num_skips
                sg = SkipgramBatchGenerator(sequences, num_skips, batch_size_sg)
                num_sampled = params.num_sampled
            else:
                sg = None
                num_sampled = 0
                batch_size_sg = 0
                num_skips = 0
            reverse_dictionary = dict(zip(unique_msgs.values(), unique_msgs.keys()))
            model = TranslationEmbeddings(num_entities, num_relations, params.embedding_size, params.batch_size,
                                          batch_size_sg, num_sampled, len(unique_msgs),
                                          leftop, rightop, fnsim)
            embs, r_embs, best_hits, best_rank, mean_rank_list, hits_10_list, gradients_E, gradients_R, loss_list = \
                model.run(tg, sg, reverse_dictionary, merged, test_tg, test_size, 300, params.learning_rate, skipgram)

            print "Best hits", best_hits
            print "Best rank", best_rank

            for i in range(len(mean_rank_list)):
                writer.writerow([params.embedding_size, params.batch_size, params.learning_rate, num_skips, num_sampled,
                                 batch_size_sg, fold, i, mean_rank_list[i], hits_10_list[i], gradients_E[i],
                                 gradients_R[i], loss_list[i]])