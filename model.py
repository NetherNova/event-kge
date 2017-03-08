import tensorflow as tf
import numpy as np
import math
import pickle

class SuppliedEmbedding(object):
    def __init__(self, W, dictionary):
        self._W = W
        self._dictionary = dictionary

    def get_embeddings(self):
        return self._W

    def get_dictionary(self):
        return self._dictionary

def save_embedding(file_name, dictionary, embeddings):
    suppl_emb = SuppliedEmbedding(embeddings, dictionary)
    pickle.dump(suppl_emb, open(file_name, "wb"))
    

class DotProduct(object):
    def __init__(self, left, right):
        self._left = left
        self._right = right

    def similarity(self):
        return tf.matmul(self._left, self._right)


class MaxMargin(object):
    """Class needs input of positive and noise/negative sample of context
        is not an exact loss-function (cannot be interpreted in a probabilistic sense"""
    def __init__(self, input_pos, input_neg):
        """TODO: how to make object itself the function?"""
        self._input_pos = input_pos
        self._input_neg = input_neg

    def loss(self):
        loss = tf.reduce_mean(tf.maximum(0., 1. - self._input_pos + self._input_neg))
        return loss

class Softmax(object):
    """Class needs input of sequence activaction vector, the context embeddings (previous lookup)
        and the actual labels
    """
    def __init__(self, context, labels, vocabulary_size, negative_sample_size, hidden_dim):
        self._context = context
        self._labels = labels
        self._vocabulary_size = vocabulary_size
        self._negative_sample_size = negative_sample_size
        self._nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, hidden_dim],
                      stddev=1.0 / math.sqrt(hidden_dim)))
        self._nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        #self._W = tf.Variable(tf.truncated_normal(shape = [vocabulary_size, hidden_dim])) # softmax layer (not interested in these representations)
        #self._bias = tf.Variable(tf.truncated_normal(shape = [vocabulary_size]))

    def loss(self):
        #logits = tf.nn.xw_plus_b(self._sequence_history, self._W, self._bias)
        #return tf.nn.softmax_cross_entropy_with_logits(logits, labels)
        #return tf.reduce_mean(
        #    tf.nn.sampled_softmax_loss(self._W, self._bias, self._context, self._labels, 5, self._vocabulary_size)) # provide the list of negative sample values
        return tf.reduce_mean(tf.nn.nce_loss(self._nce_weights, self._nce_biases, self._context, self._labels, self._negative_sample_size, self._vocabulary_size))


class RankingModel(object):
    def __init__(self, num_entities, num_hidden, num_dim):
        self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_hidden)))
        self.A = tf.Variable(tf.tuncated_normal(shape=(num_hidden, num_dim)))

    def loss(self, lookup_entities, labels):
        input_layer = tf.concat(tf.nn.embedding_lookup(self.W, lookup_entities), 0) # matrix with stacked input words
        hidden_layer = tf.nn.sigmoid(tf.reduce_sum(tf.matmul(self.W, input_layer)))
        scores = tf.nn.sigmoid(self.A * hidden_layer)

        return scores

    def rank_error(self, scores, labels):
        tf.gather(scores, tf.nn.top_k(scores, k = scores.get_shape().values[0]).indices) # sorted scores
        # loop through all combinations



class SkipgramModel(object):
    def __init__(self, label, num_entities, num_hidden, num_hidden_softmax):
        self.num_dim = num_hidden
        self.num_entities = num_entities
        self.num_hidden_softmax = num_hidden_softmax
        self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_hidden), name="W-"+label)) # hidden layer (i.e. embeddings)

    def loss(self, lookup_entities, labels):
        #TODO: embedding_lookup, sum over context
        # concatentation of previous layer --> num_hidden_softmax needs to be the size of the concatentation
        context_embeddings = tf.nn.embedding_lookup(self.W, lookup_entities)
        # label_embeddings = tf.nn.embedding_lookup(self.W, labels)
        loss = Softmax(context_embeddings, labels, self.num_entities, self.num_hidden_softmax).loss()
        return loss

    def get_normalized_embeddings(self):
        return self.W / tf.sqrt(tf.reduce_sum(tf.square(self.W), 1, keep_dims=True))

    def get_embeddings(self):
        return self.W     


class EventsWithWordsModel(object):
    def __init__(self, len_sequence, num_words_per_sequence, num_entities, embedding_size, num_label_entities, num_neg_samples):
        self._len_sequence = len_sequence
        self._num_words_per_sequence = num_words_per_sequence
        self._num_entities = num_entities
        self._embedding_size = embedding_size
        self._num_label_entities = num_label_entities
        self._num_neg_samples = num_neg_samples
        self.W = EmbeddingLayer("EventWords", num_entities, embedding_size)

    def loss(self, train_dataset, train_labels, batch_size):
        concat = IncrementalConcatLayer(self.W.get_embeddings(), train_dataset, batch_size, self._embedding_size, self._len_sequence, self._num_words_per_sequence)
        loss = Softmax(concat, train_labels, self._num_label_entities, self._num_neg_samples, (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size).loss()
        return loss


class EventsWithWordsAndVariantModel(object):
    def __init__(self, len_sequence, num_words_per_sequence, num_entities, embedding_size, num_label_events, num_label_variants, variant_index, num_neg_samples):
        self._len_sequence = len_sequence
        self._num_words_per_sequence = num_words_per_sequence
        self._num_entities = num_entities
        self._embedding_size = embedding_size
        self._num_label_events = num_label_events
        self._num_label_variants = num_label_variants
        self._num_neg_samples = num_neg_samples
        self._variant_index = variant_index
        self.W = EmbeddingLayer("EventWords", num_entities, embedding_size)

    def loss(self, train_dataset, train_labels_events, train_labels_variants, batch_size):
        concat = IncrementalConcatLayer(self.W.get_embeddings(), train_dataset, batch_size, self._embedding_size, self._len_sequence, self._num_words_per_sequence)
        variant_embeddings = tf.reshape(tf.nn.embedding_lookup(self.W.get_embeddings(), tf.slice(train_dataset, [0, self._variant_index], [batch_size, 1])), [batch_size, self._embedding_size])
        concat_variant = ConcatLayer(concat, variant_embeddings)
        concat_last_event = ConcatLayer(concat, tf.reshape(tf.nn.embedding_lookup(self.W.get_embeddings(), train_labels_events), [batch_size, self._embedding_size]))
        loss1 = Softmax(concat_variant, train_labels_events, self._num_label_events, self._num_neg_samples, (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size + self._embedding_size).loss()
        loss2 = Softmax(concat_last_event, train_labels_variants, self._num_label_variants, self._num_neg_samples - 3, (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size + self._embedding_size).loss()
        return loss1 + loss2

    def get_model(self, train_dataset, dataset_size):
        concat = IncrementalConcatLayer(self.W.get_embeddings(), train_dataset, dataset_size, self._embedding_size, self._len_sequence, self._num_words_per_sequence)
        variant_embeddings = tf.reshape(tf.nn.embedding_lookup(self.W.get_embeddings(), tf.slice(train_dataset, [0, self._variant_index], [dataset_size, 1])), [dataset_size, self._embedding_size])
        concat_variant = ConcatLayer(concat, variant_embeddings)
        return concat_variant

    def get_embeddings(self, dataset):
        return self.W.get_embeddings()
    

class EventsWithWordsAndVariantComposedModel(object):
    def __init__(self, len_sequence, num_words_per_sequence, num_entities, embedding_size, num_label_events, num_label_variants, variant_index, num_neg_samples, num_variant_parts):
        self._len_sequence = len_sequence
        self._num_words_per_sequence = num_words_per_sequence
        self._num_entities = num_entities
        self._embedding_size = embedding_size
        self._num_label_events = num_label_events
        self._num_label_variants = num_label_variants
        self._num_neg_samples = num_neg_samples
        self._variant_index = variant_index
        self._num_variant_parts = num_variant_parts
        self.W = EmbeddingLayer("EventWords", num_entities, embedding_size)

    def loss(self, train_dataset, train_labels_events, train_labels_variants, batch_size):
        # last entries in train_dataset (..., variant, part, part, ..., part)
        concat = IncrementalConcatLayer(self.W.get_embeddings(), train_dataset, batch_size, self._embedding_size, self._len_sequence, self._num_words_per_sequence)
        var_avg = AverageLayer(tf.nn.embedding_lookup(self.W.get_embeddings(), tf.slice(train_dataset, [0, self._variant_index], [batch_size, self._num_variant_parts])), axis=1)
        concat_variant_parts = ConcatLayer(concat, var_avg)
        concat_last_event = ConcatLayer(concat, tf.reshape(tf.nn.embedding_lookup(self.W.get_embeddings(), train_labels_events), [batch_size, self._embedding_size]))
        loss1 = Softmax(concat_variant_parts, train_labels_events, self._num_label_events, self._num_neg_samples, (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size + self._embedding_size).loss()
        loss2 = Softmax(concat_last_event, train_labels_variants, self._num_label_variants, self._num_neg_samples - 3, (self._len_sequence) * (self._num_words_per_sequence+1) * self._embedding_size + self._embedding_size).loss()
        return loss1 + loss2

    def get_model(self, train_dataset, dataset_size):
        concat = IncrementalConcatLayer(self.W.get_embeddings(), train_dataset, dataset_size, self._embedding_size, self._len_sequence, self._num_words_per_sequence)
        var_avg = AverageLayer(tf.nn.embedding_lookup(self.W.get_embeddings(), tf.slice(train_dataset, [0, self._variant_index], [dataset_size, self._num_variant_parts])), axis=1)
        concat_variant_parts = ConcatLayer(concat, var_avg)
        return concat_variant_parts

    def get_embeddings(self, dataset):
        return self.W.get_embeddings()

# relation between modules embedded score(e1) < score(e2)
# beginn und ende einer Sequenz (OEE Down / Stillstand)
# dependency strength (Modul -> same, directBefore, directAfter)
class EventsWithModules(object):
    def __init__(self, len_sequence):
        pass


def beforeAfterRanking(scores, indices_sorted_by_score, indices_actual_sorted):
    """for i in inidces_actual_sorted
           inner loop:
                var 0 - i scores[i] - scores[var] < 0 --> error_vector.append(1)
           inner loop:
                var i - end: scores[i] - scores[var] > 0 --> error_vector.append(1)
    """
    def body(i, x):
        a = tf.reshape(tf.nn.embedding_lookup(embeddings, tf.slice(train_dataset, [0, (num_words_per_sequence + 1) * i],
                                                                   [batch_size, (num_words_per_sequence + 1)])),
                       [batch_size, (num_words_per_sequence + 1) * embedding_size])
        # a = tf.reshape(tf.slice(train_emb, [0, i, 0], [batch_size, 1, embedding_size]), [batch_size, embedding_size])
        return i + 1, tf.concat(1, [x, a])

    def condition(i, x):
        return i < len_sequence

    i = tf.constant(1)
    init = tf.reshape(
        tf.nn.embedding_lookup(embeddings, tf.slice(train_dataset, [0, 0], [batch_size, (num_words_per_sequence + 1)])),
        [batch_size, (num_words_per_sequence + 1) * embedding_size])
    _, result = tf.while_loop(condition, body, [i, init],
                              shape_invariants=[i.get_shape(), tf.TensorShape([None, None])])
    return tf.reshape(result, [batch_size, len_sequence * (num_words_per_sequence + 1) * embedding_size])


def IncrementalConcatLayer(embeddings, train_dataset, batch_size, embedding_size, len_sequence, num_words_per_sequence):
    """Deprecated: simply use tf.reshape()"""
    def body(i, x):
        a = tf.reshape(tf.nn.embedding_lookup(embeddings, tf.slice(train_dataset, [0, (num_words_per_sequence+1)*i], [batch_size, (num_words_per_sequence+1)])), [batch_size, (num_words_per_sequence+1)*embedding_size])
        #a = tf.reshape(tf.slice(train_emb, [0, i, 0], [batch_size, 1, embedding_size]), [batch_size, embedding_size])
        return i+1, tf.concat(1, [x, a])

    def condition(i, x):
        return i < len_sequence
    
    i = tf.constant(1)
    init = tf.reshape(tf.nn.embedding_lookup(embeddings, tf.slice(train_dataset, [0,0], [batch_size, (num_words_per_sequence+1)])), [batch_size, (num_words_per_sequence+1)*embedding_size])
    _, result = tf.while_loop(condition, body, [i, init], shape_invariants=[i.get_shape(), tf.TensorShape([None, None])])
    return tf.reshape(result, [batch_size, len_sequence*(num_words_per_sequence+1)*embedding_size])

def ConcatLayer(left, right):
    return tf.concat(1, [left, right])

def AverageLayer(tensor, axis):
    #for 3-dim tensor with batches --> use axis=1
    return tf.reduce_mean(tensor, axis)

class EmbeddingLayer(object):
    def __init__(self, label, num_entities, num_hidden):
        self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_hidden), name="W-"+label))

    def get_embeddings(self):
        return self.W


class TranslationModelShared(object):
    """ TransE Model with """
    def __init__(self, num_entities, num_relations, num_dim, shared_entity_layer):
        # left entities come from shared lower layer (e.g. skipgram)
        self._shared_entity_layer = shared_entity_layer
        #self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_dim), name="WEnt"))
        self.R = tf.Variable(tf.truncated_normal(shape=(num_relations, num_dim), name="R"))

    def loss(self, left_entity, relation, pos_right_entity, neg_right_entity):
        embed_left = tf.nn.embedding_lookup(self._shared_entity_layer, left_entity)
        embed_relation = tf.nn.embedding_lookup(self.R, relation)

        
        left = embed_left + embed_relation
        pos_distance = tf.sqrt(tf.square(left) + tf.square(embed_pos_right))
        neg_distance = tf.sqrt(tf.square(left) + tf.square(embed_neg_right))

        return MaxMargin(pos_distance, neg_distance).loss()

    def get_normalized_embeddings(self, relation=False):
        if relation:
            return self.R / tf.sqrt(tf.reduce_sum(tf.square(self.R), 1, keep_dims=True))
        else:
            return self.W / tf.sqrt(tf.reduce_sum(tf.square(self.W), 1, keep_dims=True))


class EventEmbedding(object):
    def __init__(self, label, num_entities, num_events, num_dim):
        self.num_dim = num_dim
        self.num_entities = num_entities
        self.num_events = num_events
        self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_dim), name="W-"+label))
        self.W_events = tf.Variable(tf.truncated_normal(shape=(num_events, num_dim), name="W-events"))

    def loss(self, lookup_entity, negative_entity, event_entities):
        # works on batch of entities?
        embed_pos = tf.nn.embedding_lookup(self.W, lookup_entity)
        embed_neg = tf.nn.embedding_lookup(self.W, negative_entity)
        embed_context = tf.reduce_sum(tf.nn.embedding_lookup(self.W_events, event_entities), 1)

        # regularisieren / normalisieren
        sim_pos = DotProduct(embed_pos, tf.transpose(embed_context)).similarity()
        sim_neg = DotProduct(embed_neg, tf.transpose(embed_context)).similarity()

        loss = MaxMargin(sim_pos, sim_neg).loss() + 0.01*tf.nn.l2_loss(self.W) + 0.01*tf.nn.l2_loss(self.W_events)
        return loss

    def get_normalized_embeddings(self):
        return self.W / tf.sqrt(tf.reduce_sum(tf.square(self.W), 1, keep_dims=True))

    def evaluate_cosine_similarity(self, valid_dataset):
        normalized_embeddings = self.get_normalized_embeddings()
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        return tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

class ContextEventEmbedding(object):
    """For attached context information to labels (set of labels)"""
    def __init__(self, label, num_entities, num_events, num_dim):
        self.num_dim = num_dim
        self.num_entities = num_entities
        self.num_events = num_events
        self.W = tf.Variable(tf.truncated_normal(shape=(num_entities, num_dim), name="W-"+label))
        self.W_events = tf.Variable(tf.truncated_normal(shape=(num_events, num_dim), name="W-events"))

    def loss(self, lookup_entity, negative_entity, event_entities):
        # works on batch of entities?
        embed_pos = tf.reduce_sum(tf.nn.embedding_lookup(self.W, lookup_entity), 1)
        embed_neg = tf.reduce_sum(tf.nn.embedding_lookup(self.W, negative_entity), 1)
        embed_context = tf.reduce_sum(tf.nn.embedding_lookup(self.W_events, event_entities), 1)

        # regularisieren / normalisieren
        sim_pos = DotProduct(embed_pos, tf.transpose(embed_context)).similarity()
        sim_neg = DotProduct(embed_neg, tf.transpose(embed_context)).similarity()

        loss = MaxMargin(sim_pos, sim_neg).loss() + 0.01*tf.nn.l2_loss(self.W) + 0.01*tf.nn.l2_loss(self.W_events)
        return loss

    def get_normalized_embeddings(self):
        return self.W / tf.sqrt(tf.reduce_sum(tf.square(self.W), 1, keep_dims=True))

    def evaluate_cosine_similarity(self, valid_dataset):
        normalized_embeddings = self.get_normalized_embeddings()
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        return tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


class RecurrentEventEmbedding(EventEmbedding):
    def __init__(self, label, num_entities, num_dim):
        # here num_entities is only number of variants
        super(RecurrentEventEmbedding, self).__init__(label, num_entities, 0, num_dim)

    def loss(self, lookup_entity, negative_entity, context_entities):
        #cell = tf.nn.rnn_cell.BasicRNNCell(self.num_dim)
        cell = tf.nn.rnn_cell.LSTMCell(self.num_dim)
        #context_embedding = tf.nn.embedding_lookup(self.W, context_entities)
        #context_embedding =tf.unstack(context_embeddings)
        context_embedding = tf.pack(context_entities)
        context_embedding = tf.one_hot(context_embedding, 3)
        context_embedding = tf.unstack(context_embedding)
        outputs, state = tf.nn.rnn(cell, context_embedding, dtype=tf.float32)
        embed_context = state[1] # take last state only
        embed_pos = tf.nn.embedding_lookup(self.W, lookup_entity)
        embed_neg = tf.nn.embedding_lookup(self.W, negative_entity)
        score_pos = tf.matmul(embed_pos, tf.transpose(embed_context))
        score_neg = tf.matmul(embed_neg, tf.transpose(embed_context))

        loss = tf.reduce_mean(tf.maximum(0., 1. - score_pos + score_neg))
        return loss 

    def get_normalized_embeddings(self):
        return self.W / tf.sqrt(tf.reduce_sum(tf.square(self.W), 1, keep_dims=True))

    def evaluate_cosine_similarity(self, valid_dataset):
        """TODO: with sequences"""
        normalized_embeddings = self.get_normalized_embeddings()
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        return tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
