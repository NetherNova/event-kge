import tensorflow as tf
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import EventEmbedding, EmbeddingLayer, ConcatLayer, Softmax, EventsWithWordsModel, EventsWithWordsAndVariantModel, EventsWithWordsAndVariantComposedModel
from experiments import SimpleEmbeddingExperiment, plot_with_labels
from etl import read_data, read_metadata, context_window_for_classification, context_window_last_event
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from model import save_embedding


def split_message(message):
    return message.split(' ')


def increment_dict_values(dictionary, value):
    for k,v in dictionary.iteritems():
        dictionary[k] = v + value


class EventWordSkipgramExperiment(SimpleEmbeddingExperiment):
    """
    SKIPGRAM-Model in this case, sequence size is window before and after
    """
    def __init__(self, input_data, meta_data, sequence_size, sample_words_size):
        super(EventWordSkipgramExperiment, self).__init__(input_data, sequence_size)
        self._word_dictionary = dict()
        self._word_reverse_dictionary = dict()
        self._variant_dictionary = dict()
        self._variant_reverse_dictionary = dict()
        self._variant_parts_dictionary = dict()
        self._variant_parts_reverse_dictionary = dict()
        self._module_dictionary = dict()
        self._module_reverse_dictionary = dict()
        self._sample_words_size = sample_words_size
        self._meta_data = meta_data

    def prepare_data(self):
        # Message Dict
        for message, module, variant, _ in self._input_data:
            self._event_dictionary.setdefault(message, len(self._event_dictionary))
            self._variant_dictionary.setdefault(variant, len(self._variant_dictionary))
            for part in self._meta_data[variant]:
                self._variant_parts_dictionary.setdefault(part, len(self._variant_parts_dictionary))
            self._module_dictionary.setdefault(module, len(self._module_dictionary))
            for w in split_message(message):
                self._word_dictionary.setdefault(w, len(self._word_dictionary))    
        self._event_dictionary["UNK_E"] = len(self._event_dictionary)
        self._word_dictionary["UNK_W"] =  len(self._word_dictionary)
        self._variant_dictionary["UNK_V"] = len(self._variant_dictionary)
        self._variant_parts_dictionary["UNK_VP"] = len(self._variant_parts_dictionary)
        self._module_dictionary["UNK_M"] = len(self._module_dictionary)
        # increment all entries
        increment_dict_values(self._word_dictionary, len(self._event_dictionary))
        value = len(self._event_dictionary) + len(self._word_dictionary)
        increment_dict_values(self._variant_dictionary, value)
        value += len(self._variant_dictionary)
        increment_dict_values(self._variant_parts_dictionary, value)
        value += len(self._variant_parts_dictionary)
        increment_dict_values(self._module_dictionary, value)
        
        self._event_reverse_dictionary = dict(zip(self._event_dictionary.values(), self._event_dictionary.keys()))
        self._module_reverse_dictionary = dict(zip(self._event_dictionary.values(), self._event_dictionary.keys()))
        self._word_reverse_dictionary = dict(zip(self._word_dictionary.values(), self._word_dictionary.keys()))
        self._variant_reverse_dictionary = dict(zip(self._variant_dictionary.values(), self._variant_dictionary.keys()))
        self._variant_parts_reverse_dictionary = dict(zip(self._variant_parts_dictionary.values(), self._variant_parts_dictionary.keys()))
        print(max(self._event_dictionary.values()))
        print(min(self._word_dictionary.values()), max(self._word_dictionary.values()))
        print(min(self._variant_dictionary.values()), max(self._variant_dictionary.values()))
        # (message_id, variant_id) train and labels for embeddings!
        train, labels = context_window_last_event(self._sequence_size,
                                                  [(self._event_dictionary[message], self._variant_dictionary[variant], self._module_dictionary[module], timestamp)
                                                   for (message, module, variant, timestamp) in self._input_data])
        # train_instance = [event_1 word11 word12 word13 event_2 word21 word22 word23 event_3 word31 word32 word33, variant]
        for i in range(len(labels)): # for every sequence
            train_tmp = []
            for j in range(len(train[0]) - 1): # for every event
                train_tmp.append(train[i][j])
                words = split_message(self._event_reverse_dictionary[train[i][j]])
                indices = np.random.choice(len(words), np.min([self._sample_words_size, len(words)]), replace=False)
                sampled_words = [words[sampled_index] for sampled_index in indices]
                train_tmp += [self._word_dictionary[w] - len(self._module_dictionary) for w in sampled_words]
                for z in range(self._sample_words_size-len(sampled_words)):
                    train_tmp.append(self._word_dictionary["UNK_W"])
            train_tmp.append(train[i][len(train[0]) - 1]) # last thing to append is variant
            num_parts = 2
            parts = self._meta_data[self._variant_reverse_dictionary[train[i][len(train[0]) - 1]]]
            part_indices = np.random.choice(len(parts), np.min([num_parts, len(parts)]), replace=False)
            for p in part_indices:
                part_index = self._variant_parts_dictionary[parts[p]]
                train_tmp.append(part_index) #(+ composed parts num_parts )
            self._train_data_set.append(train_tmp)
            self._train_labels.append([labels[i][0], labels[i][1] - min(self._variant_dictionary.values())])
        print("Original events: ", len(self._input_data))
        print(self._input_data[:10])
        print("Train data sequences: ", len(self._train_data_set))
        print(self._train_data_set[:10])
        print("Train labels: ", len(self._train_labels))
        print(self._train_labels[:10])

    def populate_and_transform_sequences(self, train, labels):
        pass
        
    def get_vocabulary(self):
        merge = dict(zip(self.get_event_vocabulary().keys() + self.get_word_vocabulary().keys(), self.get_event_vocabulary().values() + self.get_word_vocabulary().values()))
        merge = dict(zip(merge.keys() + self.get_variant_parts_vocabulary().keys(), merge.values() + self.get_variant_parts_vocabulary().values()))
        merge = dict(zip(merge.keys() + self.get_module_vocabulary().keys(), merge.values() + self.get_module_vocabulary().values()))
        return dict(zip(merge.keys() + self.get_variant_vocabulary().keys(), merge.values() + self.get_variant_vocabulary().values()))

    def get_event_vocabulary(self):
        return self._event_dictionary

    def get_word_vocabulary(self):
        return self._word_dictionary

    def get_variant_vocabulary(self):
        return self._variant_dictionary

    def get_variant_parts_vocabulary(self):
        return self._variant_parts_dictionary

    def get_module_vocabulary(self):
        return self._module_dictionary

    def generate_batch(self, batch_size):
        """can stay the same?"""
        if self._data_index + batch_size > len(self._train_data_set):
            self._data_index = 0
        #self._data_index = (self._data_index + 1) % len(self._train_data_set) # begins at zero again, after bigger than len(data)
        batch = np.array(self._train_data_set[self._data_index : self._data_index + batch_size])
        batch_labels = np.array(self._train_labels[self._data_index : self._data_index + batch_size])
        batch_labels = batch_labels.reshape((batch_size, 2))
        self._data_index += batch_size
        return batch, batch_labels

    def run(self, model, batch_size, num_steps, embedding_size, classification_size):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            # batch: [event_1 word11 word12 word13 event_2 word21 word22 word23 event_3 word31 word32 word33 ... event_ni]
            # train_labels [batch_size, num_types] - loop through types for each batch
            train_dataset = tf.placeholder(tf.int32, [batch_size, None])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 2])

            classification_dataset = tf.placeholder(tf.int32, [classification_size, None]) # Fall: wenn nur words + variant, mit UNK_E events?

            train_labels_events = tf.slice(train_labels, [0, 0], [batch_size, 1])
            train_labels_variants = tf.slice(train_labels, [0, 1], [batch_size, 1])

            event_emb = EmbeddingLayer("Event", len(self.get_vocabulary()), embedding_size)
            #word_emb = EmbeddingLayer("Word", len(self.get_word_vocabulary()), embedding_size)
            #model = EventsWithWordsModel(self._sequence_size - 1 , self._sample_words_size, len(self.get_word_vocabulary()) + len(self.get_event_vocabulary()), embedding_size, len(self.get_event_vocabulary()), 5)
            #model = EventsWithWordsAndVariantModel(self._sequence_size - 1, self._sample_words_size,
            #                                       len(self.get_word_vocabulary()) + len(self.get_event_vocabulary()) + len(self.get_variant_vocabulary()),
            #                                                                             embedding_size, len(self.get_event_vocabulary()), len(self.get_variant_vocabulary()), 8, 5)
            
            model = EventsWithWordsAndVariantComposedModel(self._sequence_size - 1, self._sample_words_size,
                                                   len(self.get_word_vocabulary()) + len(self.get_event_vocabulary()) + len(self.get_variant_vocabulary()) + len(self.get_variant_parts_vocabulary()),
                                                                                         embedding_size, len(self.get_event_vocabulary()), len(self.get_variant_vocabulary()), 8, 5, 2)
            """
            #[batch_size, 4, n_dim]
            event1_withwords = tf.reshape(tf.nn.embedding_lookup(event_emb.get_embeddings(),tf.slice(train_dataset, [0, 0], [batch_size, 4])), [batch_size, 4*embedding_size])
            event2_withwords = tf.reshape(tf.nn.embedding_lookup(event_emb.get_embeddings(),tf.slice(train_dataset, [0, 4], [batch_size, 4])), [batch_size, 4*embedding_size])
            variant = tf.reshape(tf.nn.embedding_lookup(event_emb.get_embeddings(),tf.slice(train_dataset, [0, 8], [batch_size, 1])), [batch_size, embedding_size])

            c_final = ConcatLayer(event1_withwords, event2_withwords)
            c_final = ConcatLayer(c_final, variant)

            c_final_variant = ConcatLayer(event1_withwords, event2_withwords)
            c_final_variant = ConcatLayer(c_final_variant, tf.reshape(tf.nn.embedding_lookup(event_emb.get_embeddings(), train_labels_events), [batch_size, embedding_size]))

            # for every "type" a separate softmax
            # at each step sample a type, then train this softmax
            # conditional switch in training data
            model = Softmax(c_final, train_labels_events, len(self.get_event_vocabulary()), 5, c_final.get_shape()[1].value)
            model2 = Softmax(c_final_variant, train_labels_variants, len(self.get_variant_vocabulary()), 2, c_final_variant.get_shape()[1].value) # predict variant from all events
            
            loss = model.loss()
            loss2 = model2.loss()
            joint_loss = loss + loss2
            """
            #learning_rate = tf.placeholder(tf.float32, shape=[])
            #loss = model.loss(train_dataset, train_labels_events, batch_size)
            #joint_loss = model.loss(train_dataset, train_labels_events, train_labels_variants, batch_size)
            joint_loss = model.loss(train_dataset, train_labels_events, train_labels_variants, batch_size)
            optimizer1 = tf.train.AdagradOptimizer(1.0).minimize(joint_loss) # Adagrad penalizes learning rate for parameters that get updated frequently
            #optimizer2 = tf.train.AdagradOptimizer(2.0).minimize(loss2)
            
            # Classification data preparation
            c_final_classification_dataset = model.get_model(classification_dataset, classification_size)
            """
            event1_withwords_classification = tf.reshape(tf.nn.embedding_lookup(event_emb.get_embeddings(),tf.slice(classification_dataset, [0, 0], [classification_size, 4])), [classification_size, 4*embedding_size])
            event2_withwords_classification = tf.reshape(tf.nn.embedding_lookup(event_emb.get_embeddings(),tf.slice(classification_dataset, [0, 4], [classification_size, 4])), [classification_size, 4*embedding_size])
            event3_withwords_classification = tf.reshape(tf.nn.embedding_lookup(event_emb.get_embeddings(),tf.slice(classification_dataset, [0, 8], [classification_size, 4])), [classification_size, 4*embedding_size])
            variant_classification = tf.reshape(tf.nn.embedding_lookup(event_emb.get_embeddings(),tf.slice(classification_dataset, [0, 12], [classification_size, 1])), [classification_size, embedding_size])
            c_final_classification_train = ConcatLayer(event1_withwords_classification, event2_withwords_classification)
            c_final_classification_train = ConcatLayer(c_final_classification_train, event3_withwords_classification)
            c_final_classification_train = ConcatLayer(c_final_classification_train, variant_classification)
            # spit out, then in numpy use cross-validation with SVM/LogisticRegression to classify
            """
        with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          classification_dataset_feed, classification_labels = self.classification(classification_data, classification_size)
          average_loss = 0
          for step in range(num_steps):
            batch_data, batch_labels = self.generate_batch(batch_size)
            feed_dict = dict()
            feed_dict[train_dataset] = batch_data
            feed_dict[train_labels] = batch_labels
            _, l = session.run(
                [optimizer1, joint_loss], feed_dict=feed_dict)
            average_loss += l
            if step % 200 == 0:
              if step > 0:
                average_loss = average_loss / 200
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step %d: %f' % (step, average_loss))
              average_loss = 0
            """
            if step % 1000 == 0:
               feed_dict[classification_dataset] = np.array(classification_dataset_feed)
               classification_dataset_np = session.run([c_final_classification_dataset], feed_dict=feed_dict)
               print np.average(cross_validate(classification_dataset_np[0], classification_labels))
            #second function
            _, l = session.run(
                [optimizer2, loss2], feed_dict=feed_dict)
            average_loss += l
            if step % 200 == 0:
              if step > 0:
                average_loss = average_loss / 200
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step %d: %f' % (step, average_loss))
              average_loss = 0
            """
          final_embeddings = event_emb.get_embeddings().eval()
        return final_embeddings, self.get_event_vocabulary(), c_final_classification_dataset, classification_labels

    def classification(self, input_data, classification_size):
        # classify root-cause analysis, given sequence, what is the most likely cause (common cause scenario)
        # other way round, given one event, what is the sequence of causes max margin max(1.0, score_neg - score_true (RNN?) 
        classification_dataset_l = []
        tmp_sequence = []
        for (message, module, variant, time) in classification_data:
            if message in self._event_dictionary:
                tmp_message = self._event_dictionary[message]
            else:
                tmp_message = self._event_dictionary["UNK_E"]
            if variant in self._variant_dictionary:
                tmp_variant = self._variant_dictionary[variant]
            else:
                tmp_variant = self._variant_dictionary["UNK_V"]
            if module in self._module_dictionary:
                tmp_module = self._module_dictionary[module]
            else:
                tmp_module = self._module_dictionary["UNK_M"]
            tmp_sequence.append((tmp_message, tmp_variant, tmp_module, time))
        train, labels = context_window_for_classification(tmp_sequence, 3, classification_size)
        sample_words_size = 3
        for i in range(len(labels)):
            train_tmp = []
            for j in range(len(train[0]) - 1): # for every event
                train_tmp.append(train[i][j])
                # TODO: test for event known or unknown
                words = split_message(self._event_reverse_dictionary[train[i][j]])
                if self._event_reverse_dictionary[train[i][j]] == "UNK_E":
                    for z in range(self._sample_words_size):
                        train_tmp.append(self._word_dictionary["UNK_W"])
                else:
                    indices = np.random.choice(len(words), np.min([self._sample_words_size, len(words)]), replace=False)
                    sampled_words = [words[sampled_index] for sampled_index in indices]
                    train_tmp += [self._word_dictionary[w] - len(self._module_dictionary) for w in sampled_words]
                    for z in range(self._sample_words_size-len(sampled_words)):
                        train_tmp.append(self._word_dictionary["UNK_W"])
            train_tmp.append(train[i][len(train[0]) - 1]) # last thing to append is variant
            num_parts = 2
            parts = self._meta_data[self._variant_reverse_dictionary[train[i][len(train[0]) - 1]]]
            part_indices = np.random.choice(len(parts), np.min([num_parts, len(parts)]), replace=False)
            for p in part_indices:
                if parts[p] in self._variant_parts_dictionary:
                    part = self._variant_parts_dictionary[parts[p]]
                else:
                    part = self._variant_parts_dictionary["UNK_VP"]
                train_tmp.append(part) #(+ composed parts num_parts )
            classification_dataset_l.append(train_tmp)
        return classification_dataset_l, labels


def cross_validate(data, labels):
    clf = LogisticRegression()
    scores = cross_val_score(clf, data, labels, cv=5, scoring="f1_macro")
    return scores


if __name__ == '__main__':
    data = read_data("./test_data/")
    print(len(data))
    max_events=len(data)
    data = np.array(data[:max_events])
    
    meta_data = read_metadata("./test_data/variant_info_parsed.txt")
    classification_data = np.array(read_data("./test_data/"))
    classification_size = 500
    sequence_size = 5
    sample_words_size = 3
    variant_index = 8
    batch_size = 128
    embedding_size = 128
    num_steps = 10000
    experiment = EventWordSkipgramExperiment(data, meta_data, sequence_size, sample_words_size) #SimpleSkipgramExperiment(data, sequence_size)
    experiment.prepare_data()
    final_embeddings, dictionary, classification_dataset, classification_labels = experiment.run("model", batch_size, num_steps, embedding_size, classification_size)
    save_embedding("embeddings.pickle", dictionary, final_embeddings)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    np.savetxt("final_embeddings.txt", final_embeddings, delimiter=",")
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #plot_only = len(context_reverse_dictionary)
    plot_only = 20
    low_dim_embs = tsne.fit_transform(final_embeddings[40:40+plot_only, :])
    plot_labels = [reverse_dictionary[40 + i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, plot_labels)
