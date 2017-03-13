import tensorflow as tf
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import EventEmbedding, ContextEventEmbedding, RecurrentEventEmbedding, SkipgramModel, TranslationModelShared


# # # DBLP # # #
def read_data(path):
  # Sequence not author overlapping?
  pass


# TODO:
# data preparation + part model
# events as bag of terms vector


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(28, 28))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 fontsize=12,
                 ha='right',
                 va='bottom')
  plt.show()  
  plt.savefig(filename)


class Experiment(object):
    def __init__(self, input_data):
        """
        basic inputs are always list of (message, module, variant)
        possibly sorted in any order
        """
        self._input_data = input_data

    def prepare_data(self):
        pass

    def run(self, model):
        #model.loss()
        #model.optimize()
        pass
    

class SimpleEmbeddingExperiment(Experiment):
    def __init__(self, input_data, sequence_size):
        super(SimpleEmbeddingExperiment, self).__init__(input_data)
        self._train_data_set = []    # events
        self._train_labels = []  # context
        self._event_dictionary = dict()
        self._event_reverse_dictionary = dict()
        self._context_dictionary = dict()
        self._context_reverse_dictionary = dict()
        self._sequence_size = sequence_size
        self._data_index = 0

    def prepare_data(self):
      """positive and negative context"""
      dictionary = dict()
      # populate dicts with unique ids
      for message, module, variant in self._input_data:
          self._event_dictionary.setdefault(message, len(self._event_dictionary))
          self._context_dictionary.setdefault(variant, len(self._context_dictionary))
      self._context_reverse_dictionary = dict(zip(self._context_dictionary.values(), self._context_dictionary.keys()))
      self._event_reverse_dictionary = dict(zip(self._event_dictionary.values(), self._event_dictionary.keys()))
      for i in range(len(data)):
        if i+self._sequence_size >= len(data):
            break
        window = data[i: i+self._sequence_size]
        first_variant = window[0][2]
        # verify always same variant for window
        if len(np.unique([window[x][2] for x in range(sequence_size)])) > 1:
            #print("non-unique variant: %s" %(window))
            continue
        #if all([(d[2] == first_variant) for d in window]):
        self._train_data_set.append([self._event_dictionary[d[0]] for d in window])
        self._train_labels.append(self._context_dictionary[first_variant])

    def generate_batch(self, batch_size):
        self._data_index = (self._data_index + 1) % len(self._train_data_set) # begins at zero again, after biggern than len(data)
        batch = np.array(self._train_data_set[self._data_index : self._data_index + batch_size])
        #neg_batch_labels = np.array(train_label_s)[np.random.randint(0, len(train_data_set), batch_size)]
        batch_labels = np.array(self._train_labels[self._data_index : self._data_index + batch_size])
        neg_batch_labels = self.random_sample(batch_labels)
        return batch, neg_batch_labels, batch_labels

    def random_sample(self, valid_batch_labels):
        # take a random context label from unique_labels
        result = []
        for i in range(len(valid_batch_labels)):
            unique_labels = self.get_unique_labels().copy()
            current_label = valid_batch_labels[i]
            unique_labels.remove(current_label)
            unique_labels = list(unique_labels)
            result.append(unique_labels[np.random.randint(0, len(unique_labels))])
        return np.array(result)
        
    def run(self, model, batch_size, num_steps):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            model = EventEmbedding("Test", self.get_vocabulary_size(), self.get_vocabulary_size(context=False), 64)
            train_dataset = tf.placeholder(tf.int32, [batch_size, self._sequence_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size])
            neg_dataset = tf.placeholder(tf.int32, shape=[batch_size])
            
            loss = model.loss(train_labels, neg_dataset, train_dataset)
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
        num_steps = 500
        with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          average_loss = 0
          for step in range(num_steps):
            batch_data, neg_batch, batch_labels = self.generate_batch(batch_size)
            # batch_labels = zip(batch_labels.to_list(), [w[0] for w in batch_labels[1:]])
            #feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            feed_dict = dict()
            feed_dict[train_dataset] = batch_data
            feed_dict[train_labels] = batch_labels
            feed_dict[neg_dataset] = neg_batch
            _, l = session.run(
                [optimizer, loss], feed_dict=feed_dict)
            #feed_dict = {train_dataset : batch_data, neg_dataset : neg_batch, train_labels : batch_labels}
            #_, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 20 == 0:
              if step > 0:
                average_loss = average_loss / 20
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step %d: %f' % (step, average_loss))
              average_loss = 0
          final_embeddings = model.get_normalized_embeddings().eval()
        return final_embeddings

    def get_vocabulary_size(self, context=True):
        if context:
            return len(self._context_dictionary.keys())
        else:
            return len(self._event_dictionary.keys())

    def get_unique_labels(self):
        return set(self._context_dictionary.values())

""" SKIPGRAM-Model """
class SimpleSkipgramExperiment(SimpleEmbeddingExperiment):
    # in this case, sequence size is window before and after...
    def __init__(self, input_data, sequence_size):
        super(SimpleSkipgramExperiment, self).__init__(input_data, sequence_size)

    def prepare_data(self):
        for message, module, variant in self._input_data:
            self._event_dictionary.setdefault(message, len(self._event_dictionary))
        self._event_dictionary["UNK"] = len(self._event_dictionary)
        self._event_reverse_dictionary = dict(zip(self._event_dictionary.values(), self._event_dictionary.keys()))
        train, labels = context_window(self._sequence_size, [self._event_dictionary[message] for (message, module, variant) in self._input_data])
        for i in range(len(labels)):
            for j in range(len(train[0])):
                self._train_data_set.append(labels[i])
                self._train_labels.append(train[i][j])

    def generate_batch(self, batch_size):
        """can stay the same?"""
        self._data_index = (self._data_index + 1) % len(self._train_data_set) # begins at zero again, after biggern than len(data)
        batch = np.array(self._train_data_set[self._data_index : self._data_index + batch_size])
        batch_labels = np.array(self._train_labels[self._data_index : self._data_index + batch_size])
        batch_labels = batch_labels.reshape((batch_size, 1))
        return batch, batch_labels

    def run(self, model, batch_size, num_steps):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            model = SkipgramModel("Skipgram", self.get_vocabulary_size(context=False), 128, 128)
            train_dataset = tf.placeholder(tf.int32, [batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            
            loss = model.loss(train_dataset, train_labels)
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
        with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          average_loss = 0
          for step in range(num_steps):
            batch_data, batch_labels = self.generate_batch(batch_size)
            feed_dict = dict()
            feed_dict[train_dataset] = batch_data
            feed_dict[train_labels] = batch_labels
            _, l = session.run(
                [optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 20 == 0:
              if step > 0:
                average_loss = average_loss / 20
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step %d: %f' % (step, average_loss))
              average_loss = 0
            final_embeddings = model.get_normalized_embeddings().eval()
        return final_embeddings, self._event_reverse_dictionary


""" Share Translagion and SKIPGRAM-Model """
class SharedSkipgramExperiment(SimpleEmbeddingExperiment):
    # in this case, sequence size is window before and after...
    def __init__(self, input_data, sequence_size):
        super(SharedSkipgramExperiment, self).__init__(input_data, sequence_size)
        self._relation_dictionary = dict()

    def prepare_data(self):
        skip = self._sequence_size // 2
        for message, module, variant in self._input_data:
            self._event_dictionary.setdefault(message, len(self._event_dictionary))
            #self._context_dictionary.setdefault(module, len(self._context_dictionary))
            self._context_dictionary.setdefault(variant, len(self._context_dictionary))
        self._event_dictionary["UNK"] = len(self._event_dictionary)
        self._relation_dictionary["hasVariant"] = 0
        #self._relation_dictionary["hasModule"] = 1
        self._event_reverse_dictionary = dict(zip(self._event_dictionary.values(), self._event_dictionary.keys()))
        self._context_reverse_dictionary = dict(zip(self._context_dictionary.values(), self._context_dictionary.keys()))
        labels, train = context_window(self._sequence_size, [self._event_dictionary[message] for (message, module, variant) in self._input_data])
        for i in range(len(train)):
            for j in range(len(labels[0])):
                self._train_data_set.append((train[i], 0, self._context_dictionary[self._input_data[i + skip][2]], self.negative_sample(self._input_data[i + skip][2])))
                self._train_labels.append(labels[i][j])
        print("Training data: ", self._train_data_set[:20])

    def negative_sample(self, pos_entity):
        unique_indices = set(self._context_dictionary.values()) # makes copy
        pos_index = self._context_dictionary[pos_entity]
        unique_indices.remove(pos_index)
        return list(unique_indices)[np.random.randint(0, len(unique_indices))]

    def generate_batch(self, batch_size):
        """can stay the same?"""
        self._data_index = (self._data_index + 1) % len(self._train_data_set) # begins at zero again, after biggern than len(data)
        batch = np.array(self._train_data_set[self._data_index : self._data_index + batch_size])
        batch_labels = np.array(self._train_labels[self._data_index : self._data_index + batch_size])
        batch_labels = batch_labels.reshape((batch_size, 1))
        return batch, batch_labels

    def run(self, model, batch_size, num_steps, embedding_size):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            model1 = SkipgramModel("Skipgram", self.get_vocabulary_size(context=False), embedding_size, embedding_size)
            model2 = TranslationModelShared(self.get_vocabulary_size(context=True), 2, embedding_size, model1.get_embeddings())
            
            train_dataset = tf.placeholder(tf.int32, [batch_size, 4])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

            train_data_model1 = tf.slice(train_dataset, [0, 0], [batch_size, 1])
            train_data_model2_relation = tf.slice(train_dataset, [0, 1], [batch_size, 1])
            train_data_model2_pos_right_entity = tf.slice(train_dataset, [0, 2], [batch_size, 1])
            train_data_model2_neg_right_entity = tf.slice(train_dataset, [0, 3], [batch_size, 1])
            
            loss1 = model1.loss(tf.reshape(train_data_model1, [batch_size]), train_labels)
            loss2 = model2.loss(train_data_model1, train_data_model2_relation, train_data_model2_pos_right_entity, train_data_model2_neg_right_entity)
            joint_loss = loss1 + loss2
            
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(joint_loss)
            
        with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          average_loss = 0
          for step in range(num_steps):
            batch_data, batch_labels = self.generate_batch(batch_size)
            feed_dict = dict()
            feed_dict[train_dataset] = batch_data
            feed_dict[train_labels] = batch_labels
            _, l = session.run(
                [optimizer, joint_loss], feed_dict=feed_dict)
            average_loss += l
            if step % 20 == 0:
              if step > 0:
                average_loss = average_loss / 20
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step %d: %f' % (step, average_loss))
              average_loss = 0
            final_embeddings = model1.get_normalized_embeddings().eval()
            final_embeddings_context = model2.get_normalized_embeddings().eval()
        return final_embeddings, self._event_reverse_dictionary, final_embeddings_context, self._context_reverse_dictionary

class ContextualSimpleEventEmbeddingExperiment(SimpleEmbeddingExperiment):
    def __init__(self, input_data, sequence_size, meta_dict):
        super(ContextualSimpleEventEmbeddingExperiment, self).__init__(input_data, sequence_size)
        self._meta_dict = meta_dict
        print(self._meta_dict)
        #self.prepare_data()
    
    def prepare_data(self):
      """augment labels with variant information"""
      # populate dicts with unique ids
      for message, module, variant in self._input_data:
          self._event_dictionary.setdefault(message, len(self._event_dictionary))
          if variant in self._meta_dict:
              for w in self._meta_dict[variant]:
                  self._context_dictionary.setdefault(w, len(self._context_dictionary))
          else:
              continue
      self._context_reverse_dictionary = dict(zip(self._context_dictionary.values(), self._context_dictionary.keys()))
      self._event_reverse_dictionary = dict(zip(self._event_dictionary.values(), self._event_dictionary.keys()))
      for i in range(len(self._input_data)):
        if i+self._sequence_size >= len(self._input_data):
            break
        window = self._input_data[i: i+self._sequence_size]
        first_variant = window[0][2]
        # verify always same variant for window
        if len(np.unique([window[x][2] for x in range(sequence_size)])) > 1:
            #print("non-unique variant: %s" %(window))
            continue
        #if all([(d[2] == first_variant) for d in window]):
        self._train_data_set.append([self._event_dictionary[d[0]] for d in window])
        self._train_labels.append([self._context_dictionary[w] for w in self._meta_dict[first_variant]])

    def generate_batch(self, batch_size):
        """can stay the same?"""
        self._data_index = (self._data_index + 1) % len(self._train_data_set) # begins at zero again, after biggern than len(data)
        batch = np.array(self._train_data_set[self._data_index : self._data_index + batch_size])
        #neg_batch_labels = np.array(train_label_s)[np.random.randint(0, len(train_data_set), batch_size)]
        batch_labels = np.array(self._train_labels[self._data_index : self._data_index + batch_size])
        neg_batch_labels = self.random_sample(batch_labels)
        return batch, neg_batch_labels, batch_labels

    def random_sample(self, valid_batch_labels):
        """have to augment sample with contex information"""
        # take a random context label from unique_labels
        result = []
        for i in range(len(valid_batch_labels)):
            unique_labels = self.get_unique_labels().copy()
            current_labels = valid_batch_labels[i] # [0,6,8]
            unique_labels = list(unique_labels)
            result.append(np.random.randint(0, len(unique_labels), len(current_labels))) # sample |len| current label replacements
        return np.array(result)
        
    def run(self, model, batch_size, num_steps):
        print('Running Model')
        graph = tf.Graph()
        with graph.as_default():
            model = ContextEventEmbedding("Test", self.get_vocabulary_size(), self.get_vocabulary_size(context=False), 64)
            train_dataset = tf.placeholder(tf.int32, [batch_size, self._sequence_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, None])
            neg_dataset = tf.placeholder(tf.int32, shape=[batch_size, None])
            
            loss = model.loss(train_labels, neg_dataset, train_dataset)
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
        num_steps = 200
        with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          average_loss = 0
          for step in range(num_steps):
            batch_data, neg_batch, batch_labels = self.generate_batch(batch_size)
            # batch_labels = zip(batch_labels.to_list(), [w[0] for w in batch_labels[1:]])
            #feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            feed_dict = dict()
            feed_dict[train_dataset] = batch_data
            feed_dict[train_labels] = batch_labels
            feed_dict[neg_dataset] = neg_batch
            _, l = session.run(
                [optimizer, loss], feed_dict=feed_dict)
            #feed_dict = {train_dataset : batch_data, neg_dataset : neg_batch, train_labels : batch_labels}
            #_, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 20 == 0:
              if step > 0:
                average_loss = average_loss / 20
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step %d: %f' % (step, average_loss))
              average_loss = 0
            final_embeddings = model.get_normalized_embeddings().eval()
        return final_embeddings



class RecurrentEmbeddingExperiment(SimpleEmbeddingExperiment):
    def __init__(self, data, sequence_size):
        super(RecurrentEmbeddingExperiment, self).__init__(data, sequence_size)
        
    def prepare_data(self):
      """positive and negative context"""
      # populate dicts with unique ids
      for message, module, variant in self._input_data:
          self._event_dictionary.setdefault(message, len(self._event_dictionary))
          #self._event_dictionary.setdefault(variant, len(self._event_dictionary))
          self.context_dictionary.setdefault(variant, len(dictionary))
      self.reverse_event_dictionary = dict(zip(self._event_dictionary.values(), self._event_dictionary.keys()))
      self.reverse_context_dictionary = dict(zip(self.context_dictionary.values(), self.context_dictionary.keys()))
      for i in range(len(data)):
        if i+sequence_size >= len(data):
            break
        window = data[i: i+sequence_size]
        first_variant = window[0][2]
        # verify always same variant for window
        if len(np.unique([window[x][2] for x in range(sequence_size)])) > 1:
            #print("non-unique variant: %s" %(window))
            continue
        #if all([(d[2] == first_variant) for d in window]):
        self._train_data_set.append([self._event_dictionary[d[0]] for d in window])
        self._train_labels.append(self._context_dictionary[first_variant]) 


    def generate_batch(self, batch_size):
        # negative sampling of event objects for one target that is to be replaced with a random one
        self._data_index = (self._data_index + 1) % len(self._train_data_set) # begins at zero again, after biggern than len(data)
        batch = np.array(self._train_data_set[self._data_index : self._data_index + batch_size])
        batch_seq = []
        for i in range(sequence_size):
          batch_seq.append([])
        for t_entry in batch:
          for i in range(sequence_size):
              batch_seq[i].append(t_entry[i])
        #neg_batch_labels = np.array(train_label_s)[np.random.randint(0, len(train_data_set), batch_size)]
        batch_labels = np.array(self._train_labels[self._data_index : self._data_index + batch_size])
        neg_batch_labels = self.random_sample(batch_labels)
        return batch_seq, neg_batch_labels, batch_labels

    def random_sample(self, valid_batch_labels):
        # take a random context label from unique_labels
        result = []
        for i in range(len(valid_batch_labels)):
            unique_labels = set(self._train_labels) #unique_train_labels.copy()
            current_label = valid_batch_labels[i]
            unique_labels.remove(current_label)
            unique_labels = list(unique_labels)
            result.append(unique_labels[np.random.randint(0, len(unique_labels))])
        return np.array(result)

    def run(self, model, batch_size, num_steps):
        num_steps = 200
        graph = tf.Graph()
        unique_entities = set(self._train_labels)
        with graph.as_default():
            model = RecurrentEventEmbedding("Test", len(unique_entities), 64)
            train_dataset = list()
            for _ in range(sequence_size):
                train_dataset.append(tf.placeholder(tf.int32, shape=[batch_size]))
            train_labels = tf.placeholder(tf.int32, shape=[batch_size])
            neg_dataset = tf.placeholder(tf.int32, shape=[batch_size])
            valid_dataset = tf.constant(np.array(list(unique_entities)), dtype=tf.int32)
            
            loss = model.loss(train_labels, neg_dataset, train_dataset)
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
        with tf.Session(graph=graph) as session:
          tf.global_variables_initializer().run()
          print('Initialized')
          average_loss = 0
          for step in range(num_steps):
            batch_data, neg_batch, batch_labels = self.generate_batch(
              batch_size)
            # batch_labels = zip(batch_labels.to_list(), [w[0] for w in batch_labels[1:]])
            #feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
            feed_dict = dict()
            for i in range(sequence_size):
                feed_dict[train_dataset[i]] = batch_data[i]
            feed_dict[train_labels] = batch_labels
            feed_dict[neg_dataset] = neg_batch
            _, l = session.run(
                [optimizer, loss], feed_dict=feed_dict)
            #feed_dict = {train_dataset : batch_data, neg_dataset : neg_batch, train_labels : batch_labels}
            #_, l = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += l
            if step % 20 == 0:
              if step > 0:
                average_loss = average_loss / 20
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step %d: %f' % (step, average_loss))
              average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 200 == 0:
              sim = similarity.eval()
              for i in range(valid_size):
                valid_word = idx_to_e[valid_examples[i]]
                top_k = 2 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                  close_word = idx_to_e[nearest[k]]
                  log = '%s %s,' % (log, close_word)
                print(log)
            final_embeddings = model.get_normalized_embeddings().eval()
        return final_embeddings

if __name__ == '__main__':
    data = read_data("./test data/")
    print(len(data))
    #meta_data = read_metadata("./variant_info_parsed.txt")
    sequence_size = 5
    batch_size = 64
    embedding_size = 64
    num_steps = 500
    experiment = SharedSkipgramExperiment(data, sequence_size) #SimpleSkipgramExperiment(data, sequence_size) #RecurrentEmbeddingExperiment(data, sequence_size)#ContextualSimpleEventEmbeddingExperiment(data, sequence_size, meta_data) #SimpleEmbeddingExperiment(data, sequence_size)
    experiment.prepare_data()
    final_embeddings, reverse_dictionary, final_embeddings_context, context_reverse_dictionary = experiment.run("model", batch_size, num_steps, embedding_size)
    np.savetxt("final_embeddings.txt", final_embeddings, delimiter=",")
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = len(context_reverse_dictionary)
    low_dim_embs = tsne.fit_transform(final_embeddings_context[:plot_only, :])
    labels = [context_reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)
    

