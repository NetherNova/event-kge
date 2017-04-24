import tensorflow as tf
from model import skipgram_loss
import numpy as np
from model import SuppliedEmbedding, normalize


class EmbeddingPreTrainer(object):
    def __init__(self, ent_dict, batch_generator, embedding_size, vocab_size, num_sampled, batch_size, file_name):
        self.ent_dict = ent_dict
        self.batch_generator = batch_generator
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_sampled = num_sampled
        self.batch_size = batch_size
        self.model = SkipgramModel(embedding_size, batch_size, num_sampled, vocab_size)
        self.embs = None
        self.file_name = file_name

    def train(self, num_steps):
        print "Pre-training embeddings..."
        with tf.Session() as session:
            self.model.create_graph()
            self.normalized = normalize(self.model.variables())
            tf.global_variables_initializer().run()
            average_loss = 0
            for b in range(1, num_steps + 1):
                # Event batches
                batch_x, batch_y = self.batch_generator.next()
                batch_y = np.array(batch_y).reshape((self.batch_size, 1))
                feed_dict = {
                             self.model.train_inputs: batch_x, self.model.train_labels: batch_y
                             }
                _, l = session.run(self.model.train(), feed_dict=feed_dict)
                average_loss += l
                if b % 100 == 0:
                    print "Step %d - average loss %.2f " %(b, average_loss / 100.0)
                    average_loss = 0
            # TODO: normalize?
            self.embs = session.run(self.normalized)


    def save(self):
        if self.embs is None:
            print "No embeddings defined yet"
            return
        sup_embs = SuppliedEmbedding(self.embs, self.ent_dict)
        sup_embs.save_embedding(self.file_name)
        return sup_embs


class SkipgramModel(object):
    def __init__(self, embedding_size, batch_size, num_sampled, vocab_size):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.vocab_size = vocab_size

    def create_graph(self):
        w_bound = 0.5 * self.embedding_size
        self.E = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -w_bound, w_bound))
        self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
        sg_embed = tf.nn.embedding_lookup(self.E, self.train_inputs)
        self.loss = skipgram_loss(self.vocab_size, self.num_sampled, sg_embed, self.embedding_size, self.train_labels)
        self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

    def train(self):
        return [self.optimizer, self.loss]

    def variables(self):
        return self.E