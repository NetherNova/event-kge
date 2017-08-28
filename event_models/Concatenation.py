
from event_models.LinearEventModel import LinearEventModel
import tensorflow as tf


class Concatenation(LinearEventModel):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skips, shared=True, alpha=1.0):
        super(Concatenation, self).__init__(num_entites, vocab_size, embedding_size, num_skips, shared, alpha)

        self.combine_op = lambda x: tf.reshape(x, [x.get_shape()[0].value, self.num_skips * self.embedding_size])

    def create_graph(self):
        if not self.shared:
            # own input layer
            self.V = tf.Variable(tf.truncated_normal([self.num_entities, self.embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32))))

            self.update = tf.scatter_update(self.V, range(self.vocab_size, self.num_entities),
                                            tf.zeros([self.num_entities-self.vocab_size, self.embedding_size]))

        # Output layer
        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.num_skips * self.embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32))))
        self.nce_biases = tf.Variable(tf.truncated_normal([self.vocab_size]))

    @staticmethod
    def name():
        return "Concat"