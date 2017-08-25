
from EventModel import EventModel


class Skipgram(EventModel):
    def __init__(self, vocab_size, embedding_size, shared=True):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.shared = shared

        if not shared:
            # own input layer
            self.V = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))

        # Output layer
        self.nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
        self.nce_biases = tf.Variable(tf.truncated_normal([vocab_size]))

    def loss(self, num_sampled, train_labels, train_indeces, embeddings=None):
        if embeddings is None:
            embeddings = self.V
        skipgram_loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=self.nce_weights,
                           biases=self.nce_biases,
                           labels=train_labels,
                           inputs=embeddings,
                           num_sampled=num_sampled,
                           num_classes=self.vocab_size,
                           remove_accidental_hits=True))
        return skipgram_loss

    def variables(self):
        if not self.shared:
            return [self.V]
        else:
            return []