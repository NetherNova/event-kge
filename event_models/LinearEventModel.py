import tensorflow as tf


class LinearEventModel(object):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skips, shared=True, alpha=1.0):
        self.num_entities = num_entites
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_skips = num_skips
        self.shared = shared
        self.alpha = alpha

    def create_graph(self):
        if not self.shared:
            # own input layer
            self.V = tf.Variable(tf.truncated_normal([self.num_entities, self.embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32))), name="V")

            self.update = tf.scatter_update(self.V, range(self.vocab_size, self.num_entities),
                                            tf.zeros([self.num_entities-self.vocab_size, self.embedding_size]))

        # Output layer
        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32))), name="Wnce")
        self.nce_biases = tf.Variable(tf.truncated_normal([self.vocab_size]), name="bnce")

    def loss(self, num_sampled, train_labels, train_indices, embeddings=None):
        if embeddings is None:
            embeddings = self.V

        train_embeddings = tf.nn.embedding_lookup(embeddings, train_indices)

        train_embeddings = self.combine_op(train_embeddings)

        skipgram_loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=embeddings,
                           biases=self.nce_biases,
                           labels=train_labels,
                           inputs=train_embeddings,
                           num_sampled=num_sampled,
                           num_classes=self.vocab_size,
                           remove_accidental_hits=True))
        return skipgram_loss

    def variables(self):
        if not self.shared:
            return [self.V]
        else:
            return []


class Concatenation(LinearEventModel):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skips, shared=True, alpha=1.0):
        super(Concatenation, self).__init__(num_entites, vocab_size, embedding_size, num_skips, shared, alpha)

        self.combine_op = lambda x: tf.reshape(x, [x.get_shape()[0].value, self.num_skips * self.embedding_size])

    def create_graph(self):
        if not self.shared:
            # own input layer
            self.V = tf.Variable(tf.truncated_normal([self.num_entities, self.embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32))), name="V")

            self.update = tf.scatter_update(self.V, range(self.vocab_size, self.num_entities),
                                            tf.zeros([self.num_entities-self.vocab_size, self.embedding_size]))

        # Output layer
        self.nce_weights = tf.Variable(tf.truncated_normal([self.vocab_size, self.num_skips * self.embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(self.embedding_size, dtype=tf.float32))), name="Wnce")
        self.nce_biases = tf.Variable(tf.truncated_normal([self.vocab_size]), name="bnce")

    @staticmethod
    def name():
        return "Concat"


class Skipgram(LinearEventModel):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skpis, shared=True, alpha=1.0):
        super(Skipgram, self).__init__(num_entites, vocab_size, embedding_size, num_skpis, shared, alpha)
        self.combine_op = lambda x: x

    @staticmethod
    def name():
        return "Skipgram"


class Average(LinearEventModel):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skpis, shared=True, alpha=1.0):
        super(Average, self).__init__(num_entites, vocab_size, embedding_size, num_skpis, shared, alpha)
        self.combine_op = lambda x: tf.reduce_mean(x, axis=1)

    @staticmethod
    def name():
        return "Average"