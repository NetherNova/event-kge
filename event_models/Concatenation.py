
from EventModel import EventModel


class Concatentation(EventModel):
    def __init__(self, vocab_size, embedding_size, shared=True):
        if not shared:
            # own input layer
            self.V = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # Output layer
        self.nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
        self.nce_biases = tf.Variable(tf.truncated_normal([vocab_size]))

    def loss(self, num_sampled, train_labels, embeddings=None):
        embed_context = tf.reshape(embed, [embed.get_shape()[0].value, embed.get_shape()[1].value * embedding_size])

        # embed_context = tf.concat_v2([embed_context, sequence_vectors], axis=1)

        # embed_context = tf.Print(embed_context.get_shape(),[embed_context], "context: ")
        W = tf.Variable(
            tf.truncated_normal([vocab_size, embed_context.get_shape()[1].value],
                                stddev=1.0 / tf.sqrt(tf.constant(embedding_size, dtype=tf.float32))))
        bias = tf.Variable(tf.truncated_normal([vocab_size]))

        # Softmax of concatenated vectors
        concat_loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=W,
                           biases=bias,
                           labels=train_labels,
                           inputs=embed_context,
                           num_sampled=num_sampled,
                           num_classes=vocab_size,
                           remove_accidental_hits=True))
        return concat_loss

    def variables(self):
        if not shared:
            return [self.V]
        else:
            return []