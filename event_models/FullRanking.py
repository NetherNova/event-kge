
import tensorflow as tf
from models.model import max_margin


class FullRanking(object):
    def __init__(self, vocab_size, embedding_size, shared=True):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.shared = shared

    def create_graph(self):
        if not self.shared:
            self.V = tf.Variable()

    def loss(self, train_labels, train_indices, negative_labels, embeddings=None):
        if embeddings is None:
            embeddings = self.V

        embeddings = tf.nn.embedding_lookup(embeddings, train_indices)
        # average
        context_embs = tf.reduce_mean(embeddings, axis=1)

        pos_embs = tf.nn.embedding_lookup(embeddings, train_labels)
        neg_embs = tf.nn.embedding_lookup(embeddings, negative_labels)
        score_pos = tf.multiply(context_embs, pos_embs)
        score_neg = tf.multiply(context_embs, neg_embs)
        ranking_loss = max_margin(score_pos, score_neg)
        return ranking_loss
