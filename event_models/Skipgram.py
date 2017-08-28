
from event_models.LinearEventModel import LinearEventModel


class Skipgram(LinearEventModel):
    def __init__(self, num_entites, vocab_size, embedding_size, num_skpis, shared=True, alpha=1.0):
        super(Skipgram, self).__init__(num_entites, vocab_size, embedding_size, num_skpis, shared, alpha)
        self.combine_op = lambda x: x

    @staticmethod
    def name():
        return "Skipgram"