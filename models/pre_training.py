import numpy as np
import tensorflow as tf
import itertools
from scipy.sparse import lil_matrix
from models.model import SuppliedEmbedding, normalize, SkipgramModel
import pickle


class EmbeddingPreTrainer(object):
    def __init__(self, ent_dict, batch_generator, file_name):
        self.ent_dict = ent_dict
        self.batch_generator = batch_generator
        self.embs = None
        self.file_name = file_name
        self.file_store = dict()    # params -> file

    def get(self, num_steps, embedding_size, batch_size, num_sampled, vocab_size, num_entities):
        print("Checking for pre-trained embeddings...")
        params = (num_steps, embedding_size, batch_size, num_sampled, vocab_size)
        if params in self.file_store:
            print("Loading from previous run...")
            return self.load(self.file_store[params], embedding_size, num_entities)
        print("Pre-training embeddings...")
        model = SkipgramModel(embedding_size, batch_size, num_sampled, vocab_size)
        with tf.Session() as session:
            model.create_graph()
            normalized = normalize(model.variables())
            tf.global_variables_initializer().run()
            average_loss = 0
            for b in range(1, num_steps + 1):
                # Event batches
                batch_x, batch_y = self.batch_generator.next(batch_size)
                batch_y = np.array(batch_y).reshape((batch_size, 1))
                feed_dict = {
                             model.train_inputs: batch_x, model.train_labels: batch_y
                             }
                _, l = session.run(model.train(), feed_dict=feed_dict)
                average_loss += l
                if b % 1000 == 0:
                    print("Step {0} - average loss {1} ".format(b, average_loss / 1000.0))
                    average_loss = 0
            # TODO: normalize - yup
            self.embs = session.run(normalized)
        self.save(params)
        return self.load(self.file_name + '_'.join([str(p) for p in params]), embedding_size, num_entities)

    def save(self, params):
        if self.embs is None:
            raise("No embeddings defined yet")
        self.file_store[params] = self.file_name + '_'.join([str(p) for p in params])
        sup_embs = SuppliedEmbedding(self.embs, self.ent_dict)
        sup_embs.save_embedding(self.file_name + '_'.join([str(p) for p in params]))

    def load(self, file_name, embedding_size, num_entities):
        w_bound = np.sqrt(6. / embedding_size)
        initE = np.random.uniform(-w_bound, w_bound, (num_entities, embedding_size))
        print("Loading supplied embeddings...")
        with open(file_name, "rb") as f:
            supplied_embeddings = pickle.load(f)
            supplied_dict = supplied_embeddings.get_dictionary()
            for event_id, emb_id in supplied_dict.iteritems():
                if event_id in self.ent_dict:
                    new_id = self.ent_dict[event_id]
                    initE[new_id] = supplied_embeddings.get_embeddings()[emb_id]
        return initE


class TEKEPreparation(object):
    """
    Take word2vec embeddings and sequences to update them using co-occurrence of entities
    """
    def __init__(self, sequences, pre_embeddings, num_entities):
        self.sequences = sequences
        self.pre_embeddings = pre_embeddings
        # all kg entities
        # set static entity entries to zero vector
        self.num_entities = num_entities
        self.calculate_cooc()
        self.calculate_pointwise()
        self.calculate_pairwise()

    def calculate_cooc(self):
        window_size = 5
        self.cooc = lil_matrix((self.num_entities, self.num_entities))
        for seq in self.sequences:
            for i, tmp_entity in enumerate(seq):
                if i + window_size <= len(seq):
                    window = seq[i: i + window_size]
                    for c in itertools.combinations(window, 2):
                        self.cooc[c[0], c[1]] += 1

    def calculate_pointwise(self):
        """
        matrix of pointwise context embedding
        :return:
        """
        self.X = self.cooc.dot(self.pre_embeddings)
        sum_cooc = self.cooc.sum(1)
        non_zeros = sum_cooc.nonzero()[0]
        self.X[non_zeros,:] = self.X[non_zeros,:] / sum_cooc[non_zeros]

    def calculate_pairwise(self):
        """
        list of matrices of pairwise embeddings (i.e. tensor)
        :return:
        """
        self.embedding_size = self.pre_embeddings.shape[1]
        self.XY = [lil_matrix((self.num_entities, self.embedding_size))] * self.num_entities

        indices = self.cooc.nonzero()
        for i in np.unique(indices[0]):
            neighbors_i = indices[1][np.where(indices[0] == i)]
            for j in np.unique(indices[0]):
                neighbors_j = indices[1][np.where(indices[0] == j)]
                common_neighbors = np.intersect1d(neighbors_i, neighbors_j)
                # each neighbour contributes with min_cooc of i and j
                sum_weights = 0
                sum_embeddings = np.zeros((self.embedding_size))
                for neighbor in common_neighbors:
                    cooc_i_to_neighbor = self.cooc[i, neighbor]
                    cooc_j_to_neighbor = self.cooc[j, neighbor]

                    if cooc_i_to_neighbor < cooc_j_to_neighbor:
                        weight = cooc_i_to_neighbor
                    else:
                        weight = cooc_j_to_neighbor
                    sum_weights += weight
                    sum_embeddings += weight * self.pre_embeddings[neighbor, :]
                if sum_weights == 0:
                    continue
                self.XY[i][j,:] = (1/(1.0*sum_weights)) * sum_embeddings

    def get_pointwise_batch(self, batch_pos, batch_neg):
        result_h = self.X[batch_pos[1, :], :]
        result_t = self.X[batch_pos[0, :], :]

        result_hn = self.X[batch_neg[1, :], :]
        result_tn = self.X[batch_neg[0, :], :]
        return result_h, result_t, result_hn, result_tn

    def get_pairwise_batch(self, batch_pos, batch_neg):
        result_xy = []
        result_xyn = []
        for h, t in zip(batch_pos[1, :], batch_pos[0, :]):
            result_xy.append(self.XY[h][t, :].todense().tolist()[0])
        for h, t in zip(batch_neg[1, :], batch_neg[0, :]):
            result_xyn.append(self.XY[h][t, :].todense().tolist()[0])
        return np.array(result_xy), np.array(result_xyn)

    def get_pointwise(self, indices=None):
        if indices is None:
            return self.X
        else:
            return self.X[indices]

    def get_pairwise(self, index_left=None, index_right=None):
        if index_left is None:
            # get everything from left to specified right
            return self.XY[0:len(self.XY)][index_right, :].todense()
        else:
            return np.array(self.XY)[index_left]


if __name__ == '__main__':
    sequences = [[2,10,20,30,10,5], [10,2,21,10,11]] * 2000
    pre_embeddings = np.random.random((5000, 80))
    # calculate batch_wise
    num_entities = 5000
    batch_pos = np.array([[1,2], [2,3], [4, 4]])
    batch_neg = np.array([[1,2], [2,3], [4, 5]])
    teke = TEKEPreparation(sequences, pre_embeddings, num_entities)
    print(teke.get_pointwise_batch(batch_pos, batch_neg)[0].shape)
    print(teke.get_pairwise_batch(batch_pos, batch_neg)[0].shape)

    print(teke.get_pointwise([1,2,3]).shape)
    print(teke.get_pointwise().shape)

    print(teke.get_pairwise(index_left=[1,2,3]))
    print(teke.get_pairwise(index_left=None, index_right=[12,3]))