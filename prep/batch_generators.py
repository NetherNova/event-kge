import numpy as np


class SkipgramBatchGenerator(object):
    def __init__(self, sequences, num_skips, rnd):
        """
        center word is target, context should predict center word
        :param sequences: list of lists of event entities
        :param num_skips:  window left and right of target
        :param batch_size:
        """
        self.sequences = sequences
        self.sequence_index = 0
        self.num_skips = num_skips
        self.event_index = num_skips
        self.rnd = rnd
        self.prepare_target_skips()

    def prepare_target_skips(self):
        self.data_index = 0
        self.data = []
        for seq in self.sequences:
            for target_ind in range(self.num_skips, len(seq) - self.num_skips):
                for i in range(-self.num_skips, self.num_skips+1):
                    if i == 0:
                        # avoid the target_ind itself
                        continue
                    self.data.append( (seq[target_ind], seq[target_ind + i]) )
        self.rnd.shuffle(self.data)

    def next(self, batch_size):
        batch_x = []
        batch_y = []
        for b in range(batch_size):
            self.data_index = self.data_index % len(self.data)  #
            batch_x.append(self.data[self.data_index][0])
            batch_y.append(self.data[self.data_index][1])
            self.data_index += 1
        return batch_x, batch_y


class FuturePredictiveBatchGenerator(object):
    def __init__(self, sequences, num_skips, rnd):
        """
        center word is target, context should predict center word
        :param sequences: list of lists of event entities
        :param num_skips:  window left and right of target
        :param batch_size:
        """
        self.sequences = sequences
        self.sequence_index = 0
        self.num_skips = num_skips
        self.event_index = num_skips
        self.rnd = rnd
        self.prepare_target_skips()

    def prepare_target_skips(self):
        self.data_index = 0
        self.data = []
        for seq in self.sequences:
            for target_ind in range(self.num_skips, len(seq) - self.num_skips):
                target_context = []
                for i in range(-self.num_skips, self.num_skips+1):
                    if i == 0:
                        # avoid the target_ind itself
                        continue
                    target_context.append(seq[target_ind + i])
                self.data.append( (target_context, seq[target_ind]) )
        self.rnd.shuffle(self.data)

    def next(self, batch_size):
        batch_x = []
        batch_y = []
        for b in range(batch_size):
            self.data_index = self.data_index % len(self.data)
            batch_x.append(self.data[self.data_index][0])
            batch_y.append(self.data[self.data_index][1])
            self.data_index += 1
        return batch_x, batch_y


class AutoEncoderBatchGenerator(object):
    def __init__(self, sequences, num_skips, rnd):
        """
        center word is target, context should predict center word
        :param sequences: list of lists of event entities
        :param num_skips:  window left and right of target
        :param batch_size:
        """
        self.sequences = sequences
        self.sequence_index = 0
        self.num_skips = num_skips
        self.event_index = num_skips
        self.rnd = rnd
        self.prepare_target_skips()

    def prepare_target_skips(self):
        self.data_index = 0
        self.data = []
        for seq in self.sequences:
            for target_ind in range(self.num_skips, len(seq) - self.num_skips):
                target_context = []
                for i in range(-self.num_skips, self.num_skips+1):
                    target_context.append(seq[target_ind + i])
                self.data.append( (target_context, seq[target_ind]) )
        self.rnd.shuffle(self.data)

    def next(self, batch_size):
        batch_x = []
        batch_y = []
        for b in range(batch_size):
            self.data_index = self.data_index % len(self.data)
            batch_x.append(self.data[self.data_index][0])
            batch_y.append(self.data[self.data_index][1])
            self.data_index += 1
        return batch_x, batch_y


class PredictiveEventBatchGenerator(object):
    def __init__(self, sequences, num_skips, rnd):
        """
        center word is target, context should predict center word
        :param sequences: list of lists of event entities
        :param num_skips:  window left and right of target
        :param batch_size:
        :param include_seq_ids: add sequence number to labels (for concatentation layer)
        """
        self.sequences = sequences
        self.sequence_index = 0
        self.num_skips = num_skips
        self.event_index = num_skips
        self.rnd = rnd
        self.prepare_target_skips()

    def prepare_target_skips(self):
        self.data_index = 0
        self.data = []
        for n, seq in enumerate(self.sequences):
            for target_ind in range(self.num_skips, len(seq)):
                tmp_list = []
                for i in range(-self.num_skips, 0):
                    tmp_list.append(seq[target_ind + i])
                self.data.append( (tmp_list, seq[target_ind]) )
        self.rnd.shuffle(self.data)

    def next(self, batch_size):
        batch_x = []
        batch_y = []
        for b in range(batch_size):
            self.data_index = self.data_index % len(self.data)
            batch_x.append(self.data[self.data_index][0])
            batch_y.append(self.data[self.data_index][1])
            self.data_index += 1
        return batch_x, batch_y


class TripleBatchGenerator(object):
    def __init__(self, triples, entity_dictionary, relation_dictionary, num_neg_samples, rnd, sample_negative=True,
                 bern_probs=None):
        self.all_triples = []
        self.batch_index = 0
        self.num_neg_samples = num_neg_samples
        self.rnd = rnd
        self.entity_dictionary = entity_dictionary
        self.relation_dictionary = relation_dictionary
        self.sample_negative = sample_negative
        self.bern_probs = bern_probs
        self.ent_array = np.array(list(self.entity_dictionary.values()))
        self.ids_in_ent_dict = dict(list(zip(list(self.ent_array), range(0, self.ent_array.size))))

        for (s, p, o) in sorted(triples):
            s = str(s)
            p = str(p)
            o = str(o)
            if s not in self.entity_dictionary:
                continue
            if o not in self.entity_dictionary:
                continue
            if p not in self.relation_dictionary:
                continue
            s_ind = self.entity_dictionary[s]
            p_ind = self.relation_dictionary[p]
            o_ind = self.entity_dictionary[o]
            self.all_triples.append((s_ind, p_ind, o_ind))

    def next(self, batch_size):
        # return lists of entity and reltaion indices
        inpr = []
        inpl = []
        inpo = []

        inprn = []
        inpln = []
        inpon = []
        if self.sample_negative:
            batch_size_tmp = batch_size // self.num_neg_samples
        else:
            batch_size_tmp = batch_size

        for b in range(batch_size_tmp):
            if self.batch_index >= len(self.all_triples):
                self.batch_index = 0
            current_triple = self.all_triples[self.batch_index]
            # Append current triple with *num_neg_samples* triples
            if self.sample_negative:
                for i in range(self.num_neg_samples):
                    inpl.append(current_triple[0])
                    inpr.append(current_triple[2])
                    inpo.append(current_triple[1])
                    rn, ln, on = self.get_negative_sample(*current_triple)
                    inpln.append(ln)
                    inprn.append(rn)
                    inpon.append(on)
            else:
                inpl.append(current_triple[0])
                inpr.append(current_triple[2])
                inpo.append(current_triple[1])
            self.batch_index += 1
        return np.array([inpr, inpl, inpo]), np.array([inprn, inpln, inpon])

    def get_negative_sample(self, s_ind, p_ind, o_ind, left_probability=0.5):
        """
        Uniform sampling (avoiding correct triple from being sampled again)
        :param left_probability:
        :return:
        """
        if self.bern_probs:
            # with (tph / (tph + hpt)) probability we sample a *head*
            left_probability = self.bern_probs[p_ind]
        if self.rnd.binomial(1, left_probability) > 0:
            mask = np.ones(self.ent_array.size, dtype=bool)
            mask[self.ids_in_ent_dict[s_ind]] = 0
            sample_set = self.ent_array[mask]
            s_ind = self.rnd.choice(sample_set, 1)[0]
        else:
            mask = np.ones(self.ent_array.size, dtype=bool)
            mask[self.ids_in_ent_dict[o_ind]] = 0
            sample_set = self.ent_array[mask]
            o_ind = self.rnd.choice(sample_set, 1)[0]
        return o_ind, s_ind, p_ind