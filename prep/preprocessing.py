import pandas as pd
import numpy as np
from rdflib import ConjunctiveGraph, URIRef, RDF, RDFS, OWL, Literal
import operator
import pickle


schema_relations = [RDFS.subClassOf, RDFS.subPropertyOf, OWL.inverseOf, OWL.disjointWith, OWL.imports]


def remove_rel_triples(g, relation_list):
    for rel in relation_list:
        to_remove_triples = []
        for s, p, o in g.triples((None, URIRef(rel), None)):
            to_remove_triples.append((s, p, o))
        for triple in to_remove_triples:
            g.remove(triple)
    return g


class PreProcessor(object):
    def __init__(self, kg_path, sequence_path, ent_dict=None):
        self.kg_path = kg_path
        self.sequence_path = sequence_path
        if ent_dict is None:
            self.ent_dict = dict()
        else:
            self.ent_dict = ent_dict
        self.g = ConjunctiveGraph()
        self.unique_msgs = self.ent_dict.copy()
        self.vocab_size = len(self.ent_dict)

    def load_knowledge_graph(self, format, exclude_rels=[], clean_schema=True):
        self.g.load(self.kg_path, format = format)
        # remove triples with excluded relation
        remove_rel_triples(self.g, exclude_rels)
        # remove triples with relations between class-level constructs
        if clean_schema:
            remove_rel_triples(self.g, schema_relations)
            
    def load_unique_msgs_from_txt(self, path):
        """
        Assuming csv text files with two columns
        :param path:
        :return:
        """
        with open(path, "rb") as f:
            for line in f:
                split = line.split(',')
                try:
                    emb_id = int(split[1].strip())
                except:
                    print "Error reading id of %s in given dictionary" %line
                    # skip this event entitiy, treat it as common entitiy later on
                    continue
                self.ent_dict[split[0]] = emb_id
        # sort ascending w.r.t. embedding id, in case of later stripping
        # self.ent_dict = sorted(self.ent_dict.items(), key=operator.itemgetter(1), reverse=False)
        self.unique_msgs = self.ent_dict.copy()
        self.vocab_size = len(self.unique_msgs)

    def prepare_sequences(self, path_to_input, path_to_output):
        """
        Dumps pickle for sequences and dictionary
        :param data_frame:
        :param file_name:
        :param index:
        :param classification_event:
        :return:
        """
        with open(path_to_input, "rb") as f:
            result = []
            for line in f:
                entities = line.split(',')
                result.append([int(e.strip()) for e in entities if int(e.strip()) in self.unique_msgs.values()])
        print "Preparing sequential data..."
        print result[:10]
        pickle.dump(result, open(path_to_output + ".pickle", "wb"))
        print "Processed %d sequences" % (len(result))
        return len(result)

    def get_vocab_size(self):
        return self.vocab_size

    def get_ent_dict(self):
        return self.ent_dict

    def get_kg(self):
        return self.g