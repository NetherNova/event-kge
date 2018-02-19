from rdflib import ConjunctiveGraph, URIRef, RDF, RDFS, OWL, Literal
from prep.etl import get_merged_dataframe, get_unique_entities, update_amberg_ontology
import operator
from utils import url_parse

schema_relations = [RDFS.subClassOf, RDFS.subPropertyOf, OWL.inverseOf, OWL.disjointWith, OWL.imports]


def remove_rel_triples(g, relation_list):
    for rel in relation_list:
        to_remove_triples = []
        for s, p, o in g.triples((None, URIRef(rel), None)):
            to_remove_triples.append((s, p, o))
        for triple in to_remove_triples:
            g.remove(triple)
    return g


def remove_ent_triples(g, excluded_ents):
    remove_triples = []
    for e in excluded_ents:
        for (s, p, o) in g.triples((e, None, None)):
            remove_triples.append((s, p, o))
        for (s, p, o) in g.triples((None, None, e)):
            remove_triples.append((s, p, o))
    for s, p, o in remove_triples:
        g.remove((s, p, o))


class PreProcessor(object):
    def __init__(self, kg_path):
        self.kg_path = kg_path
        self.ent_dict = dict()
        self.rel_dict = dict()
        self.g = ConjunctiveGraph()
        self.unique_msgs = self.ent_dict.copy()

    def load_knowledge_graph(self, format='xml', exclude_rels=[], clean_schema=True, amberg_params=None,
                             excluded_entities=None):
        self.g.load(self.kg_path, format=format)
        # remove triples with excluded relation
        remove_rel_triples(self.g, exclude_rels)
        # remove triples with relations between class-level constructs
        if clean_schema:
            remove_rel_triples(self.g, schema_relations)
        if excluded_entities is not None:
            remove_ent_triples(self.g, excluded_entities)
        if amberg_params:
            path_to_events = amberg_params[0]
            max_events = amberg_params[1]
            self.merged = get_merged_dataframe(path_to_events, max_events)
            self.unique_msgs, unique_vars, unique_mods, unique_fes = get_unique_entities(self.merged)
            update_amberg_ontology(self.g, self.ent_dict, self.unique_msgs, unique_mods, unique_fes, unique_vars,
                                   self.merged)

        self.update_entity_relation_dictionaries()

    def update_entity_relation_dictionaries(self):
        """
        Given an existing entity dictionary, update it to *ontology*
        :param ontology:
        :param ent_dict: the existing entity dictionary
        :return:
        """
        ent_counter = 0
        fixed_ids = set([id for id in self.ent_dict.values()])
        # sorting ensures equal random splits on equal seeds
        for h in sorted(set(self.g.subjects(None, None)).union(set(self.g.objects(None, None)))):
            # parse to handle special characters in the url
            uni_h = url_parse(str(h))
            uni_h_frag = uni_h.split('#')[1]
            # check to avoid duplicate urls in the dictionary
            if uni_h_frag not in self.ent_dict:
                while ent_counter in fixed_ids:
                    ent_counter += 1
                self.ent_dict.setdefault(uni_h, ent_counter)
                ent_counter += 1
            else:
                # replace url fragment in the dictionary with complete url
                self.ent_dict[uni_h] = self.ent_dict[uni_h_frag]
                del self.ent_dict[uni_h_frag]

        # add new relations to dict
        for r in sorted(set(self.g.predicates(None, None))):
            uni_r = str(r)
            if uni_r not in self.rel_dict:
                self.rel_dict.setdefault(uni_r, len(self.rel_dict))

    def load_unique_msgs_from_txt(self, path, max_events=None):
        """
        Assuming csv text files with two columns
        :param path:
        :return:
        """
        with open(path, "r") as f:
            for line in f:
                split = line.split(',')
                try:
                    emb_id = int(split[1].strip())
                except:
                    print("Error reading id of {0} in given dictionary".format(line))
                    # skip this event entitiy, treat it as common entitiy later on
                    continue
                self.ent_dict[split[0]] = emb_id
        # sort ascending w.r.t. embedding id, in case of later stripping
        # self.ent_dict = sorted(self.ent_dict.items(), key=operator.itemgetter(1), reverse=False)
        self.unique_msgs = self.ent_dict.copy()
        if max_events is not None:
            all_msgs = sorted(self.unique_msgs.items(), key=operator.itemgetter(1), reverse=False)
            self.unique_msgs = dict(all_msgs[:max_events])
            excluded_events = dict(all_msgs[max_events:]).keys()
            return excluded_events

    def prepare_sequences(self, path_to_input, use_dict=True):
        """
        Dumps pickle for sequences and dictionary
        :param data_frame:
        :param file_name:
        :param index:
        :param classification_event:
        :return:
        """
        print("Preparing sequential data...")
        with open(path_to_input, "r") as f:
            result = []
            for line in f:
                entities = line.split(',')
                if use_dict:
                    result.append([int(e.strip()) for e in entities if int(e.strip()) in self.unique_msgs.values()])
                else:
                    result.append([int(e.strip()) for e in entities])
        print("Processed {0} sequences".format(len(result)))
        return result

    def get_vocab_size(self):
        return len(self.unique_msgs)

    def get_ent_dict(self):
        return self.ent_dict

    def get_rel_dict(self):
        return self.rel_dict

    def get_kg(self):
        return self.g

    def get_unique_msgs(self):
        return self.unique_msgs

    def get_merged(self):
        return self.merged