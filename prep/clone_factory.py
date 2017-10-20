from etl import amberg_ns, base_ns, occursOn, message_column, fe_column, module_column, get_unique_entities, get_merged_dataframe, update_amberg_ontology
from rdflib import URIRef, RDF, RDFS, OWL, ConjunctiveGraph
import numpy as np
import pandas as pd


def unify_dicts(a, b):
    for key, value in b.iteritems():
        a[key] = len(a)


ont_path = '../clones/Ontology/amberg_inferred_v2.xml'
event_path = '../test_data/Sequences'

original_events = get_merged_dataframe(event_path, None)
original_g = ConjunctiveGraph()
original_g.load(ont_path)

clones = ['A', 'B', 'C', 'D']
clone_products = ['3SU14001AA101BA0', '3SU14001AA101CA0', '3SU14001AA103HA0', '3SU14002AA103CA0']

device = URIRef('http://purl.oclc.org/NET/ssnx/ssn#Device')
process = URIRef('http://www.loa-cnr.it/ontologies/DUL.owl#Process')
hasPart = URIRef('http://www.loa-cnr.it/ontologies/DUL.owl#hasPart')
follows = URIRef('http://www.loa-cnr.it/ontologies/DUL.owl#follows')
involvedEquipment = base_ns['involvedEquipment']

global_msgs_dict = dict()
global_variants_dict = dict()
global_modules_dict = dict()
global_fes_dict = dict()
entities_dict = dict()

remove_original = False
clone_g = ConjunctiveGraph()

# TODO: remove NamedIndividual for all entities
for s,p,o in original_g.triples((None, RDF.type, OWL.NamedIndividual)):
    original_g.remove((s,p,o))


for i, clone in enumerate(clones):
    if i == len(clones) - 1:
        remove_original = True
    # copy all device entities
    for dev in original_g.subjects(RDF.type, device):
        # their associated triples
        for s,p,o in original_g.triples((dev, None, None)):
            new_s = clone + '-' + str(s).split('#')[1]
            new_s = amberg_ns[new_s]
            if p in (RDF.type, amberg_ns['hasSkill']):
                if remove_original:
                    original_g.remove((s, p, o))
                clone_g.add((new_s, p, o))
            elif p in (hasPart, amberg_ns['connectsTo']):
                new_o = clone + '-' + str(o).split('#')[1]
                new_o = amberg_ns[new_o]
                if remove_original:
                    original_g.remove((s, p, o))
                clone_g.add((new_s, p, new_o))

        for s,p,o in original_g.triples((None, None, dev)):
            new_o = clone + '-' + str(o).split('#')[1]
            new_o = amberg_ns[new_o]

            if p in (hasPart, amberg_ns['connectsTo'], base_ns['observedBy'], base_ns['involvedEquipment']):
                new_s = clone + '-' + str(s).split('#')[1]
                new_s = amberg_ns[new_s]
                if remove_original:
                    original_g.remove((s, p, o))
                clone_g.add((new_s, p, new_o))

    # TODO: clone processes
    for proc in original_g.subjects(RDF.type, process):
        # involvedEquipment, follows, typing
        for s,p,o in original_g.triples((None, None, proc)):
            new_s = clone + '-' + str(s).split('#')[1]
            new_s = amberg_ns[new_s]
            if p in (involvedEquipment, follows):
                new_o = clone + '-' + str(o).split('#')[1]
                new_o = amberg_ns[new_o]
                clone_g.add((new_s, p, new_o))
        for s,p,o in original_g.triples((proc, None, None)):
            new_s = clone + '-' + str(s).split('#')[1]
            new_s = amberg_ns[new_s]
            if p == follows:
                new_o = clone + '-' + str(o).split('#')[1]
                new_o = amberg_ns[new_o]
                clone_g.add((new_s, p, new_o))
            elif p == RDF.type:
                clone_g.add((new_s, p, o))

    clone_g.add((amberg_ns[clone + '-Linie'], amberg_ns['produces'], amberg_ns[clone_products[i][:7] + '-' + clone_products[i][7:12] + '-' + clone_products[i][12:]]))

    clone_events = original_events[original_events['MLFB'] == clone_products[i]]
    clone_events[fe_column] = clone + '-' + clone_events[fe_column]
    clone_events[module_column] = clone + '-' + clone_events[module_column]
    clone_events[message_column] = clone + '-' + clone_events[message_column]

    del clone_events[fe_column]
    clone_events.to_csv(open('../clones/Sequences/clones'+clone+'.csv', 'wb'), sep=';', index=False, date_format='%d.%m.%Y %H:%M:%S')

    # msgs_dict, variants_dict, modules_dict, fes_dict = get_unique_entities(clone_events)

    # unify_dicts(global_msgs_dict, msgs_dict)
    # unify_dicts(global_variants_dict, variants_dict)
    # unify_dicts(global_modules_dict, modules_dict)
    # unify_dicts(global_fes_dict, fes_dict)
    #
    # update_amberg_ontology(clone_g, entities_dict, global_msgs_dict, global_modules_dict, global_fes_dict, global_variants_dict, clone_events)

# add rest
for s,p,o in original_g.triples((None, None, None)):
    clone_g.add((s,p,o))

clone_g.serialize('../clones/Ontology/amberg_clone.rdf')
print("Serialized %d triples for clone" %(len(clone_g)))