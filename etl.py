import os
import pandas as pd
import glob
import numpy as np
import pickle
import csv
from rdflib import ConjunctiveGraph, RDF, RDFS, OWL, Literal, URIRef, Namespace


# indices for pandas dataframe
time_column = 'Timestamp'
module_column = 'Module'
variant_column = 'MLFB'
message_column = 'Meldetext'
fe_column = 'FE'
# indices for numpy array
message_index = 0
module_index = 1
variant_index = 2
time_index = 3

base_ns = Namespace("http://www.siemens.com/ontology/demonstrator#")
amberg_ns = Namespace("http://www.siemens.com/ontologies/amberg#")
isMadeOf = base_ns["isMadeOf"]
material = base_ns["Material"]
hasPart = URIRef("http://www.loa-cnr.it/ontologies/DUL.owl#hasPart")
occursOn = base_ns["occursOn"]
event = base_ns["Event"]

#TODO: Define uri_to_index dictionary (if OWL does not allow exact content in comments)
#TODO: Parse ontology w/o events --> add them with ids from extracted unique_events_dict
#TODO: Relate modules, fes and events to ids

#Training data: window size 3
#Batch: (e1, e2, e3, m1, m2, m3) : (e2 -> e1, e2 -> e3, m2 -> m1, m3 -> m1)
#(h1, r1, t1), (h2, r2, t2) : [h1, r1, neg t1], [h1, r1, neg t1]...

def read_data(path, max_events=None):
    """Read all csv-files in *path*, merge and sort by time"""
    all_files = glob.glob(os.path.join(path, '*.csv'))
    print "Number of files: %s" %len(all_files)
    df_from_each_file = (pd.read_csv(f, sep=';') for f in all_files)
    table = pd.concat(df_from_each_file, ignore_index=True)
    print "Number of events: %s" %table.shape[0]
    # set timestamp as index and sort ascending
    table[time_column] = pd.to_datetime(table[time_column], format="%d.%m.%Y %H:%M:%S")
    table = table.set_index(pd.DatetimeIndex(table[time_column]))
    table = table.sort_index(ascending=True)
    if max_events and max_events < table.shape[0]:
        return table.iloc[:max_events]
    return table[[message_column, module_column, variant_column, time_column]]


def time_window(df, window_size, include_time=False):
    """
    Extract list of lists of events using sliding window of *window_size*
    df: sorted dataframe
    window_size: time window size in minutes
    """
    train = []
    labels_dict = dict()
    off = pd.DateOffset(minutes=window_size)
    for i in xrange(df.shape[0]):
        window_start = df.iloc[i][time_column]
        window_end = window_start + off
        local_window = df[window_start : window_end]
        # one sequence is a list of 3-tuples [[message, module, variant], [mesage, module, variant] ...]
        if include_time:
            train.append(local_window[[message_column, module_column, variant_column, time_column]].values.tolist())
        else:
            train.append(local_window[[message_column, module_column, variant_column]].values.tolist())
    return train


def read_metadata(path):
    meta_dict = dict()
    f = open(path, 'r')
    for line in f:
        line = line.strip()
        line_elements = line.split(',')
        variant = line_elements[0].replace('-', '')
        meta_dict[variant] = ['Part-' + str(part) for part in line_elements[1:]]
    meta_dict['UNK_V'] = ['Part-0', 'Part-1']
    f.close()
    return meta_dict


def read_ontology(path):
    #Use defined dictionary to refer to events, modules and fe's
    #one process implemented with connections
    g = ConjunctiveGraph()
    g.load(path)
    print "Read %d number of triples" %len(g)
    #remove not needed triples
    for (s,p,o) in g.triples((None, None, None)):
        if p == URIRef('http://www.siemens.com/ontology/demonstrator#tagAlias'):
            g.remove((s,p,o))
        if o == OWL["NamedIndividual"]:
            g.remove((s, p, o))
    return g

def update_ontology(ont, msg_dict, mod_dict, fe_dict, var_dict, data):
    #update entities in ontology to ids of dictionaries
    #Variants (parts), modules (fes), events
    #Give new Ids to entities never seen before
    entity_uri_to_data_id = dict()
    for msg, id in msg_dict.iteritems():
        fe_or_module_id = None
        fe_or_module = np.unique(data[data[message_column] == msg][fe_column])[0]
        fe_or_module_id = fe_dict[fe_or_module]
        if not fe_or_module:
            fe_or_module = np.unique(data[data[message_column] == msg][module_column])[0]
            fe_or_module_id = mod_dict[fe_or_module]
            fe_or_module = fe_or_module.replace('odule', '').replace(' ', '')
        ont.add((URIRef(amberg_ns['Event-'+str(id)]), RDF.type, base_ns['Event']))
        ont.add((URIRef(amberg_ns['Event-'+str(id)]), occursOn, amberg_ns[fe_or_module]))
        entity_uri_to_data_id[str(amberg_ns['Event-'+str(id)])] = id
        entity_uri_to_data_id[str(amberg_ns[fe_or_module])] = fe_or_module_id
    return ont, entity_uri_to_data_id

    # same procedure for modules, fes, ...
    """
    #max_id to continue for new values
    all_ids = msg_dict.values() + mod_dict.values() + fe_dict.values() + var_dict.values()
    new_id_start = np.max(all_ids) + 1
    for (s,p,o) in ont.triples():
        if s not in entity_uri_to_data_id:
            #add
            pass
    """


def context_window(window_size, sequence):
    skip = window_size // 2  # only odd window_size makes sense
    train = []
    labels = []
    for i in range(skip, len(sequence) - skip):
        current_target = i
        context = sequence[current_target - skip: (current_target)]
        context = context + sequence[current_target + 1: current_target + 1 + skip]
        train.append(context)
        labels.append(sequence[current_target])
    return train, labels


def context_windows(sequence, window_size, max_gap="5 minutes"):
    result = []
    table = pd.DataFrame(sequence)
    off = pd.Timedelta(max_gap)
    message_index = 0
    variant_index = 1
    module_index = 2
    time_index = 3
    # keine luecken in der reihe > max_gap in mins
    table[time_index] = pd.to_datetime(table[time_index], format="%d.%m.%Y %H:%M:%S")
    table = table.set_index(pd.DatetimeIndex(table[time_index]))
    table = table.sort_index()
    indices_list = [range(i, i + window_size) for i in xrange(0, len(sequence)-window_size)]
    for i, indices in enumerate(indices_list):
        window = table.iloc[indices]
        if window.iloc[-1][time_index] - window.iloc[0][time_index] > off:
            continue
        # one sequence [[message, variant, module], [mesage, variant, module] ...]
        result.append(window[[0, 1, 2]].values.tolist())
    return result


def context_window_for_classification(sequence, window_size, classification_size, classification_events=[]):
    # every sequence, classify what critical event is going to happen (True/False) binary classification
    train = []
    labels = []
    sequence = context_windows(sequence, window_size)
    for i, seq in enumerate(sequence):
        if i >= classification_size:
            return train, labels
        tmp_list = []
        for index in xrange(len(seq)-1):
            tmp_list.append(seq[index][0]) # event
        tmp_list.append(seq[0][1]) # append the first seen variant
        train.append(tmp_list)
        previous_sequence = sequence[i - 1]
        labels.append(previous_sequence[-1][2]) # append module of the last event
    return train, labels


def context_window_last_event(window_size, sequence):
    train = []
    labels = []
    sequence = context_windows(sequence, window_size)
    for seq in sequence:
        tmp_list = []
        for index in xrange(len(seq)-1):
            tmp_list.append(seq[index][0]) # event
        tmp_list.append(seq[0][1]) # append the first seen variant
        train.append(tmp_list) # window_size - 1 events
        # the n-th event , module, and, n-th variant (assuming 1st and n-th variant are the same)
        labels.append(seq[len(seq)-1])
    return train, labels


def get_unique_entities(data):
    #Extract from pandas dataframe
    unique_msgs = np.unique(data[message_column])
    unique_msgs_dict = dict(zip(unique_msgs, range(len(unique_msgs))))
    unique_variants = np.unique(data[variant_column])
    unique_variants_dict = dict(zip(unique_variants, range(len(unique_msgs), len(unique_msgs) + len(unique_variants))))
    unique_modules = np.unique(data[module_column])
    unique_modules_dict = dict(zip(unique_modules, range(len(unique_msgs) + len(unique_variants),
                                                len(unique_msgs) + len(unique_variants) + len(unique_modules))))
    unique_fes = np.unique(data[fe_column])
    unique_fes_dict = dict(zip(unique_fes, range(len(unique_msgs) + len(unique_variants) + len(unique_modules),
                                    len(unique_msgs) + len(unique_variants) + len(unique_modules) + len(unique_fes))))
    return unique_msgs_dict, unique_variants_dict, unique_modules_dict, unique_fes_dict


path = "/home/nether-nova/Documents/Amberg Events/test_data/"
max_events = 5000
window_size = 3
df = read_data(path, max_events)
fe_df = pd.read_csv(path + "messages_to_fe.csv")
merged = pd.merge(df, fe_df, on="Meldetext")
#merged[module_column] = merged["FE"]    # replace module column with FE-Module
merged = merged.set_index(pd.DatetimeIndex(merged[time_column]))
merged = merged.sort_index(ascending=True)
unique_msgs, unique_vars, unique_mods, unique_fes = get_unique_entities(merged)
#includes relations
ontology = read_ontology(path + "./amberg_inferred.xml")
ont, uri_to_id = update_ontology(ontology, unique_msgs, unique_mods, unique_fes, unique_vars, merged)
for k in uri_to_id:
    print k, [t for t in ont.triples((URIRef(k), None, None))]


def binary_sequences(sequences, index, classification_event=None):
    train = []
    labels = []
    lookup_dict = None
    if index == message_index:
        lookup_dict = unique_msgs
    elif index == variant_index:
        lookup_dict = unique_vars
    else:
        lookup_dict = unique_mods
    for i, seq in enumerate(sequences):
        local_entities = [event[index] for event in seq]
        x = 0
        hit = False
        if classification_event:
            for j, m in enumerate(local_entities):
                #what if multiple matches?
                if m == classification_event:
                    train.append(' '.join([str(lookup_dict[msg]) for msg in local_entities[x:j]]))
                    labels.append(1)
                    x = j + 1
                    hit = True
        if not hit:
            train.append(' '.join([str(lookup_dict[msg]) for msg in local_entities]))
            labels.append(0)
    return train, labels


def prepare_sequences(data_frame, file_name, index, classification_event='Dropping OEE'):
    train_data = time_window(data_frame, window_size)
    train, labels = binary_sequences(train_data, index, classification_event)
    print train[0:10]
    print labels[0:10]
    pickle.dump(train, open(path + file_name + ".pickle", "wb"))
    pickle.dump(labels, open(path + file_name + "_labels.pickle", "wb"))
    lookup_dict = None
    if index == message_index:
        lookup_dict = unique_msgs
    elif index == variant_index:
        lookup_dict = unique_vars
    else:
        lookup_dict = unique_mods
    reverse_lookup = dict(zip(lookup_dict.values(), lookup_dict.keys()))
    pickle.dump(reverse_lookup, open(path + file_name + "_dictionary.pickle", "wb"))


def prepare_fe_log_file(file_name):
    sequences = time_window(merged, 3, include_time=True)
    with open(file_name, "wb") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Case ID", "Activity", "Start Date", "End Date"])
        for i, s in enumerate(sequences):
            for fe in s:
                writer.writerow([i, fe[module_index], fe[time_index], fe[time_index]])


#prepare_sequences(df, "train_sequences", message_index, classification_event=None)

#prepare_sequences(df, "train_sequences", message_index, classification_event=None)
#prepare_fe_log_file("./test_data/fe_log.txt")

