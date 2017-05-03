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

# rdf vocabulary
base_ns = Namespace("http://www.siemens.com/ontology/demonstrator#")
amberg_ns = Namespace("http://www.siemens.com/ontologies/amberg#")
isMadeOf = base_ns["isMadeOf"]
material = base_ns["Material"]
hasPart = URIRef("http://www.loa-cnr.it/ontologies/DUL.owl#hasPart")
occursOn = base_ns["occursOn"]
event = base_ns["Event"]


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


def read_ontology(path, format='xml'):
    """

    :param path:s
    :return:
    """
    g = ConjunctiveGraph()
    g.load(path, format=format)
    for s,p,o in g.triples((None, URIRef('http://www.siemens.com/ontology/demonstrator#tagAlias'), None)):
        g.remove((s,p,o))
    return g


def load_text_file(path):
    g = ConjunctiveGraph()
    with open(path, "rb") as file:
        for line in file:
            s, p, o = line.split("\t")
            s = s.strip()
            p = p.strip()
            o = o.strip()
            g.add((URIRef("http://" + s), URIRef("http://" + p), URIRef("http://" + o)))
    return g


def update_ontology(ont, msg_dict, mod_dict, fe_dict, var_dict, data):
    """
    Update entities in ontology to ids of dictionaries
    Give new Ids to entities never seen before
    :param ont:
    :param msg_dict: event messages
    :param mod_dict: modules
    :param fe_dict: fes
    :param var_dict: Variants
    :param data:
    :return:
    """
    ont.add((base_ns['Material-Event'], RDFS.subClassOf, base_ns['Event']))
    ont.add((base_ns['Axis-Event'], RDFS.subClassOf, base_ns['Event']))
    ont.add((base_ns['Jam-Event'], RDFS.subClassOf, base_ns['Event']))
    entity_uri_to_data_id = dict()
    for msg, id in msg_dict.iteritems():
        fe_or_module_id = None
        fe_or_module = np.unique(data[data[message_column] == msg][fe_column])[0]
        fe_or_module_id = fe_dict[fe_or_module]
        if not fe_or_module:
            fe_or_module = np.unique(data[data[message_column] == msg][module_column])[0]
            fe_or_module_id = mod_dict[fe_or_module]
            fe_or_module = fe_or_module.replace('odule', '').replace(' ', '')
        ont.add((amberg_ns['Event-'+str(id)], RDF.type, base_ns['Event']))
        ont.add((amberg_ns['Event-'+str(id)], occursOn, amberg_ns[fe_or_module]))
        # TODO: if both entries -> occursOn Module and FE
        ont.add((amberg_ns[fe_or_module], RDF.type, amberg_ns['ProductionUnit']))
        entity_uri_to_data_id[str(amberg_ns['Event-'+str(id)])] = id
        entity_uri_to_data_id[str(amberg_ns[fe_or_module])] = fe_or_module_id
        if "Stau" in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Jam-Event']))
        elif "Achse" in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Axis-Event']))
        elif "F?llstand" in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Material-Event']))
    return ont, entity_uri_to_data_id


def get_unique_entities(data):
    """
    Extract from pandas dataframe
    :param data:
    :return:
    """
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


def get_messages_to_module(dataframe):
    """
    Get unique messages with modules as dataframe
    :param dataframe:
    :return:
    """
    dataframe = dataframe[[message_column, module_column]].drop_duplicates()
    message_to_module = np.array(dataframe)
    #message_to_module = pd.read_csv("/home/nether-nova/Documents/Amberg Events/test_data/unique_messages.txt", sep=",",
    #                                header=None)
    me2m = dict(zip(message_to_module[:, 0], message_to_module[:, 1]))
    print me2m
    return me2m


def get_messages_to_fe(message_to_module_dict):
    messages_to_fe = dict()
    for message, module in message_to_module_dict.iteritems():
        terms = message.split(' ')
        for i, t in enumerate(terms):
            t = t.replace(':', '').strip()
            if t.startswith('FE'):
                fe_num = None
                last_char = None
                if i < len(terms) - 1:
                    try:
                        fe_num = int(terms[i + 1])
                    except ValueError:
                        pass
                try:
                    last_char = int(t[-1])
                except ValueError:
                    pass
                if fe_num and not last_char:
                    t = t + str(fe_num)
                messages_to_fe[message] = module.replace('odule', '').replace(' ', '') + '_' + t
        if message not in messages_to_fe:
            messages_to_fe[message] = module.replace('odule', '').replace(' ', '') + '_' + 'UNK-FE'
    fe_to_index = dict()
    for i, fe in enumerate(np.unique(messages_to_fe.values())):
        fe_to_index[fe] = i

    messages_to_fe_df = pd.DataFrame(zip(messages_to_fe.keys(), messages_to_fe.values()), columns=["Meldetext", "FE"])

    print messages_to_fe_df.head(10)
    # messages_to_fe_df.to_csv("./test_data/messages_to_fe.txt", sep=",", quotechar='"')
    # reverse_messages_to_fe = dict(zip(messages_to_fe.values(), messages_to_fe.keys()))

    return messages_to_fe_df


# path = "/home/nether-nova/Documents/Amberg Events/test_data/"
# max_events = 5000
# window_size = 3
# df = read_data(path, max_events)
# fe_df = pd.read_csv(path + "messages_to_fe.txt")
# merged = pd.merge(df, fe_df, on="Meldetext")
# # merged[module_column] = merged["FE"]    # replace module column with FE-Module
# merged = merged.set_index(pd.DatetimeIndex(merged[time_column]))
# merged = merged.sort_index(ascending=True)s
# unique_msgs, unique_vars, unique_mods, unique_fes = get_unique_entities(merged)
# # includes relations
# ontology = read_ontology(path + "./amberg_inferred.xml")
# print "Read %d number of triples" % len(ontology)
# ont, uri_to_id = update_ontology(ontology, unique_msgs, unique_mods, unique_fes, unique_vars, merged)
# print "Number of triples: ", len(ont)
# for k in uri_to_id:
#     print k, [t for t in ont.triples((URIRef(k), None, None))]


def get_merged_dataframe(path, max_events):
    df = read_data(path, max_events)
    message_to_module_dict = get_messages_to_module(df)
    fe_df = get_messages_to_fe(message_to_module_dict)
    merged = pd.merge(df, fe_df, on="Meldetext")
    # merged[module_column] = merged["FE"]    # replace module column with FE-Module
    merged = merged.set_index(pd.DatetimeIndex(merged[time_column]))
    merged = merged.sort_index(ascending=True)
    return merged


def binary_sequences(sequences, index, unique_dict, classification_event=None):
    train = []
    labels = []
    for i, seq in enumerate(sequences):
        local_entities = [event[index] for event in seq]
        x = 0
        hit = False
        if classification_event:
            for j, m in enumerate(local_entities):
                # what if multiple matches?
                if m == classification_event:
                    train.append(' '.join([str(unique_dict[msg]) for msg in local_entities[x:j]]))
                    labels.append(1)
                    x = j + 1
                    hit = True
        if not hit:
            # append indices of entities as strings (needed for tf.VocabProcessor)
            train.append(' '.join([str(unique_dict[entity]) for entity in local_entities]))
            labels.append(0)
    return train, labels


def prepare_sequences(data_frame, path_to_file, index, unique_dict, window_size, classification_event=None):
    """
    Dumps pickle for sequences and dictionary
    :param data_frame:
    :param file_name:
    :param index:
    :param classification_event:
    :return:
    """
    train_data = time_window(data_frame, window_size)
    result = []
    print "Preparing sequential data..."
    for i, seq in enumerate(train_data):
        local_entities = [event[index] for event in seq]
        result.append([unique_dict[entity] for entity in local_entities])
    print result[:10]
    pickle.dump(result, open(path_to_file + ".pickle", "wb"))
    reverse_lookup = dict(zip(unique_dict.values(), unique_dict.keys()))
    pickle.dump(reverse_lookup, open(path_to_file + "_dictionary.pickle", "wb"))
    print "Processed %d sequences" %(len(result))
    return len(result)


def prepare_sequences_nba(path_to_input, path_to_output):
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
            result.append([int(e.strip()) for e in entities])
    print "Preparing sequential data..."
    print result[:10]
    pickle.dump(result, open(path_to_output + ".pickle", "wb"))
    print "Processed %d sequences" %(len(result))
    return len(result)


def prepare_fe_log_file(merged, file_name):
    sequences = time_window(merged, 3, include_time=True)
    with open(file_name, "wb") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Case ID", "Activity", "Start Date", "End Date"])
        for i, s in enumerate(sequences):
            for fe in s:
                writer.writerow([i, fe[module_index], fe[time_index], fe[time_index]])


def prepare_sensor_data(file_path):
    df = pd.read_csv(file_path)
    sensor_mapping = {}
    for k in df:
        sensor_mapping[k].setdefault(len(sensor_mapping))
    return df, sensor_mapping


def etl_sensor_data(module_file, line_file):
    df = pd.read_csv(module_file, sep=";", error_bad_lines=False, decimal=',', parse_dates=[0])
    df = df.groupby(by='Zeitspalte', as_index=True).mean()

    df2 = pd.read_csv(line_file, sep=";", error_bad_lines=False, decimal=',', parse_dates=[0])
    df2 = df2.set_index("Zeitspalte")
    merged = df.merge(df2, left_index=True, right_index=True)
    X = merged.corr()
    X["std"] = merged.std()
    X["mean"] = merged.mean()
    X["max"] = merged.max()
    X["min"] = merged.min()

#prepare_sequences(df, "train_sequences", message_index, classification_event=None)

#prepare_sequences(df, "train_sequences", message_index, classification_event=None)
#prepare_fe_log_file("./test_data/fe_log.txt")

