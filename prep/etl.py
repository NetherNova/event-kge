import os
import pandas as pd
import glob
import numpy as np
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
    print("Number of files: {0}".format(len(all_files)))
    df_from_each_file = (pd.read_csv(f, sep=';') for f in all_files)
    table = pd.concat(df_from_each_file, ignore_index=True)
    print("Number of events: {0}".format(table.shape[0]))
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
    off = pd.Timedelta(minutes=window_size)
    window_start = df.iloc[0][time_column]
    for i in range(1, len(df) - 1):
        entry_time = df.iloc[i][time_column]
        next_entry_time = df.iloc[i + 1][time_column]
        diff = next_entry_time - entry_time
        if diff > off:
            local_window = df[window_start : entry_time]
            # one sequence is a list of 3-tuples [[message, module, variant], [mesage, module, variant] ...]
            if include_time:
                train.append(local_window[[message_column, module_column, variant_column, time_column]].values.tolist())
            else:
                train.append(local_window[[message_column, module_column, variant_column]].values.tolist())
            window_start = entry_time
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


def update_amberg_ontology(ont, ent_dict, msg_dict, mod_dict, fe_dict, var_dict, data):
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
    for i, (msg, id) in enumerate(msg_dict.iteritems()):
        fe_or_module_id = None
        fe_or_module = np.unique(data[data[message_column] == msg][fe_column])
        if fe_or_module.shape[0] > 0:
            fe_or_module_id = fe_dict[fe_or_module[0]]
        if fe_or_module_id is None:
            fe_or_module = np.unique(data[data[message_column] == msg][module_column])
            if fe_or_module.shape[0] > 0:
                fe_or_module_id = mod_dict[fe_or_module[0]]
                fe_or_module = fe_or_module[0].replace('odule', '').replace(' ', '')
        if not fe_or_module_id:
            continue
        ont.add((amberg_ns['Event-'+str(id)], RDF.type, base_ns['Event']))
        ont.add((amberg_ns['Event-'+str(id)], occursOn, amberg_ns[fe_or_module[0]]))
        if "Stau" in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Jam-Event']))
        elif "Achse" in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Axis-Event']))
        elif "F?llstand" in msg or "fehlt" in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Material-Event']))
        elif "Schutzt?re" in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Door-Event']))
        elif 'Variantenwechsel' in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Changeover-Event']))
        elif 'Staubsauger' in msg:
            ont.add((URIRef(amberg_ns['Event-' + str(id)]), RDF.type, base_ns['Cleaning-Event']))
        ont.add((amberg_ns[fe_or_module], RDF.type, amberg_ns['ProductionUnit']))
        ent_dict[str(amberg_ns['Event-'+str(id)])] = id
        ent_dict[str(amberg_ns[fe_or_module])] = fe_or_module_id


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

    print(messages_to_fe_df.head(10))
    # messages_to_fe_df.to_csv("./test_data/messages_to_fe.txt", sep=",", quotechar='"')
    # reverse_messages_to_fe = dict(zip(messages_to_fe.values(), messages_to_fe.keys()))

    return messages_to_fe_df


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


def prepare_sequences(data_frame, index, unique_dict, window_size, max_seq, g_train):
    """

    :param data_frame:
    :param index:
    :param unique_dict:
    :param window_size:
    :param max_seq: for amberg maximum number of separated sequences to consider
    :param g_train:
    :return:
    """
    train_data = time_window(data_frame, window_size)
    result = []
    overall_length = 0
    zero_shot_dict = dict()
    non_zero_dict = dict()
    print("Preparing sequential data...")
    for i, seq in enumerate(train_data[:max_seq]):
        local_entities = [event[index] for event in seq]
        tmp_list = [unique_dict[entity] for entity in local_entities]
        result.append(tmp_list)
        overall_length += len(tmp_list)
        for event_entity in tmp_list:
            tmp_uri = amberg_ns['Event-' + str(event_entity)]
            if (tmp_uri, None, None) not in g_train and (None, None, tmp_uri) not in g_train:
                if tmp_uri in zero_shot_dict:
                    continue
                else:
                    zero_shot_dict[tmp_uri] = True
            else:
                non_zero_dict[tmp_uri] = True
    print(result[:10])
    print("Zero shot events: ", len(zero_shot_dict))
    print("Non zero shot events: ", len(non_zero_dict))
    print("Processed {0} sequences: ".format(len(result)))
    print("Overall length of sequence: ", overall_length)
    return result


def embs_to_df(embs, reverse_dictionary):
    colnames = ['x' + str(i) for i in range(embs.shape[1])]
    df = pd.DataFrame(embs, columns=colnames)
    df["id"] = [i for i in range(embs.shape[0])]
    df["uri"] = [reverse_dictionary[k] for k in reverse_dictionary]
    df = df.set_index("id")
    return df


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

