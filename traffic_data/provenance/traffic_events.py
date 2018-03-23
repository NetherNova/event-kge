__author__ = 'martin'

import pandas as pd
from rdflib import ConjunctiveGraph, URIRef, Literal, RDF, RDFS, Namespace, OWL


base_uri = Namespace('http://www.siemens.com/citypulse#')
path_processed = 'processed/'
path_src = 'original_data/'
path_kg = 'kg/'


def generate_events(tr_df, result, mapping):
    # Vehicle count events (implicit semantics)
    # TODO: events for average traffic decreases drastically, increases drastically over last n-steps...
    time_column = "TIMESTAMP"
    tr_df[time_column] = pd.to_datetime(tr_df[time_column], format="%Y-%m-%dT%H:%M:%S")
    tr_df = tr_df.set_index(pd.DatetimeIndex(tr_df[time_column]))
    tr_df = tr_df.sort_index(ascending=True)
    mean_n_vs = tr_df["vehicleCount"].mean()
    std_n_vs = tr_df["vehicleCount"].std()
    sliding_window = pd.DateOffset(minutes=30)

    for i in xrange(tr_df.shape[0]):
        window_start = tr_df.iloc[i][time_column]
        window_end = window_start + sliding_window
        local_window = tr_df[window_start : window_end]
        diff = local_window.iloc[0]['vehicleCount'] - local_window.iloc[-1]['vehicleCount']
        if diff < -1 * std_n_vs:
            id = "Event" + str(diff) + "-" + str(local_window['REPORT_ID'][0])
            result.append([local_window[time_column][-1], id, "DecreasedEvent", str(local_window['REPORT_ID'][0])])
            mapping[id] = {'type':  'DecreasedEvent', 'occursAt': str(local_window['REPORT_ID'][0])}
        elif diff > std_n_vs:
            id = "Event" + str(diff) + "-" + str(local_window['REPORT_ID'][0])
            result.append([local_window[time_column][-1], id, "IncreasedEvent", str(local_window['REPORT_ID'][0])])
            mapping[id] = {'type':  'IncreasedEvent', 'occursAt': str(local_window['REPORT_ID'][0])}
    """
    tr_df.loc[tr_df["vehicleCount"] > (mean_n_vs + 0.5 * std_n_vs), "EventType"] = "HighTrafficEvent"
    tr_df.loc[(tr_df["vehicleCount"] <= (mean_n_vs + 0.5 * std_n_vs)) & (tr_df["vehicleCount"] >= (mean_n_vs - 0.5 * std_n_vs)), "EventType"] = "NormalTrafficEvent"
    tr_df.loc[tr_df["vehicleCount"] < (mean_n_vs - 0.5 * std_n_vs), "EventType"] = "LowTrafficEvent"

    tr_df["Event"] = ["Event" + str(tr_df.loc[i, "vehicleCount"]) + "-" + str(tr_df.loc[i, "REPORT_ID"]) for i in range(len(tr_df))]
    return tr_df[["Event", "EventType", "TIMESTAMP", "REPORT_ID"]]
    """


def start_event_generation():
    metadata = pd.read_csv(path_src + 'trafficMetaData.csv', sep=',')
    metadata = metadata[metadata['POINT_1_CITY'] == 'Aarhus']
    frames = []
    mapping = {}
    for i, entry in enumerate(metadata["REPORT_ID"]):
        print('Report ID: ' + str(entry) + " - " + str(int(100 * (1.0 + i) / metadata.shape[0])) + '%')
        traffic_file = 'traffic_feb_june/trafficData' + str(entry) + '.csv'
        traffic_data = pd.read_csv(path_src + traffic_file)
        generate_events(traffic_data, frames, mapping)
    merged = pd.DataFrame(frames, columns=["Timestamp", "Id", "Type", "At"])
    merged = merged.set_index(pd.DatetimeIndex(merged["Timestamp"]))
    merged = merged.sort_index(ascending=True)
    merged.to_csv(path_processed + "events.csv")
    pd.DataFrame.from_dict(mapping, orient='index').to_csv(path_processed + "mapping.csv")


def parse_log():
    log_file = path_processed + 'location_types.log'
    poi = []
    with open(log_file, 'rb') as f:
        c = 0
        for line in f:
            right_part = line.split(':')[1]
            if right_part.find('OVER_QUERY_LIMIT'):
                continue
            if (c % 2) == 0:
                poi_tmp = []
                entry = []
                point = right_part.split(' ')[1].strip()
                entry.append(point)
                print('POI: ', point)
            else:
                entities = right_part[right_part.find('['):]
                entities = [e.strip().replace('"', '') for e in entities.replace('[', '').replace(']', '').split(',')]
                entry.append(entities)
                poi_tmp.append(entry)
                print('Entities: ', entities)
            c += 1
            poi.append(poi_tmp)
    return poi


def remove_non_ascii(str):
    char_arr = []
    for char in str:
        if char == ' ':
            continue
        try:
            char.decode()
        except UnicodeDecodeError:
            continue
        char_arr.append(char)
    return ''.join(char_arr)


def populate_from_metadata(metadata, g):
    for i in xrange(metadata.shape[0]):
        row = metadata.iloc[i]
        path_id = str(row['REPORT_ID'])
        path_uri = base_uri[path_id]
        g.add((path_uri, RDF.type, base_uri['Path']))
        from_id = str(row['POINT_1_NAME'])
        from_uri = base_uri[from_id]
        from_street = remove_non_ascii(str(row['POINT_1_STREET']))
        from_street_uri = base_uri[from_street]
        from_city = remove_non_ascii(row['POINT_1_CITY'])
        from_city_uri = base_uri[from_city]
        to_id = str(row['POINT_2_NAME'])
        to_uri = base_uri[to_id]
        to_street = remove_non_ascii(str(row['POINT_2_STREET']))
        to_street_uri = base_uri[to_street]
        to_city = remove_non_ascii(row['POINT_2_CITY'])
        to_city_uri = base_uri[to_city]
        g.add((path_uri, base_uri['hasStart'], from_uri))
        g.add((path_uri, base_uri['hasEnd'], to_uri))
        g.add((from_uri, base_uri['locatedAt'], from_street_uri))
        g.add((to_uri, base_uri['locatedAt'], to_street_uri))
        g.add((from_street_uri, RDF.type, base_uri['Route']))
        g.add((to_street_uri, RDF.type, base_uri['Route']))
        # evt explicitly add street in city ...
        g.add((from_uri, base_uri['locatedAt'], from_city_uri))
        g.add((to_uri, base_uri['locatedAt'], to_city_uri))
        g.add((from_uri, RDF.type, base_uri['Address']))
        g.add((to_uri,  RDF.type, base_uri['Address']))
        g.add((to_city_uri, RDF.type, base_uri['City']))
        g.add((from_city_uri, RDF.type, base_uri['City']))


def populate_ontology():
    ont_path = path_kg + 'traffic_ontology.xml'
    metadata = pd.read_csv(path_src + 'trafficMetaData.csv', sep=',')
    g = ConjunctiveGraph()
    g.load(ont_path)
    g.add((URIRef(base_uri), RDF.type, OWL.Ontology))
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    # g.bind("city", base_uri)
    # populate from metadata: [Path, from[name], to[name], from[has[street]], to[has[street]]]
    populate_from_metadata(metadata, g)
    poi = parse_log()
    for entry in poi:
        point = entry[0][0].split('_')[0] + "_" + entry[0][0].split('_')[1]
        metadata_entry = metadata[metadata['REPORT_ID'] == int(entry[0][0].split('_')[2])]
        address_id = metadata_entry[point + '_NAME'].values[0]

        poi_list = entry[0][1]
        for tmp_poi in poi_list:
            # generate an id for the poi
            tmp_poi_id = str(abs(hash(point + '_' + str(address_id) + '_' + tmp_poi)))
            g.add((base_uri[tmp_poi_id], RDF.type, base_uri['Point_of_interest']))
            g.add((base_uri[tmp_poi_id], RDF.type, base_uri[tmp_poi[0].upper() + tmp_poi[1:]]))
            g.add((base_uri[tmp_poi_id], base_uri['locatedAt'], base_uri[str(address_id)]))

    simple_sequence = []
    events = pd.read_csv(path_processed + 'events.csv')
    mapping = pd.read_csv(path_processed + 'mapping.csv').T.to_dict()
    for k, v in mapping.iteritems():
        g.add((base_uri[v['Unnamed: 0']], base_uri['occursAt'], base_uri[str(v['occursAt'])]))
        g.add((base_uri[v['Unnamed: 0']], RDF.type, base_uri[v['type']]))

    for e in events['Id']:
        simple_sequence.append(e)
    with open(path_processed + 'sequence.txt', "wb") as seq_file:
        seq_file.write(','.join(simple_sequence))
    g.serialize(path_kg + 'traffic_individuals.xml', format='xml')

start_event_generation()
populate_ontology()

mapping = pd.read_csv(path_processed + 'mapping.csv').T.to_dict()
events = pd.read_csv(path_processed + 'events.csv')
simple_sequence = []

unique_msg_dict = dict()
for k in mapping:
    event_id = mapping[k]['Unnamed: 0']
    unique_msg_dict[event_id] = k

for e in events['Id']:
    simple_sequence.append(str(unique_msg_dict[e]))

with open(path_processed + 'sequence.txt', "wb") as seq_file:
    seq_file.write(','.join(simple_sequence))

with open(path_processed + 'unique_msgs.txt', "wb") as msg_file:
    for k, v in unique_msg_dict.iteritems():
        msg_file.write(k + ',' + str(v) + '\n')
