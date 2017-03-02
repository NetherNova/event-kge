import os
import pandas as pd
import glob


# indices for pandas dataframe
time_column = 'Timestamp'
module_column = 'Module'
variant_column = 'MLFB'
message_column = 'Meldetext'
# indices for numpy array
message_index = 0
module_index = 1
variant_index = 2


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


def time_window(df, window_size, classification_labels=[]):
    """
    Extract list of lists of events using sliding window of *window_size*
    df: sorted dataframe
    window_size: time window size in minutes
    """
    train = []
    labels = []
    labels_dict = dict()
    off = pd.DateOffset(minutes=window_size)
    for i in xrange(df.shape[0]):
        window_start = df.iloc[i][time_column]
        window_end = window_start + off
        local_window = df[window_start : window_end]
        # one sequence is a list of 3-tuples [[message, module, variant], [mesage, module, variant] ...]
        train.append(local_window[[message_column, module_column, variant_column]].values.tolist())
        local_labels = set()
        for m in local_window[message_column]:
            if m not in labels_dict:
                labels_dict.setdefault(m, len(labels_dict))
            if m in classification_labels:
                local_labels.add(labels_dict[m])
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


if __name__ == '__main__':
    path = "/home/nether-nova/Documents/Amberg Events/Week40/"
    max_events = 5000
    df = read_data(path, max_events)
    data = time_window(df, 3)
    print len(data[0])