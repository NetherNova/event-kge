from model import save_embedding
import numpy as np
from etl import read_data, context_windows, time_window
import pandas as pd
import matplotlib.pyplot as plt
import operator
from matplotlib import ticker
import seaborn as sns


module_sort_dict = {"Module 010" : 1, "Module 020" : 2, "Module 030" : 3, "Module 200": 4,
                    "Module 040" : 4, "Module 050" : 5, "Module 070" : 6, "Module 090" : 7,
                    "Module 100" : 8, "Module 110" : 9, "Module 120" : 10 }


def correlation_matrix(sequence, entity, num_unique_entities):
    """
    input: entity: one of [0,1,2]: message, module, variant
    """
    outer = []
    index_dict = dict()
    message_index_module_dict = dict()
    # seq = [[message, module, variant], [message, module, variant], ...]
    for seq in sequence:
        entity_slice = np.array(seq)[:, entity]
        inner = np.zeros(num_unique_entities)
        for i, tmp_entity in enumerate(entity_slice):
          index_dict.setdefault(tmp_entity, len(index_dict))
          message_index_module_dict[index_dict[tmp_entity]] = seq[i][1].replace('"', '')
          tmp_entity_index = index_dict[tmp_entity]
          #inner[tmp_entity_index] = 1
          inner[tmp_entity_index] += 1
        outer.append(inner)
    df = pd.DataFrame(outer)
    df = df.reindex_axis(sorted(df.columns, key=lambda x:module_sort_dict[message_index_module_dict[x]]), axis=1)
    correlation_matrix = df.corr()
    return correlation_matrix, df, index_dict, message_index_module_dict


def co_occurrence_matix(sequence, entity, num_unique_entities):
    """
    input: entity: one of [0,1,2]: message, module, variant
    """
    outer = []
    index_dict = dict()
    message_index_module_dict = dict()
    # seq = [[message, module, variant], [message, module, variant], ...]
    for seq in sequence:
        entity_slice = np.array(seq)[:, entity]
        inner = np.zeros(num_unique_entities)
        for i, tmp_entity in enumerate(entity_slice):
          index_dict.setdefault(tmp_entity, len(index_dict))
          message_index_module_dict[index_dict[tmp_entity]] = seq[i][1].replace('"', '')
          tmp_entity_index = index_dict[tmp_entity]
          #inner[tmp_entity_index] = 1
          inner[tmp_entity_index] += 1
        outer.append(inner)
    X = np.array(outer)
    Xc = np.dot(X.T, X)
    #normalized
    Xc = np.dot(Xc, (np.diag(1./Xc.diagonal())))
    #df = pd.DataFrame(Xc)
    #df = df.reindex_axis(sorted(df.columns, key=lambda x:module_sort_dict[message_index_module_dict[x]]), axis=1)
    #correlation_matrix = df.corr()
    return Xc, index_dict, message_index_module_dict


def co_occurrence_from_to_matrix(sequence, entity, num_unique_entities):
    from_to_matrix = np.zeros((num_unique_entities, num_unique_entities))
    index_dict = dict()
    message_index_module_dict = dict()
    # seq = [[message, module, variant], [message, module, variant], ...]
    for seq in sequence:
        entity_slice = np.array(seq)[:, entity]
        for i in xrange(len(entity_slice)):
            for j in xrange(i + 1, len(entity_slice)):
                if entity_slice[i] not in index_dict:
                    index_dict.setdefault(entity_slice[i], len(index_dict))
                if entity_slice[j] not in index_dict:
                    index_dict.setdefault(entity_slice[j], len(index_dict))
                message_index_module_dict[index_dict[entity_slice[i]]] = seq[i][1].replace('"', '')
                i_index = index_dict[entity_slice[i]]
                j_index = index_dict[entity_slice[j]]
                from_to_matrix[i_index][j_index] += 1
    X = np.array(from_to_matrix)
    #X = X * (1.0/X.diagonal())
    return X, index_dict, message_index_module_dict


def co_occurrence_from_to_matrix_skips(sequence, entity, num_unique_entities, num_skips):
    from_to_matrix = np.zeros((num_unique_entities, num_unique_entities))
    index_dict = dict()
    message_index_module_dict = dict()
    # seq = [[message, module, variant], [message, module, variant], ...]
    for seq in sequence:
        entity_slice = np.array(seq)[:, entity]
        for i in xrange(len(entity_slice) - num_skips):
            for j in xrange(i + 1, i + num_skips):
                if entity_slice[i] not in index_dict:
                    index_dict.setdefault(entity_slice[i], len(index_dict))
                if entity_slice[j] not in index_dict:
                    index_dict.setdefault(entity_slice[j], len(index_dict))
                message_index_module_dict[index_dict[entity_slice[i]]] = seq[i][1].replace('"', '')
                i_index = index_dict[entity_slice[i]]
                j_index = index_dict[entity_slice[j]]
                from_to_matrix[i_index][j_index] += 1
    X = np.array(from_to_matrix)
    #X = X * (1.0/X.diagonal())
    return X, index_dict, message_index_module_dict


def plot_correlation_matrix(input_matrix, labels, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    cax = ax.matshow(input_matrix, interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    fig = plt.gcf()
    plt.show()
    fig.savefig("./Plots/" + file_name + ".svg", format='svg', dpi=1200)


def plot_matrix_sns(input_matrix, labels, file_name):
    sns.set(style="white")
    mask = np.zeros_like(input_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(input_matrix, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=5, yticklabels=5,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    sns_plot.get_figure().savefig("./Plots/" + file_name + "_sns" + ".eps", format='eps', dpi=1200)



def join_messages_to_fe(file_name, df):
    data = pd.read_csv(file_name)
    merged = pd.merge(df, data, left_on="Meldetext", right_on="Meldetext")
    return merged


if __name__ == '__main__':
    data = read_data("./test_data/")
    print(len(data))
    max_events=5000 #len(data)
    data = data[:max_events]

    entity = 0
    corr = "corr"
    from_to = "from_to"
    cooc = "cooc"
    mat_type = corr
    if mat_type == from_to:
        file_name = "from_to_embeddings"
        matrix, index_dict, message_index_module_dict = co_occurrence_from_to_matrix(
            time_window(data, 5), entity, len(np.unique(np.array(data)[:, entity]))
           )
        norm = np.sum(matrix, axis=1)
        norm_matrix = matrix / norm[:, np.newaxis]
        norm_matrix = matrix / norm
        for k, v in index_dict.iteritems():
            print("For Module", k)
            top_k_inds = (-norm_matrix[v, :]).argsort()[:5 + 1]
            print("Top k: ")
            for i in top_k_inds:
                module = [m for m in index_dict if index_dict[m] == i][0]
                print(module + ":" + str(norm_matrix[v, i]))
    elif mat_type == corr:
        file_name = "corr_embeddings"
        matrix, df, index_dict, message_index_module_dict = correlation_matrix(
        time_window(data, 5), entity, len(np.unique(np.array(data)[:, entity]))
        )
    else:
        file_name = "cooc_embeddings"
        matrix, index_dict, message_index_module_dict = co_occurrence_matix(
        time_window(data, 5), entity, len(np.unique(np.array(data)[:, entity]))
        )
    num_dim = 128
    U,S,V = np.linalg.svd(matrix)
    #reconstructed = np.dot(U[:,:45], np.dot(np.diag(S[:45]), V[:45,:]))
    reduced_embeddings = np.dot(U[:,:num_dim], np.diag(S[:num_dim]))
    save_embedding("./Embeddings/" + file_name + ".pickle", index_dict, reduced_embeddings)
    labels = [l[0] for l in sorted(index_dict.items(), key=operator.itemgetter(1))]
    #plot_correlation_matrix(matrix, labels, file_name)
    g=sns.clustermap(matrix)
    g.savefig("test.png", format='png')
    #plot_matrix_sns(matrix, labels, file_name)