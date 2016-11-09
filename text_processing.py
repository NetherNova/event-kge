import numpy as np
import sklearn

"""
Class for text processing
"""
def tokenize(text):
    text = text.replace(':', '')
    return text.split(' ')

if __name__ == '__main__':
    test_data_path = "./test_data/"
    event_file = test_data_path + "test_data.csv"
    f = open(event_file, "rb")
    for i, line in enumerate(f):
        if i == 0:
            continue
        cols = line.split("\t")
        event = cols[1]
        module = cols[2]
        time = cols[3]
        print tokenize(event)
