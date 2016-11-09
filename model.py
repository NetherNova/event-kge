import numpy as np
import scipy
import scipy.sparse
import theano
import theano.sparse as S
import theano.tensor as T
from collections import OrderedDict
import sys
import time
from rdflib import ConjunctiveGraph, URIRef, Literal, RDF, RDFS, OWL

"""
Text-Local context word embedding + global event-level embeddings
+ Module-local

Local process model: concatenate events per module together
Global process model: concatenate event by time

Variant-specific?
"""
class Embeddings(object):
    def __init__(vocab_size, dim):
        wbound = np.sqrt(6. /vocab_size)
        W = theano.shared(np.random.uniform(low=-wbound, high=wbound),
                          size=(vocab_size, dim))
        

def get_core_model(path):
    """
    path: path to the semantic model
    Provided relations hasPart, hasFollower, connectedTo (functional_dependence)
    """
    
    


if __name__ == '__main__':
    
