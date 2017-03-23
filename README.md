# Knowledge Graph and Sequence Data Embedding Models

Implementation of translation-based models (TransE [Bordes et al. 2013]) in tensorflow.
For the use of sequential data (e.g. text corpora, events, etc.),
there is the option to use a joint model (i.e. TransE + Skipgram trained jointly)

Data preparation works directly with ontologies (RDF or OWL)
For triple processing, rdflib is used.

## Requirements:
- rdflib
- pandas
- numpy
- scikit-learn