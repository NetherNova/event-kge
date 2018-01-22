# Knowledge Graph and Sequence Data Embedding Models

Implementation of translation-based relational learning models (TransE [Bordes et al. 2013], TransH) in Tensorflow.

In addition to original source code:
More efficient triple scoring and mini-batch processing (Adagrad SGD).

Data preparation works directly with ontologies (RDF or OWL)
For triple processing, rdflib is used.

For the use of sequential data (e.g. text corpora, events, etc.),
there is the option to use a joint model (i.e. TransE + Skipgram (TEKE) either pre-trained or joint training)

## EKL
More advanced sequential embedding models (event models) can be plugged-in (Convolutional-Autoencoder, Concatenation, ...)

## How to run:
Put the *path_to_kg* and optional *path_to_sequence* in [ekl_experiment.py](./ekl_experiment.py)

Invoke: >
python ekl_experiment.py

## Bring your own data
- Put an rdf/xml file into your *path_to_kg*
- Put a *sequence.txt* file of comma-separated event IDs into *path_to_sequence*
- Supply a *unique_msgs.txt* mapping of the form: Event-URI fragment identifier | ID (starting from 0)

## Requirements:
- rdflib (4.1.2)
- pandas (0.19.2)
- numpy (1.13.0)
- TensorFlow (1.1.0)
