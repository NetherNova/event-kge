# Knowledge Graph and Sequence Data Embedding Models

Implementation of translation-based models (TransE [Bordes et al. 2013], TransH) in tensorflow.

For the use of sequential data (e.g. text corpora, events, etc.),
there is the option to use a joint model (i.e. TransE + Skipgram (TEKE) either pre-trained or joint training)

More advanced sequential embedding models (event models) can be plugged-in (Convolutional-Autoencoder, Concatenation, ...)

Data preparation works directly with ontologies (RDF or OWL)
For triple processing, rdflib is used.

## How to run:
Put the *path_to_kg* and optional *path_to_sequence* in ekl_experiment.py
python ekl_experiment.py

## Requirements:
- rdflib (4.1.2)
- pandas (0.19.2)
- numpy (1.13.0)
- TensorFlow (1.1.0)
