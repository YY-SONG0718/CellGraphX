# CellGraphX

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit/)

## Overview

CellGraphX is a method for cross-species cell type prediction. It uses a heterogeneous graph neural network to learn the relationships between genes and cell types across different species.

The heterogeneous graph currently contains 3 types of nodes: 

- gene
- orthologous group
- cell type

and 2 types of edges: 

- gene-orthologous group
- gene-cell type

The gene-orthologous group edges are used to connect genes that belong to the same orthologous group, while the gene-cell type edges are used to connect genes to the cell types in which they are highly expressed (marker genes). The weights of the gene-cell type edges are the log2FC values of the gene as a marker in the cell type, allowing for a more nuanced understanding of the relationships between genes and cell types. Gene-orthologous group edges are unweighted.

Originally, the heterogeneous graph is built from querying a GeneSpectraKG Neo4j database and parsing the results to PyG HeteroData. However, Yuyao is also working on more general ways to contract a graph from a tabular data.

## Running

First generate the heterogeneous graph data from a GeneSpectraKG Neo4j database. You can do this by running the following command:

```bash
python data/generate_input_from_KG_script.py
```

Run CellGraphX with the following command:

```bash
python CellGraphX/main.py 
```

This will perform training, evaluation and generate a t-SNE visualisation using the default configuration. You can modify the configuration in `configs/config.py`.

If hyperparameter optimization is desired, run:

```bash
python notebooks/optuna_example.py
```


## LICENSE
MIT license. See LICENSE file for details.

## Get in touch

Yuyao Song <ysong@ebi.ac.uk> wrote all the code here. 
