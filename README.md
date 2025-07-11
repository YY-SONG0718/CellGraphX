# CellGraphX

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit/)

- Cross-species Integration: the method is aimed for cross-species integration analysis
- Graph Learning: the method is based on heterogeneous graph neural networks
- Weighted edges: the gene-cell type edges are weighted by log2FC of the gene in the cell type

## Overview

CellGraphX is a method for cross-species integration analysis. It uses a heterogeneous graph neural network to learn the relationships between genes and cell types across different species.

The heterogeneous graph currently contains 3 types of nodes: 
- gene
- orthologous group
- cell type

and 2 types of edges: 
- gene-orthologous group
- gene-cell type

The gene-orthologous group edges are used to connect genes that are orthologous to each other across species, while the gene-cell type edges are used to connect genes to the cell types in which they are highly expressed (marker genes). The weights of the gene-cell type edges are determined by the log2FC of the gene in the cell type, allowing for a more nuanced understanding of the relationships between genes and cell types. Gene-orthologous group edges are unweighted.

Originally, the heterogeneous graph is built from querying a GeneSpectraKG Neo4j database and parsing the results to PyG HeteroData. However, Yuyao is also working on more traditional ways to contract a graph from a tabular data.

## Running

First generate the heterogeneous graph data from a GeneSpectraKG Neo4j database. You can do this by running the following command:

```bash
python data/generate_input_from_KG_script.py
```

Run CellGraphX with the following command:

```bash
python CellGraphX/main.py 
```

## LICENSE
MIT license. See LICENSE file for details.

## Get in touch

Yuyao Song <ysong@ebi.ac.uk> wrote all the code here. 
