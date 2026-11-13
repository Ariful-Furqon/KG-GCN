# KG-GCN: graph convolution on a rice pest–disease knowledge graph

Code for the manuscript *Graph convolution on a rice pest–disease knowledge graph: receptive
fields and exact bounds for ranking metrics* (submitted to the Proceedings of the Jangjeon
Mathematical Society, ICCGANT 2025).

The knowledge graph links rice symptoms (*Gejala*) to pests (*HamaPadi*) and diseases
(*PenyakitPadi*). A graph convolutional network classifies each vertex as a diagnostic entity
(pest or disease) or a symptom, and ranks held-out vertices by the predicted score.

## Repository layout

| Path | Contents |
|---|---|
| `Ontology/OntologiHamaPenyakitPadi_v1.1.rdf` | Source ontology: 84 vertices (43 pests/diseases, 41 symptoms), 136 arcs |
| `Model/KG-GCN_evaluated.py` | Original experiment: Neo4j → Node2Vec → 3-layer GCN, stratified 10-fold CV. Produces the manuscript results |
| `Model/KG-GCN_testing.py`, `Model/KG-GCN.ipynb` | Earlier development versions |
| `Model/Ref/` | Earlier exploratory notebooks |
| `experiments/node_classification.py` | Re-implementation from the RDF file (no Neo4j), ablations and baselines |
| `experiments/link_prediction.py` | Symptom-conditioned diagnosis posed as link prediction |
| `experiments/results/` | Saved outputs of the two experiment scripts |
| `paper/make_figures.py` | Generates Figures 2 and 3 of the manuscript from the per-fold rank configurations |
| `*.png` | Figures produced by `Model/KG-GCN_evaluated.py` |

## Running the original experiment

The original script reads the graph from a Neo4j instance into which the ontology has been
imported. Credentials are read from environment variables:

```bash
export NEO4J_URI=bolt://localhost:7687   # optional, this is the default
export NEO4J_USER=neo4j                  # optional, this is the default
export NEO4J_PASSWORD=...                # required
python Model/KG-GCN_evaluated.py
```

Dependencies: `neo4j`, `pandas`, `numpy`, `networkx`, `node2vec`, `torch`, `torch_geometric`,
`scikit-learn`, `matplotlib`.

`RESTORE_BEST_COPY = False` (the default) reproduces the manuscript results. In that setting
`model.state_dict()` is stored without a copy, so the early-stopping "best" state is overwritten
by later steps and the final weights are evaluated. Set it to `True` to restore the actual best
checkpoint.

## Experiments without Neo4j

Both scripts read the ontology directly (override the path with `KG_RDF`) and need only
`rdflib`, `numpy` and `torch`. Run them from the `experiments/` directory:

```bash
cd experiments
python node_classification.py 5    # 5 seeds x 10 folds -> results/node_classification.json
python link_prediction.py 3        # 3 seeds x 10 folds -> results/link_prediction.json
```

The Node2Vec random walks and the GCN layer are re-implemented in plain PyTorch with the same
normalisation as PyG's `GCNConv`. The numbers are therefore close to, but not bit-identical
with, the Neo4j/PyG pipeline.

### Node classification (5 seeds × 10 folds)

| Configuration | AP | Accuracy | Log loss | Inverted folds |
|---|---|---|---|---|
| Original protocol (E0) | 0.715 | 0.517 | 0.688 | 34% |
| Inner validation split, checkpoint copy fixed (E1) | 0.706 | 0.510 | 0.690 | 32% |
| Tuned: lr 1e-2, dropout 0.5, weight decay 5e-4, 300 epochs (E2) | 0.777 | 0.549 | 0.710 | 20% |
| Separate self/neighbour weights (E6) | 1.000 | 0.988 | 0.123 | 0% |
| Baseline: rule `in-degree > 0` (B1) | 0.999 | 0.988 | 0.197 | 0% |
| Baseline: MLP on Node2Vec, no graph (B2) | 0.993 | 0.922 | 0.707 | 0% |

The graph is strongly heterophilous (edge homophily 11/136 ≈ 0.08), and the class of a vertex
is determined, with one exception, by its in-degree in the message direction. A layer with
separate self and neighbour weights therefore separates the classes, but so does the one-line
in-degree rule. See Proposition 6.1 of the manuscript.

### Link prediction (3 seeds × 10 folds, 43 candidates per query)

| Method | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---|---|---|---|
| Random | 0.123 | 0.029 | 0.093 | 0.304 |
| Node2Vec dot product | 0.130 | 0.021 | 0.107 | 0.333 |
| GCN encoder | 0.155 | 0.043 | 0.136 | 0.408 |
| GCN with separate self/neighbour weights | 0.159 | 0.048 | 0.128 | 0.429 |

All structure-based methods are close to random. Symptom-conditioned diagnosis needs a larger
graph or textual symptom features.

## Manuscript figures

```bash
cd paper
python make_figures.py   # writes fig2_metrics_vs_k.pdf and fig4_per_fold.pdf
```

The per-fold rank configurations hard-coded in the script reproduce every metric reported by
`Model/KG-GCN_evaluated.py` to four decimal places.
