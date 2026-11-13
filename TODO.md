# TODO for Adding Testing for @1, @3, @5, @10, @20 and Visualization in KG-GCN_evaluated.py

- [x] Change k from single value to list ks = [1, 3, 5, 10, 20]
- [x] Modify metric storage: Use dictionaries for k-dependent metrics (precisions, recalls, f1s, ndcgs, hits)
- [x] Update fold loop: Compute metrics for each k in ks
- [x] Update printing: Print results for each k per fold
- [x] Update averaging: Compute averages and stds for each k
- [x] Update visualizations: Add line plot for k-dependent metrics vs k, keep existing plots for k=20
- [x] Run the script and verify outputs

# Open items

- [ ] Rotate the Neo4j passwords that were committed in earlier revisions (they remain in the git history)
- [ ] Save per-epoch losses (`losses_per_fold`, `val_losses_per_fold`) to disk so the loss figure can be regenerated
- [ ] Use a separate validation fold for early stopping (nested cross-validation)
- [ ] Evaluate a layer with separate self and neighbour weights in the original Neo4j/PyG pipeline
