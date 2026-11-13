# TODO for Adding Testing for @1, @3, @5, @10, @20 and Visualization in KG-GCN_evaluated.py

- [ ] Change k from single value to list ks = [1, 3, 5, 10, 20]
- [ ] Modify metric storage: Use dictionaries for k-dependent metrics (precisions, recalls, f1s, ndcgs, hits)
- [ ] Update fold loop: Compute metrics for each k in ks
- [ ] Update printing: Print results for each k per fold
- [ ] Update averaging: Compute averages and stds for each k
- [ ] Update visualizations: Add line plot for k-dependent metrics vs k, keep existing plots for k=20
- [ ] Run the script and verify outputs
