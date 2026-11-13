from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ndcg_score, average_precision_score, log_loss
import matplotlib.pyplot as plt

uri = "bolt://localhost:7687"
username = "neo4j"
password = "***REMOVED***"

driver = GraphDatabase.driver(uri, auth=(username, password))

class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 2)  # 2 classes: 0 and 1
        self.dropout = torch.nn.Dropout(p=0.7)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

def extract_node_properties_and_relationships():
    with driver.session() as session:
        # Cypher query to fetch node properties and relationships
        query = """
        MATCH (n)-[r]->(m)
            WHERE any(label IN labels(n) WHERE label IN ['Gejala', 'PenyakitPadi', 'HamaPadi'])
            AND any(label IN labels(m) WHERE label IN ['Gejala', 'PenyakitPadi', 'HamaPadi'])
        RETURN
         n.label AS target_label, labels(n) AS target_class,
        type(r) AS relationship_type,
         m.label AS source_label, labels(m) AS source_class
        """
        result = session.run(query)
        # Extract properties and relationships and store in DataFrame
        df = pd.DataFrame([record.values() for record in result], columns=result.keys())
        # Clean labels
        df['source_class'] = df['source_class'].apply(lambda entry: [item for item in entry if item not in ["Resource", "NamedIndividual"]])
        df['target_class'] = df['target_class'].apply(lambda entry: [item for item in entry if item not in ["Resource", "NamedIndividual"]])
        return df

# Call the function to extract node properties and relationships
Rice_KG_df = extract_node_properties_and_relationships()

# Print the DataFrame
print(Rice_KG_df)

G = nx.DiGraph()
edges = Rice_KG_df[['source_label', 'target_label']].values.tolist()
G.add_edges_from(edges)

node2vec = Node2Vec(G, dimensions=64, walk_length=10, num_walks=50, workers=1, seed=42)
n2v_model = node2vec.fit(window=5, min_count=1, batch_words=4)

nodes = list(G.nodes())
node2idx = {node: idx for idx, node in enumerate(nodes)}
x = torch.tensor([n2v_model.wv[node] for node in nodes], dtype=torch.float)

edge_index = Rice_KG_df[['source_label', 'target_label']].applymap(lambda x: node2idx[x]).values.T
edge_index = torch.tensor(edge_index, dtype=torch.long)

def is_gejala(label):
    kelas = Rice_KG_df.loc[Rice_KG_df['source_label'] == label, 'source_class']
    if kelas.empty:
        kelas = Rice_KG_df.loc[Rice_KG_df['target_label'] == label, 'target_class']
    kelas = kelas.iloc[0] if not kelas.empty else []
    return 'Gejala' in kelas

def is_hama_penyakit(label):
    kelas = Rice_KG_df.loc[Rice_KG_df['source_label'] == label, 'source_class']
    if kelas.empty:
        kelas = Rice_KG_df.loc[Rice_KG_df['target_label'] == label, 'target_class']
    kelas = kelas.iloc[0] if not kelas.empty else []
    return any(k in ['HamaPadi', 'PenyakitPadi'] for k in kelas)

# Label: 0 = bukan hama/penyakit, 1 = hama/penyakit
y = torch.tensor([1 if is_hama_penyakit(node) else 0 for node in nodes], dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

# Cross-validation evaluation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# accuracies = []
ks = [1, 3, 5, 10, 20]
precisions = {k: [] for k in ks}
recalls = {k: [] for k in ks}
f1s = {k: [] for k in ks}
ndcgs = {k: [] for k in ks}
hits = {k: [] for k in ks}
maps = []
mean_ranks = []
mrrs = []
log_losses = []
losses_per_fold = []  # To store training loss values per epoch for each fold
val_losses_per_fold = []  # To store validation loss values per epoch for each fold

for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(nodes)), y.numpy())):
    print(f"Fold {fold + 1}:")

    # Create model for each fold
    model = SimpleGCN(in_channels=x.shape[1], hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.05)  # Increased L2 regularization

    # Training loop with early stopping
    losses = []  # To store training loss for each epoch in this fold
    val_losses = []  # To store validation loss for each epoch in this fold
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    best_model_state = None

    for epoch in range(50):  # Increased max epochs
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Compute validation loss
        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = F.cross_entropy(val_out[test_idx], data.y[test_idx])
            val_losses.append(val_loss.item())

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        model.train()

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    losses_per_fold.append(losses)
    val_losses_per_fold.append(val_losses)

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        probs = torch.softmax(out, dim=1)[:,1]
        y_true = data.y[test_idx].numpy()
        y_pred = pred[test_idx].numpy()
        y_score = probs[test_idx].numpy()

    # Calculate metrics for this fold
    accuracy = accuracy_score(y_true, y_pred)

    # Compute k-independent metrics once
    map_val = average_precision_score(y_true, y_score)
    if np.sum(y_true) > 0:
        ranks = np.argsort(np.argsort(-y_score)) + 1  # Rank of each item (1-based)
        mean_rank_val = np.mean(ranks[y_true == 1])  # Mean rank for positive samples
        mrr_val = np.mean(1.0 / ranks[y_true == 1])  # MRR for positive samples
    else:
        mean_rank_val = 0.0
        mrr_val = 0.0
    log_loss_val = log_loss(y_true, probs[test_idx])

    maps.append(map_val)
    mean_ranks.append(mean_rank_val)
    mrrs.append(mrr_val)
    log_losses.append(log_loss_val)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  MAP: {map_val:.4f}")
    print(f"  Mean Rank: {mean_rank_val:.4f}")
    print(f"  MRR: {mrr_val:.4f}")
    print(f"  Log Loss: {log_loss_val:.4f}")

    # Compute metrics for each k
    for k in ks:
        # Precision@k, Recall@k, F1@k, NDCG@k
        top_k_indices = np.argsort(y_score)[::-1][:k]
        y_true_topk = y_true[top_k_indices]
        precision_k_val = np.sum(y_true_topk) / k
        recall_k_val = np.sum(y_true_topk) / np.sum(y_true) if np.sum(y_true) > 0 else 0
        f1_k_val = 2 * (precision_k_val * recall_k_val) / (precision_k_val + recall_k_val) if (precision_k_val + recall_k_val) > 0 else 0
        ndcg_k_val = ndcg_score([y_true], [y_score], k=k)

        # Hits@k
        hits_k_val = precision_k_val  # Hits@k is equivalent to Precision@k in this binary ranking context

        precisions[k].append(precision_k_val)
        recalls[k].append(recall_k_val)
        f1s[k].append(f1_k_val)
        ndcgs[k].append(ndcg_k_val)
        hits[k].append(hits_k_val)

        print(f"  Precision@{k}: {precision_k_val:.4f}")
        print(f"  Recall@{k}: {recall_k_val:.4f}")
        print(f"  F1Score@{k}: {f1_k_val:.4f}")
        print(f"  NDCG@{k}: {ndcg_k_val:.4f}")
        print(f"  Hits@{k}: {hits_k_val:.4f}")
    print()

# Average metrics across folds
avg_map = np.mean(maps)
avg_mrr = np.mean(mrrs)
avg_log_loss = np.mean(log_losses)

print("Cross-Validation Results:")
print(f"Average MAP: {avg_map:.4f} ± {np.std(maps):.4f}")
print(f"Average MRR: {avg_mrr:.4f} ± {np.std(mrrs):.4f}")
print(f"Average Log Loss: {avg_log_loss:.4f} ± {np.std(log_losses):.4f}")

for k in ks:
    avg_precision_k = np.mean(precisions[k])
    avg_recall_k = np.mean(recalls[k])
    avg_f1_k = np.mean(f1s[k])
    avg_ndcg_k = np.mean(ndcgs[k])
    avg_hits_k = np.mean(hits[k])
    print(f"Average Precision@{k}: {avg_precision_k:.4f} ± {np.std(precisions[k]):.4f}")
    print(f"Average Recall@{k}: {avg_recall_k:.4f} ± {np.std(recalls[k]):.4f}")
    print(f"Average F1Score@{k}: {avg_f1_k:.4f} ± {np.std(f1s[k]):.4f}")
    print(f"Average NDCG@{k}: {avg_ndcg_k:.4f} ± {np.std(ndcgs[k]):.4f}")
    print(f"Average Hits@{k}: {avg_hits_k:.4f} ± {np.std(hits[k]):.4f}")

# Visualization per fold
metrics = ['Precision@10', 'Recall@10', 'NDCG@10', 'Hits@10', 'MAP', 'Mean Rank', 'MRR', 'Log Loss']
folds = list(range(1, 11))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    if metric == 'Precision@10':
        values = precisions[10]
    elif metric == 'Recall@10':
        values = recalls[10]
    elif metric == 'NDCG@10':
        values = ndcgs[10]
    elif metric == 'Hits@10':
        values = hits[10]
    elif metric == 'MAP':
        values = maps
    elif metric == 'Mean Rank':
        values = mean_ranks
    elif metric == 'MRR':
        values = mrrs
    elif metric == 'Log Loss':
        values = log_losses

    axes[i].plot(folds, values, marker='o', linestyle='-', color='b')
    axes[i].set_title(f'{metric} per Fold')
    axes[i].set_xlabel('Fold')
    axes[i].set_ylabel(metric)
    axes[i].grid(True)

plt.tight_layout()
plt.savefig('evaluation_metrics_per_fold.png')
plt.show()

# Overall average horizontal bar plot for head-to-head comparison (for k=10, best performing k)
avg_precision_10 = np.mean(precisions[10])
avg_recall_10 = np.mean(recalls[10])
avg_ndcg_10 = np.mean(ndcgs[10])
avg_hits_10 = np.mean(hits[10])
metrics_avg = ['Precision@10', 'Recall@10', 'NDCG@10', 'Hits@10', 'MAP', 'MRR', 'Log Loss']
averages = [avg_precision_10, avg_recall_10, avg_ndcg_10, avg_hits_10, avg_map, avg_mrr, avg_log_loss]
stds = [np.std(precisions[10]), np.std(recalls[10]), np.std(ndcgs[10]), np.std(hits[10]), np.std(maps), np.std(mrrs), np.std(log_losses)]

plt.figure(figsize=(12, 8))
bars = plt.barh(metrics_avg, averages, xerr=stds, capsize=5, color='skyblue', align='center')
plt.title('Average Cross-Validation Evaluation Metrics for KG-GCN Model (10-Fold)')
plt.xlabel('Score')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels on the bars
for bar, avg, std in zip(bars, averages, stds):
    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{avg:.4f} ± {std:.4f}', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('evaluation_metrics_average.png')
plt.show()

# Loss visualization: Training and Testing Loss per Fold
fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 5 columns for 10 folds
axes = axes.flatten()

for fold_idx in range(10):
    ax = axes[fold_idx]
    train_losses = losses_per_fold[fold_idx]
    val_losses = val_losses_per_fold[fold_idx]
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label='Training Loss', color='blue')
    ax.plot(epochs, val_losses, label='Validation Loss', color='red')
    ax.set_title(f'Fold {fold_idx + 1}: Training vs Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig('loss_comparison_per_fold.png')
plt.show()

# New visualization: Metrics vs k
plt.figure(figsize=(12, 8))
precision_vals = [np.mean(precisions[k]) for k in ks]
recall_vals = [np.mean(recalls[k]) for k in ks]
ndcg_vals = [np.mean(ndcgs[k]) for k in ks]
hits_vals = [np.mean(hits[k]) for k in ks]

plt.plot(ks, precision_vals, marker='o', label='Precision@k', color='blue')
plt.plot(ks, recall_vals, marker='s', label='Recall@k', color='green')
plt.plot(ks, ndcg_vals, marker='^', label='NDCG@k', color='red')
plt.plot(ks, hits_vals, marker='d', label='Hits@k', color='orange')

plt.xlabel('k')
plt.ylabel('Score')
plt.title('Average Metrics vs k for KG-GCN Model')
plt.legend()
plt.grid(True)
plt.xticks(ks)
plt.tight_layout()
plt.savefig('metrics_vs_k.png')
plt.show()

