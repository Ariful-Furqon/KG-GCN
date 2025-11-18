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
        self.conv2 = GCNConv(hidden_channels, 2)  # 2 classes: 0 and 1
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
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
precisions_k = []
recalls_k = []
# f1s_k = []
ndcgs_k = []
hits_k = []
maps = []
mean_ranks = []
mrrs = []
log_losses = []
losses_per_fold = []  # To store training loss values per epoch for each fold
val_losses_per_fold = []  # To store validation loss values per epoch for each fold
k = 20

for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(nodes)), y.numpy())):
    print(f"Fold {fold + 1}:")

    # Create model for each fold
    model = SimpleGCN(in_channels=x.shape[1], hidden_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)  # Added L2 regularization

    # Training loop
    losses = []  # To store training loss for each epoch in this fold
    val_losses = []  # To store validation loss for each epoch in this fold
    for epoch in range(10):
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
        model.train()

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

    # Precision@k, Recall@k, F1@k, NDCG@k
    top_k_indices = np.argsort(y_score)[::-1][:k]
    y_true_topk = y_true[top_k_indices]
    precision_k_val = np.sum(y_true_topk) / k
    recall_k_val = np.sum(y_true_topk) / np.sum(y_true) if np.sum(y_true) > 0 else 0
    f1_k_val = 2 * (precision_k_val * recall_k_val) / (precision_k_val + recall_k_val) if (precision_k_val + recall_k_val) > 0 else 0
    ndcg_k_val = ndcg_score([y_true], [y_score], k=k)

    # Hits@k, MAP
    hits_k_val = precision_k_val  # Hits@k is equivalent to Precision@k in this binary ranking context
    map_val = average_precision_score(y_true, y_score)

    # MRR, Log Loss
    if np.sum(y_true) > 0:
        ranks = np.argsort(np.argsort(-y_score)) + 1  # Rank of each item (1-based)
        mean_rank_val = np.mean(ranks[y_true == 1])  # Mean rank for positive samples
        mrr_val = np.mean(1.0 / ranks[y_true == 1])  # MRR for positive samples
    else:
        mean_rank_val = 0.0
        mrr_val = 0.0
    log_loss_val = log_loss(y_true, probs[test_idx])

    # accuracies.append(accuracy)
    precisions_k.append(precision_k_val)
    recalls_k.append(recall_k_val)
    # f1s_k.append(f1_k_val)
    ndcgs_k.append(ndcg_k_val)
    hits_k.append(hits_k_val)
    maps.append(map_val)
    mean_ranks.append(mean_rank_val)
    mrrs.append(mrr_val)
    log_losses.append(log_loss_val)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision@{k}: {precision_k_val:.4f}")
    print(f"  Recall@{k}: {recall_k_val:.4f}")
    print(f"  F1Score@{k}: {f1_k_val:.4f}")
    print(f"  NDCG@{k}: {ndcg_k_val:.4f}")
    print(f"  Hits@{k}: {hits_k_val:.4f}")
    print(f"  MAP: {map_val:.4f}")
    print(f"  Mean Rank: {mean_rank_val:.4f}")
    print(f"  MRR: {mrr_val:.4f}")
    print(f"  Log Loss: {log_loss_val:.4f}")
    print()

# Average metrics across folds
# avg_accuracy = np.mean(accuracies)
avg_precision_k = np.mean(precisions_k)
avg_recall_k = np.mean(recalls_k)
# avg_f1_k = np.mean(f1s_k)
avg_ndcg_k = np.mean(ndcgs_k)
avg_hits_k = np.mean(hits_k)
avg_map = np.mean(maps)
avg_mrr = np.mean(mrrs)
avg_log_loss = np.mean(log_losses)

print("Cross-Validation Results:")
# print(f"Average Accuracy: {avg_accuracy:.4f} ± {np.std(accuracies):.4f}")
print(f"Average Precision@{k}: {avg_precision_k:.4f} ± {np.std(precisions_k):.4f}")
print(f"Average Recall@{k}: {avg_recall_k:.4f} ± {np.std(recalls_k):.4f}")
# print(f"Average F1Score@{k}: {avg_f1_k:.4f} ± {np.std(f1s_k):.4f}")
print(f"Average NDCG@{k}: {avg_ndcg_k:.4f} ± {np.std(ndcgs_k):.4f}")
print(f"Average Hits@{k}: {avg_hits_k:.4f} ± {np.std(hits_k):.4f}")
print(f"Average MAP: {avg_map:.4f} ± {np.std(maps):.4f}")
print(f"Average MRR: {avg_mrr:.4f} ± {np.std(mrrs):.4f}")
print(f"Average Log Loss: {avg_log_loss:.4f} ± {np.std(log_losses):.4f}")

# Visualization per fold
metrics = ['Precision@20', 'Recall@20', 'NDCG@20', 'Hits@20', 'MAP', 'Mean Rank', 'MRR', 'Log Loss']
folds = list(range(1, 11))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    if metric == 'Precision@20':
        values = precisions_k
    elif metric == 'Recall@20':
        values = recalls_k
    elif metric == 'NDCG@20':
        values = ndcgs_k
    elif metric == 'Hits@20':
        values = hits_k
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

# Overall average horizontal bar plot for head-to-head comparison
metrics_avg = [f'Precision@{k}', f'Recall@{k}', f'NDCG@{k}', f'Hits@{k}', 'MAP', 'MRR', 'Log Loss']
averages = [avg_precision_k, avg_recall_k, avg_ndcg_k, avg_hits_k, avg_map,  avg_mrr, avg_log_loss]
stds = [np.std(precisions_k), np.std(recalls_k), np.std(ndcgs_k), np.std(hits_k), np.std(maps), np.std(mrrs), np.std(log_losses)]

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
epochs = list(range(1, 11))
fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # 2 rows, 5 columns for 10 folds
axes = axes.flatten()

for fold_idx in range(10):
    ax = axes[fold_idx]
    train_losses = losses_per_fold[fold_idx]
    val_losses = val_losses_per_fold[fold_idx]
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

