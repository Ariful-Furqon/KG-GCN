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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ndcg_score
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

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
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
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
precisions_k = []
recalls_k = []
f1s_k = []
ndcgs_k = []
k = 20

for fold, (train_idx, test_idx) in enumerate(skf.split(range(len(nodes)), y.numpy())):
    print(f"Fold {fold + 1}:")

    # Create model for each fold
    model = SimpleGCN(in_channels=x.shape[1], hidden_channels=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

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

    accuracies.append(accuracy)
    precisions_k.append(precision_k_val)
    recalls_k.append(recall_k_val)
    f1s_k.append(f1_k_val)
    ndcgs_k.append(ndcg_k_val)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision@{k}: {precision_k_val:.4f}")
    print(f"  Recall@{k}: {recall_k_val:.4f}")
    print(f"  F1Score@{k}: {f1_k_val:.4f}")
    print(f"  NDCG@{k}: {ndcg_k_val:.4f}")
    print()

# Average metrics across folds
avg_accuracy = np.mean(accuracies)
avg_precision_k = np.mean(precisions_k)
avg_recall_k = np.mean(recalls_k)
avg_f1_k = np.mean(f1s_k)
avg_ndcg_k = np.mean(ndcgs_k)

print("Cross-Validation Results:")
print(f"Average Accuracy: {avg_accuracy:.4f} ± {np.std(accuracies):.4f}")
print(f"Average Precision@{k}: {avg_precision_k:.4f} ± {np.std(precisions_k):.4f}")
print(f"Average Recall@{k}: {avg_recall_k:.4f} ± {np.std(recalls_k):.4f}")
print(f"Average F1Score@{k}: {avg_f1_k:.4f} ± {np.std(f1s_k):.4f}")
print(f"Average NDCG@{k}: {avg_ndcg_k:.4f} ± {np.std(ndcgs_k):.4f}")

# Visualization
metrics = ['Accuracy', f'Precision@{k}', f'Recall@{k}', f'F1Score@{k}', f'NDCG@{k}']
averages = [avg_accuracy, avg_precision_k, avg_recall_k, avg_f1_k, avg_ndcg_k]
stds = [np.std(accuracies), np.std(precisions_k), np.std(recalls_k), np.std(f1s_k), np.std(ndcgs_k)]

plt.figure(figsize=(10, 6))
plt.bar(metrics, averages, yerr=stds, capsize=5, color='skyblue')
plt.title('Cross-Validation Evaluation Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('evaluation_metrics.png')
plt.show()
