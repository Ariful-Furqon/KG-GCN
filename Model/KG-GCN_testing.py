from neo4j import GraphDatabase
import pandas as pd
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

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
node_lower_to_original = {node.lower(): node for node in nodes}
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

# Train the model on the full dataset
model = SimpleGCN(in_channels=x.shape[1], hidden_channels=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

# Training loop for 100 epochs
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Compute predictions (probabilities) for all nodes
model.eval()
with torch.no_grad():
    out = model(data)
    probs = torch.softmax(out, dim=1)[:, 1]  # Probability of being hama/penyakit (class 1)

# Create a dictionary of node labels to predicted probabilities
node_probs = {node: prob.item() for node, prob in zip(nodes, probs)}

# Function to get connected diseases/pests for a given symptom
def get_connected_diseases(symptom):
    # Find rows where source_label is the symptom and target_class contains 'PenyakitPadi' or 'HamaPadi'
    connected = Rice_KG_df[(Rice_KG_df['source_label'] == symptom) & 
                           (Rice_KG_df['target_class'].apply(lambda x: any(cls in ['PenyakitPadi', 'HamaPadi'] for cls in x)))]
    diseases = connected['target_label'].tolist()
    # Rank by predicted probability (higher prob first)
    ranked = sorted([(d, node_probs.get(d, 0)) for d in diseases], key=lambda x: x[1], reverse=True)
    return ranked

# Command prompt interface
print("\nModel training completed. Enter symptoms (comma-separated) to get predictions. Type 'exit' to quit.")
while True:
    user_input = input("Enter symptoms: ").strip()
    if user_input.lower() == 'exit':
        break
    symptoms = [s.strip() for s in user_input.split(',') if s.strip()]
    if not symptoms:
        print("No symptoms entered. Try again.")
        continue

    # Aggregate probabilities across all symptoms
    aggregated_scores = {}
    for symptom in symptoms:
        lower_symptom = symptom.lower()
        if lower_symptom not in node_lower_to_original:
            print(f"Symptom '{symptom}' not found in the graph.")
            continue
        original_symptom = node_lower_to_original[lower_symptom]
        results = get_connected_diseases(original_symptom)
        for disease, prob in results:
            if disease in aggregated_scores:
                aggregated_scores[disease] += prob
            else:
                aggregated_scores[disease] = prob

    if aggregated_scores:
        # Normalize scores to [0, 1] by dividing by the maximum score
        max_score = max(aggregated_scores.values())
        if max_score > 0:
            normalized_scores = {k: v / max_score for k, v in aggregated_scores.items()}
        else:
            normalized_scores = aggregated_scores
        # Sort diseases by normalized score in descending order
        sorted_diseases = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        print(f"\nFinal Diagnosis based on symptoms {', '.join(symptoms)}:")
        for disease, total_score in sorted_diseases:
            print(f"  {disease}: {total_score:.4f}")
    else:
        print("No connected diseases/pests found for the entered symptoms.")
    print()

