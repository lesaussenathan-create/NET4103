import os
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, mean_absolute_error, accuracy_score

warnings.filterwarnings("ignore")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # Normalisation L2 pour stabiliser l'apprentissage
        x = F.normalize(x, p=2, dim=1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def prepare_pyg_data(G, target_attr):
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)

    labels = []
    features = []
    all_attributes = ['dorm', 'major', 'gender', 'year']
    
    for node in G.nodes():
        data = G.nodes[node]
        labels.append(str(data.get(target_attr, '0')))
        
        feat = []
        for attr in all_attributes:
            if attr != target_attr:
                val = data.get(attr, 0)
                try:
                    feat.append(float(val))
                except ValueError:
                    feat.append(0.0)
        features.append(feat)

    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    y = torch.tensor(y_encoded, dtype=torch.long)
    x = torch.tensor(np.array(features), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, y=y)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Propagation de labels GCN (Question 5)")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--network", type=str, default="Duke14.gml")
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    set_seed(42)
    filepath = args.data_dir / args.network

    if not filepath.exists():
        print(f"[ERREUR] Fichier '{filepath}' introuvable.", file=sys.stderr)
        return

    print(f"Réseau     : {args.network}")
    print(f"Matériel   : {device.type.upper()} (PyTorch utilise nativement tous les cœurs disponibles)")
    
    G_original = nx.read_gml(str(filepath))
    lcc_nodes = max(nx.connected_components(G_original), key=len)
    G = G_original.subgraph(lcc_nodes).copy()
    
    attributes_to_test = ['dorm', 'major', 'gender']
    fractions_to_remove = [0.1, 0.2, 0.3]
    
    results_dict = {attr: [] for attr in attributes_to_test}

    total_steps = len(attributes_to_test) * len(fractions_to_remove) * args.epochs
    
    print("\nLancement des entraînements...")
    
    with tqdm(total=total_steps, desc="Calcul GCN en cours", unit="épq", leave=True) as pbar:
        for attr in attributes_to_test:
            pyg_data = prepare_pyg_data(G, target_attr=attr).to(device)
            num_nodes = pyg_data.y.size(0)
            num_classes = len(torch.unique(pyg_data.y))
            
            for frac in fractions_to_remove:
                indices = list(range(num_nodes))
                random.shuffle(indices)
                
                num_test = int(num_nodes * frac)
                test_idx = indices[:num_test]
                train_idx = indices[num_test:]
                
                train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                train_mask[train_idx] = True
                test_mask[test_idx] = True

                model = GCN(num_node_features=pyg_data.x.size(1), num_classes=num_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                
                model.train()
                for _ in range(args.epochs):
                    optimizer.zero_grad()
                    out = model(pyg_data)
                    loss = F.nll_loss(out[train_mask], pyg_data.y[train_mask])
                    loss.backward()
                    optimizer.step()
                    
                    pbar.update(1)

                model.eval()
                with torch.no_grad():
                    out = model(pyg_data)
                    pred = out.argmax(dim=1)
                    
                    y_true = pyg_data.y[test_mask].cpu().numpy()
                    y_pred = pred[test_mask].cpu().numpy()
                    
                    acc = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred, average='weighted')
                    mae = mean_absolute_error(y_true, y_pred)
                    
                    results_dict[attr].append((frac, acc, f1, mae))

    print("\n" + "="*60)
    print("RÉSULTATS FINAUX")
    print("="*60)
    for attr in attributes_to_test:
        print(f"\n--- Cible : {attr.upper()} ---")
        for frac, acc, f1, mae in results_dict[attr]:
            print(f"  Masqué: {frac*100:2.0f}% | Acc: {acc:.3f} | F1: {f1:.3f} | MAE: {mae:.3f}")

if __name__ == "__main__":
    main()