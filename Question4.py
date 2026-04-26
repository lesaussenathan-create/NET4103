import sys
import math
import random
import argparse
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

import numpy as np
import networkx as nx
import progressbar

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class LinkPrediction(ABC):
    def __init__(self, graph):
        self.graph = graph
        self.N = len(graph)

    def neighbors(self, v):
        return list(self.graph.neighbors(v))

    @abstractmethod
    def fit(self, u, v):
        pass


class CommonNeighbors(LinkPrediction):
    def fit(self, u, v):
        return len(set(self.neighbors(u)) & set(self.neighbors(v)))


class Jaccard(LinkPrediction):
    def fit(self, u, v):
        nu, nv = set(self.neighbors(u)), set(self.neighbors(v))
        union = len(nu | nv)
        return len(nu & nv) / union if union else 0.0


class AdamicAdar(LinkPrediction):
    def fit(self, u, v):
        common = set(self.neighbors(u)) & set(self.neighbors(v))
        return sum(
            1.0 / math.log(self.graph.degree(w))
            for w in common
            if self.graph.degree(w) > 1
        )


# ────────────────────
# GNN — Question 4.e
# ────────────────────

def build_gnn_predictor(graph, hidden=32, out=16, epochs=100, lr=0.01):
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import negative_sampling

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_to_idx = {n: i for i, n in enumerate(graph.nodes())}
    N = len(node_to_idx)

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = torch.cat([ei, ei[[1, 0]]], dim=1).to(device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

    feats = []
    for node in graph.nodes():
        d = graph.nodes[node]
        feats.append([
            float(d.get("dorm", 0)),
            float(d.get("major", 0)),
            float(d.get("year", 0)),
            float(d.get("gender", 0)),
        ])
    x = torch.tensor(np.array(feats), dtype=torch.float).to(device)

    class GCNEncoder(torch.nn.Module):
        def __init__(self, in_ch, hid, out_ch):
            super().__init__()
            self.conv1 = GCNConv(in_ch, hid)
            self.conv2 = GCNConv(hid, out_ch)

        def forward(self, x, ei):
            x = self.conv1(x, ei).relu()
            x = F.dropout(x, p=0.5, training=self.training)
            return self.conv2(x, ei)

    model = GCNEncoder(x.shape[1], hidden, out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    
    widgets = [f'  [GNN] Entraînement ({device}) ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(maxval=epochs, widgets=widgets).start()
    
    for i in range(epochs):
        opt.zero_grad()
        z = model(x, edge_index)
        neg_ei = negative_sampling(edge_index, num_nodes=N, num_neg_samples=edge_index.size(1))
        
        pos = (z[edge_index[0]] * z[edge_index[1]]).sum(1)
        neg = (z[neg_ei[0]] * z[neg_ei[1]]).sum(1)
        
        loss = (-torch.log(torch.sigmoid(pos) + 1e-15).mean()
                - torch.log(1 - torch.sigmoid(neg) + 1e-15).mean())
        loss.backward()
        opt.step()
        bar.update(i + 1)
        
    bar.finish()

    model.eval()
    with torch.no_grad():
        emb = model(x, edge_index).cpu()

    def score_fn(u, v):
        if u not in node_to_idx or v not in node_to_idx:
            return 0.0
        eu, ev = emb[node_to_idx[u]], emb[node_to_idx[v]]
        return torch.sigmoid(torch.dot(eu, ev)).item()

    return score_fn


def _score_chunk(args):
    chunk, neighbors_map, metric_name, degrees_map = args
    results = []
    
    for u, v in chunk:
        nu = neighbors_map.get(u, set())
        nv = neighbors_map.get(v, set())
        common = nu & nv
        
        if metric_name == "CommonNeighbors":
            s = len(common)
        elif metric_name == "Jaccard":
            union = len(nu | nv)
            s = len(common) / union if union else 0.0
        elif metric_name == "AdamicAdar":
            s = sum(
                1.0 / math.log(degrees_map[w])
                for w in common
                if degrees_map.get(w, 0) > 1
            )
        else:
            s = 0.0
            
        if s > 0:
            results.append((u, v, s))
            
    return results


def _compute_metrics(predictions, removed_set, removed_len):
    k_values = [50, 100, 200, 300, 400]
    out = {}
    for k in k_values:
        top_k = predictions[:k]
        tp = sum(1 for u, v, _ in top_k if (u, v) in removed_set)
        out[k] = {
            "precision": tp / k if k else 0,
            "recall": tp / removed_len if removed_len else 0,
        }
    return out


def evaluate_topo(G_original, metric_name, fraction, workers=4, chunk_size=5000):
    G = G_original.copy()
    all_edges = list(G.edges())
    n_remove = int(len(all_edges) * fraction)
    
    removed = random.sample(all_edges, n_remove)
    G.remove_edges_from(removed)
    removed_set = {(u, v) for u, v in removed} | {(v, u) for u, v in removed}

    neighbors_map = {n: set(G.neighbors(n)) for n in G.nodes()}
    degrees_map = {n: G.degree(n) for n in G.nodes()} if metric_name == "AdamicAdar" else {}
    
    non_edges = list(nx.non_edges(G))
    chunks = [
        (non_edges[i:i + chunk_size], neighbors_map, metric_name, degrees_map)
        for i in range(0, len(non_edges), chunk_size)
    ]

    predictions = []
    
    widgets = [f'    {metric_name:18s} ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(maxval=len(chunks), widgets=widgets).start()
    
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score_chunk, chunk) for chunk in chunks]
        for i, future in enumerate(as_completed(futures)):
            predictions.extend(future.result())
            bar.update(i + 1)
            
    bar.finish()

    predictions.sort(key=lambda x: x[2], reverse=True)
    return _compute_metrics(predictions, removed_set, len(removed))


def evaluate_gnn(G_original, fraction, **gnn_kwargs):
    G = G_original.copy()
    all_edges = list(G.edges())
    n_remove = int(len(all_edges) * fraction)
    removed = random.sample(all_edges, n_remove)
    G.remove_edges_from(removed)
    removed_set = {(u, v) for u, v in removed} | {(v, u) for u, v in removed}

    score_fn = build_gnn_predictor(G, **gnn_kwargs)
    non_edges = list(nx.non_edges(G))
    predictions = []
    
    widgets = ['    [GNN] Évaluation ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(maxval=len(non_edges), widgets=widgets).start()
    
    for i, (u, v) in enumerate(non_edges):
        s = score_fn(u, v)
        if s > 0:
            predictions.append((u, v, s))
        bar.update(i + 1)
        
    bar.finish()

    predictions.sort(key=lambda x: x[2], reverse=True)
    return _compute_metrics(predictions, removed_set, len(removed))


SMALL_GRAPHS = [
    "Caltech36.gml", "Reed98.gml", "Haverford76.gml", "Simmons81.gml",
    "Swarthmore42.gml", "Amherst41.gml", "Bowdoin47.gml", "Hamilton46.gml",
    "Trinity100.gml", "USFCA72.gml", "Williams40.gml", "Oberlin44.gml",
]

TOPO_METRICS = ["CommonNeighbors", "Jaccard", "AdamicAdar"]


def load_lcc(filepath: Path) -> nx.Graph:
    G = nx.read_gml(str(filepath))
    lcc = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc).copy()


def print_summary(name, metric, results):
    row = " | ".join(
        f"P@{k}={results[k]['precision']:.4f} R@{k}={results[k]['recall']:.4f}"
        for k in sorted(results)
    )
    print(f"  [{name}] {metric:18s} -> {row}")


def main():
    parser = argparse.ArgumentParser(description="Évaluation de la prédiction de liens")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fraction", type=float, default=0.1, choices=[0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-gnn", action="store_true")
    parser.add_argument("--gnn-epochs", type=int, default=100)
    args = parser.parse_args()

    filepaths = [args.data_dir / f for f in SMALL_GRAPHS if (args.data_dir / f).exists()]
    missing = [f for f in SMALL_GRAPHS if not (args.data_dir / f).exists()]
    
    if missing:
        print(f"[ATTENTION] Fichiers manquants: {missing}", file=sys.stderr)
    if not filepaths:
        print("[ERREUR] Aucun fichier trouvé. Vérifiez le dossier de données.", file=sys.stderr)
        sys.exit(1)

    print(f"Configuration : f={args.fraction} | {len(filepaths)} graphes | {args.workers} processus CPU\n")

    all_results = {}

    for fp in filepaths:
        graph_name = fp.stem
        print(f"\nTraitement du graphe : {graph_name}")
        G = load_lcc(fp)
        
        all_results[graph_name] = {}
        for metric in TOPO_METRICS:
            res = evaluate_topo(G, metric, args.fraction, workers=args.workers)
            all_results[graph_name][metric] = res

    print("\n" + "=" * 80)
    print(f"RÉSUMÉ -- Métriques Topologiques (f={args.fraction})")
    print("=" * 80)
    for name, res_by_metric in all_results.items():
        for metric, results in res_by_metric.items():
            print_summary(name, metric, results)

    print("\n" + "=" * 80)
    print("MOYENNE Precision@50 sur tous les graphes")
    print("=" * 80)
    for metric in TOPO_METRICS:
        scores = [
            all_results[n][metric][50]["precision"]
            for n in all_results if metric in all_results[n]
        ]
        if scores:
            print(f"  {metric:18s} : {np.mean(scores):.4f} (+/- {np.std(scores):.4f}) sur {len(scores)} graphes")

    if not args.skip_gnn:
        print("\n" + "=" * 80)
        print(f"ÉVALUATION GNN (f={args.fraction})")
        print("=" * 80)
        for fp in filepaths[:3]: 
            print(f"\nGNN sur le graphe : {fp.stem}")
            G = load_lcc(fp)
            res = evaluate_gnn(G, args.fraction, epochs=args.gnn_epochs)
            print_summary(fp.stem, "GNN", res)


if __name__ == "__main__":
    freeze_support()
    main()