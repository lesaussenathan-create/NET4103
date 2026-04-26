import sys
import argparse
from pathlib import Path

import networkx as nx
import community.community_louvain as community_louvain
from sklearn.metrics import normalized_mutual_info_score as nmi

def validate_community(filepath, name):
    print(f"\n{'='*40}")
    print(f"Analyse des communautés : {name}")
    print(f"{'='*40}")
    
    if not filepath.exists():
        print(f"[ERREUR] Fichier introuvable : {filepath}", file=sys.stderr)
        return None, None

    G = nx.read_gml(str(filepath))
    
    print("Exécution de l'algorithme de Louvain...")
    partition = community_louvain.best_partition(G)
    
    true_dorms = []
    true_years = []
    pred_labels = []
    
    for node in G.nodes():
        data = G.nodes[node]
        dorm = data.get('dorm', 0)
        year = data.get('year', 0)
        
        if dorm != 0 and year != 0:
            true_dorms.append(dorm)
            true_years.append(year)
            pred_labels.append(partition[node])
            
    nmi_dorm = nmi(true_dorms, pred_labels)
    nmi_year = nmi(true_years, pred_labels)
    
    print(f"Score NMI (Résidence) : {nmi_dorm:.4f}")
    print(f"Score NMI (Année)     : {nmi_year:.4f}")
    
    return nmi_dorm, nmi_year

def main():
    parser = argparse.ArgumentParser(description="Détection de communautés - Question 6")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), 
                        help="Dossier contenant les fichiers .gml")
    args = parser.parse_args()

    networks = {
        "Caltech36.gml": "Caltech",
        "Georgetown15.gml": "Georgetown"
    }

    for filename, name in networks.items():
        filepath = args.data_dir / filename
        validate_community(filepath, name)

if __name__ == "__main__":
    main()