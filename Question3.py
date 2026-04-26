import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import progressbar

warnings.filterwarnings('ignore', category=RuntimeWarning)

def process_single_graph(filepath):
    """Calcule l'assortativité pour un graphe."""
    G = nx.read_gml(str(filepath))
    
    res = {
        'size': len(G), 
        'student_fac': 0.0, 
        'major_index': 0.0, 
        'dorm': 0.0, 
        'gender': 0.0, 
        'degree': 0.0
    }
    
    for attr in ['student_fac', 'major_index', 'dorm', 'gender']:
        try:
            r = nx.attribute_assortativity_coefficient(G, attr)
            if not np.isnan(r):
                res[attr] = r
        except Exception:
            pass
            
    try:
        r_deg = nx.degree_assortativity_coefficient(G)
        if not np.isnan(r_deg):
            res['degree'] = r_deg
    except Exception:
        pass
        
    return res


def main():
    parser = argparse.ArgumentParser(description="Analyse de l'assortativité - Question 3")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dossier contenant les fichiers .gml")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"[ERREUR] Le dossier '{args.data_dir}' n'existe pas.", file=sys.stderr)
        sys.exit(1)

    filepaths = list(args.data_dir.glob("*.gml"))
    if not filepaths:
        print(f"[ERREUR] Aucun fichier .gml trouvé dans '{args.data_dir}'.", file=sys.stderr)
        sys.exit(1)

    assort_results = {'student_fac': [], 'major_index': [], 'dorm': [], 'gender': [], 'degree': []}
    network_sizes = []

    print(f"Traitement de {len(filepaths)} graphes en parallèle...")

    # Configuration de la barre de progression globale
    widgets = ['Progression: ', progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA()]
    bar = progressbar.ProgressBar(maxval=len(filepaths), widgets=widgets).start()

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_graph, fp) for fp in filepaths]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                res = future.result()
                network_sizes.append(res['size'])
                for attr in assort_results.keys():
                    assort_results[attr].append(res[attr])
            except Exception as e:
                # On utilise la méthode de contournement pour ne pas casser l'affichage de la barre
                sys.stdout.write(f"\n[ERREUR] Échec sur un fichier : {e}\n")
                sys.stdout.flush()
            
            # Mise à jour de la barre
            bar.update(i + 1)
            
    bar.finish()

    plot_attributes = ['student_fac', 'major_index', 'degree', 'dorm', 'gender']
    titles = ['Status (Student/Fac)', 'Major', 'Vertex Degree', 'Dorm', 'Gender']
    
    fig, axes = plt.subplots(5, 2, figsize=(12, 18))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    for i, attr in enumerate(plot_attributes):
        ax_scatter = axes[i, 0]
        ax_scatter.scatter(network_sizes, assort_results[attr], alpha=0.7, edgecolors='w')
        ax_scatter.axhline(0, color='black', linestyle='--')
        ax_scatter.set_xscale('log')
        ax_scatter.set_title(f'{titles[i]} Assortativity')
        ax_scatter.set_xlabel('Network size n (log scale)')
        ax_scatter.set_ylabel('Assortativity')
        
        ax_hist = axes[i, 1]
        ax_hist.hist(assort_results[attr], bins=20, color='skyblue', edgecolor='black', density=True)
        ax_hist.axvline(0, color='black', linestyle='--')
        ax_hist.set_title(f'Distribution of {titles[i]}')
        ax_hist.set_xlabel('Assortativity')
        ax_hist.set_ylabel('Density')

    plt.suptitle("Assortativity Analysis on 100 Facebook Networks", fontsize=16)
    plt.show()


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()