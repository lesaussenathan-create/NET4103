import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


TARGET_FILES = [
    "Caltech36.gml",
    "MIT8.gml",
    "Johns Hopkins55.gml",
]


def compute_stats(filepath: Path) -> dict:
    """Charge un graphe GML et calcule les métriques sur la LCC."""
    G = nx.read_gml(str(filepath))
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()

    degrees = [d for _, d in G_lcc.degree()]
    clustering = nx.clustering(G_lcc)

    return {
        "name": filepath.stem,
        "n": G_lcc.number_of_nodes(),
        "m": G_lcc.number_of_edges(),
        "density": nx.density(G_lcc),
        "global_cc": nx.transitivity(G_lcc),
        "avg_local_cc": nx.average_clustering(G_lcc),
        "degrees": degrees,
        "clustering": clustering,
    }


def plot_network(stats: dict, out_dir: Path) -> None:
    """Génère et sauvegarde les deux graphiques pour un réseau."""
    name = stats["name"]
    degrees = stats["degrees"]
    clustering = stats["clustering"]

    deg_list = list(degrees)
    cc_list = [clustering[n] for n in clustering]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(deg_list, bins=50, log=True, color="steelblue", edgecolor="black")
    ax1.set_title(f"Distribution des degrés — {name}")
    ax1.set_xlabel("Degré")
    ax1.set_ylabel("Fréquence (log)")

    ax2.scatter(deg_list, cc_list, alpha=0.3, s=10, color="coral")
    ax2.set_xscale("log")
    ax2.set_title(f"Degré vs Clustering local — {name}")
    ax2.set_xlabel("Degré (log)")
    ax2.set_ylabel("Coefficient de clustering local")

    fig.tight_layout()
    out_path = out_dir / f"{name}_analysis.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Figure sauvegardée : {out_path}")


def print_stats(stats: dict) -> None:
    print(f"\n{'='*50}")
    print(f"Réseau : {stats['name']}")
    print(f"  Noeuds (LCC)            : {stats['n']}")
    print(f"  Arêtes (LCC)            : {stats['m']}")
    print(f"  Densité                 : {stats['density']:.6f}")
    print(f"  Clustering global       : {stats['global_cc']:.4f}")
    print(f"  Clustering local moyen  : {stats['avg_local_cc']:.4f}")
    sparse = "oui" if stats["density"] < 0.01 else "non"
    print(f"  Réseau sparse ?         : {sparse}")


def main():
    parser = argparse.ArgumentParser(description="Analyse de réseaux FB100 — Q2")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dossier contenant les fichiers .gml (défaut : ./data)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output"),
        help="Dossier de sortie pour les figures (défaut : ./output)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Nombre de processus parallèles (défaut : 3)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    filepaths = []
    for filename in TARGET_FILES:
        fp = args.data_dir / filename
        if fp.exists():
            filepaths.append(fp)
        else:
            print(f"[WARN] Fichier introuvable, ignoré : {fp}", file=sys.stderr)

    if not filepaths:
        print("[ERROR] Aucun fichier trouvé. Vérifiez --data-dir.", file=sys.stderr)
        sys.exit(1)

    print(f"Traitement de {len(filepaths)} réseau(x) avec {args.workers} worker(s)...")

    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(compute_stats, fp): fp for fp in filepaths}
        for future in as_completed(futures):
            fp = futures[future]
            try:
                stats = future.result()
                results[fp.stem] = stats
                print(f"  [OK] {fp.stem} traité.")
            except Exception as exc:
                print(f"  [ERROR] {fp.stem} : {exc}", file=sys.stderr)

    for stem in [Path(f).stem for f in TARGET_FILES]:
        if stem in results:
            print_stats(results[stem])
            plot_network(results[stem], args.out_dir)


if __name__ == "__main__":
    main()