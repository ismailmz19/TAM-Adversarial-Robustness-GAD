"""
Relation Camouflage Attack + Jaccard Preprocessing Defense on TAM
"""
import os, sys, argparse, time
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TAM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TAM_DIR)

from model import Model
from utils import load_mat, preprocess_features, normalize_adj_tensor, \
                  calc_distance, graph_nsgt, normalize_score

def relation_camouflage_attack(adj_dense_np, features_np, labels, k):
    if k == 0:
        return adj_dense_np.copy(), 0
    adj_attacked = adj_dense_np.copy()
    anomaly_idx = np.where(labels == 1)[0]
    normal_idx  = np.where(labels == 0)[0]
    feat_norm = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8)
    normal_feats = feat_norm[normal_idx]
    n_added = 0
    for a in anomaly_idx:
        sim = np.array(feat_norm[a] @ normal_feats.T).flatten()
        existing = set(np.where(adj_attacked[a] > 0)[0])
        candidates = [(sim[i], normal_idx[i]) for i in range(len(normal_idx))
                      if normal_idx[i] not in existing]
        candidates.sort(reverse=True)
        for _, n in candidates[:k]:
            adj_attacked[a, n] = 1
            adj_attacked[n, a] = 1
            n_added += 1
    return adj_attacked, n_added

def jaccard_defense(adj_dense_np, tau):
    if tau == 0.0:
        return adj_dense_np.copy(), 0
    adj_bin = (adj_dense_np > 0).astype(float)
    np.fill_diagonal(adj_bin, 0)
    adj_cleaned = adj_dense_np.copy()
    rows, cols = np.where(np.triu(adj_bin, k=1) > 0)
    n_removed = 0
    for u, v in zip(rows, cols):
        neighbors_u = set(np.where(adj_bin[u] > 0)[0])
        neighbors_v = set(np.where(adj_bin[v] > 0)[0])
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        jaccard = intersection / union if union > 0 else 0.0
        if jaccard < tau:
            adj_cleaned[u, v] = 0
            adj_cleaned[v, u] = 0
            n_removed += 1
    return adj_cleaned, n_removed

def run_tam(features_t, raw_adj_t, raw_features_t, nb_nodes, ft_size, device,
            embedding_dim=128, num_epoch=100, lr=1e-5,
            cutting=18, N_tree=3, negsamp_ratio=2, readout='avg'):

    def inference(feature, adj_matrix):
        feature = feature / torch.norm(feature, dim=-1, keepdim=True)
        sim_matrix = torch.mm(feature, feature.T) * adj_matrix
        row_sum = torch.sum(adj_matrix, 0)
        r_inv = torch.pow(row_sum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        return torch.sum(sim_matrix, 1) * r_inv

    def max_message(feature, adj_matrix):
        msg = inference(feature, adj_matrix)
        return -torch.sum(msg), msg

    model_list, optimiser_list = [], []
    for _ in range(cutting * N_tree):
        m = Model(ft_size, embedding_dim, 'prelu', negsamp_ratio, readout).to(device)
        o = torch.optim.Adam(m.parameters(), lr=lr)
        model_list.append(m)
        optimiser_list.append(o)

    all_cut_adj = torch.cat([raw_adj_t.clone() for _ in range(N_tree)])
    dis_array = calc_distance(raw_adj_t[0], raw_features_t[0])
    index = 0
    all_scores = []

    for n_cut in range(cutting):
        print(f"        cutting {n_cut+1}/{cutting}", end='\r')
        for n_t in range(N_tree):
            cut_adj = graph_nsgt(dis_array, all_cut_adj[n_t]).unsqueeze(0)
            adj_norm = normalize_adj_tensor(cut_adj)
            model_list[index].train()
            for epoch in range(num_epoch):
                optimiser_list[index].zero_grad()
                node_emb, feat1, feat2 = model_list[index].forward(features_t, adj_norm)
                loss, _ = max_message(node_emb[0], raw_adj_t[0])
                loss.backward()
                optimiser_list[index].step()
            model_list[index].eval()
            with torch.no_grad():
                node_emb, _, _ = model_list[index].forward(features_t, adj_norm)
                scores = inference(node_emb[0], raw_adj_t[0])
                scores = 1 - normalize_score(np.array(scores.cpu().detach()))
                all_scores.append(scores)
            all_cut_adj[n_t] = torch.squeeze(cut_adj)
            index += 1

    print()
    return np.mean(all_scores, axis=0)

def run_experiment(dataset, k_values, tau_values,
                   num_epoch=100, cutting=18, N_tree=3,
                   embedding_dim=128, lr=1e-5):

    print(f"\n{'='*65}")
    print(f"  Dataset: {dataset}")
    print(f"  Attack k: {k_values}  |  Defense tau: {tau_values}")
    print(f"{'='*65}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(dataset)
    labels = ano_label

    if dataset in ['Amazon', 'YelpChi', 'Amazon-all', 'YelpChi-all']:
        features, _ = preprocess_features(features)
        raw_features_np = features
    else:
        raw_features_np = np.array(features.todense())

    nb_nodes = raw_features_np.shape[0]
    ft_size  = raw_features_np.shape[1]
    raw_adj_np = (adj + sp.eye(adj.shape[0])).toarray()

    print(f"  Nodes: {nb_nodes}, Features: {ft_size}")
    print(f"  Anomalies: {int(labels.sum())} ({100*labels.mean():.1f}%)")

    results = {}

    for k in k_values:
        print(f"\n  -- Attack k={k} --")
        results[k] = {}
        adj_attacked_np, n_added = relation_camouflage_attack(
            raw_adj_np, raw_features_np, labels, k)
        if k > 0:
            print(f"     Added {n_added} camouflage edges")

        for tau in tau_values:
            print(f"\n     Defense tau={tau}")
            t0 = time.time()
            adj_defended_np, n_removed = jaccard_defense(adj_attacked_np, tau)
            if tau > 0:
                print(f"       Removed {n_removed} edges (Jaccard < {tau})")
            raw_features_t = torch.FloatTensor(raw_features_np[np.newaxis]).to(device)
            features_t     = torch.FloatTensor(raw_features_np[np.newaxis]).to(device)
            raw_adj_t      = torch.FloatTensor(adj_defended_np[np.newaxis]).to(device)
            scores = run_tam(features_t, raw_adj_t, raw_features_t,
                             nb_nodes, ft_size, device,
                             embedding_dim=embedding_dim, num_epoch=num_epoch,
                             lr=lr, cutting=cutting, N_tree=N_tree)
            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
            elapsed = time.time() - t0
            results[k][tau] = {'auroc': auroc, 'auprc': auprc,
                               'n_edges_added': n_added,
                               'n_edges_removed': n_removed,
                               'elapsed': elapsed}
            print(f"       AUROC={auroc:.4f}  AUPRC={auprc:.4f}  ({elapsed:.0f}s)")

    print(f"\n\n{'='*75}")
    print(f"  RESULTS: {dataset}")
    print(f"{'='*75}")
    header = f"  {'k':>3}  {'Edges+':>7}  {'No Defense':>12}"
    for tau in tau_values[1:]:
        header += f"  {'tau='+str(tau):>14}"
    print(header)
    print(f"  {'-'*72}")
    clean_auroc = results[k_values[0]][0.0]['auroc']
    for k in k_values:
        n_added = results[k][0.0]['n_edges_added']
        base_auroc = results[k][0.0]['auroc']
        row = f"  {k:>3}  {n_added:>7}  {base_auroc:>12.4f}"
        for tau in tau_values[1:]:
            a = results[k][tau]['auroc']
            delta = a - base_auroc
            row += f"  {a:>8.4f}({delta:>+6.3f})"
        print(row)
    print(f"\n  Clean TAM (k=0, tau=0): {clean_auroc:.4f}")
    print(f"{'='*75}\n")

    np.save(f'results_camouflage_v2_{dataset}.npy',
            {'dataset': dataset, 'k_values': k_values,
             'tau_values': tau_values, 'results': results})
    print(f"  Saved to results_camouflage_v2_{dataset}.npy")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=str,   default='Amazon')
    parser.add_argument('--k_values',      type=int,   nargs='+', default=[0, 3, 5])
    parser.add_argument('--tau_values',    type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3])
    parser.add_argument('--num_epoch',     type=int,   default=100)
    parser.add_argument('--cutting',       type=int,   default=18)
    parser.add_argument('--N_tree',        type=int,   default=3)
    parser.add_argument('--embedding_dim', type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=1e-5)
    args = parser.parse_args()
    run_experiment(dataset=args.dataset, k_values=args.k_values,
                   tau_values=args.tau_values, num_epoch=args.num_epoch,
                   cutting=args.cutting, N_tree=args.N_tree,
                   embedding_dim=args.embedding_dim, lr=args.lr)
