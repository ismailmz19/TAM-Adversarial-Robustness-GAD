"""
Relation Camouflage Attack & Defense on TAM
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

def two_hop_defense(adj_np, base_scores):
    adj_bin = (adj_np > 0).astype(float)
    np.fill_diagonal(adj_bin, 0)
    adj2 = adj_bin @ adj_bin
    np.fill_diagonal(adj2, 0)
    adj2 = (adj2 > 0).astype(float)
    row_sums = adj2.sum(axis=1) + 1e-8
    hop2_mean = (adj2 @ base_scores) / row_sums
    return 0.5 * base_scores + 0.5 * hop2_mean

def run_tam(features_t, raw_adj_t, raw_features_t, nb_nodes, ft_size, device,
            embedding_dim=128, num_epoch=100, lr=1e-5,
            cutting=18, N_tree=3, negsamp_ratio=2, readout='avg', lamda=0):

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

def run_experiment(dataset, k_values, num_epoch=100, cutting=18, N_tree=3,
                   embedding_dim=128, lr=1e-5):
    print(f"\n{'='*60}\n  Dataset: {dataset}   k_values: {k_values}\n{'='*60}")
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
        print(f"\n  -- k={k} --")
        t0 = time.time()

        adj_attacked_np, n_added = relation_camouflage_attack(raw_adj_np, raw_features_np, labels, k)
        if k > 0:
            print(f"     Added {n_added} camouflage edges")

        raw_features_t = torch.FloatTensor(raw_features_np[np.newaxis]).to(device)
        features_t     = torch.FloatTensor(raw_features_np[np.newaxis]).to(device)
        raw_adj_t      = torch.FloatTensor(adj_attacked_np[np.newaxis]).to(device)

        base_scores = run_tam(features_t, raw_adj_t, raw_features_t,
                              nb_nodes, ft_size, device,
                              embedding_dim=embedding_dim, num_epoch=num_epoch,
                              lr=lr, cutting=cutting, N_tree=N_tree)

        auroc_base = roc_auc_score(labels, base_scores)
        auprc_base = average_precision_score(labels, base_scores)

        adj_for_defense = adj_attacked_np.copy()
        np.fill_diagonal(adj_for_defense, 0)
        defense_scores = two_hop_defense(adj_for_defense, base_scores)
        auroc_def = roc_auc_score(labels, defense_scores)
        auprc_def = average_precision_score(labels, defense_scores)

        elapsed = time.time() - t0
        results[k] = {'auroc_base': auroc_base, 'auprc_base': auprc_base,
                      'auroc_defense': auroc_def, 'auprc_defense': auprc_def,
                      'n_edges_added': n_added, 'elapsed': elapsed}

        print(f"     Base    AUROC={auroc_base:.4f}  AUPRC={auprc_base:.4f}")
        print(f"     Defense AUROC={auroc_def:.4f}  AUPRC={auprc_def:.4f}")
        print(f"     Time: {elapsed:.1f}s")

    print(f"\n{'='*72}\n  RESULTS: {dataset}\n{'='*72}")
    print(f"  {'k':>3}  {'Edges+':>7}  {'Base AUROC':>11}  {'Def AUROC':>10}  {'Δ Attack':>9}  {'Δ Defense':>10}")
    print(f"  {'-'*68}")
    clean_auroc = results[k_values[0]]['auroc_base']
    for k in k_values:
        r = results[k]
        print(f"  {k:>3}  {r['n_edges_added']:>7}  "
              f"{r['auroc_base']:>11.4f}  {r['auroc_defense']:>10.4f}  "
              f"{r['auroc_base']-clean_auroc:>+9.4f}  "
              f"{r['auroc_defense']-r['auroc_base']:>+10.4f}")
    print(f"{'='*72}\n")

    np.save(f'results_camouflage_{dataset}.npy',
            {'dataset': dataset, 'k_values': k_values, 'results': results})
    print(f"  Saved to results_camouflage_{dataset}.npy")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       type=str,   default='Amazon')
    parser.add_argument('--k_values',      type=int,   nargs='+', default=[0, 1, 2, 3, 5])
    parser.add_argument('--num_epoch',     type=int,   default=100)
    parser.add_argument('--cutting',       type=int,   default=18)
    parser.add_argument('--N_tree',        type=int,   default=3)
    parser.add_argument('--embedding_dim', type=int,   default=128)
    parser.add_argument('--lr',            type=float, default=1e-5)
    args = parser.parse_args()
    run_experiment(dataset=args.dataset, k_values=args.k_values,
                   num_epoch=args.num_epoch, cutting=args.cutting,
                   N_tree=args.N_tree, embedding_dim=args.embedding_dim, lr=args.lr)
