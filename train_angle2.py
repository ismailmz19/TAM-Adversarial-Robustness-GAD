# -*- coding: utf-8 -*-
# TAM Extension: Angle 2 (Neighbor-Trust Weighted Affinity)
# Core idea: weight each neighbor's contribution by its own affinity score.
# Neighbors that look "normal" (high affinity) get more trust.
# This prevents anomalous neighbors from poisoning normal nodes' scores.
import torch.nn as nn
from model import Model
from utils import *
from sklearn.metrics import average_precision_score, roc_auc_score
import os
import argparse
from tqdm import tqdm
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Amazon')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--subgraph_size', type=int, default=15)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--margin', type=int, default=2)
parser.add_argument('--negsamp_ratio', type=int, default=2)
parser.add_argument('--cutting', type=int, default=4)
parser.add_argument('--N_tree', type=int, default=3)
parser.add_argument('--lamda', type=int, default=0)
parser.add_argument('--trust_iters', type=int, default=2,
                    help='Number of neighbor-trust refinement iterations')
args = parser.parse_args()

args.lr = 1e-5
args.num_epoch = 500

LAMBDA_MAP = {'BlogCatalog': 1, 'ACM': 1, 'Amazon': 0, 'Facebook': 0, 'Reddit': 0, 'YelpChi': 0}
if args.dataset in LAMBDA_MAP:
    args.lamda = LAMBDA_MAP[args.dataset]

if args.dataset in ['ACM', 'YelpChi']:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

print(f'\n{"="*60}')
print(f'  Dataset     : {args.dataset}')
print(f'  Trust iters : {args.trust_iters}')
print(f'{"="*60}\n')

os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

if args.dataset in ['Amazon', 'YelpChi', 'Amazon-all', 'YelpChi-all']:
    features, _ = preprocess_features(features)
    raw_features = features
else:
    raw_features = features.todense()
    features = raw_features

dgl_graph = adj_to_dgl_graph(adj)
nb_nodes = features.shape[0]
ft_size = features.shape[1]

raw_adj = adj
raw_adj = (raw_adj + sp.eye(adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()

raw_features = torch.FloatTensor(raw_features[np.newaxis])
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])

optimiser_list = []
model_list = []
for i in range(args.cutting * args.N_tree):
    model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if torch.cuda.is_available():
        model = model.cuda()
    optimiser_list.append(optimiser)
    model_list.append(model)

if torch.cuda.is_available():
    print('Using CUDA')
    features = features.cuda()
    raw_features = raw_features.cuda()
    adj = adj.cuda()
    raw_adj = raw_adj.cuda()


def reg_edge(emb, adj):
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    sim_u_u = torch.mm(emb, emb.T) * (1 - adj)
    row_sum = torch.sum(1 - adj, 1)
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    return torch.sum(torch.sum(sim_u_u, 1) * r_inv)


def inference_original(feature, adj_matrix):
    """Original TAM: uniform neighbor averaging."""
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.squeeze(torch.mm(feature, feature.T)) * adj_matrix
    r_inv = torch.pow(torch.sum(adj_matrix, 0), -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    return torch.sum(sim_matrix, 1) * r_inv


def inference_trust(feature, adj_matrix, n_iters=2):
    """
    Angle 2: Neighbor-Trust Weighted Affinity.

    Iteration 0 (bootstrap):
        h0(vi) = (1/|N(vi)|) * sum_j sim(xi, xj)   [same as original TAM]

    Iteration t > 0 (refinement):
        trust_weight(vi->vj) = h_{t-1}(vj) * adj(vi,vj)
        h_t(vi) = sum_j trust_weight(vi->vj) * sim(xi,xj)
                  / sum_j trust_weight(vi->vj)

    Intuition: neighbors with high affinity (likely normal) get more trust.
    Anomalous neighbors (low affinity) get down-weighted automatically.
    No labels needed — the signal is self-supervised.
    """
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.squeeze(torch.mm(feature, feature.T)) * adj_matrix

    # Bootstrap: standard uniform average (h0)
    r_inv = torch.pow(torch.sum(adj_matrix, 0), -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    h = torch.sum(sim_matrix, 1) * r_inv  # shape: [N]

    for _ in range(n_iters):
        # Trust weight matrix: W[i,j] = h[j] * adj[i,j]
        # h[j] acts as neighbor j's trustworthiness
        trust = h.unsqueeze(0) * adj_matrix  # [N, N]
        trust_sum = torch.sum(trust, 1, keepdim=True)  # [N, 1]
        trust_sum[trust_sum == 0] = 1.0  # avoid div by zero
        trust_norm = trust / trust_sum  # row-normalized trust weights
        # Weighted affinity score
        h = torch.sum(trust_norm * sim_matrix, 1)

    return h


def max_message_trust(feature, adj_matrix, n_iters=2):
    """Training loss using trust-weighted affinity."""
    h = inference_trust(feature, adj_matrix, n_iters)
    return -torch.sum(h), h


start = time.time()

with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    all_cut_adj = torch.cat([raw_adj for _ in range(args.N_tree)])
    print('<<<<<<Start to calculate distance<<<<<')
    dis_array = calc_distance(raw_adj[0, :, :], raw_features[0, :, :])
    index = 0

    # Store scores from both methods
    message_list_orig = []    # original TAM scores per model
    message_list_trust = []   # Angle 2 scores per model

    for n_cut in range(args.cutting):
        print(f'\n--- Truncation depth k={n_cut+1}/{args.cutting} ---')
        batch_orig = []
        batch_trust = []

        for n_t in range(args.N_tree):
            cut_adj = graph_nsgt(dis_array, all_cut_adj[n_t, :, :]).unsqueeze(0)
            optimiser_list[index].zero_grad()
            model_list[index].train()
            adj_norm = normalize_adj_tensor(cut_adj)

            for epoch in range(args.num_epoch):
                node_emb, feat1, feat2 = model_list[index].forward(features, adj_norm)
                # Train with trust-weighted loss (Angle 2)
                loss, _ = max_message_trust(node_emb[0, :, :], raw_adj[0, :, :], args.trust_iters)
                loss = loss + args.lamda * reg_edge(feat1[0, :, :], raw_adj[0, :, :])
                loss.backward()
                optimiser_list[index].step()
                if epoch % 100 == 0:
                    print(f"  [k={n_cut+1}, t={n_t+1}] epoch {epoch:3d}  loss={loss.detach().cpu().numpy():.4f}")

            # Compute both original and trust scores at inference
            model_list[index].eval()
            with torch.no_grad():
                node_emb, _, _ = model_list[index].forward(features, adj_norm)
                score_orig = inference_original(node_emb[0, :, :], raw_adj[0, :, :])
                score_trust = inference_trust(node_emb[0, :, :], raw_adj[0, :, :], args.trust_iters)

            batch_orig.append(score_orig)
            batch_trust.append(score_trust)
            all_cut_adj[n_t, :, :] = torch.squeeze(cut_adj)
            index += 1

        # Per-model AUC logging
        for i, (so, st) in enumerate(zip(batch_orig, batch_trust)):
            so_np = 1 - normalize_score(np.array(so.cpu().detach()))
            st_np = 1 - normalize_score(np.array(st.cpu().detach()))
            auc_o = roc_auc_score(ano_label, so_np)
            auc_t = roc_auc_score(ano_label, st_np)
            print(f'  Model [k={n_cut+1}, t={i+1}]  AUC_orig={auc_o:.4f}  AUC_trust={auc_t:.4f}  delta={auc_t-auc_o:+.4f}')

        message_list_orig.append(torch.mean(torch.stack(batch_orig), 0))
        message_list_trust.append(torch.mean(torch.stack(batch_trust), 0))

end = time.time()

# Final aggregation
score_final_orig = 1 - normalize_score(np.array(torch.mean(torch.stack(message_list_orig), 0).cpu().detach()))
score_final_trust = 1 - normalize_score(np.array(torch.mean(torch.stack(message_list_trust), 0).cpu().detach()))

auc_orig = roc_auc_score(ano_label, score_final_orig)
ap_orig  = average_precision_score(ano_label, score_final_orig, average='macro', pos_label=1)
auc_trust = roc_auc_score(ano_label, score_final_trust)
ap_trust  = average_precision_score(ano_label, score_final_trust, average='macro', pos_label=1)

print(f'\n{"="*60}')
print(f'  FINAL RESULTS --- {args.dataset}')
print(f'{"="*60}')
print(f'  Original TAM   AUROC: {auc_orig:.4f}  AUPRC: {ap_orig:.4f}')
print(f'  Angle 2 Trust  AUROC: {auc_trust:.4f}  AUPRC: {ap_trust:.4f}  ({auc_trust-auc_orig:+.4f})')
print(f'  Runtime: {end - start:.1f}s')
print(f'{"="*60}\n')

np.save(f'results_angle2_{args.dataset}.npy', {
    'dataset': args.dataset, 'ano_label': ano_label,
    'score_original': score_final_orig,
    'score_trust': score_final_trust,
    'auc_original': auc_orig, 'ap_original': ap_orig,
    'auc_trust': auc_trust,   'ap_trust': ap_trust,
    'trust_iters': args.trust_iters,
})
print(f'  Saved results_angle2_{args.dataset}.npy')
