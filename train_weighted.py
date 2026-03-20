# -*- coding: utf-8 -*-
import torch.nn as nn
from model import Model
from utils import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import os
import argparse
from tqdm import tqdm
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, [0]))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Amazon")
parser.add_argument("--lr", type=float)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--num_epoch", type=int)
parser.add_argument("--drop_prob", type=float, default=0.0)
parser.add_argument("--subgraph_size", type=int, default=15)
parser.add_argument("--readout", type=str, default="avg")
parser.add_argument("--margin", type=int, default=2)
parser.add_argument("--negsamp_ratio", type=int, default=2)
parser.add_argument("--cutting", type=int, default=4)
parser.add_argument("--N_tree", type=int, default=3)
parser.add_argument("--lamda", type=int, default=0)
parser.add_argument("--sep_percentile", type=float, default=0.1)
parser.add_argument("--sep_temp", type=float, default=1.0)
args = parser.parse_args()

args.lr = 1e-5
args.num_epoch = 500

LAMBDA_MAP = {"BlogCatalog": 1, "ACM": 1, "Amazon": 0, "Facebook": 0, "Reddit": 0, "YelpChi": 0}
if args.dataset in LAMBDA_MAP:
    args.lamda = LAMBDA_MAP[args.dataset]
if args.dataset in ["ACM", "YelpChi"]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

print("="*60)
print(f"  Dataset  : {args.dataset}")
print(f"  Sep temp : {args.sep_temp}")
print("="*60)

os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["OMP_NUM_THREADS"] = "1"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

if args.dataset in ["Amazon", "YelpChi", "Amazon-all", "YelpChi-all"]:
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
    model = Model(ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if torch.cuda.is_available():
        model = model.cuda()
    optimiser_list.append(optimiser)
    model_list.append(model)

if torch.cuda.is_available():
    print("Using CUDA")
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
    return torch.sum(sim_u_u * r_inv.unsqueeze(1))


def max_message(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.squeeze(torch.mm(feature, feature.T)) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0
    r_inv = torch.pow(torch.sum(adj_matrix, 0), -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    message = torch.sum(sim_matrix, 1) * r_inv
    return -torch.sum(message), message


def inference(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.squeeze(torch.mm(feature, feature.T)) * adj_matrix
    r_inv = torch.pow(torch.sum(adj_matrix, 0), -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    return torch.sum(sim_matrix, 1) * r_inv


def compute_separation_score(score_vec, percentile=0.10):
    n = len(score_vec)
    k = max(1, int(n * percentile))
    s = np.sort(score_vec)
    return float(s[-k:].mean() - s[:k].mean())


def weighted_aggregate(score_list, percentile=0.10, temperature=1.0):
    scores_np = [s.cpu().detach().numpy() if isinstance(s, torch.Tensor) else np.array(s) for s in score_list]
    sep_scores = np.array([compute_separation_score(sv, percentile) for sv in scores_np])
    scaled = sep_scores / temperature - sep_scores.max() / temperature
    weights = np.exp(scaled)
    weights /= weights.sum()
    stacked = torch.stack([s if isinstance(s, torch.Tensor) else torch.FloatTensor(s) for s in score_list], dim=0)
    weights_t = torch.FloatTensor(weights).unsqueeze(1).to(stacked.device)
    return (stacked * weights_t).sum(dim=0), weights, sep_scores


start = time.time()

with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description("Training")
    all_cut_adj = torch.cat([raw_adj for _ in range(args.N_tree)])
    print("<<<<<<Start to calculate distance<<<<<")
    dis_array = calc_distance(raw_adj[0, :, :], raw_features[0, :, :])
    index = 0
    message_mean_list = []
    message_mean_list_weighted = []

    for n_cut in range(args.cutting):
        print(f"\n--- Truncation depth k={n_cut+1}/{args.cutting} ---")
        message_list = []

        for n_t in range(args.N_tree):
            cut_adj = graph_nsgt(dis_array, all_cut_adj[n_t, :, :]).unsqueeze(0)
            optimiser_list[index].zero_grad()
            model_list[index].train()
            adj_norm = normalize_adj_tensor(cut_adj)

            for epoch in range(args.num_epoch):
                node_emb, feat1, feat2 = model_list[index].forward(features, adj_norm)
                loss, _ = max_message(node_emb[0, :, :], raw_adj[0, :, :])
                message_sum = inference(node_emb[0, :, :], raw_adj[0, :, :])
                loss = loss + args.lamda * reg_edge(feat1[0, :, :], raw_adj[0, :, :])
                loss.backward()
                optimiser_list[index].step()
                if epoch % 100 == 0:
                    print(f"  [k={n_cut+1}, t={n_t+1}] epoch {epoch:3d}  loss={loss.detach().cpu().numpy():.4f}")

            message_list.append(torch.unsqueeze(message_sum, 0))
            all_cut_adj[n_t, :, :] = torch.squeeze(cut_adj)
            index += 1

        for i, mes in enumerate(message_list):
            mes_np = 1 - normalize_score(np.array(torch.squeeze(mes).cpu().detach()))
            auc_i = roc_auc_score(ano_label, mes_np)
            sep_i = compute_separation_score(np.array(torch.squeeze(mes).cpu().detach()), args.sep_percentile)
            print(f"  Model [k={n_cut+1}, t={i+1}]  AUC={auc_i:.4f}  sep={sep_i:.4f}")

        message_mean_list.append(torch.unsqueeze(torch.mean(torch.cat(message_list), 0), 0))

        msg_w, w_t, sep_t = weighted_aggregate([torch.squeeze(m) for m in message_list], args.sep_percentile, args.sep_temp)
        message_mean_list_weighted.append(torch.unsqueeze(msg_w, 0))
        print(f"  T-weights k={n_cut+1}: " + "  ".join([f"t{i+1}={w:.3f}" for i, w in enumerate(w_t)]))

        score_orig = 1 - normalize_score(np.array(torch.mean(torch.cat(message_mean_list), 0).cpu().detach()))
        auc_orig = roc_auc_score(ano_label, score_orig)
        ap_orig = average_precision_score(ano_label, score_orig, average="macro", pos_label=1)

        msg_k, w_k, sep_k = weighted_aggregate([torch.squeeze(m) for m in message_mean_list_weighted], args.sep_percentile, args.sep_temp)
        score_w = 1 - normalize_score(np.array(msg_k.cpu().detach()))
        auc_w = roc_auc_score(ano_label, score_w)
        ap_w = average_precision_score(ano_label, score_w, average="macro", pos_label=1)

        print(f"  Cumulative k={n_cut+1}: Orig AUROC={auc_orig:.4f} | Weighted AUROC={auc_w:.4f}")
        print(f"  K-weights: " + "  ".join([f"k{i+1}={w:.3f}" for i, w in enumerate(w_k)]))

end = time.time()

score_final_orig = 1 - normalize_score(np.array(torch.mean(torch.cat(message_mean_list), 0).cpu().detach()))
auc_fo = roc_auc_score(ano_label, score_final_orig)
ap_fo = average_precision_score(ano_label, score_final_orig, average="macro", pos_label=1)

msg_fw, fw_k, fsep_k = weighted_aggregate([torch.squeeze(m) for m in message_mean_list_weighted], args.sep_percentile, args.sep_temp)
score_final_w = 1 - normalize_score(np.array(msg_fw.cpu().detach()))
auc_fw = roc_auc_score(ano_label, score_final_w)
ap_fw = average_precision_score(ano_label, score_final_w, average="macro", pos_label=1)

print("="*60)
print(f"  FINAL RESULTS - {args.dataset}")
print("="*60)
print(f"  Original TAM  AUROC: {auc_fo:.4f}  AUPRC: {ap_fo:.4f}")
print(f"  Weighted TAM  AUROC: {auc_fw:.4f}  AUPRC: {ap_fw:.4f}")
print(f"  Delta AUROC : {auc_fw-auc_fo:+.4f}")
print(f"  Delta AUPRC : {ap_fw-ap_fo:+.4f}")
print(f"  K-weights: " + "  ".join([f"k{i+1}={w:.3f}" for i, w in enumerate(fw_k)]))
print(f"  Runtime: {end-start:.1f}s")
print("="*60)

np.save(f"results_{args.dataset}.npy", {
    "dataset": args.dataset, "ano_label": ano_label,
    "score_original": score_final_orig, "score_weighted": score_final_w,
    "auc_original": auc_fo, "auc_weighted": auc_fw,
    "ap_original": ap_fo, "ap_weighted": ap_fw,
    "final_k_weights": fw_k, "final_sep_scores": fsep_k,
})
print(f"  Saved results_{args.dataset}.npy")
