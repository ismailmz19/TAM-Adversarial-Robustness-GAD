import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
## import dgl

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj_tensor(raw_adj):
    adj = raw_adj[0, :, :]
    row_sum = torch.sum(adj, 0)
    r_inv = torch.pow(row_sum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    adj = torch.mm(adj, torch.diag_embed(r_inv))
    adj = torch.mm(torch.diag_embed(r_inv), adj)
    adj = adj.unsqueeze(0)
    return adj


def normalize_score(ano_score):
    ano_score = ((ano_score - np.min(ano_score)) / (
            np.max(ano_score) - np.min(ano_score)))
    return ano_score


def process_dis(init_value, cutting_dis_array):
    r_inv = np.power(init_value, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    cutting_dis_array = cutting_dis_array.dot(sp.diags(r_inv))
    cutting_dis_array = sp.diags(r_inv).dot(cutting_dis_array)
    return cutting_dis_array


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mat(dataset):
    """Load .mat dataset."""

    data = sio.loadmat("./data/{}.mat".format(dataset))
    label = data['Label'] if ('Label' in data) else data['gnd']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    return adj, feat, ano_labels, str_ano_labels, attr_ano_labels





# compute the distance between each node
def calc_distance(adj, seq):
    adj = adj.cuda() if torch.cuda.is_available() else adj if torch.cuda.is_available() else adj
    seq = seq.cuda() if torch.cuda.is_available() else seq

    N = adj.shape[0]
    dis_array = torch.zeros((N, N), device="cuda" if torch.cuda.is_available() else "cpu")

    print("=== Starting calc_distance ===")
    for i in range(N):

        # guaranteed progress print
        if i % 100 == 0 or i == N-1:
            print(f"calc_distance progress: {i}/{N}")

        neighbors = torch.nonzero(adj[i] > 0, as_tuple=False).flatten()

        if len(neighbors) == 0:
            continue

        diffs = seq[i] - seq[neighbors]
        dists = torch.norm(diffs, dim=1)

        dis_array[i, neighbors] = dists

    print("=== Finished calc_distance ===")
    return dis_array




def calc_sim(adj_matrix, attr_matrix):
    row = adj_matrix.shape[0]
    col = adj_matrix.shape[1]
    dis_array = np.zeros((row, col))
    for i in range(row):
        # print(i)
        node_index = np.argwhere(adj_matrix[i, :] > 0)[:, 0]
        for j in node_index:
            dis = get_cos_similar(attr_matrix[i].tolist(), attr_matrix[j].tolist())
            dis_array[i][j] = dis

    return dis_array


def graph_nsgt(dis_array, adj):
    # move tensors to GPU only if CUDA is available
    if torch.cuda.is_available():
        dis_array = dis_array.cuda() if torch.cuda.is_available() else dis_array
        adj = adj.cuda() if torch.cuda.is_available() else adj if torch.cuda.is_available() else adj
    else:
        dis_array = dis_array.to('cpu')
        adj = adj.to('cpu')

    row = dis_array.shape[0]
    dis_array_u = dis_array * adj
    mean_dis = dis_array_u[dis_array_u != 0].mean()

    for i in range(row):
        node_index = torch.argwhere(adj[i, :] > 0)
        if node_index.shape[0] != 0:
            max_dis = dis_array[i, node_index].max()
            min_dis = mean_dis
            if max_dis > min_dis:
                random_value = (max_dis - min_dis) * np.random.random_sample() + min_dis
                cutting_edge = torch.argwhere(dis_array[i, node_index[:, 0]] > random_value)
                if cutting_edge.shape[0] != 0:
                    adj[i, node_index[cutting_edge[:, 0]]] = 0

    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj



import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (8.5, 7.5)
from matplotlib.backends.backend_pdf import PdfPages

def draw_pdf(message, ano_label, dataset):
    with PdfPages('{}-TAM.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        abnormal_message_all = message[ano_label == 1]
        message_all = [normal_message_all, abnormal_message_all]
        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of abnormal {}'.format(mu_1))
        print('The std of abnormal {}'.format(sigma_1))
        n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Abnormal'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)
        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=7.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=7.5)
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
        plt.legend(loc='upper left', fontsize=30)
        plt.title(''.format(dataset), fontsize=25)
        plt.show()


def draw_pdf_str_attr(message, ano_label, str_ano_label, attr_ano_label, dataset):
    with PdfPages('{}-TAM.pdf'.format(dataset)) as pdf:
        normal_message_all = message[ano_label == 0]
        str_abnormal_message_all = message[str_ano_label == 1]
        attr_abnormal_message_all = message[attr_ano_label == 1]
        message_all = [normal_message_all, str_abnormal_message_all, attr_abnormal_message_all]

        mu_0 = np.mean(message_all[0])
        sigma_0 = np.std(message_all[0])
        print('The mean of normal {}'.format(mu_0))
        print('The std of normal {}'.format(sigma_0))
        mu_1 = np.mean(message_all[1])
        sigma_1 = np.std(message_all[1])
        print('The mean of str_abnormal {}'.format(mu_1))
        print('The std of str_abnormal {}'.format(sigma_1))
        mu_2 = np.mean(message_all[2])
        sigma_2 = np.std(message_all[2])
        print('The mean of attt_abnormal {}'.format(mu_2))
        print('The std of attt_abnormal {}'.format(sigma_2))
        n, bins, patches = plt.hist(message_all, bins=30, normed=1, label=['Normal', 'Structural Abnormal', 'Contextual Abnormal'])
        y_0 = mlab.normpdf(bins, mu_0, sigma_0)
        y_1 = mlab.normpdf(bins, mu_1, sigma_1)
        y_2= mlab.normpdf(bins, mu_2, sigma_2)  #

        plt.plot(bins, y_0, color='steelblue', linestyle='--', linewidth=3.5)
        plt.plot(bins, y_1, color='darkorange', linestyle='--', linewidth=3.5)
        plt.plot(bins, y_2, color='green', linestyle='--', linewidth=3.5)

        plt.xlabel('TAM-based Affinity', fontsize=25)
        plt.ylabel('Number of Samples', size=25)
        plt.yticks(fontsize=25)
        plt.xticks(fontsize=25)
        plt.legend(loc='upper left', fontsize=18)
        plt.title('{}'.format(dataset), fontsize=25)
        plt.show()




def adj_to_dgl_graph(adj):
    return None

