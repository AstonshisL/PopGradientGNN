import os

import scipy.io as sio
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
from os.path import join, isfile
from os import listdir
import numpy as np
import os.path as osp
from networkx.convert_matrix import from_numpy_array
import networkx as nx
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import coalesce
import pandas as pd
from gradient import get_gradients, sparse_top_percentage
from networkx.convert_matrix import to_scipy_sparse_array

data_folder = "/..../"


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def read_sigle_data(
        data, nonimg, label
):  # 将一个输入的网络连接强度矩阵转换为一个包含节点特征、边索引和边属性的图数据对象。
    """
    graph包含：  x直接在数据data里获取
    x: 节点特征矩阵，形状为 (num_nodes, num_features)，其中 num_nodes 是节点数量，num_features 是每个节点的特征维度
    edge_index: 边索引张量，形状为 (2, num_edges)，其中 num_edges 是边的数量，每一列表示一条边连接的两个节点的索引。
    edge_attr: 边属性矩阵，形状为 (num_edges, num_edge_features)，其中 num_edge_features 是每条边的属性维度。
    """
    # print(data)
    data = np.abs(data)  # ? 输入是一个网络连接强度的矩阵
    """pcorr = sparse_top_percentage(data, percentage=10)
    matrix = pcorr
    mean = np.mean(matrix)
    std = np.std(matrix)
    # 进行Z-Score标准化
    zscore_matrix = (matrix - mean) / std
    pcorr = zscore_matrix"""

    pcorr = sparse_top_percentage(data, percentage=100)
    # pcorr = np.where(np.abs(pcorr) < 0.8, 0, data)#阈值稀疏化

    num_nodes = pcorr.shape[0]  # 获取 pcorr 矩阵的节点数量 num_nodes
    G = from_numpy_array(pcorr)  # 将 pcorr 矩阵转换为 NetworkX 图对象 G
    A = nx.to_scipy_sparse_array(G)  # 将 NetworkX 图对象 G 转换为 SciPy 稀疏矩阵 A。
    adj = A.tocoo()  # 将稀疏矩阵 A 转换为 COO 格式的稀疏矩阵 adj
    edge_att = adj.data  # 初始化一个全零数组 edge_att用于存储边的属性
    for i in range(
            len(adj.row)
    ):  # 遍历 adj.row，将 pcorr 矩阵中对应位置的值赋给 edge_att，得到边的属性。
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    edge_index = np.stack(
        [adj.row, adj.col]
    )  # 将 adj.row 和 adj.col 堆叠在一起，形成一个形状为 (2, num_edges) 的边索引张量 edge_index，其中 num_edges 是边的数量。
    edge_index, edge_att = remove_self_loops(
        torch.from_numpy(edge_index), torch.from_numpy(edge_att)
    )
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
    att = pcorr  # 原有的代码为att = data
    att[att == float("inf")] = 0
    # print(edge_att)
    att_torch = torch.from_numpy(att).float()
    gradient = get_gradients(att)
    gradient = torch.tensor(gradient)
    gradient = gradient.reshape(246, 4)
    #以下代码意义为将gradient拼接

    #C = np.hstack((att_torch, gradient))
    #C = torch.from_numpy(C)

    # 以上代码意义为将gradient拼接
    #C = torch.tensor(C, dstype=torch.float32)
    nan_cnt = gradient.isnan().sum().item()

    if nan_cnt > 0:
        return None

    # gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())

    graph = Data(
        x=gradient,
        edge_index=edge_index.long(),
        edge_attr=edge_att,
        y=label,
        nonimg=nonimg,
    )
    #print(graph)
    # 包含节点特征 att_torch、边索引   edge_index 和边属性 edge_att。
    return graph


def read_feature_data(data, nonimg, label, graphnoimg):  # 将noimg作为
    """
    graph包含：  x直接在数据data里获取
    x: 节点特征矩阵，形状为 (num_nodes, num_features)，其中 num_nodes 是节点数量，num_features 是每个节点的特征维度
    edge_index: 边索引张量，形状为 (2, num_edges)，其中 num_edges 是边的数量，每一列表示一条边连接的两个节点的索引。
    edge_attr: 边属性矩阵，形状为 (num_edges, num_edge_features)，其中 num_edge_features 是每条边的属性维度。
    """
    import numpy as np
    import networkx as nx
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import remove_self_loops, coalesce

    # 假设你已经有了以下变量
    # features: 形状为(246, 47)的NumPy数组，表示节点的特征
    # pcorr: 形状为(246, 246)的NumPy数组，表示节点间的皮尔森相关性
    # label: 你的目标标签
    data = np.abs(data)
    features = data
    pcorr = np.corrcoef(features)  # 在归一化之前构建相关性矩阵
    # 对特征进行归一化
    scaler_standard = StandardScaler()
    features = scaler_standard.fit_transform(features)

    pcorr = sparse_top_percentage(pcorr, percentage=100)

    num_nodes = pcorr.shape[0]

    # 构建图
    G = from_numpy_array(pcorr)  # 将 pcorr 矩阵转换为 NetworkX 图对象 G
    A = nx.to_scipy_sparse_array(G)
    adj = A.tocoo()

    # 处理边属性
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]

    # 构建边索引
    edge_index = np.stack([adj.row, adj.col])

    # 移除自环并合并边
    edge_index, edge_att = remove_self_loops(
        torch.from_numpy(edge_index), torch.from_numpy(edge_att)
    )
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
    att = pcorr  # 原有的代码为att = data
    att[att == float("inf")] = 0
    # print(edge_att)
    att_torch = torch.from_numpy(att).float()

    gradient = get_gradients(pcorr)
    gradient = torch.tensor(gradient)
    gradient = gradient.reshape(246, 1)#根据梯度的维数进行调整
    #gradient_One = torch.ones((246, 4))
    #print(graphnoimg)

    # gradient = (gradient - gradient.min()) / (
    #     gradient.max() - gradient.min()
    # )  # 归一化梯度特征
    graph = Data(
        x=gradient,
        edge_index=edge_index.long(),
        edge_attr=edge_att,
        y=label,
        nonimg=nonimg,
        graphnoimg=graphnoimg
    )
    return graph


def get_networks():  # 将graph的节点特征
    data_folder = R"E:\Papers\Datasets\MCI diagnosis\R2SN-dataset\IDs_matrices"  # 通过feature构建
    data_path = "E:\Papers\Datasets\MCI diagnosis\R2SN-dataset\duomotai/NCMCI_MMSE.csv"
    subject_list = pd.read_csv(data_path, usecols=[0], header=None).values.flatten()
    nonimg = pd.read_csv(data_path, header=None).iloc[:, 1:5].values
    graphnoimg = pd.read_csv(data_path, header=None).iloc[:, 5:13].values
    labels = pd.read_csv(data_path, header=None).iloc[:, 13].values
    graphs = []
    lis_error_index = []
    for index, (subject, label) in enumerate(zip(subject_list, labels)):
        try:
            fl = os.path.join(data_folder, subject + ".csv")
            matrix = pd.read_csv(fl, header=None, dtype=float).values

            # 获取当前样本的特征
            current_features = nonimg[index, :]
            graphnoimg_row = graphnoimg[index, :]

            # 这里假设read_sigle_data是一个函数，用于创建图数据结构
            # 你需要将特征和标签传递给这个函数
            graph = read_feature_data(matrix, current_features, label,graphnoimg_row)
            if graph:
                # graph = read_feature_data(matrix, current_features, label)
                print(graph)
                graphs.append(graph)
                print(len(graphs))
        except:
            lis_error_index.append(index)
    torch.save(graphs, "data/NCMCI1_8_4nonimg.pt")
    print("数据保存成功！")
    print("错误索引：", lis_error_index)

    return graphs


if __name__ == "__main__":
    get_networks()
