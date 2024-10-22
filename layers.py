from torch_geometric.data import Data
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from torch_sparse import spspmm
from torch_geometric.nn import (
    GATConv,
    TransformerConv,
    global_mean_pool,
    global_max_pool,
    ASAPooling,
    TopKPooling,
    EdgePooling,
    global_add_pool,
    global_sort_pool,
    ChebConv,
    SAGEConv,
)
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.connect import FilterEdges
from torch_geometric.nn.pool.select import SelectTopK
from torch import nn
from torch_geometric.utils import (
    add_self_loops,
    sort_edge_index,
    remove_self_loops,
    add_random_edge,
)
from torch_geometric.utils import contains_self_loops
from torch_geometric.utils import dropout_adj, dropout_node, dropout_path, dropout_edge
from torch_geometric.utils import degree
from torch_geometric.utils import to_networkx
from torch_geometric.nn.norm import GraphNorm, PairNorm
import opt

# from lglayer import SABP
from utils_.lggnn_utils import EDGE, get_inputs

opt = opt.OptInit().initialize()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义一个MLP进行分类
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()  # 添加sigmoid激活函数
        self.softmax = nn.Softmax()

    def forward(self, x):
        # print('x的维度为', x.shape)
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)

        x = self.bn1(x)
        x = F.dropout(x, p=opt.dropout, training=self.training)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.bn2(x)
        x = F.dropout(x, p=opt.dropout, training=self.training)

        x = self.fc3(x)
        x = F.dropout(x, p=opt.dropout, training=self.training)
        x = self.sigmoid(x)
        # = F.log_softmax(x, dim=-1)
        return x


class Local_GAT(torch.nn.Module):
    # 2层图卷积+SABP+图卷积----
    def __init__(self):
        super(Local_GAT, self).__init__()
        self.indim = opt.num_features
        self.dim1 = opt.dim1
        self.atten1 = opt.atten1  # 8 32 8 16/8目前效果最好，性能很不稳定
        self.dim2 = opt.dim2
        self.atten2 = opt.atten2
        # self.dim3 = opt.dim3
        # self.atten3 = opt.atten3
        self.gradients = 492
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        # self.relu = nn.ReLU()

        self.conv1 = GATConv(self.indim, self.dim1, self.atten1)
        self.gcn = GCNConv(self.indim, self.dim1 * self.atten1)
        self.pooling1 = TopKPooling(self.dim1 * self.atten1, ratio=opt.TopK)
        # self.pooling1 = ASAPooling(self.dim1 * self.atten1, ratio=opt.ASAP_ratio)
        # self.pooling1 = TopKPooling(self.dim1, ratio=0.3)
        # self.bn1 = torch.nn.BatchNorm1d(self.dim1 * self.atten1)
        self.bn1 = torch.nn.BatchNorm1d(self.dim1 * self.atten1)
        self.gn1 = GraphNorm(self.dim1 * self.atten1)
        self.pn1 = PairNorm(self.dim1 * self.atten1)
        self.conv2 = GATConv(self.dim1 * self.atten1, self.dim2, self.atten2)
        # self.conv2 = GCNConv(self.dim1,self.dim2)

        self.pooling2 = TopKPooling(self.dim2 * self.atten2, ratio=opt.TopK)
        # self.pooling2 = ASAPooling(self.dim2 * self.atten2, ratio=opt.ASAP_ratio)
        self.bn2 = torch.nn.BatchNorm1d(self.dim2 * self.atten2)
        self.gn2 = GraphNorm(self.dim2 * self.atten2)
        self.pn2 = PairNorm(self.dim2 * self.atten2)

        self.cnn1 = nn.Conv1d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.cnn2 = nn.Conv1d(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.cnnpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lin1 = nn.Linear(492, 246)
        self.lin2 = nn.Linear(246, 64)

        self.GAT_MLP = MLP(
            input_size=(self.dim1 * self.atten1 + self.dim2 * self.atten2),
            hidden_size1=opt.hidden_size1,
            hidden_size2=opt.hidden_size2,
            output_size=2,
        )
        self.MLP = MLP(
            input_size=(self.dim1 * self.atten1 + self.dim2 * self.atten2),
            hidden_size1=opt.hidden_size1,
            hidden_size2=opt.hidden_size2,
            output_size=2,
        )

        # self.drop_adj = dropout_adj()使用伯努利分布中的样本以概率从邻接矩阵中随机删除边。(edge_index, edge_attr)p
        # self.drop_node = dropout_node()使用伯努利分布中的样本edge_index以概率从邻接矩阵中随机删除节点
        # self.drop_path = dropout_path()edge_index基于随机游走从邻接矩阵中删除边
        # self.drop_edge = dropout_edge()使用伯努利分布中的样本edge_index以概率从邻接矩阵中随机删除边
        # add_random_edge

    def forward(self, data):
        #print(f"x content: {data.x}, type: {type(data.x)}")
        x, edge_index, edge_attr = (
            data.x.to(torch.float32),
            data.edge_index.long(),
            data.edge_attr.to(torch.float32),
        )
        # shape x[15744, 246] edge_index[2, 1134246]  edge_attr[1134246]
        batch = data.batch

        # gradient = data.gradient.float()
        # gradient = gradient.view(len(torch.unique(batch)), 492)
        """gradient = self.relu(self.lin1(gradient))
        gradient = self.relu(self.lin2(gradient))"""

        # 将节点的度作为特征拼接
        # node_degree = self.generate_degree_matrix(data)
        # x = torch.cat([x, node_degree.view(-1, 1)], dim=-1)
        edge_attr = edge_attr.squeeze()
        # x2 = self.gcn(x, edge_index, edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))
        x1, edge_index1, edge_attr1, batch1, _, _ = self.pooling1(
            x, edge_index, None, batch
        )
        # edge_index, _ = dropout_edge(edge_index, p=opt.edropout, training=self.training)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.relu(x1)
        # x = self.gn1(x,batch1,opt.batch)
        x = self.bn1(x)
        x = self.conv2(x, edge_index1, edge_attr1)
        x2, edge_index2, edge_attr2, batch2, _, _ = self.pooling2(
            x, edge_index1, None, batch1
        )

        # x = F.dropout(x, p=0.5, training=self.training)
        x2 = self.relu(x2)
        x2 = self.bn2(x2)
        # x2 = self.gn2(x, batch2, opt.batch)
        # x = self.conv3(x, edge_index)
        # x3, edge_index, _, batch3, _, _ = self.pooling3(x, edge_index, None, batch2)

        embedding1 = global_max_pool(x1, batch1)
        embedding2 = global_max_pool(x2, batch2)
        # embedding3 = global_max_pool(x3, batch3)

        embedding_cat = torch.cat((embedding1, embedding2), dim=1)
        # embedding_g = self.Transformer(gradient)
        # embedding = torch.cat((embedding_cat,embedding_g),dim=1)

        # embedding = self.bn3(embedding_cat.double())
        # embedding = F.dropout(embedding, p=0.5, training=self.training)
        # embedding = self.relu(embedding)
        # G_embedding = torch.cat((embedding, gradient), dim=1)
        # prediction = self.MLP(embedding_cat)
        return embedding_cat

    def augment_adj(self, edge_index, edge_attr, num_nodes):
        edge_index, edge_attr = add_self_loops(
            edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes
        )  # 在邻接矩阵中增加自环,num_nodes=11000
        edge_index, edge_attr = sort_edge_index(edge_index, edge_attr, num_nodes)
        edge_index, edge_attr = spspmm(
            edge_index,
            edge_attr,
            edge_index,
            edge_attr,
            num_nodes,
            num_nodes,
            num_nodes,
        )
        # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        # print(edge_index.size(),edge_attr.size())

        return edge_index, edge_attr

    def generate_degree_matrix(self, data):
        # 获取edge_index，即图中所有边的两个端点的索引
        edge_index = data.edge_index
        # 计算每个节点的度
        node_degree = degree(edge_index[1], dtype=torch.long)

        return node_degree


class Global_GNN(nn.Module):
    def __init__(self):
        super(Global_GNN, self).__init__()
        self.convdim = opt.dim1 * opt.atten1 + opt.dim2 * opt.atten2+8#增加了8维的noimg
        self.num_layers = 4
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(
            ChebConv(self.convdim, 20, K=5, normalization="sym", node_dim=0)
        )
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20, K=5, normalization="sym", node_dim=0))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20, K=5, normalization="sym", node_dim=0))
        self.bns.append(nn.BatchNorm1d(20))
        self.convs.append(ChebConv(20, 20, K=5, normalization="sym", node_dim=0))
        self.bns.append(nn.BatchNorm1d(20))
        self.out_fc = nn.Linear(20, 2)
        self.weights = torch.nn.Parameter(torch.randn((len(self.convs))))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.out_fc.reset_parameters()
        torch.nn.init.normal_(self.weights)

    def forward(self, features, edges, edge_weight):
        x = torch.Tensor(features)
        x = x.float()
        layer_out = []
       # print("edges dtype:", edges.dtype)
        #print("x dtype:", x.dtype)

        x = self.convs[0](x, edges)
        x = self.bns[0](x)
        x = F.relu(x, inplace=True)
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[1](x, edges)
        x = self.bns[1](x)
        x = F.relu(x, inplace=True)
        x = x + 0.7 * layer_out[0]
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[2](x, edges)
        x = self.bns[2](x)
        x = F.relu(x, inplace=True)
        x = x + 0.7 * layer_out[1]
        layer_out.append(x)
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.convs[3](x, edges)
        x = self.bns[3](x)
        x = F.relu(x, inplace=True)
        x = x + 0.7 * layer_out[2]
        layer_out.append(x)
        weight = F.softmax(self.weights, dim=0)
        for i in range(len(layer_out)):
            layer_out[i] = layer_out[i] * weight[i]
        emb = sum(layer_out)
        x = self.out_fc(emb)
        return x


class lg_gnn(torch.nn.Module):

    def __init__(self):
        super(lg_gnn, self).__init__()
        self._setup()

    def _setup(self):
        self.edge = EDGE(4, dropout=0.2)  # 选择nonimg的维数
        self.graph_level_model = Local_GAT()
        self.hierarchical_model = Global_GNN()

    def forward(self, graphs):
        embedding = self.graph_level_model(graphs)
        graphnoimg = graphs.graphnoimg     #导入8维的MMSE评分等非图像信息
        # 将列表转换为张量，并调整形状为 (12, 1)
        graphnoimg_tensor = torch.tensor(graphnoimg)
        # 拼接 embedding 和 graphnoimg_tensor，沿着列（dim=1）
        graphnoimg_tensor = graphnoimg_tensor.to(embedding.device)
        embedding_noimg = torch.cat((embedding, graphnoimg_tensor), dim=1) #将非图像信息与GNN提取出的特征进行拼接

        nonimg = graphs.nonimg
        nonimg = torch.tensor(nonimg)
        # 通过数据加载器对象获取边的输入特征。
        edge_index, edge_input = get_inputs(nonimg, embedding)
        # 对边的输入特征进行标准化处理。
        std = edge_input.std(axis=0)
        std[std == 0] = 0.000001  # 避免除以零
        edge_input = (edge_input - edge_input.mean(axis=0)) / std
        # 将边的索引转换为 PyTorch 的长整型张量。
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
        # 将边的输入特征转换为 PyTorch 的浮点型张量。
        edge_input = torch.tensor(edge_input, dtype=torch.float32).to(device)
        # 使用模型 edge 计算边的权重。
        edge_weight = torch.squeeze(self.edge(edge_input))
        # 使用全局 GNN 模型对嵌入向量进行预测。
        predictions = self.hierarchical_model(
            embedding_noimg, edge_index, edge_weight
        )  # Global GNN
        # print(predictions)
        return predictions

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_layers = 2
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_nodes = 246
        num_gcn_hid = 24
        self.num_gcn_final = 8
        # self.convs = torch.nn.ModuleList()
        # self.convs.extend([ for _ in range(num_layers)])
        self.gcn1 = GCNConv(num_features, num_gcn_hid)
        self.gcn2 = GCNConv(num_gcn_hid, self.num_gcn_final)
        self.fc = torch.nn.Linear(self.num_gcn_final+8, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index.long(), data.batch

        # x = x.reshape(-1, self.num_features)
        x = self.gcn1(x, edge_index)
        x = self.gcn2(x, edge_index)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = x.reshape(-1, self.num_nodes, self.num_gcn_final)
        x = torch.max(x, dim=1)[0].squeeze()
        graphnoimg = data.graphnoimg  # 导入8维的MMSE评分等非图像信息
        # 将列表转换为张量，并调整形状为 (12, 1)
        graphnoimg_tensor = torch.tensor(graphnoimg)
        # 拼接 embedding 和 graphnoimg_tensor，沿着列（dim=1）
        graphnoimg_tensor = graphnoimg_tensor.to(x.device)
        graphnoimg_tensor = graphnoimg_tensor.float()
        x = torch.cat((x, graphnoimg_tensor), dim=1)  # 将非图像信息与GNN提取出的特征进行拼接

        x = self.fc(x)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, dim_features, dim_target):
        super().__init__()

        num_layers = 2
        dim_embedding = 24
        self.aggregation = "max"  # can be mean or max

        if self.aggregation == "max":
            self.fc_max = nn.Linear(dim_embedding, dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            # Overwrite aggregation method (default is set to mean
            conv = SAGEConv(dim_input, dim_embedding, aggr=self.aggregation)

            self.layers.append(conv)

        # For graph classification
        self.fc1 = nn.Linear(num_layers * dim_embedding+8, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index.long(), data.batch

        x_all = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.aggregation == "max":
                x = torch.relu(self.fc_max(x))
            x_all.append(x)

        x = torch.cat(x_all, dim=1)
        x = global_max_pool(x, batch)
        graphnoimg = data.graphnoimg  # 导入8维的MMSE评分等非图像信息
        # 将列表转换为张量，并调整形状为 (12, 1)
        graphnoimg_tensor = torch.tensor(graphnoimg)
        # 拼接 embedding 和 graphnoimg_tensor，沿着列（dim=1）
        graphnoimg_tensor = graphnoimg_tensor.to(x.device)
        graphnoimg_tensor = graphnoimg_tensor.float()
        x = torch.cat((x, graphnoimg_tensor), dim=1)  # 将非图像信息与GNN提取出的特征进行拼接
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

