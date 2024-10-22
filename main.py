from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import KFold, StratifiedKFold
import torch
import copy
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch.utils.data as Data
from layers import lg_gnn, opt,GCN,GraphSAGE
#from models.simple_gnn import GCN, GraphSAGE

#from models.DGCNN import DGCNN
#from models.DiffPool import DiffPool
import sys

#from models.DGCNN import DGCNN

sys.path.append('C:/Users/Administrator/Desktop/MCI')


from config import DATA_DIR
from torch.optim import lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_val_data_process(data):
    dataset = data
    # 划分训练集和验证集
    train_data, val_data = Data.random_split(
        dataset, [round(0.8 * len(dataset)), round(0.2 * len(dataset))]
    )
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=opt.batch, shuffle=True, num_workers=1
    )
    val_dataloader = DataLoader(
        dataset=val_data, batch_size=opt.batch, shuffle=True, num_workers=1
    )
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    seed = 2024
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)

    lambda_l1 = opt.l1
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
    )  # lr=0.0005,weight_decay=0.01;weight_decay的值越大
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=opt.step_size, gamma=opt.gamma
    )  # step_size=30, gamma=0.1

    criterion1 = (
        nn.CrossEntropyLoss()
    )  # F.nll_loss   #nn.CrossEntropyLoss()   nn.BCELoss()
    criterion2 = F.nll_loss

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    since = time.time()
    num_node = opt.num_node

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0
        model.train()

        for data in train_dataloader:
            data = data.to(device)
            label = torch.tensor(data.y)
            label = label.to(device)
            # print('label的值为',label.shape)
            # print('data的值为',data)
            # print('data.batch的值为',data.batch)

            output = model(data)  # torch.Size([1, 32])
            # print('output为', output)
            # print(print('label为', label))
            # output = torch.squeeze(output)#转换为[32]
            # print('output的维数为',output.shape)
            # print('label的维数为',label.shape)

            loss = criterion1(
                output, label
            )  # +criterion2(F.log_softmax(output, dim=-1),label)
            # 添加L1正则化
            l1_regularization = torch.tensor(0.0)
            l1_regularization = l1_regularization.to(device)
            for param in model.parameters():
                l1_regularization += torch.norm(param, 1)
            loss = loss + lambda_l1 * l1_regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0) / num_node

            # print(data.size(0)/num_node)

            train_corrects += torch.sum(torch.argmax(output, dim=1) == label)
            train_num += data.size(0) / num_node
        scheduler.step()  # 更新学习率

        labels, preds, pred_probs = [], [], []
        for data in val_dataloader:
            data = data.to(device)
            labels.append(data.y)
            label = torch.tensor(data.y)
            label = label.to(device)
            # print('label的值为',label.shape)
            # print('data的值为',data)
            # print('data.batch的值为',data.batch)
            model.eval()
            # model.train()
            output = model(data)  # torch.Size([1, 32])
            # output = torch.squeeze(output)#转换为[32]
            # print('output的维数为',output.shape)
            # print('label的维数为',label.shape)
            loss = criterion1(
                output, label
            )  # +criterion2(F.log_softmax(output, dim=-1),label)
            val_loss += loss.item() * data.size(0) / num_node
            pred = torch.argmax(output, dim=1)
            preds.append(pred.cpu().detach().numpy())
            pred_probs.append(output.cpu().detach().numpy())
            val_corrects += torch.sum(pred == label)
            val_num += data.size(0) / num_node
        labels, preds, pred_probs = (
            np.concatenate(labels),
            np.concatenate(preds),
            np.concatenate(pred_probs),
        )

        # 计算指标
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        # Precision
        precision = precision_score(labels, preds, labels=[0, 1])
        print(precision)

        # Recall
        recall = recall_score(labels, preds, labels=[0, 1])
        print(recall)

        try:
            auc = roc_auc_score(labels, pred_probs[:, 1], labels=[0, 1])
        except:
            auc = None
        f1 = f1_score(labels, preds, labels=[0, 1])

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(acc)

        print(
            "{} train loss:{:.4f} train acc: {:.4f}".format(
                epoch, train_loss_all[-1], train_acc_all[-1]
            )
        )
        print(
            "{} val loss:{:.4f} val acc: {:.4f}".format(
                epoch, val_loss_all[-1], val_acc_all[-1]
            )
        )

        if acc > best_acc:
            best_acc = acc
            final_metrics = [acc, precision, recall, auc, f1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_use = time.time() - since
        print("Time elapsed: {:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "model/best_model.pth")

    train_process = pd.DataFrame(
        data={
            "epoch": range(num_epochs),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all,
        }
    )
    final_metrics = pd.DataFrame(final_metrics)

    return train_process, final_metrics


def matplot_acc_loss(train_process, save_prefix=True):
    # 设置 seaborn 主题
    sns.set_theme(style="whitegrid")

    # 创建一个图形框架
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制训练和验证损失
    sns.lineplot(
        x=train_process["epoch"],
        y=train_process.train_loss_all,
        marker="o",
        dashes=False,
        color="red",
        label="Train Loss",
        ax=axs[0],
    )
    sns.lineplot(
        x=train_process["epoch"],
        y=train_process.val_loss_all,
        marker="s",
        dashes=False,
        color="blue",
        label="Val Loss",
        ax=axs[0],
    )
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    # 绘制训练和验证准确率
    sns.lineplot(
        x=train_process["epoch"],
        y=train_process.train_acc_all,
        marker="o",
        dashes=False,
        color="red",
        label="Train Acc",
        ax=axs[1],
    )
    sns.lineplot(
        x=train_process["epoch"],
        y=train_process.val_acc_all,
        marker="s",
        dashes=False,
        color="blue",
        label="Val Acc",
        ax=axs[1],
    )
    axs[1].set_title("Training and Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    # 显示图表
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_features = opt.num_features
    # pt = DATA_DIR.joinpath(f"data{num_features}.pt")
    # pt = DATA_DIR.joinpath(f"nonimg4NCMCI.SGR100.pt")
    pt = DATA_DIR.joinpath(f"./NCMCI2_8_4nonimg.pt")
    data = torch.load(pt)
    dataset = data
    labels = [dataset[i]["y"] for i in range(len(dataset))]
    # for i in range(len(dataset)):
    #     dataset[i]["x"] = dataset[i]["x"].mean(1)

    # 划分训练集和验证集
    kf = KFold(n_splits=8, shuffle=True, random_state=2024)
    # kf = StratifiedKFold(n_splits=8, shuffle=True, random_state=2024)
    last_accs = []
    metric_df = pd.DataFrame(index=["acc", "precision", "recall", "auc", "f1"])
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        # for fold, (train_idx, test_idx) in enumerate(kf.split(dataset, labels)):
        print(f"FOLD {fold}")
        print("--------------------------------")

        # 为每个fold重新初始化模型

        #model = GCN(num_features, 2)
        model = GraphSAGE(num_features, 2)
        # model = DiffPool(num_features, 2)
        # model = DGCNN(num_features, 2)
        #model = lg_gnn()
        # 划分数据集
        train_dataset = [dataset[i] for i in train_idx]
        val_dataset = [dataset[i] for i in test_idx]
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=opt.batch,
            shuffle=True,
            num_workers=8,
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=opt.batch,
            shuffle=False,
            num_workers=8,
        )

        # 训练模型
        train_process, metrics = train_model_process(
            model, train_dataloader, val_dataloader, num_epochs=opt.num_epochs
        )
        metrics.index = metric_df.index
        metrics.columns = [fold]
        metric_df = pd.concat([metric_df, metrics], axis=1)
        matplot_acc_loss(train_process)

    # print("Last 5-fold validation accuracies:", last_accs)
    # print("Average accuracy:", sum(last_accs) / len(last_accs))
    print(metric_df)
    print("--------------AVG METRICS------------------")
    print(metric_df.mean(1))
