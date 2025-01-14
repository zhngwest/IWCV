import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from model.model_iwci import *
from data import datalaoder
from sklearn.model_selection import KFold
from densityratio.density_ratio import *
from evaluate.evaluate_test import *
from densityratio.kmm import compute_density_ratio



# set the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--corruption_type', type=str, default='gaussian_noise', help='covariate shift type')
parser.add_argument('--corruption_level', type=float, default=3, help='covariate shift level, severity levels 1-5')
parser.add_argument('--imbalance_rate', type=float, default=0.4, help='class imbalance rate, should be less than 1')
parser.add_argument('--n_splits', type=float, default=10, help='n-Fold Cross-Validation ')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--step', type=float, default=100, help='period of learning rate decay')
parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--num_val', type=int, default=1000, help='total number of validation data')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training data')
parser.add_argument('--batch_size_val', type=int, default=256, help='batch size for validation data')
parser.add_argument('--num_epoch', type=int, default=10, help='total number of training epoch')
parser.add_argument('--seed', type=int, default=100, help='random seed')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.environ['PYTHONHASHSEED'] = '0'
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def build_model():
    net = LeNet(n_out=2)
    if torch.cuda.is_available():
        net.cuda()
    opt = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.step, gamma=args.gamma)
    return net, opt, scheduler

def main():
    # no imbalance
    # data_loader
    cifar10_path = 'data/cifar_data/cifar10'
    cifar10c_path = 'data/cifar_data/cifar-10-c'
    category_1, category_2 = 0, 1

    test_loader = datalaoder.load_cifar_data(cifar10_path, category_1, category_2, args.imbalance_rate,  args.corruption_type,
                                                 args.corruption_level,  args.batch_size)

    train_loader, val_loader= datalaoder.load_cifarC_data(cifar10c_path, category_1, category_2, args.imbalance_rate,  args.corruption_type,
                                                 args.corruption_level,  args.batch_size)

    # define the model, optimizer, and lr decay scheduler
    net, opt, scheduler = build_model()

    # prepare1： pos_indices & neg_indices
    pos_indices = []
    neg_indices = []
    for i, (images, labels) in enumerate(train_loader):
        batch_pos_indices = (labels == 0).nonzero(as_tuple=True)[0]
        batch_neg_indices = (labels != 0).nonzero(as_tuple=True)[0]
        pos_indices.extend(batch_pos_indices + i * len(labels))
        neg_indices.extend(batch_neg_indices + i * len(labels))
    pos_indices = torch.tensor(pos_indices)
    neg_indices = torch.tensor(neg_indices)

    # prepare2： compute density_ratio
    # directly use the train and test data
    # w = compute_density_ratio(net, train_loader, test_loader)
    # print("the shape of w : ",w.shape)

    n_splits = args.n_splits

    kf = KFold(n_splits=n_splits)

    total_accuracy = 0.0
    fold = 0

    for train_idx, val_idx in kf.split(train_loader.dataset):
        fold += 1
        print(f"Training fold {fold}/{n_splits}...")

        # 根据索引划分训练集和验证集
        train_subset = torch.utils.data.Subset(train_loader.dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_loader.dataset, val_idx)

        train_loader_fold = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size)
        val_loader_fold = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size)

        # use the test data to compute the density ratio
        w = compute_density_ratio(net, train_loader_fold, test_loader, fold)
        assert len(w) == len(train_loader_fold.dataset), "Length of weights must match training data."

        # train model
        net.train()
        for epoch in tqdm(range(args.num_epoch)):
            # define some metric
            train_acc_tmp = []
            val_acc_tmp = []
            for batch_idx, (images, labels) in enumerate(train_loader_fold):
                y_train = labels
                image = to_cuda(images)
                labels = to_cuda(labels)
                out_train = net(image)

                # 调用全局正负类索引，动态调整损失
                S_pos_size = len(pos_indices) if len(pos_indices) > 0 else 1
                S_neg_size = len(neg_indices) if len(neg_indices) > 0 else 1

                # 手动计算batch的incides起止
                start_idx = batch_idx * args.batch_size
                end_idx = start_idx + len(y_train)
                indices_in_train = list(range(start_idx, end_idx))  # 当前 batch 的全局索引
                w_batch = w[indices_in_train]  # 获取当前 batch 对应的权重

                loss = 0.0
                for idx in range(len(y_train)):
                    if idx in pos_indices:
                        loss += len(y_train) * (w_batch[idx] * F.cross_entropy(out_train[idx].unsqueeze(0),
                                                          labels[idx].unsqueeze(0))) / S_pos_size
                    else:
                        loss += len(y_train) * (w_batch[idx] * F.cross_entropy(-out_train[idx].unsqueeze(0),
                                                          labels[idx].unsqueeze(0))) / S_neg_size
                loss = loss / len(y_train)

                l_tr_wc = F.cross_entropy(out_train, labels, reduction='none')
                l_tr_wc_weighted = torch.sum(l_tr_wc * w_batch)

                opt.zero_grad()
                # loss.backward()
                l_tr_wc_weighted.backward()
                opt.step()

                # 计算准确率
                train_correct = 0
                train_total = 0
                _, predicted = torch.max(out_train, 1)
                train_total += labels.size(0)

                train_correct += (predicted == labels).sum().item()
                train_accuracy = train_correct / train_total
                train_acc_tmp.append(train_accuracy)
        train_accuracy_mean = np.mean(train_acc_tmp)
        print(f"Epoch {epoch+1}/{args.num_epoch}, train accuracy mean is {train_accuracy_mean:.4f}")

       # evaluate the model on val data
        net.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            for val_images, val_labels in val_loader_fold:
                val_images = to_cuda(val_images)
                val_labels = to_cuda(val_labels)
                out_val = net(val_images)
                loss = F.cross_entropy(out_val, val_labels, reduction='none')

                _, predicted = torch.max(out_val, 1)
                # the problem is that the length of val doesn't match the predicted?

                if predicted.size(0) != val_labels.size(0):
                    print(
                        f"Skipping batch with size mismatch: predicted={predicted.size(0)}, labels={val_labels.size(0)}")
                    continue  # Skip this batch if sizes don't match

                total_val += val_labels.size(0)
                correct_val += (predicted == val_labels).sum().item()
                val_accuracy = correct_val / total_val
                val_acc_tmp.append(val_accuracy)

            print(f"Validation Accuracy: {val_accuracy:.4f}")
            total_accuracy += val_accuracy
    # 计算平均指标
    average_accuracy = total_accuracy / n_splits
    print("The average accuracy is :", average_accuracy)
    # 在测试集表现
    accuracy, auc = evaluate_on_test_set(net, test_loader)


if __name__ == "__main__":
    main()


