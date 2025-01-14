import numpy as np
import torch
import torch.nn.functional as F
from densityratio.kmm import get_density_ratios
from model.train import train_model

# reweight the loss function according to loss_collection
def custom_loss(model, X_train, X_val, y_train, y_val, pos_indices, neg_indices):

    #计算weight
    out_train = model(X_train)
    out_val = model(X_val)

    l_tr = F.cross_entropy(out_train, y_train, reduction='none')
    l_val = F.cross_entropy(out_val, y_val, reduction='none')

    l_tr_reshape = np.array(l_tr.detach().cpu()).reshape(-1, 1)
    l_val_reshape = np.array(l_val.detach().cpu()).reshape(-1, 1)

    density_ratio = get_density_ratios(l_tr_reshape, l_val_reshape)

    # 获取正样本和负样本的索引
    pos_indices = torch.where(y_train == 1)[0]
    neg_indices = torch.where(y_train == 0)[0]

    pos_loss = 0.0
    neg_loss = 0.0

    for i in range(len(X_train)):
        X_leave_out = torch.cat((X_train[:i], X_train[i+1:]))
        y_leave_out = torch.cat((y_train[:i], y_train[i+1:]))
        train_model(model, X_leave_out, y_leave_out)

        input_image = X_train[i]
        input_image = input_image.unsqueeze(0)

        pred = model(input_image)

        if y_train[i] == 1:
            pos_loss += len(y_train)*density_ratio[i] * F.cross_entropy(pred, y_train[i].view(1))/len(pos_indices)
        else:
            neg_loss += len(y_train)*density_ratio[i] * F.cross_entropy(-pred.view(1, -1), y_train[i].view(1))/len(neg_indices)
    return (pos_loss + neg_loss) / len(X_train)