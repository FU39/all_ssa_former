import torch


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(0)
    loss_es = 0
    for t in range(T):
        loss_es += criterion(outputs[t, ...], labels)
    loss_es = loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        loss_mmd = 0
    return (1 - lamb) * loss_es + lamb * loss_mmd  # L_Total
