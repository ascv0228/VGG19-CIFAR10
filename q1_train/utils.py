import torch

from hyperparameters import *

def eval(model, loss_func, dataloader):

    model.eval()
    loss, accuracy = 0, 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            error = loss_func(logits, batch_y)
            loss += error.item()

            probs, pred_y = logits.data.max(dim=1)
            accuracy += (pred_y==batch_y.data).float().sum()/batch_y.size(0)

    loss /= len(dataloader)
    accuracy = accuracy*100.0/len(dataloader)
    return loss, accuracy


def train_epoch(model, loss_func, optimizer, dataloader):

    model.train()
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        error = loss_func(logits, batch_y)
        error.backward()
        optimizer.step()


import matplotlib.pyplot as plt

def plot_loss_acc(train_loss, val_loss, train_acc, val_acc):
    assert len(train_loss) == len(val_loss) == len(train_acc) == len(val_acc), "length of data are different."
    epochs = range(1, len(train_loss)+1)
    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 1. 繪製 Loss 圖
    ax1.plot(epochs, train_loss, label='train_loss', color='tab:blue')
    ax1.plot(epochs, val_loss, label='val_loss', color='tab:orange')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss')
    ax1.legend()

    # 2. 繪製 Accuracy 圖
    ax2.plot(epochs, train_acc, label='train_acc', color='tab:blue')
    ax2.plot(epochs, val_acc, label='val_acc', color='tab:orange')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy(%)')
    ax2.set_title('accuracy')
    ax2.legend()

    # 優化布局並顯示
    plt.tight_layout()
    plt.savefig("q1_train_loss_acc.jpg")
    plt.show()