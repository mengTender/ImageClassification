# _*_ coding: utf-8 _*_
# 在cifar10上训练VGG16
import os
import argparse
import torch
import torch.nn as nn
from torch import optim
import torch.cuda as cuda
import torch.utils.data as udata
import torchvision as tv
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm

import configuration_cifar10 as bc  # basic config
import utils.plot_utils as pu

# 参数获取
args_p = argparse.ArgumentParser()

args_p.add_argument('-model_name', default='VGG16', help="Pre-defined Model Name")
args_p.add_argument('-batch_size', default=128, help="Batch Size for Training")  # 批大小
args_p.add_argument('-lr', default=0.01, help="Learning Rate")  # 学习率
args_p.add_argument('-num_epoches', default=250, help="Training Epoches")  # 训练周期数
args_p.add_argument('-result_savepath', default='./models/results.txt', help="result save path")  # 训练周期数
args = args_p.parse_args()

# 数据集下载
train_set = datasets.CIFAR10('./data', train=True, download=True, transform=bc.transforms_dict['train_transform'])
train_loader = udata.DataLoader(dataset=train_set, shuffle=True, batch_size=args.batch_size)
test_set = datasets.CIFAR10('./data', train=False, download=True, transform=bc.transforms_dict['test_transform'])
test_loader = udata.DataLoader(dataset=test_set, shuffle=False, batch_size=int(args.batch_size))

# 定义模型
model = bc.models[args.model_name]
use_gpu = cuda.is_available()

if use_gpu:
    model = model.cuda()

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.005)

# 损失和准确率记录
best_epoch = 1
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# 模型训练
for epoch in range(args.num_epoches):
    print('\n', '*' * 25, 'Epoch {}'.format(epoch + 1), '*' * 25)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in tqdm(enumerate(train_loader, 1)):
        img, label = data
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img = Variable(img)
        label = Variable(label)
        # 前向传播
        out = model(img)
        loss = criterion(out, label)
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        accuracy = (pred == label).float().mean()
        running_acc += num_correct.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 一个周期训练结束
    loss_epoch = running_loss / len(train_set)
    acc_epoch = running_acc / len(train_set)
    print("Finish epoch: {}, Loss: {:.6f}, Acc:{:.6f}".format(epoch + 1, running_loss / len(train_set),
                                                              running_acc / len(train_set)))
    train_losses.append(loss_epoch)
    train_accuracies.append(acc_epoch)

    # 模型评估
    model.eval()
    eval_loss = 0
    eval_acc = 0
    with torch.no_grad():
        for data in test_loader:
            img, label = data
            if use_gpu:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, dim=1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / len(test_set), eval_acc / len(test_set)))
        test_losses.append(eval_loss / len(test_set))
        test_accuracies.append(eval_acc / len(test_set))
    model.train()
    # 模型数据保存
    torch.save(model.state_dict(), './models/epoch_{}.pth'.format(epoch + 1))
with open(args.result_savepath, mode='a') as result_file:
    result_file.write('model_name: {}'.format(args.model_name))
    result_file.write('batch_size: {}'.format(args.batch_size))
    result_file.write('lr: {}'.format(args.lr))
    result_file.write('num_epoches: {}'.format(args.num_epoches))
    result_file.write('train_losses: {}'.format(str(train_losses)))
    result_file.write('train_accuracies: {}'.format(str(train_accuracies)))
    result_file.write('test_losses: {}'.format(str(test_losses)))
    result_file.write('test_accuracies: {}'.format(str(test_accuracies)))
# 作图
pu.plot_figures(train_losses, 'Train Loss', ["Epoch", "Loss"], "./models/train_loss.png")
pu.plot_figures(train_accuracies, 'Train Accuracy', ["Epoch", "Accuracy"], "./models/train_acc.png")
pu.plot_figures(test_losses, 'Test Loss', ["Epoch", "Loss"], "./models/test_loss.png")
pu.plot_figures(test_accuracies, 'Test Accuracy', ["Epoch", "Accuracy"], "./models/test_acc.png")
print("Training process finished!")
