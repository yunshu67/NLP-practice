import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

VOCAB = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
train_data = [1, 2, 3, 4, 5]
train_label = [1, 1, 1, 0, 0]


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(7, 3, padding_idx=0)
        self.fc = nn.Linear(3, 1)

    def forward(self, x):
        embed = self.embedding(x)
        out = self.fc(embed)
        # print(out)
        sigmoid_out = torch.sigmoid(out)
        # print(sigmoid_out)
        return sigmoid_out


myModel = MyModel()
myModel.forward(torch.tensor(1))

lossFn = nn.functional.binary_cross_entropy

# print(lossFn(torch.tensor(0.1),torch.tensor(1.0)))


learning_rate = 1e-2
optimizer = torch.optim.Adam(myModel.parameters(), lr=learning_rate)

epoch = 10

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))
    # 训练步骤开始
    myModel.train()
    for idx in range(len(train_label)):
        data = torch.tensor(train_data[idx])
        label = torch.tensor([train_label[idx]],dtype=torch.float)
        output = myModel(data)  # 求模型的输出
        loss = lossFn(output, label)  # 求loss


        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 求梯度
        optimizer.step()  # 更新参数


