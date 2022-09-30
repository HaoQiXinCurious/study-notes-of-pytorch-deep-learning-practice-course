"""
Curious
"""
import torch
import matplotlib.pyplot as plt


x_data = torch.Tensor([[4.0], [2.0], [3.0]])
y_data = torch.Tensor([[8.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()  # 用于调用基类构造函数，若单继承，super避免了基类的显式调用；若多继承还有其它好处
        self.Linear = torch.nn.Linear(1, 1)  # 构造一个Liner类的对象，相当于y=wx+b函数，两个输入分别是样本的输入及输出的大小

    def forward(self, x):  # forward函数会覆盖基类中的forward函数
        y_pre = self.Linear(x)  # Linear对象为可调用对象
        return y_pre


model = LinearModel()
criterion = torch.nn.MSELoss(reduction='sum')  # 生成一个求Loss的均方差的对象，size_average为False表示求完方差的和后不求均值
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 生成一个会对模型参数优化的执行随机梯度下降的对象，lr为学习率

epoch_list = []
loss_list = []
for epoch in range(1000):
    y_pre = model(x_data)  # 前馈函数
    loss = criterion(y_pre, y_data)  # 求Loss
    print(epoch, loss)  # 打印，loss是一个对象，被打印时自动生成loss表

    optimizer.zero_grad()  # 将梯度清零
    loss.backward()  # 根据loss反向传播
    print((optimizer.param_groups[0])['params'])
    optimizer.step()  # 更新参数

    epoch_list.append(epoch)
    loss_list.append(loss.detach().numpy())

print('w = ', model.Linear.weight.item())
print('b = ', model.Linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pre = ', y_test.data)
plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
