"""
Curious
"""
import matplotlib.pyplot as plt
from random import sample

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0


def forward(x):
    return x * w


def cost(xd, yd):  # xd, yd表示x_data和y_data在这个函数的形参
    cos = 0
    for x, y in zip(xd, yd):
        y_pre = forward(x)
        cos += (y_pre - y) ** 2
    return cos / len(xd)


def gradient(formal_batch_data):  # formal_batch_data表示batch_data的形参
    grad = 0
    for x, y in formal_batch_data:
        grad += 2 * x * (x * w - y)
    return grad / len(formal_batch_data)


print('Predict (before training)', 'w=', w,  4, forward(4), '\n')
epoch_list = []
cost_list = []
batch_size = 2  # 设置batch_size
for epoch in range(100):
    cost_val = cost(x_data, y_data)  # 依然使用cost作为度量
    batch_data = sample(tuple(zip(x_data, y_data)), batch_size)  # 根据随机选取batch_size个样本放入列表batch_data
    grad_val = gradient(batch_data)  # 使用随机选取的batch_data计算梯度
    w -= 0.01 * grad_val
    print('\tgrad: ', batch_data, grad_val)

    epoch_list.append(epoch)
    cost_list.append(cost_val)
    print('Epoch: ', epoch, 'w=', w, 'cost=', cost_val)

print('\nPredict (after training)', 'w=', w, 4, forward(4))


plt.plot(epoch_list, cost_list)
plt.title('Mini-Batch Gradient Descent(batch size: %d)' % batch_size)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()
