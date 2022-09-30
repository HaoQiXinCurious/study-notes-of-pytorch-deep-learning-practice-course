"""
Curious
"""
import matplotlib.pyplot as plt
from random import choice

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):  # xs, ys表示x_data和y_data在这个函数的形参
    cos = 0
    for x, y in zip(xs, ys):
        y_pre = forward(x)
        cos += (y_pre - y) ** 2
    return cos / len(xs)


def gradient(x, y):
    return 2 * x * (x * w - y)


print('Predict (before training)', 'w=', w, 4, forward(4), '\n')
epoch_list = []
cost_list = []
for epoch in range(100):
    x_random, y_random = choice(tuple(zip(x_data, y_data)))  # 修改1：仅随机选择一个x和y作为梯度计算的输入
    grad = gradient(x_random, y_random)
    w -= 0.01 * grad
    cost_val = cost(x_data, y_data)  # 修改2：依然使用cost作为度量
    print('\tgrad: ', x_random, y_random, grad)

    epoch_list.append(epoch)
    cost_list.append(cost_val)
    print('Epoch: ', epoch, 'w=', w, 'cost=', cost_val)

print('\nPredict (after training)', 'w=', w, 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.title('Stochastic Gradient Descent')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()
