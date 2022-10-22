"""
Curious
"""
import matplotlib.pyplot as plt
from random import shuffle

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
iteration_list = []
cost_list_iter = []
for epoch in range(30):  # 小一点方便看图
    i = 0
    # 修改1：打乱数据原本的顺序
    data_list = list(zip(x_data, y_data))  # 先将数据保存在列表中以便使用shuffle
    shuffle(data_list)
    for x, y in data_list:
        grad = gradient(x, y)
        w -= 0.01 * grad
        cost_val = cost(x_data, y_data)  # 修改2：依然使用cost作为度量
        print('\tgrad: ', x, y, grad)
        cost_list_iter.append(cost_val)
        iteration_list.append(epoch * 3 + i)
        i += 1

    epoch_list.append(epoch)
    cost_list.append(cost_val)
    print('Epoch: ', epoch, 'w=', w, 'cost=', cost_val)

print('\nPredict (after training)', 'w=', w, 4, forward(4))

plt.subplot(211)
plt.title('Stochastic Gradient Descent')
plt.plot(epoch_list, cost_list)
plt.xlabel('Epoch')
plt.ylabel('Cost')

plt.subplot(212)
plt.plot(iteration_list, cost_list_iter)
plt.xlabel('iteration')
plt.ylabel('Cost')
print('cost1: ', cost_list, '\ncost2: ', cost_list_iter)
plt.show()
