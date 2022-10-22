"""
Curious
"""
import matplotlib.pyplot as plt
from random import shuffle

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [2.0, 4.0, 6.0, 8.0]  # 修改1：增加了一个样本
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


print('Predict (before training)', 'w=', w, 4, forward(4), '\n')
epoch_list = []
cost_list = []
iteration_list = []
cost_list_iter = []
batch_size = 2  # 设置batch_size
for epoch in range(30):  # 小一点方便看图
    i = 0
    data_list = list(zip(x_data, y_data))
    shuffle(data_list)
    while i < len(data_list) / batch_size:
        # 修改2：
        # 每次从打乱顺序的样本中选出batch_size个样本，直至每个样本都被选取过一遍后这次的while循环结束
        # 此处是先拿前两个，再拿后两个，结束此次while循环
        batch_data = data_list[i * batch_size: (i + 1) * batch_size]
        grad_val = gradient(batch_data)
        w -= 0.01 * grad_val
        cost_val = cost(x_data, y_data)  # 依然使用cost作为度量
        cost_list_iter.append(cost_val)
        iteration_list.append(epoch * len(data_list) / batch_size + i)
        i += 1

    epoch_list.append(epoch)
    cost_list.append(cost_val)
    print('Epoch: ', epoch, 'w=', w, 'cost=', cost_val)

print('\nPredict (after training)', 'w=', w, 4, forward(4))

plt.subplot(211)
plt.title('Mini-Batch Gradient Descent(batch size: %d)' % batch_size)
plt.plot(epoch_list, cost_list)
plt.xlabel('Epoch')
plt.ylabel('Cost')

plt.subplot(212)
plt.plot(iteration_list, cost_list_iter)
plt.xlabel('iteration')
plt.ylabel('Cost')
plt.show()
