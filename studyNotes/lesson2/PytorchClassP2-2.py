"""
Curious
"""
import numpy as np
import matplotlib.pyplot as plt
import visdom

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w


def loss(x, y):
    y_pre = forward(x)
    return (y_pre - y) * (y_pre - y)


w_list = []
mse_list = []
vis = visdom.Visdom()
win = vis.line(Y=[0], X=[0])
for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pre_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pre_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)
    vis.line(Y=mse_list, X=w_list, win=win, update='append')


plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
# plt.show()
