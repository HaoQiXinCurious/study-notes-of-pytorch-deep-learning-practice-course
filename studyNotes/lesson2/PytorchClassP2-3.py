"""
Curious
"""
import numpy as np
import visdom
from matplotlib import pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w + b


def loss(x, y):
    y_pre = forward(x)
    return (y_pre - y) * (y_pre - y)


w_list = []
b_list = []
mse_list = []
vis = visdom.Visdom()
# win = 0
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0, 4.1, 0.1):
        print('w =', w, 'b =', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pre_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pre_val, loss_val)
        print('MSE=', l_sum / 3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum / 3)
        # vis.scatter(X=np.vstack((w_list, b_list, mse_list)).T,
        #             win=win,
        #             name='pre',
        #             update='append')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(w_list, b_list, mse_list)
vis.matplot(plt)
vis.scatter(X=np.vstack((w_list, b_list, mse_list)).T)

# print(np.vstack((w_list, b_list, mse_list)).T)
# bOrw = np.arange(0.0, 4.1, 0.1)
# print('b or w: ', bOrw)
# w_list_mesh, b_list_mesh = np.meshgrid(bOrw, bOrw)
# print('w: ', w_list_mesh, '\n')
# print('b: ', b_list_mesh)
