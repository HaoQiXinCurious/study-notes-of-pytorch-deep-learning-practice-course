"""
Curious
"""
import numpy as np
import matplotlib.pyplot as plt
import visdom

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

w_list_mesh, b_list_mesh = np.meshgrid(np.arange(0.0, 4.1, 0.1), np.arange(0.0, 4.1, 0.1))
mse_list_mesh = (np.array(mse_list)).reshape(41, 41)
print('w', np.shape(w_list_mesh), 'b', np.shape(b_list_mesh), 'mse', np.shape(mse_list_mesh))

fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.scatter(w_list_mesh, b_list_mesh, mse_list_mesh)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('mse')
ax.set_title('Loss(scatter)')
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_wireframe(w_list_mesh, b_list_mesh, mse_list_mesh)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('mse')
ax.set_title('Loss(wireframe)')
plt.show()

# vis = visdom.Visdom()
# vis.matplot(plt)
