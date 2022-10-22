"""
Curious
"""
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


print('Predict (before training)', 'w=', w,  4, forward(4), '\n')
epoch_list = []
loss_list = []
iteration_list = []
loss_list_iter = []
for epoch in range(30):  # 改小一点方便看图
    i = 0
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01 * grad
        l = loss(x, y)
        print('\tgrad: ', x, y, grad)
        i += 1
        loss_list_iter.append(l)
        iteration_list.append(epoch * 3 + i)

    loss_list.append(l)
    epoch_list.append(epoch)
    print('Epoch: ', epoch, 'w=', w, 'loss=', l)

print('\nPredict (after training)', 'w=', w, 4, forward(4))


plt.subplot(211)
plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(212)
plt.plot(iteration_list, loss_list_iter)
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.show()
