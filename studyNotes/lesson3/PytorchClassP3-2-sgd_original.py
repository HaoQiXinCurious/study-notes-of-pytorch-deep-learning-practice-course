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
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01 * grad
        l = loss(x, y)
        print('\tgrad: ', x, y, grad)

    loss_list.append(l)
    epoch_list.append(epoch)
    print('Epoch: ', epoch, 'w=', w, 'loss=', l)

print('\nPredict (after training)', 'w=', w, 4, forward(4))


plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
