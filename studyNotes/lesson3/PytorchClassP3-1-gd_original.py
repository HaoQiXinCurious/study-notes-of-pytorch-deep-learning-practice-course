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


def cost(xs, ys):
    cos = 0
    for x, y in zip(xs, ys):
        y_pre = forward(x)
        cos += (y_pre - y) ** 2
    return cos / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 'w=', w, 4, forward(4), '\n')
epoch_list = []
cost_list = []
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val

    epoch_list.append(epoch)
    cost_list.append(cost_val)
    print('Epoch: ', epoch, 'w=', w, 'cost=', cost_val)

print('\nPredict (after training)', 'w=', w, 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.title('Stochastic Gradient Descent')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()
