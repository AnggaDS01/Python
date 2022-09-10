import os
import numpy as np
import matplotlib.pyplot as plt

'''
    os.path.dirname(os.path.realpath(__file__)) # digunakan untuk .py
    os.getwcd() # digunakan untuk .ipynb
'''

script_path = os.path.dirname(
    os.path.realpath(__file__))  # digunakan untuk .py

history_path = os.path.join(script_path, 'history')

loss_history_path = os.path.join(history_path, 'loss_history')
prediction_history_path = os.path.join(history_path, 'prediction_history')
accuracy_history_path = os.path.join(history_path, 'accuracy')

data_source_path = os.path.join(script_path, 'data_source')

X = np.load(f'{data_source_path}/X.npy')
y = np.load(f'{data_source_path}/y.npy')


sorted_prediction_dir = sorted(os.listdir(
    prediction_history_path), key=lambda x: (len(x), x))
sorted_loss_dir = sorted(os.listdir(loss_history_path),
                         key=lambda x: (len(x), x))
sorted_accuracy_dir = sorted(os.listdir(
    accuracy_history_path), key=lambda x: (len(x), x))


def plot_decision_boundary(X, y):
    np.random.seed(43)
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    for prediction_file, loss_file, accuracy_file in zip(sorted_prediction_dir, sorted_loss_dir, sorted_accuracy_dir):
        plt.clf()

        y_pred = np.load(f'{prediction_history_path}/{prediction_file}')
        loss = np.load(f'{loss_history_path}/{loss_file}')
        accuracy = np.load(f'{accuracy_history_path}/{accuracy_file}')

        plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                    edgecolors='white', linewidths=.5)

        plt.title(prediction_file.split('.')[0])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        t = plt.text(
            x=x_max * 0.8,
            y=y_max * 0.8,
            s=f'loss: {np.round(loss, 3)}\naccuracy: {np.round(accuracy, 3)}'
        )

        t.set_bbox(dict(facecolor='white', alpha=0.5))

        plt.pause(.0001)
    plt.show()


plot_decision_boundary(X, y)
