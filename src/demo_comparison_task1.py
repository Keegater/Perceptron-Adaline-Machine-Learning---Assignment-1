import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Perceptron import Perceptron
from Adaline import AdalineGD

# Loading and standardizing data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None, encoding='utf-8')
# First 100 samples
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
# Extracting sepal length (col 0) and petal length (col 2)
X = df.iloc[0:100, [0, 2]].values
# Standardizing
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


# run and generate graphs for each combo of learning rate and epochs
etas = [0.8, 0.15, 0.001]
iters = [10, 100, 1000]
for eta in etas:
    print("########################################################")
    print("Testing with Learning Rate: {}\n".format(eta))
    for n_iter in iters:

        print("--------------------------------------------------------")
        print("{} Epochs".format(n_iter))

        # remove for loop and uncomment below for solo runs

        # init models with same params
        ## eta=0.03, n_iter=1000 good results for high iter
        #eta = 0.15         # Learning rate
        #n_iter = 50        # Num epochs
        ppn = Perceptron( n_iter=n_iter, eta=eta)
        ada = AdalineGD( n_iter=n_iter, eta=eta)

        ppn.fit(X_std, y)
        ada.fit(X_std, y)

        # Compute the margin
        def compute_margin(classifier, X):
            net_input = classifier.net_input(X)
            norm_w = np.linalg.norm(classifier.w_)
            if norm_w == 0:
                return np.nan
            return np.min(np.abs(net_input)) / norm_w

        margin_ppn = compute_margin(ppn, X_std)
        margin_ada = compute_margin(ada, X_std)


        # Print results
        # Loss
        #   Perceptron: # updates per epoch (step func)
        #   Adaline: Mean Squared Error (continious)
        # Margin
        # Convergence

        # Print Perceptron results
        print("= Perceptron =")
        print("Final weights:\t", ppn.w_)
        print("Final bias:\t", ppn.b_)
        print("Misclassifications in final epoch:\t", ppn.errors_[-1])
        print("Margin:\t{:.4f}".format(margin_ppn))
        print()

        # Print Adaline results
        print("= Adaline =")
        print("Final weights:\t", ada.w_)
        print("Final bias:\t", ada.b_)
        print("Final MSE:\t{:.4f}".format(ada.losses_[-1]))
        print("Margin:\t{:.4f}".format(margin_ada))
        print("--------------------------------------------------------\n")


        # Plot convergence curves
        plt.figure(figsize=(12, 10))

        # Perceptron: Plot number of updates per epoch
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Number of Updates")
        plt.title("Perceptron Convergence")

        # Adaline: Plot MSE per epoch
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Mean Squared Error")
        plt.title("Adaline Convergence")

        plt.suptitle("Epochs: {}    Learning rate: {}".format(n_iter, eta))

        plt.tight_layout()
        #plt.show()

        # Plot final regions
        def plot_decision_regions(X, y, classifier, resolution=0.02):
            from matplotlib.colors import ListedColormap

            # setup marker generator and color map
            markers = ('o', 's', '^', 'v', '<')
            colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
            cmap = ListedColormap(colors[:len(np.unique(y))])

            # plot decision surface
            x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))
            Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
            Z = Z.reshape(xx1.shape)
            plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
            plt.xlim(xx1.min(), xx1.max())
            plt.ylim(xx2.min(), xx2.max())

            # plot class examples
            for idx, cl in enumerate(np.unique(y)):
                plt.scatter(x=X[y == cl, 0],
                            y=X[y == cl, 1],
                            alpha=0.8,
                            c=colors[idx],
                            marker=markers[idx],
                            label=f'Class {cl}',
                            edgecolor='black')

        #plt.figure(figsize=(10, 4))

        plt.subplot(2, 2, 3)
        plot_decision_regions(X_std, y, classifier=ppn)
        plt.xlabel('Sepal length [cm]')
        plt.ylabel('Petal length [cm]')
        plt.title("Perceptron Decision Regions")
        plt.legend(loc="upper left")

        plt.subplot(2, 2, 4)
        plot_decision_regions(X_std, y, classifier=ada)
        plt.xlabel('Sepal length [cm]')
        plt.ylabel('Petal length [cm]')
        plt.title("Adaline Decision Regions")
        plt.legend(loc="upper left")
        #plt.suptitle("Epochs: {}    Learning rate: {}".format(n_iter, eta))


        plt.tight_layout()
        plt.show()