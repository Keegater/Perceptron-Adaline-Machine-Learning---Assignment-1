import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from PerceptronMod import PerceptronMod

# =============================================================================
# Multi-class Perceptron using One-Vs-Rest strategy
# =============================================================================
class MultiClassPerceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.classifiers = {}  # dictionary to store one PerceptronMod per class
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            # For one-vs-rest: current class is 1, others 0.
            y_binary = np.where(y == cls, 1, 0)
            clf = PerceptronMod(eta=self.eta, n_iter=self.n_iter, random_state=self.random_state)
            clf.fit(X, y_binary)
            self.classifiers[cls] = clf
        return self

    def predict(self, X):
        # Compute net inputs for all classifiers. Each row corresponds to a class.
        net_inputs = np.array([clf.netinput(X) for clf in self.classifiers.values()])
        # Choose the class with the highest net input.
        indices = np.argmax(net_inputs, axis=0)
        classes = np.array(list(self.classifiers.keys()))
        return classes[indices]

# =============================================================================
# Load and prepare the Iris dataset (using only 2 features)
# =============================================================================
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Define a mapping from class names to numeric labels
label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
inv_map = {v: k for k, v in label_map.items()}
# Use only 2 features: sepal length (col 0) and petal length (col 2)
X = df.iloc[:, [0, 2]].values
y = df.iloc[:, 4].map(label_map).values  # Apply the mapping so that y is numeric

# Standardize features
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

# =============================================================================
# Train the multi-class perceptron on the two features
# =============================================================================
mcp = MultiClassPerceptron(eta=0.10, n_iter=50, random_state=1)
mcp.fit(X_std, y)
y_pred = mcp.predict(X_std)
accuracy = np.mean(y_pred == y)
print("Training accuracy with 2 features: {:.2f}%".format(accuracy * 100))

# =============================================================================
# Plotting decision regions for the 2-feature data
# =============================================================================
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # Setup marker generator and color map.
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Determine boundaries.
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # Predict class labels for all points in the meshgrid.
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Plot the decision surface.
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples.
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=inv_map[cl],
                    edgecolor='black')

plt.figure(figsize=(8, 6))
plot_decision_regions(X_std, y, classifier=mcp)
plt.xlabel('Sepal Length [standardized]')
plt.ylabel('Petal Length [standardized]')
plt.title('Multi-Classification Perceptron Decision Regions')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
for cls in mcp.classes_:
    clf = mcp.classifiers[cls]
    plt.plot(range(1, len(clf.errors)+1), clf.errors, marker='o', label=inv_map[cls])



plt.xlabel('Epochs')
plt.ylabel('Misclassifications')
plt.title('Convergence of Perceptrons')
plt.legend()
plt.tight_layout()
plt.show()
