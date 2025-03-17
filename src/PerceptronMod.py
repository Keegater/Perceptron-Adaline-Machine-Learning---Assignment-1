import numpy as np

class PerceptronMod:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1] + 1)  # Bias included
        self.errors = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w += update * np.insert(xi, 0, 1)  # Insert bias term
                errors += int(update != 0.0)
            self.errors.append(errors)
        return self

    def netinput(self, X):
        """Calculate net input"""
        if X.ndim == 1:
            X = np.insert(X, 0, 1)
        else:
            X = np.insert(X, 0, 1, axis=1)
        return np.dot(X, self.w)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.netinput(X) >= 0.0, 1, 0)