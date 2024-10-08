import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
from sklearn.metrics import zero_one_loss
from sklearn.inspection import DecisionBoundaryDisplay

# Define your Gaussian Bayes Classifier
class GaussianBayesClassifier:
    def __init__(self):
        """Initialize the Gaussian Bayes Classifier"""
        self.pY = []  # class prior probabilities, p(Y=c)
        self.pXgY = []  # class-conditional probabilities, p(X|Y=c)
        self.classes_ = []  # list of possible class values

    def fit(self, X, y):
        """Fits a Gaussian Bayes classifier with training features X and training labels y."""
        from sklearn.mixture import GaussianMixture
        self.classes_ = np.unique(y)  # Identify the class labels
        for c in self.classes_:
            self.pY.append(np.mean(y == c))  # Estimate p(Y=c) (a float)
            model_c = GaussianMixture(1)  # Gaussian for p(X|Y=c)
            model_c.fit(X[y == c, :])  # Fit a Gaussian for class c
            self.pXgY.append(model_c)  # Store Gaussian model for each class

    def predict(self, X):
        """Makes predictions with the Gaussian Bayes classifier on the features in X."""
        pXY = np.stack([np.exp(p.score_samples(X)) for p in self.pXgY]).T
        pXY *= np.array(self.pY).reshape(1, -1)  # Multiply p(X=x|Y=c) by p(Y=c)
        pYgX = pXY / pXY.sum(1, keepdims=True)  # Normalize to p(Y=c|X=x)
        return self.classes_[np.argmax(pYgX, axis=1)]  # Return class with max probability


# Load the NYC housing data from the provided URL
url = 'https://ics.uci.edu/~ihler/classes/cs178/data/nyc_housing.txt'
with requests.get(url) as link:
    datafile = StringIO(link.text)
    nych = np.genfromtxt(datafile, delimiter=',')
    nych_X, nych_y = nych[:, :-1], nych[:, -1]  # Features and labels

# Plot the decision boundary for the Gaussian Bayes Classifier
def plot_decision_boundary(X, y, learner, plot_title="Gaussian Bayes Classifier"):
    # Some keyword arguments for making nice looking plots.
    plot_kwargs = {'cmap': 'jet',     # Another option: 'viridis'
                   'response_method': 'predict',
                   'plot_method': 'pcolormesh',
                   'shading': 'auto',
                   'alpha': 0.5,
                   'grid_resolution': 100}

    # Create a plot
    figure, axes = plt.subplots(1, 1, figsize=(4, 4))

    # Get just the first two features of X (for 2D plotting)
    X2 = X[:, :2]

    # Fit the learner on the two-feature dataset
    learner.fit(X2, y)

    # Predict labels for the dataset
    y_pred = learner.predict(X2)

    # Evaluate error
    err = zero_one_loss(y, y_pred)
    print(f'Error Rate (0/1): {err}')

    # Plot decision boundary
    DecisionBoundaryDisplay.from_estimator(learner, X2, ax=axes, **plot_kwargs)

    # Scatter plot the actual data points
    axes.scatter(X2[:, 0], X2[:, 1], c=y, edgecolor=None, s=12)

    # Set plot title
    axes.set_title(plot_title)

    # Show plot
    plt.show()


# Instantiate the Gaussian Bayes Classifier
learner = GaussianBayesClassifier()

# Plot the decision boundary for the Gaussian Bayes Classifier
plot_decision_boundary(nych_X, nych_y, learner)
