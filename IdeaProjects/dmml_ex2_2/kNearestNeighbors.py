import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(0)


def load_reduced_iris_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Lädt den Iris-Datensatz, der auf seine ersten beiden Merkmale reduziert
    wurde (Kelchblattbreite, Kelchblattlänge)."""
    iris = load_iris()
    return iris.data[:, :2], iris.target


def plot_iris_dataset(X: np.ndarray, y: np.ndarray) -> None:
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors="k")
    plt.title("Reduziertes Iris Dataset")
    plt.xlabel("Kelchblattbreite")
    plt.ylabel("Kelchblattlänge")
    plt.show()


X, y = load_reduced_iris_dataset()
# Iris Visualisierung
plot_iris_dataset(X, y)


def evaluate(k: int, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Führt eine 10-fache Kreuzvalidierung eines KNN-Modells
    durch und gibt die durchschnittliche Trainingsgenauigkeit
    und die durchschnittliche Testgenauigkeit zurück."""
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_validate(estimator=model, X=X, y=y, scoring="accuracy", cv=10, return_train_score=True)
    return np.mean(scores["train_score"]), np.mean(scores["test_score"])


def evaluate_ks(ks: range, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bewertet jeden k-Wert des ks-Arrays und gibt deren jeweilige Trainings- und Testgenauigkeit"""
    accuracies = np.array([evaluate(k, X, y) for k in
                           ks])  # accuracies is a list of tuples with mean train_score, mean test_score for all k in ks
    return accuracies[:, 0], accuracies[:, 1]  # return all training and testing results


def plot_k_to_acc(ks: range, acc_train: np.ndarray, acc_test: np.ndarray) -> None:
    """Plottet die k-Werte in Relation zu ihrer jeweiligen Trainings- und
    Testgenauigkeit."""
    plt.figure()
    plt.scatter(ks, acc_train, label="Accuracy on Training Data", marker="^")
    plt.scatter(ks, acc_test, label="Accuracy on Testing Data", marker="X")
    plt.xlabel("k Neighbors")
    plt.ylabel("Accuracy")
    plt.title("kNN: Accuracy depending on Number of Neighbors")
    plt.legend()
    plt.show()


def get_best_k(ks: range, acc_test: np.ndarray) -> int:
    """Basierend auf der höchsten Testgenauigkeit gibt diese
    Funktion den besten Wert für k zurück."""
    max_index = 0
    max_value = 0
    for i in range(0, len(ks)):
        if acc_test[i].max() > max_value:
            max_index = i
            max_value = acc_test[i].max()
    return ks[max_index]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier

# Cache to store pre-computed results (optional)
cache = {}

def plot_decision_boundary(model: KNeighborsClassifier, X: np.ndarray, y: np.ndarray) -> None:
    """Plot the decision boundary for a KNN model along with training data."""

    # Step size for the mesh grid
    h = 0.05

    # Color maps for the decision boundary and data points
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

    # Determine the min/max range for the plot, with some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create a mesh grid for plotting the decision boundary
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    k = model.n_neighbors

    # Check if results are already cached
    if k in cache:
        Z = cache[k]
    else:
        # Predict the outcome for each point in the mesh grid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        # Store the result in cache
        cache[k] = Z

    # Reshape the predictions to match the mesh grid
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)  # Background coloring for regions

    # Plot the training data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors="k", s=20)  # s=20 sets point size
    plt.title(f"2D KNN Decision Boundaries (k={k})")  # Title with dynamic k value

    # Set the plot limits
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Display the plot
    plt.show()


def plot_decision_boundary_for_k(k: int, X: np.ndarray, y: np.ndarray) -> None:
    """Erstellt und passt ein KNN-Modell mit dem Wert k an und plottet
    dessen Entscheidungsgrenze."""
    knn_classifier = KNeighborsClassifier(k)
    knn_classifier.fit(X, y)  # train the classifier
    plot_decision_boundary(knn_classifier, X, y)


# Auswertung der Trainings und Testgenauigkeit für verschiedene K-Werte
ks = range(1, 100, 1)
acc_train, acc_test = evaluate_ks(ks, X, y)
plot_k_to_acc(ks, acc_train, acc_test)
get_best_k(ks, acc_test)
for k in range(1, 101, 20):
    plot_decision_boundary_for_k(k, X, y)
