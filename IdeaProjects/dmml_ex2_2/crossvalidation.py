import numpy as np  # for ndarray and numerical operations
from sklearn.base import BaseEstimator  # base class for classifiers and regressors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (LeaveOneOut, LeavePOut, RepeatedStratifiedKFold, StratifiedShuffleSplit,
                                     StratifiedKFold, cross_validate, _split)
from sklearn.tree import DecisionTreeClassifier
from rich.console import Console
from rich.table import Table


def train_test_split(X_data: np.ndarray, y_data: np.ndarray, test_size: float, seed: int = 2023) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Teilt die Daten in ein Train- und Testsatz. `test_size` gibt die Proportion des Testsatzes an"""
    rng = np.random.default_rng(seed)  # create a random number generator

    test_samples = round(len(X_data) * test_size)  # create an array which specifies which data to include into test
    # and which to data
    test_indices = np.zeros(len(X_data), dtype=bool)  # create an array of length of X_data initialized with False
    test_indices[:test_samples] = True
    rng.shuffle(test_indices)  # indices designated for the test set are randomized -> more robust randomization
    train_indices = ~test_indices  # create the inverse of test_indices specifying the indices to use for training

    return X_data[train_indices], X_data[test_indices], y_data[train_indices], y_data[test_indices]


def acc_kfold_cross_val(clf: BaseEstimator, cv: _split._BaseKFold, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Führt eine Kreuzvalidierung für den gegebenen Klassifizierer
    und die Kreuzvalidierungsmethode durch und gibt die durchschnittliche Trainings- und
    """
    scores = cross_validate(clf, X, y, cv=cv,
                            return_train_score=1)  # cv = Determines the cross-validation splitting strategy

    test_mean = scores["test_score"].mean()
    train_mean = scores["train_score"].mean()

    return test_mean, train_mean


# Definiert die Kreuzvalidierungsmethoden
cvs = [
    (StratifiedKFold(n_splits=10, shuffle=True, random_state=0), "KFold"),
    (
        RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=0),
        "Repeated KFold",
    ),
    (LeaveOneOut(), "Leave One Out"),
    (LeavePOut(p=5), "Leave P Out (p = 5)"),
    (
        StratifiedShuffleSplit(n_splits=10, test_size=0.33, random_state=0),
        "Shuffle Split",
    ),
]

# Output
console = Console()
# Classifiermodell
clf = DecisionTreeClassifier(random_state=0)


# Gibt die Genauigkeit fuer jede Method aus.
# 70/30 Split
def fit_and_score(clf: BaseEstimator, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
                  y_test: np.ndarray) -> tuple[float, float]:
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    return acc_test, acc_train


# Gibt das Trainings/Evaluierungsergebnis auf der Ausgabe aus",
def print_table(console: Console, name: str, mean_train: float, mean_test: float):
    table = Table(title=name)
    table.add_column("Train Accuracy Mean")
    table.add_column("Test Accuracy Mean")
    table.add_row(f"{mean_train * 100:3.2f} %", f"{mean_test * 100:3.2f} %")
    console.print(table)


