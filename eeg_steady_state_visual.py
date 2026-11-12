# -*- coding: utf-8 -*-
"""EEG steady-state visual signal analysis with local, runnable Python code."""

from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, RANSACRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def find_dataset(base_dir: Path) -> Path | None:
    preferred = [
        base_dir / "A001SB1_1.csv",
        base_dir / "data.csv",
        base_dir / "eeg.csv",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    csv_files = sorted(base_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    return None


def make_synthetic_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = 200
    data = pd.DataFrame(rng.normal(size=(rows, 8)))
    data[0] = (
        0.6 * data[1]
        + 0.3 * data[2]
        - 0.4 * data[3]
        + 0.2 * data[4]
        + rng.normal(0, 0.1, rows)
    )
    return data


def load_dataset() -> pd.DataFrame:
    base_dir = Path(__file__).resolve().parent
    dataset_path = find_dataset(base_dir)

    if dataset_path is not None:
        print(f"Using dataset: {dataset_path.name}")
        df = pd.read_csv(dataset_path, header=None)
    else:
        print("No .csv file found in the project directory. Using a synthetic dataset instead.")
        df = make_synthetic_dataset()

    if df.empty or df.shape[1] < 2:
        raise ValueError("The dataset must contain at least two columns.")

    return df


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if df.isna().any().any():
        imputer = SimpleImputer(strategy="median")
        df = pd.DataFrame(imputer.fit_transform(df), columns=range(df.shape[1]))

    target = df[0].astype(float)
    features = df.drop(columns=[0])

    if features.empty:
        raise ValueError("No feature columns remain after dropping the target column.")

    return features, target


# Load and inspect the data
raw_df = load_dataset()
features, target = prepare_data(raw_df)

print("Dataset shape:", raw_df.shape)
print("Missing values:", int(raw_df.isna().sum().sum()))

scatter_matrix = pd.plotting.scatter_matrix(
    features.iloc[:, : min(4, features.shape[1])],
    alpha=0.8,
    figsize=(10, 10),
    diagonal="hist",
)
plt.tight_layout()
plt.savefig("pairplot.png", dpi=150)
plt.close()

corr_matrix = features.corr()
print("Correlation matrix:\n", corr_matrix.head())
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.close()

# Regression task
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.3, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Linear Regression slope:", lr.coef_[:5])
print("Linear Regression intercept:", lr.intercept_)

train_predictions = lr.predict(X_train)
test_predictions = lr.predict(X_test)
print("Train MSE:", mean_squared_error(y_train, train_predictions))
print("Train R2:", r2_score(y_train, train_predictions))
print("Test R2:", r2_score(y_test, test_predictions))

plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Regression prediction quality")
plt.tight_layout()
plt.savefig("regression_scatter.png", dpi=150)
plt.close()

ran = RANSACRegressor(estimator=LinearRegression())
ran.fit(X_train, y_train)
print("RANSAC train R2:", r2_score(y_train, ran.predict(X_train)))
print("RANSAC test R2:", r2_score(y_test, ran.predict(X_test)))

# Classification task using the same features
class_target = np.where(target >= target.median(), "high", "low")
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    features, class_target, test_size=0.3, random_state=42
)

perceptron = Perceptron(max_iter=1000, random_state=42)
perceptron.fit(Xc_train, yc_train)
print("Perceptron train accuracy:", accuracy_score(yc_train, perceptron.predict(Xc_train)))
print("Perceptron test accuracy:", accuracy_score(yc_test, perceptron.predict(Xc_test)))

logistic = LogisticRegression(max_iter=1000, random_state=42)
logistic.fit(Xc_train, yc_train)
print("Logistic regression train accuracy:", accuracy_score(yc_train, logistic.predict(Xc_train)))
print("Logistic regression test accuracy:", accuracy_score(yc_test, logistic.predict(Xc_test)))

classifier = DecisionTreeClassifier(criterion="entropy", random_state=42)
classifier.fit(Xc_train, yc_train)
print("Decision tree train accuracy:", accuracy_score(yc_train, classifier.predict(Xc_train)))
print("Decision tree test accuracy:", accuracy_score(yc_test, classifier.predict(Xc_test)))

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=min(2, features.shape[1]))),
    ("svm", SVC(kernel="rbf", random_state=42)),
])
pipe.fit(Xc_train, yc_train)
print("SVM train accuracy:", accuracy_score(yc_train, pipe.predict(Xc_train)))
print("SVM test accuracy:", accuracy_score(yc_test, pipe.predict(Xc_test)))

param_grid = [
    {"kernel": ["linear"], "C": [0.01, 0.1, 1, 10, 100]},
    {"kernel": ["poly"], "C": [0.01, 0.1, 1, 10, 100]},
    {"kernel": ["rbf"], "C": [0.01, 0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10, 100]},
]

search = GridSearchCV(SVC(), param_grid=param_grid, scoring="accuracy", cv=4)
search.fit(Xc_train, yc_train)
print("Best SVC parameters:", search.best_params_)
print("Best SVC CV accuracy:", search.best_score_)

print("Script completed successfully.")

