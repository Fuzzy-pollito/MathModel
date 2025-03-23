import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    X, y = df.iloc[:, :2].values, df.iloc[:, 2].values  # First two columns as features, last column as labels
    return X, y


# Preprocess data
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Define neural network model
def create_model(hidden_layers):
    return MLPClassifier(hidden_layer_sizes=hidden_layers, activation='relu', solver='adam', max_iter=1000,
                         random_state=42)


# Train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test, hidden_layers):
    model = create_model(hidden_layers)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return model, acc


# Plot decision boundary
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.show()


# Main function
def main():
    # Load and split data
    filepath = '/mnt/data/Libian_desert_data.csv'  # Use correct file path
    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train models
    deep_model, deep_acc = train_and_evaluate(X_train, X_test, y_train, y_test, (8, 8, 8, 8))
    shallow_model, shallow_acc = train_and_evaluate(X_train, X_test, y_train, y_test, (4, 4))

    print(f"Deep Model Accuracy: {deep_acc:.2f}")
    print(f"Shallow Model Accuracy: {shallow_acc:.2f}")

    # Plot decision boundaries
    plot_decision_boundary(deep_model, X, y, "Deep Model Decision Boundary")
    plot_decision_boundary(shallow_model, X, y, "Shallow Model Decision Boundary")


if __name__ == "__main__":
    main()