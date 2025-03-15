import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = "/Users/maximcrucirescu/Desktop/dtu notes/sem 6/math_modelling/githubvenv/Project 2/Libian_desert_data.csv"
df = pd.read_csv(file_path)
df.columns = ["x", "y", "label"]


def apply_rotate(input_vector, angle):
    """Rotates the input vector by a given angle in radians."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, input_vector)


def apply_bias(input_vector, bias):
    """Adds a bias term to the input vector."""
    return input_vector + bias


def apply_activation(input_vector, activation='sigmoid'):
    """Applies the specified activation function."""
    if activation == 'abs':
        return np.abs(input_vector)
    elif activation == 'relu':
        return np.maximum(0, input_vector)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-input_vector))
    else:
        raise ValueError("Unsupported activation function")


class NeuralNetwork:
    def __init__(self, input_size, layers=8, activation='sigmoid', learning_rate=0.01):
        self.layers = layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.weights = [np.random.randn(2, 2) * 0.01 for _ in range(layers)]  # Smaller weights for stability
        self.biases = [np.random.randn(2) * 0.1 for _ in range(layers)]
        self.angles = [np.random.uniform(-np.pi, np.pi) for _ in range(layers)]

    def forward(self, x):
        """Performs a forward pass through the network."""
        for i in range(self.layers):
            x = apply_rotate(x, self.angles[i])
            x = apply_bias(x, self.biases[i])
            x = apply_activation(x, self.activation)
        return x[0]  # Ensure single output for classification

    def train(self, X, y, epochs=1000):
        """Trains the network using a simple gradient descent approach."""
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                x = X[i]
                target = y[i]
                output = self.forward(x)
                loss = (output - target) ** 2
                total_loss += loss

                # Compute gradient
                grad = 2 * (output - target)
                for j in range(self.layers):
                    self.biases[j] -= self.learning_rate * grad
                    self.weights[j] -= self.learning_rate * grad * np.outer(x, x)  # Weight update

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    def predict(self, X):
        """Predicts labels for input data."""
        return np.array([1 if self.forward(x) > 0.5 else 0 for x in X])


def plot_data(df):
    """Plots the dataset."""
    plt.figure(figsize=(8, 6))
    plt.scatter(df[df['label'] == 0]['x'], df[df['label'] == 0]['y'], label='Land', color='blue', alpha=0.5)
    plt.scatter(df[df['label'] == 1]['x'], df[df['label'] == 1]['y'], label='Sea', color='red', alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Libyan Desert Dataset")
    plt.show()


def evaluate_model(nn, X, y):
    """Evaluates the model accuracy."""
    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Prepare data for training
X = df[['x', 'y']].values
y = df['label'].values

# Example usage
if __name__ == "__main__":
    nn = NeuralNetwork(input_size=2, layers=8, activation='sigmoid', learning_rate=0.01)
    nn.train(X, y, epochs=1000)
    evaluate_model(nn, X, y)
    plot_data(df)
