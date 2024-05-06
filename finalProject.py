import pandas as pd
import numpy as np

# Helper Functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Logistic Regression Functions
def logistic_regression_train(X, y, learning_rate=0.01, n_iterations=1000):
    weights = np.zeros(X.shape[1])
    for i in range(n_iterations):
        scores = np.dot(X, weights)
        predictions = sigmoid(scores)
        errors = y - predictions
        gradient = np.dot(X.T, errors)
        weights += learning_rate * gradient
    return weights

def logistic_regression_predict(X, weights):
    scores = np.dot(X, weights)
    predictions = sigmoid(scores)
    return (predictions >= 0.5).astype(int)

# K-Nearest Neighbors Functions
def knn_train(X, y, k=3):
    return X, y, k

def knn_predict(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        distances = np.array([euclidean_distance(test_point, x) for x in X_train])
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_indices]
        prediction = np.bincount(k_nearest_labels).argmax()
        predictions.append(prediction)
    return np.array(predictions)

# Naive Bayes Functions
def calculate_prior(y_train):
    classes = np.unique(y_train)
    prior = np.zeros(len(classes))
    for i, c in enumerate(classes):
        prior[i] = np.sum(y_train == c) / len(y_train)
    return prior

def calculate_likelihood(X_train, y_train, epsilon=1e-9):
    n_features = X_train.shape[1]
    classes = np.unique(y_train)
    likelihood = np.zeros((len(classes), n_features))
    for i, c in enumerate(classes):
        X_train_c = X_train[y_train == c]
        mean = X_train_c.mean(axis=0)
        variance = X_train_c.var(axis=0)
        likelihood[i, :] = 1 / (np.sqrt(2 * np.pi * variance + epsilon)) * np.exp(-0.5 * (mean**2 / (variance + epsilon)))
    return likelihood

def naive_bayes_predict(X_test, prior, likelihood, classes):
    predictions = []
    for x in X_test:
        posteriors = []
        for i, c in enumerate(classes):
            posterior = np.log(prior[i]) + np.sum(np.log(likelihood[i]) * x)
            posteriors.append(posterior)
        predictions.append(classes[np.argmax(posteriors)])
    return np.array(predictions)

# Load and preprocess data
def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data):
    X = data.drop('spam', axis=1).values
    y = data['spam'].values
    return X, y

def split_data(X, y, test_size=0.3, random_state=42):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X))
    test_set_size = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def main():
    data_path = 'spambase.csv'
    data = load_data(data_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train Logistic Regression
    lr_weights = logistic_regression_train(X_train, y_train)
    lr_predictions = logistic_regression_predict(X_test, lr_weights)
    lr_accuracy = np.mean(lr_predictions == y_test)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.2f}")

    # Train K-Nearest Neighbors
    _, _, k = knn_train(X_train, y_train)
    knn_predictions = knn_predict(X_train, y_train, X_test, k)
    knn_accuracy = np.mean(knn_predictions == y_test)
    print(f"KNN Accuracy: {knn_accuracy:.2f}")

    # Train Naive Bayes
    classes = np.unique(y_train)
    prior = calculate_prior(y_train)
    likelihood = calculate_likelihood(X_train, y_train)
    nb_predictions = naive_bayes_predict(X_test, prior, likelihood, classes)
    nb_accuracy = np.mean(nb_predictions == y_test)
    print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")

if __name__ == "__main__":
    main()
