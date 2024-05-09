import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self, laplace=1):
        self.prior = {}
        self.likelihood = {}
        self.laplace = laplace

    def fit(self, X, y):
        # Calculate prior probabilities P(c)
        class_counts = y.value_counts().to_dict()
        total_samples = len(y)
        self.prior = {cls: count / total_samples for cls, count in class_counts.items()}

        # Filter features and prepare for likelihood calculation
        word_presence = X > 0
        # Calculate P(a|c) with Laplace smoothing
        for cls in np.unique(y):
            cls_samples = word_presence[y == cls]
            word_counts = cls_samples.sum(axis=0)
            total_words = cls_samples.sum().sum()
            # Apply Laplace smoothing
            self.likelihood[cls] = (word_counts + self.laplace) / (total_words + self.laplace * len(X.columns))

    def predict(self, X):
        # Use log probabilities to avoid underflow
        predictions = []
        word_presence = X > 0
        for _, sample in word_presence.iterrows():
            class_probs = {}
            for cls in self.prior:
                log_prob = np.log(self.prior[cls])
                log_likelihoods = np.log(self.likelihood[cls])
                # Sum log probabilities of words present in the email
                log_prob += (sample & log_likelihoods).sum()
                class_probs[cls] = log_prob
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions

class KNearestNeighborsOptimized:
    def __init__(self, K=3):
        self.K = K
        self.train_features = None
        self.train_labels = None

    def fit(self, X, y):
        self.train_features = X.values
        self.train_labels = y.values

    def predict(self, X):
        predictions = []
        test_points = X.values
        for test_point in test_points:
            distances = np.sqrt(np.sum((self.train_features - test_point) ** 2, axis=1))
            nearest_neighbor_idxs = np.argsort(distances)[:self.K]
            nearest_neighbor_labels = self.train_labels[nearest_neighbor_idxs]
            majority_vote = np.bincount(nearest_neighbor_labels).argmax()
            predictions.append(majority_vote)
        return predictions

class LogisticRegressionWithScaling:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Scale features
        self.means = X.mean()
        self.stds = X.std()
        X_scaled = (X - self.means) / self.stds
        
        # Initialize weights and bias
        self.weights = np.zeros(X_scaled.shape[1])
        self.bias = 0

        # Gradient descent
        for _ in range(self.num_iterations):
            z = np.dot(X_scaled, self.weights) + self.bias
            y_hat = 1 / (1 + np.exp(-z))
            dw = np.dot(X_scaled.T, (y_hat - y)) / len(y)
            db = np.sum(y_hat - y) / len(y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X_scaled = (X - self.means) / self.stds
        z = np.dot(X_scaled, self.weights) + self.bias
        y_hat = 1 / (1 + np.exp(-z))
        return [1 if prob > 0.5 else 0 for prob in y_hat]

def k_fold_cross_validation(data, K=5):
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    fold_size = len(data) // K
    for k in range(K):
        start, end = k * fold_size, (k + 1) * fold_size if (k + 1) * fold_size < len(data) else len(data)
        test_data = shuffled_data.iloc[start:end]
        train_data = shuffled_data.drop(test_data.index)
        yield train_data, test_data

def main():
    data = pd.read_csv('spambase.csv')
    models = {
        "Naive Bayes": NaiveBayesClassifier(),
        "KNN": KNearestNeighborsOptimized(K=3),
        "Logistic Regression": LogisticRegressionWithScaling(learning_rate=0.01, num_iterations=1000)
    }
    model_accuracies = {model_name: [] for model_name in models}
    
    fold_number = 0
    for train_set, test_set in k_fold_cross_validation(data, K=5):
        fold_number += 1
        train_features, train_labels = train_set.iloc[:, :-1], train_set.iloc[:, -1]
        test_features, test_labels = test_set.iloc[:, :-1], test_set.iloc[:, -1]
        
        print(f"Results for Fold {fold_number}:")
        for model_name, model in models.items():
            model.fit(train_features, train_labels)
            predictions = model.predict(test_features)
            accuracy = np.mean(predictions == test_labels)
            model_accuracies[model_name].append(accuracy)
            print(f"  Accuracy for {model_name}: {accuracy:.2f}")

    for model_name, accuracies in model_accuracies.items():
        average_accuracy = np.mean(accuracies)
        print(f'Average Accuracy for {model_name}: {average_accuracy:.2f}')

if __name__ == "__main__":
    main()
