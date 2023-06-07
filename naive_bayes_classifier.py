from iris_classifier import *
from collections import defaultdict
import pprint
import math

class NaiveBayesClassifier:
    def __init__(self) -> None:
        self.classes = {}
        self.priors = {}
        self.means = defaultdict(list)
        self.variances = defaultdict(list)

    def train(self, X, y):
        total_samples = len(X)
        unique_classes = set(y)

        for cls in unique_classes:
            class_samples = [X[i] for i in range(total_samples) if y[i] == cls]
            self.classes[cls] = class_samples
            self.priors[cls] = len(class_samples) / total_samples

        for cls in self.classes:
            features =  zip(*self.classes[cls])

            for feature in features:
                mean = sum(feature) / len(feature)
                variance = sum([(x-mean) ** 2 for x in feature]) / len(feature)

                self.means[cls].append(mean)
                self.variances[cls].append(variance)

    def predict(self, X):
        predictions = []
        for features in X:
            posteriors = {}

            for cls in self.classes:
                posterior = self.priors[cls]

                for i, feature in enumerate(features):
                    mean = self.means[cls][i]
                    variance = self.variances[cls][i]
                    chance_of_occuring = self.calculate_gaussian_distribution(feature, mean, variance)
                    posterior *= chance_of_occuring

                posteriors[cls] = posterior

            predicted_class = max(posteriors, key=posteriors.get)
            predictions.append(predicted_class)

        return predictions

    def calculate_gaussian_distribution(self,x, mean, variance):
        exponent =  math.exp(-(x-mean) ** 2 / (2 * variance))
        return (1/ math.sqrt(2 * math.pi * variance)) * exponent
    


data_x, data_y = load_csv('iris .csv')
print(len(data_x), len(data_y))
data_x_train, data_y_train, data_x_test, data_y_test = split_data_set(data_x, data_y)

classifier = NaiveBayesClassifier()

classifier.train(data_x_train, data_y_train)


data_predictions = classifier.predict(data_x_test)
wrong_predictions = []
pp = pprint.PrettyPrinter(indent=4)

for predictions in range(len(data_predictions)):
    if data_predictions[predictions] != data_y_test[predictions]:
        wrong_predictions.append(f"{data_predictions[predictions]} and expected was {data_y_test[predictions]}") 

print(f"Accuracy: {100 - ((len(wrong_predictions) / len(data_predictions)) * 100)}%")
print("----------------------------------------------------------------------------")
pp.pprint(wrong_predictions)