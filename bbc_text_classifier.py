from bbc_setup import *
import pprint
import math

class NaiveBayes:
    def __init__(self) -> None:
        self.class_count = defaultdict(int)
        self.class_probabilities = {}
        self.feature_counts = {}


    def train(self, X_train, y_train):
        for label in y_train:
            self.class_count[label] += 1

        for features, label in zip(X_train, y_train):
            if label not in self.feature_counts:
                self.feature_counts[label] = [0] * len(features)

                for i, feature in enumerate(features):
                    if feature == 1:
                        self.feature_counts[label][i] += 1

        total_sample = len(y_train)
        for label, count in self.class_count.items():
            self.class_probabilities[label] = count / total_sample


    def predict(self,X_test):
        predictions = []

        for features in X_test:
            class_scores = {}

            for cls in self.class_count:
                score = 0
                
                for i, feature in enumerate(features):
                    if feature == 1:
                        feature_count = self.feature_counts[cls][i]
                        feature_probability = (feature_count + 1) / (self.class_count[cls] + 2)
                        #added the probabilities with log probability so it doesn't under perform
                        score += math.log(feature_probability)

                score += math.log(self.class_probabilities[cls])
                class_scores[cls] = score


            prediction = max(class_scores, key=class_scores.get)
            predictions.append(prediction)
            
        return predictions
    





X, y = process_data()
X = [refine_data(text) for text in X]

X_train, y_train, X_test, y_test = split_data(X, y)
classifier = NaiveBayes()

classifier.train(X_train, y_train)

predictions = classifier.predict(X_test)


wrong_predicitions = []
pp = pprint.PrettyPrinter(indent=4)


for i in range(len(predictions)):
    if predictions[i] != y_test[i]:
        wrong_predicitions.append(f"{predictions[i]} was expected to be {y_test[i]}")

print(len(X),len(y_test))
print(f"Accuracy: {100 -((len(wrong_predicitions) / len(y_test)) * 100)} %")
print('------------------------------------------------------------------- ')
pp.pprint(wrong_predicitions)