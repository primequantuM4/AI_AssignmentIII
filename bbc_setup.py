import csv, random
from collections import defaultdict
import string
#define the y(labels) and the X(features)

def process_data(filename = 'bbc-text.csv'):
    X = []
    y = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)

        for row in csv_reader:
            label = row[0]
            text = row[1]

            X.append(text)
            y.append(label)

    return X, y

def refine_data(text:str):
    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()

    common_words = {'the', 'is', 'are','and','to','a'}
    words = [word for word in words if word not in common_words]

    processed_text = ''.join(words)

    return processed_text


def split_data(X, y, test_size=0.2):
    num_test_samples = int(len(X) * test_size)
    
    data = list(zip(X, y))
    random.shuffle(data)

    X_train = [sample[0] for sample in data[:-num_test_samples]]
    y_train = [sample[1] for sample in data[:-num_test_samples]]

    X_test = [sample[0] for sample in data[-num_test_samples:]]
    y_test = [sample[1] for sample in data[-num_test_samples:]]

    return X_train, y_train, X_test, y_test