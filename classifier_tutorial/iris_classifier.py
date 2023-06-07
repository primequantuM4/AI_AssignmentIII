import csv
import random

def load_csv(filename):
    with open(filename, 'r') as file:
        data = list(csv.reader(file))
        random.shuffle(data)
        X, y = [], []   
        
        for row in data:
            X.append(list(map(float, row[:-1])))
            y.append(row[-1])

            
    return X, y

def split_data_set(X, y, split_ratio=0.7):
    train_size = int(len(X) * split_ratio)
    X_train_data_set = X[:train_size]
    y_train_data_set = y[:train_size]

    X_test_data_set = X[train_size:]
    y_test_data_set = y[train_size:]

    return X_train_data_set, y_train_data_set, X_test_data_set, y_test_data_set


