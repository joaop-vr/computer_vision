import yaml
import json
import os
import argparse
import itertools
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import SGDClassifier
from datetime import datetime
import matplotlib.pyplot as plt

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration YAML file", required=True)
    args = parser.parse_args()

    config = args.config
    try:
        with open(config, "r") as file:
            config = yaml.safe_load(file)
    except Exception as ex:
        print("Failed to open configuration file:")
        print(ex)
        exit()

    return config 

def getOutputName(dest_dir):
    today = datetime.today()
    today_date = today.strftime('%Y_%m_%d')
    today_time = today.strftime('%H_%M_%S')

    dest_dir = os.path.join(dest_dir, today_date)
    os.makedirs(dest_dir, exist_ok=True)

    json_name = f'{today_time}.json'
    json_output = os.path.join(dest_dir, json_name)

    return json_output

class KNN:
    def __init__(self, x_train_scaled, x_test_scaled, y_train, y_test): 
        self.x_train = x_train_scaled
        self.x_test = x_test_scaled
        self.y_train = y_train
        self.y_test = y_test

    def knn(self, n: int, json_obj: dict):
        knn = KNeighborsClassifier(n_neighbors = n)
        knn.fit(self.x_train, self.y_train)

        y_pred = knn.predict(self.x_test)

        acc = accuracy_score(self.y_test, y_pred)

        print("Accuracy: ", acc)
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))

        json_obj['accuracy'] = acc

def main():
    config = arguments()

    dest_dir = os.path.abspath(config.get('dest_dir')) + '/'
    json_output = getOutputName(dest_dir)
    json_obj = {}

    json_obj['KNN'] = {}
    json_obj['KNN']['parameters'] = None

    digits = load_digits()
    x = digits.data
    y = digits.target

    scaler = StandardScaler()

    N = config.get('neighbors')
    sizes = config.get('test_size')
    states = config.get('random_state')

    idx = 0
    for test_size in sizes:
        for random_state in states:
            if random_state:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.fit_transform(x_test)

            # KNN

            knn = KNN(x_train_scaled, x_test_scaled, y_train, y_test)
            for n in N:
                print(f"Experiment {idx}:\nNeighbors: {n}\n Test Size\n {test_size}\n Random State: {random_state}\n")

                json_obj['KNN'][f'{n}, {test_size}, {random_state}'] = {
                        'accuracy': None
                        }

                knn.knn(n, json_obj['KNN'][f'{n}, {test_size}, {random_state}'])
                idx += 1

            # LINEAR

            json_obj['Linear'] = {}

            clf = SGDClassifier(random_state=42)
            clf.fit(x_train_scaled, y_train)

            y_pred = clf.predict(x_test_scaled)
            acc = accuracy_score(y_test, y_pred)

            json_obj['Linear']['accuracy'] = acc

    json_out = json.dumps(json_obj, indent=4)
    with open(json_output, 'w') as file:
        file.write(json_out)

if __name__=="__main__":
    main()
