import yaml
import argparse
import itertools
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
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

class KNN:
    def __init__(self, n: int, test_size: float, random_state: bool):
        self.n = n
        self.test_size = test_size
        self.random = random_state

    def knn(self):
        digits = load_digits()
        x = digits.data
        y = digits.target

        #plt.imshow(digits.images[0], cmap='gray')
        #plt.title(f"Label: {digits.target[0]}")
        #plt.show()

        if self.random:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size, random_state=42)
        else:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_size)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.fit_transform(x_test)

        knn = KNeighborsClassifier(n_neighbors = self.n)
        knn.fit(x_train_scaled, y_train)

        y_pred = knn.predict(x_test_scaled)

        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # inserir os resultados em um csv

def main():
    config = arguments()

    dest_file = config.get('dest_file')

    csv_file = open(dest_file, 'w', newline='')
    csv_writer = cvs.writer(csvi_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    N = [1, 2, 3, 4, 5]
    sizes = [0.1, 0.2, 0.3]
    states = [True, False]

    id = 0
    for n, test_size, random_state in itertools.product(N, sizes, states):
        print(f"Experiment {id}:\nNeighbors: {n}\n Test Size\n {test_size}\n Random State: {random_state}\n")
        knn = KNN(n, test_size, random_state)
        knn.knn()
        id += 1


if __name__=="__main__":
    main()
