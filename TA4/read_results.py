from matplotlib import pyplot as plt
import os
import sys
import yaml
import json

if len(sys.argv) != 3:
    print("Usage: python read_results.py <results_json> <config_file>")
    sys.exit(1)

results_json_path = sys.argv[1]
config_file = sys.argv[2]
output = "./_plots"
os.makedirs(output, exist_ok=True)

config = yaml.safe_load(open(config_file, "r"))
results_json = json.load(open(results_json_path, "r"))

#### GERANDO GRÁFICOS LINEAR
NEIGHBORS = config.get('neighbors')
DISTANCE_METRIC = config.get('distance_metric')
TEST_SIZE = config.get('test_size')

train_sizes = [1 - ts for ts in TEST_SIZE]
linear_accs = [results_json[f"Linear {ts}"]["accuracy"] for ts in TEST_SIZE]

plt.figure()
plt.plot(train_sizes, linear_accs, marker='o', linestyle='-')
plt.xlabel('Train Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.savefig(os.path.join(output, 'linear_accuracy.png'))
plt.close()

#### GERANDO GRÁFICOS KNN
for distance_m in DISTANCE_METRIC:
    plt.figure()
    for n in NEIGHBORS:
        accs = [results_json['KNN'][f"{n}, {ts} {distance_m}"]["accuracy"] for ts in TEST_SIZE]
        plt.plot(train_sizes, accs, marker='o', linestyle='-', label=f'n={n}')
    plt.xlabel('Train Size')
    plt.ylabel('Accuracy')
    plt.title(f'KNN Accuracy ({distance_m})')
    plt.legend(title='Neighbors')
    plt.grid(True)
    fname = f'knn_accuracy_{distance_m.replace(" ", "_")}.png'
    plt.savefig(os.path.join(output, fname))
    plt.close()