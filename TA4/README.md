
# Documentação do Repositório de Classificação de Dígitos Manuscritos

Este repositório contém scripts para executar um experimento de classificação de dígitos manuscritos utilizando o dataset Digits do scikit-learn. O fluxo principal envolve:
1. Treinamento de um classificador KNN com diferentes métricas de distância e tamanhos de conjunto de teste.
2. Treinamento de um classificador linear (SGDClassifier) para comparação.
3. Armazenamento dos resultados em arquivos JSON.
4. Geração de gráficos de acurácia a partir dos resultados.

---

## Estrutura de Diretórios

```
computer_vision/
└── TA4/
    ├── config_file.yaml     # Arquivo de configuração YAML
    ├── main.py              # Script principal para executar os experimentos
    ├── read_results.py      # Script para ler o JSON de resultados e gerar gráficos
    ├── _results/            # Diretório onde os arquivos JSON de resultados serão salvos
    └── _plots/              # Diretório onde os gráficos gerados serão salvos
```

---

## 1. Como Executar

1. **Instalar Dependências**  
   ```bash
   pip install scikit-learn pyyaml matplotlib
   ```

2. **Executar Experimentos (main.py)**  
   ```bash
   python main.py --config config_file.yaml
   ```
   - Isso irá criar uma nova pasta em `_results/YYYY_MM_DD/` e gerar um arquivo JSON com os resultados.

3. **Gerar Gráficos (read_results.py)**  
   Após executar `main.py` e obter o JSON:
   ```bash
   python read_results.py _results/YYYY_MM_DD/HH_MM_SS.json config_file.yaml
   ```
   - Os gráficos serão salvos em `_plots/`.

---

## 2. Arquivo de Configuração (config_file.yaml)

```yaml
dest_dir: './_results'
neighbors: [1, 3, 5, 7, 9]
test_size: [0.3, 0.4, 0.5, 0.6, 0.7]
distance_metric: ['euclidean', 'manhattan', 'cosine']
```

- **dest_dir**: Caminho (relativo) para o diretório onde serão salvos os arquivos JSON de resultados.  
- **neighbors**: Lista de valores para o parâmetro *n\_neighbors* do KNN.  
- **test_size**: Lista de frações que determinam a proporção de dados reservados para o conjunto de teste (30%, 40%, 50%, 60% e 70%). O restante é utilizado no treino.  
- **distance_metric**: Lista de métricas de distância a serem utilizadas pelo KNN.

---

## 3. main.py

Este script realiza os seguintes passos:

1. **Leitura de Argumentos**  
   - Recebe via linha de comando o parâmetro `--config` apontando para o arquivo YAML de configuração.
   - Exemplo de execução:
     ```
     python main.py --config config_file.yaml
     ```

2. **Carregamento e Preparação do Dataset**  
   - Utiliza `load_digits()` do scikit-learn para carregar o dataset de dígitos manuscritos (1.797 amostras, 64 features cada).  
   - Separa as features (`x`) e os rótulos (`y`).

3. **Divisão Treino \& Teste**  
   - Para cada valor em `test_size` (0.3, 0.4, 0.5, 0.6, 0.7), chama:
     ```python
     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
     ```
   - Garante que `test_size = 0.3` → 30% das amostras para teste e 70% para treino; e assim por diante.

4. **Pré-processamento**  
   - Aplica `StandardScaler` em cada partição:
     ```python
     x_train_scaled = scaler.fit_transform(x_train)
     x_test_scaled  = scaler.fit_transform(x_test)
     ```
   - Obs.: Aqui o scaler é ajustado separadamente em treino e teste (para obter média zero e variância unitária em cada conjunto).

5. **Experimentos KNN**  
   - Para cada métrica em `distance_metric` e cada valor de `n` em `neighbors`, executa:
     ```python
     knn = KNeighborsClassifier(n_neighbors=n, metric=distance_m)
     knn.fit(x_train_scaled, y_train)
     y_pred = knn.predict(x_test_scaled)
     acc = accuracy_score(y_test, y_pred)
     ```
   - Armazena o resultado no objeto `json_obj['KNN'][f"{n}, {test_size} {distance_m}"]['accuracy']`.

6. **Classificador Linear (SGD)**  
   - Para cada `test_size`, além dos KNN, treina:
     ```python
     clf = SGDClassifier()
     clf.fit(x_train_scaled, y_train)
     y_pred = clf.predict(x_test_scaled)
     acc = accuracy_score(y_test, y_pred)
     ```
   - Armazena em `json_obj[f"Linear {test_size}"]['accuracy']`.

7. **Gerar Arquivo JSON de Resultados**  
   - O JSON é salvo em:
     ```
     _results/YYYY_MM_DD/HH_MM_SS.json
     ```
   - Onde `YYYY_MM_DD` e `HH_MM_SS` vêm da data e hora atuais.

---

## 4. read_results.py

Este script lê um arquivo JSON resultante do `main.py` e gera gráficos de acurácia:

1. **Uso**  
   ```
   python read_results.py <caminho_para_resultados.json> config_file.yaml
   ```
   - `<caminho_para_resultados.json>`: Caminho para o JSON gerado pelo `main.py`.
   - `config_file.yaml`: Mesmo arquivo de configuração para ler parâmetros (neighbors, test_size, distance_metric).

2. **Geração de Gráficos Linear**  
   - Calcula `train_sizes = [1 - ts for ts in TEST_SIZE]`.  
   - Obtém acurácias do classificador Linear:
     ```python
     linear_accs = [results_json[f"Linear {ts}"]["accuracy"] for ts in TEST_SIZE]
     ```
   - Plota `train_sizes` vs. `linear_accs` e salva em `_plots/linear_accuracy.png`.

3. **Geração de Gráficos KNN**  
   - Para cada métrica em `DISTANCE_METRIC`, plota uma curva para cada valor de `n` em `NEIGHBORS`:
     ```python
     accs = [results_json["KNN"][f"{n}, {ts} {distance_m}"]["accuracy"] for ts in TEST_SIZE]
     ```
   - Salva em `_plots/knn_accuracy_{distance_metric}.png`.

---

## 5. Observações Finais

- Caso queira alterar os parâmetros do experimento (valores de `neighbors`, `test_size` ou `distance_metric`), basta editar `config_file.yaml`.
- Os diretórios `_results/` e `_plots/` são criados automaticamente se não existirem.
- Verifique permissão de escrita nos diretórios para garantir que JSON e imagens sejam salvos corretamente.

---