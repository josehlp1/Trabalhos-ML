import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

dados = pd.read_csv("data_preg.csv", header=None)
x_dados = dados[0].values
y_dados = dados[1].values

plt.scatter(x_dados, y_dados, color='blue', label='Dados Originais')

def criar_matriz_design(x, grau):
    return np.array([[xi**i for i in range(grau+1)] for xi in x])

def calcular_coeficientes_polinomiais(x, y, grau):
    X = criar_matriz_design(x, grau)
    coeficientes = np.linalg.solve(X.T @ X, X.T @ y)
    return coeficientes

def prever_valores(coeficientes, x):
    y_pred = np.array([sum(c * xi ** i for i, c in enumerate(coeficientes)) for xi in x])
    return y_pred

for grau, cor in zip([1, 2, 3, 8], ['red', 'green', 'black', 'yellow']):
    coeficientes = calcular_coeficientes_polinomiais(x_dados, y_dados, grau)
    y_pred = prever_valores(coeficientes, x_dados)
    plt.plot(x_dados, y_pred, color=cor, label=f'Grau={grau}')
    print(f'EQM para Grau={grau}:', mean_squared_error(y_dados, y_pred))

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regressão Polinomial')
plt.legend()
plt.grid(True)
plt.show()

x_teste, y_teste = x_dados[:int(len(x_dados)*0.1)], y_dados[:int(len(y_dados)*0.1)]
x_treino, y_treino = x_dados[int(len(x_dados)*0.1):], y_dados[int(len(y_dados)*0.1):]

plt.scatter(x_treino, y_treino, color='blue', label='Dados de Treinamento')

for grau, cor in zip([1, 2, 3, 8], ['red', 'green', 'black', 'yellow']):
    coeficientes = calcular_coeficientes_polinomiais(x_treino, y_treino, grau)
    y_treino_pred = prever_valores(coeficientes, x_treino)
    plt.plot(x_treino, y_treino_pred, color=cor, label=f'Grau={grau}')
    print(f'EQM para Grau={grau} (treinamento):', mean_squared_error(y_treino, y_treino_pred))

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regressão Polinomial')
plt.legend()
plt.grid(True)
plt.show()

for grau in [1, 2, 3, 8]:
    coeficientes = calcular_coeficientes_polinomiais(x_treino, y_treino, grau)
    y_teste_pred = prever_valores(coeficientes, x_teste)
    print(f'EQM para Grau={grau} (teste):', mean_squared_error(y_teste, y_teste_pred))

for grau in [1, 2, 3, 8]:
    coeficientes = calcular_coeficientes_polinomiais(x_treino, y_treino, grau)
    y_treino_pred = prever_valores(coeficientes, x_treino)
    y_teste_pred = prever_valores(coeficientes, x_teste)
    print(f'R² para Grau={grau} (treino):', r2_score(y_treino, y_treino_pred))
    print(f'R² para Grau={grau} (teste):', r2_score(y_teste, y_teste_pred))
