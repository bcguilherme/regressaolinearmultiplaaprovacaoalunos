# regressaolinearmultiplaaprovacaoalunos

# Regressão Linear Múltipla para Aprovação de Alunos

Este repositório contém um notebook de análise de dados que explora a relação entre as aprovações de alunos em anos consecutivos e realiza uma regressão linear múltipla para prever as aprovações em um ano com base nas aprovações do ano anterior.

## Conteúdo

- [Passo 1: Carregando Bibliotecas e Dados](#passo-1-carregando-bibliotecas-e-dados)
- [Passo 2: Exploração Inicial dos Dados](#passo-2-exploração-inicial-dos-dados)
- [Passo 3: Visualização dos Dados](#passo-3-visualização-dos-dados)
- [Passo 4: Normalização dos Dados](#passo-4-normalização-dos-dados)
- [Passo 5: Regressão Linear](#passo-5-regressão-linear)
- [Passo 6: Dividindo os Dados para Treino e Teste](#passo-6-dividindo-os-dados-para-treino-e-teste)
- [Passo 7: Treinamento e Avaliação do Modelo](#passo-7-treinamento-e-avaliação-do-modelo)

## Passo 1: Carregando Bibliotecas e Dados

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
alunos = pd.read_excel('/content/C+¦pia de aprovacao_alunos.xlsx')
alunos.head()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Inicializar o modelo de regressão linear
lr = LinearRegression()

# Dividir os dados em conjuntos de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# Treinar o modelo com os dados de treino
lr.fit(x_treino, y_treino)

# Avaliar o desempenho do modelo nos dados de treino e teste
r_sq = lr.score(x_treino, y_treino)
mae_treino = metrics.mean_absolute_error(y_treino, y_pred_treino)
mse_treino = metrics.mean_squared_error(y_treino, y_pred_treino)
rmse_treino = np.sqrt(mse_treino)

mae_teste = metrics.mean_absolute_error(y_teste, y_pred_teste)
mse_teste = metrics.mean_squared_error(y_teste, y_pred_teste)
rmse_teste = np.sqrt(mse_teste)

