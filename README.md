# PCA-implementation

Estrutura de Arquivos para um Projeto de PCA em Python

Aqui está uma estrutura de arquivos organizada para seu projeto de cadeira de inteligência artificial, focando na implementação do PCA:

'''
/projeto_pca/
│
├── pca/
│   ├── __init__.py             # Tornando o diretório um pacote Python
│   ├── my_pca.py               # Nossa implementação customizada do PCA
│   └── utils.py                # Funções auxiliares (opcional)
│
├── notebooks/
│   ├── 01-pca_implementation.ipynb  # Notebook explorando nossa implementação
│   └── 02-sklearn_comparison.ipynb  # Notebook comparando com scikit-learn
│
├── tests/
│   ├── __init__.py
│   └── test_my_pca.py          # Testes unitários para nossa implementação
│
├── data/
│   └── iris.csv                # Dataset de exemplo (opcional)
│
├── docs/
│   └── explanation.md          # Documentação/explicação do projeto
│
├── requirements.txt            # Dependências do projeto
└── README.md                   # Descrição geral do projeto
'''

Detalhamento dos Arquivos Principais
1. pca/my_pca.py (Implementação principal)
python

import numpy as np

class MyPCA:
    """Implementação customizada do PCA."""
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
    
    def fit(self, X):
        # Centralização dos dados
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # Cálculo da matriz de covariância
        cov_matrix = np.cov(X_centered.T)
        
        # Decomposição em autovalores e autovetores
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Ordenação dos componentes
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Armazenamento dos resultados
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = eigenvalues[:self.n_components] / eigenvalues.sum()
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

2. tests/test_my_pca.py (Testes unitários)
python

import numpy as np
from pca.my_pca import MyPCA
from sklearn.datasets import load_iris

def test_pca_implementation():
    # Carrega dados de teste
    X, _ = load_iris(return_X_y=True)
    
    # Testa nossa implementação
    my_pca = MyPCA(n_components=2)
    X_my_pca = my_pca.fit_transform(X)
    
    # Verifica formas básicas
    assert X_my_pca.shape == (X.shape[0], 2)
    assert my_pca.components.shape == (X.shape[1], 2)
    assert len(my_pca.explained_variance) == 2
    
    # Verifica que a transformação reduz a dimensionalidade
    assert np.allclose(X_my_pca[:, 0], np.dot(X - my_pca.mean, my_pca.components[:, 0]))

3. notebooks/01-pca_implementation.ipynb (Jupyter Notebook de exemplo)
python

# Célula 1: Importações
import numpy as np
import matplotlib.pyplot as plt
from pca.my_pca import MyPCA
from sklearn.datasets import load_iris

# Célula 2: Carregar dados
data = load_iris()
X = data.data
y = data.target

# Célula 3: Aplicar nosso PCA
pca = MyPCA(n_components=2)
X_pca = pca.fit_transform(X)

# Célula 4: Visualização
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1 (%.2f%%)' % (pca.explained_variance_ratio[0]*100))
plt.ylabel('PC2 (%.2f%%)' % (pca.explained_variance_ratio[1]*100))
plt.title('PCA Implementado Manualmente')
plt.colorbar()
plt.show()

4. requirements.txt

numpy>=1.20.0
scikit-learn>=1.0.0
matplotlib>=3.0.0
jupyter>=1.0.0
pytest>=7.0.0

5. README.md
markdown

# Projeto de Implementação do PCA

Implementação didática do algoritmo PCA (Principal Component Analysis) para a disciplina de Inteligência Artificial.

## Como usar

1. Clone o repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute os notebooks Jupyter ou os testes unitários

## Estrutura do Projeto

- `pca/`: Implementação do algoritmo PCA
- `notebooks/`: Análises exploratórias e comparações
- `tests/`: Testes unitários da implementação

Como Executar o Projeto

    Crie um ambiente virtual (recomendado):

bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

    Instale as dependências:

bash

pip install -r requirements.txt

    Execute os testes:

bash

pytest tests/

    Explore os notebooks Jupyter:

bash

jupyter notebook notebooks/

Esta estrutura oferece uma base sólida para seu projeto, permitindo tanto o desenvolvimento da implementação do PCA quanto a experimentação e comparação com bibliotecas estabelecidas como o scikit-learn.
