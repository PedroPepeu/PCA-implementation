import pandas as pd
from sklearn.datasets import load_iris

def load_iris_data(scale=True):
    """
    Carrega o dataset Iris e aplica pré-processamento básico
    
    Parâmetros:
    scale (bool): Se True, aplica padronização aos dados
    
    Retorna:
    tuple: (X, y, feature_names, target_names)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return X, y, iris.feature_names, iris.target_names

def load_custom_data(filepath, target_column=None, scale=True):
    """
    Carrega dados de um arquivo CSV
    
    Parâmetros:
    filepath (str): Caminho para o arquivo CSV
    target_column (str): Nome da coluna alvo (opcional)
    scale (bool): Se True, aplica padronização aos dados
    
    Retorna:
    tuple: (X, y, feature_names) ou (X, feature_names) se target_column for None
    """
    data = pd.read_csv(filepath)
    
    if target_column is not None:
        X = data.drop(target_column, axis=1).values
        y = data[target_column].values
        feature_names = list(data.drop(target_column, axis=1).columns)
    else:
        X = data.values
        y = None
        feature_names = list(data.columns)
    
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    return (X, y, feature_names) if target_column is not None else (X, feature_names)