import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

class PCAAnalyzer:
    def __init__(self, n_components=None):
        """
        Inicializa o analisador PCA
        
        Parâmetros:
        n_components (int): Número de componentes a manter
        """
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
    
    def fit_transform(self, X):
        """
        Ajusta o modelo PCA aos dados e aplica a transformação
        
        Parâmetros:
        X (array-like): Dados de entrada
        
        Retorna:
        array: Dados transformados
        """
        X_scaled = self.scaler.fit_transform(X)
        return self.pca.fit_transform(X_scaled)
    
    def plot_variance_explained(self, show_cumulative=True):
        """
        Plota a variância explicada por cada componente principal
        
        Parâmetros:
        show_cumulative (bool): Se True, mostra a variância acumulada
        """
        explained_var = self.pca.explained_variance_ratio_
        components = range(1, len(explained_var) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.bar(components, explained_var, alpha=0.6, label='Individual')
        
        if show_cumulative:
            cumulative = np.cumsum(explained_var)
            plt.step(components, cumulative, where='mid', 
                     label='Acumulada', color='red')
            plt.axhline(y=0.95, color='gray', linestyle='--')
            plt.text(x=1, y=0.96, s='95% limiar', color='gray')
        
        plt.xlabel('Componentes Principais')
        plt.ylabel('Variância Explicada')
        plt.title('Variância Explicada pelos Componentes Principais')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_components_heatmap(self, feature_names):
        """
        Plota um heatmap mostrando a contribuição das features originais
        
        Parâmetros:
        feature_names (list): Nomes das features originais
        """
        components = self.pca.components_
        df_components = pd.DataFrame(components, columns=feature_names)
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_components, cmap='coolwarm', annot=True, center=0)
        plt.title('Contribuição das Features Originais para os Componentes Principais')
        plt.xlabel('Features Originais')
        plt.ylabel('Componentes Principais')
        plt.show()
    
    def get_feature_contributions(self, feature_names):
        """
        Retorna um DataFrame com a contribuição de cada feature para os componentes
        
        Parâmetros:
        feature_names (list): Nomes das features originais
        
        Retorna:
        DataFrame: Contribuições das features
        """
        return pd.DataFrame(self.pca.components_, 
                          columns=feature_names,
                          index=[f'PC{i+1}' for i in range(self.pca.n_components_)])