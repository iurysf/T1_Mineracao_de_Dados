import pandas as pd
import re
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
import joblib
import json
import ast
import numpy as np
import os

# 1. Carregar CSV
df = pd.read_csv('imdb_filmes.csv')
print(f"Dataset original: {df.shape[0]} filmes, {df.shape[1]} colunas")

# 2. Selecionar colunas úteis
df = df[['title', 'year', 'duration', 'rating', 'votes', 'budget', 'genres', 'languages']]

# 3. Funções de conversão (mantidas)
def converter_duracao(valor):
    """
    Converte uma string de duração para um número inteiro 
    representando o total de minutos.
    """
    try:
        if pd.isna(valor):
            return None
        h, m = 0, 0
        valor = str(valor).strip().lower()
        if 'h' in valor:
            partes = valor.split('h')
            h = int(partes[0].strip())
            if 'm' in partes[1]:
                m = int(partes[1].replace('m', '').strip())
        elif 'm' in valor:
            m = int(valor.replace('m', '').strip())
        return h * 60 + m
    except:
        return None
    
def converter_valor(valor):
    """
    Converte uma string que representa um valor numérico para 
    um float.
    """
    try:
        if pd.isna(valor):
            return None
        valor = str(valor).upper()
        valor_limpo = re.sub(r'[^0-9KM\.]', '', valor)
        match = re.match(r'^(\d*\.?\d*)([KM]?)$', valor_limpo)
        if not match:
            return None
        numero, multiplicador = match.groups()
        if numero == '':
            return None
        numero = float(numero)
        if multiplicador == 'K':
            return numero * 1_000
        elif multiplicador == 'M':
            return numero * 1_000_000
        else:
            return numero
    except:
        return None

def str_para_lista(s):
    """
    Converte de forma segura uma string que representa uma lista Python em um objeto de lista real.
    """
    try:
        if pd.isna(s):
            return []
        return ast.literal_eval(s)
    except:
        return []

# 4. Aplicar conversões e tratamento de dados faltantes

# Aplica a função 'converter_valor' em cada célula da coluna 'votes'. 250k
df['votes'] = df['votes'].apply(converter_valor)

# Aplica a mesma lógica para a coluna 'budget', convertendo valores como '$1.5M'
df['budget'] = df['budget'].apply(converter_valor)

# converte a duração para minutos
df['duration'] = df['duration'].astype(str).apply(converter_duracao)


# -------------------- tratamento de dados faltantes ------------------------

# remove as linhas onde as colunas 'year', 'rating' ou 'duration' não tem valor
df = df.dropna(subset=['year', 'rating', 'duration'])

# para a coluna 'votes', se algum valor for nulo após a conversão, preenchemos com 0
df['votes'] = df['votes'].fillna(0)

# para a coluna 'budget', se algum valor for nulo, preenchemos com a mediana de todos os orçamentos
df['budget'] = df['budget'].fillna(df['budget'].median())

# converte uma lista propria para o python
df['languages'] = df['languages'].apply(str_para_lista)

# aplica a mesma lógica para a coluna 'genres'
df['genres'] = df['genres'].apply(str_para_lista)

# tratamento de gêneros e idiomas vazios
filmes_sem_genero = df['genres'].apply(len) == 0
# a função .any() verifica se existe pelo menos um 'True' na máscara
if filmes_sem_genero.any():
    # Isso garante que todos os filmes tenham pelo menos um gênero, evitando problemas em etapas futuras
    df.loc[filmes_sem_genero, 'genres'] = df.loc[filmes_sem_genero, 'genres'].apply(lambda x: ['Unknown'])

# aplica a mesma lógica para a coluna de idiomas, colocando 'English'
filmes_sem_idioma = df['languages'].apply(len) == 0
if filmes_sem_idioma.any():
    df.loc[filmes_sem_idioma, 'languages'] = df.loc[filmes_sem_idioma, 'languages'].apply(lambda x: ['English'])


# --------- Processando Gêneros ------------

# A função .explode() transforma cada item de uma lista em uma nova linha. Ex: Filme com índice 10 e gêneros ['Action', 'Drama'] vira duas linhas
genres_exploded = df.explode('genres')

# pega a coluna de texto e cria uma nova coluna para cada genero e é preenchida com 1 ou 0
genres_dummies = pd.get_dummies(genres_exploded['genres'])

# agrupa as linhas pelo índice original do filme e soma os vetores para criar uma representação final com todos os gêneros de cada filme.
genres_dummies = genres_dummies.groupby(genres_exploded.index).sum()


# --- Processando Idiomas ---

# Repetimos exatamente o mesmo processo de 3 passos para a coluna 'languages'.
lang_exploded = df.explode('languages')
lang_dummies = pd.get_dummies(lang_exploded['languages'])
lang_dummies = lang_dummies.groupby(lang_exploded.index).sum()


# ------------------------ Montando o DataFrame Final para Modelagem ---------------------------

# agora colocamos essas novas colunas criadas na base de dados
df_processed = df.drop(columns=['genres', 'languages'])
df_processed = pd.concat([df.drop(columns=['genres', 'languages']), genres_dummies, lang_dummies], axis=1)


# salva cada indice que foi criado e o filme referente a ele
movie_titles = df_processed[['title']].to_dict()['title']

# define o alvo que queremos prever
y = (df_processed['rating'] >= 7).astype(int)

# 2. define os dados que usaremos para prever.
X = df_processed.drop(columns=['rating', 'title'])

# Imprime o número final de filmes que serão usados para o treinamento,
# após toda a limpeza e pré-processamento.
print(f"Dataset final: {df.shape[0]} filmes")

# imprime a distribuição das classes de sucesso e fracasso
print(f"Distribuição do target: Sucessos={y.sum()} ({y.mean():.1%}), Não sucessos={(y==0).sum()} ({(y==0).mean():.1%})")

# --------------------- Divisão dos Dados em Conjuntos de Treino e Teste ---------------------

# divide o dataset em duas partes: uma para treinar o modelo e outra para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Salvar os índices do conjunto de treino ---
# Estes são os índices originais do DataFrame que foram para o treino
train_indices = X_train.index.tolist()

# Escalonar os dados
scaler = StandardScaler()

# Define quais colunas do nosso DataFrame devem ser escalonadas.
colunas_para_escalar = ['year', 'duration', 'votes', 'budget']

# Cria uma lista das colunas que realmente existem no X_train
colunas_existentes = [col for col in colunas_para_escalar if col in X_train.columns]

# Cria cópias dos DataFrames de treino
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Converte para o tipo 'float64' para evitar erros no pandas
for col in colunas_existentes:
    X_train_scaled[col] = X_train_scaled[col].astype('float64')
    X_test_scaled[col] = X_test_scaled[col].astype('float64')

# --------------------- Aplicação do Escalonamento -----------------------

# 1. Ajustar e Transformar os Dados de Treino:
X_train_scaled.loc[:, colunas_existentes] = scaler.fit_transform(X_train[colunas_existentes])

# 2. Transformar os Dados de Teste:
X_test_scaled.loc[:, colunas_existentes] = scaler.transform(X_test[colunas_existentes])

# ===== TREINAMENTO E AVALIAÇÃO DE TODOS OS MODELOS =====

print(f"\n{'='*60}")
print("TREINANDO E AVALIANDO TODOS OS MODELOS NO CONJUNTO DE TESTE")
print(f"{'='*60}")

# Definir os modelos
modelos = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Dicionários para guardar os artefatos
modelos_treinados = {}
metricas = {}
feature_importances = {}

for nome, modelo in modelos.items():
    print(f"--- Treinando {nome} ---")
    
    # Treinar o modelo
    # compara os dados de treino com os dados corretos para buscar padrões
    modelo.fit(X_train_scaled, y_train)
    modelos_treinados[nome] = modelo
    
    # faz previsões em dados que ele nunca viu
    y_pred = modelo.predict(X_test_scaled)
    
    # Comparamos as previsões do y_pred com os resultados y_test e calculamos métricas de perfomance
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    
    # F1-Score: Uma média harmônica entre Precisão e Recall
    f1 = f1_score(y_test, y_pred)
    
    # Armazena as métricas calculadas em um dicionário, associadas ao nome do modelo.
    metricas[nome] = {'accuracy': acc, 'precision': prec, 'f1_score': f1}
    
    print(f"Modelo {nome} treinado e avaliado.")
    print(f"Acurácia: {acc:.3f} | Precisão: {prec:.3f} | F1-Score: {f1:.3f}")
    
    # Extrair Feature Importantes
    # Apenas para modelos baseados em árvores
    if hasattr(modelo, 'feature_importances_'):
        # Criar um dicionário mapeando o nome da feature à sua importância
        importances = dict(zip(X_train_scaled.columns, modelo.feature_importances_))
        feature_importances[nome] = importances
        print(f"Feature importances extraídas para {nome}.")
    
    print("-" * 50)


# ===== SALVANDO TODOS OS ARTEFATOS =====

print(f"\n{'='*40}")
print("SALVANDO TODOS OS ARTEFATOS")
print(f"{'='*40}")

# Criar um diretório para salvar os modelos se não existir
output_dir = "artefatos_modelo"
os.makedirs(output_dir, exist_ok=True)

# Salvar o dicionário de modelos treinados
caminho_modelos = os.path.join(output_dir, 'todos_os_modelos.joblib')
joblib.dump(modelos_treinados, caminho_modelos)
print(f"✔ Dicionário com todos os modelos treinados salvo em: {caminho_modelos}")

# Salvar o scaler
caminho_scaler = os.path.join(output_dir, 'scaler.joblib')
joblib.dump(scaler, caminho_scaler)
print(f"✔ Scaler salvo em: {caminho_scaler}")

# Salvar listas de features
generos = genres_dummies.columns.tolist()
idiomas = lang_dummies.columns.tolist()

caminho_generos = os.path.join(output_dir, 'generos_lista.json')
with open(caminho_generos, 'w') as f:
    json.dump(sorted(generos), f)
print(f"✔ Lista de gêneros salva em: {caminho_generos}")

caminho_idiomas = os.path.join(output_dir, 'idiomas_lista.json')
with open(caminho_idiomas, 'w') as f:
    json.dump(sorted(idiomas), f)
print(f"✔ Lista de idiomas salva em: {caminho_idiomas}")

# Salvar as métricas de performance
caminho_metricas = os.path.join(output_dir, 'metricas_modelos.json')
with open(caminho_metricas, 'w') as f:
    json.dump(metricas, f, indent=2)
print(f"✔ Métricas de performance salvas em: {caminho_metricas}")

caminho_importances = os.path.join(output_dir, 'feature_importances.json')
with open(caminho_importances, 'w') as f:
    json.dump(feature_importances, f, indent=2)
print(f"✔ Importância das features salva em: {caminho_importances}")

with open(os.path.join(output_dir, 'movie_titles.json'), 'w') as f:
    json.dump(movie_titles, f, indent=2)
print("✔ Dicionário de títulos de filmes salvo.")

with open(os.path.join(output_dir, 'train_indices.json'), 'w') as f:
    json.dump(train_indices, f)
print("✔ Índices do conjunto de treino salvos.")