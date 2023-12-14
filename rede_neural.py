# Importa as bibliotecas necessárias
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

# Carrega o conjunto de dados
df = pd.read_excel('base_train.xlsx')

# Dividir os dados em treino e teste
X = df["news"]
y = df["rotulo"]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=42)

# Cria o vetorizador de TF-IDF
tfidf = TfidfVectorizer(stop_words=["O", "A", "É", "DO", "DA", "DE", "EM", "PARA", "QUE", "SE", "QUEM", "COM", "COMO", "QUANDO", "ONDE", "PORQUÊ"],
                        ngram_range=(1, 2), encoding="utf-8")

# Transforma os dados em vetores TF-IDF
X_treino_tfidf = tfidf.fit_transform(X_treino)
X_teste_tfidf = tfidf.transform(X_teste)

# Treina o modelo (rede neural)
modelo = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
modelo.fit(X_treino_tfidf, y_treino)

# Avalia o modelo
predicoes_teste = modelo.predict(X_teste_tfidf)
acuracia_teste = accuracy_score(y_teste, predicoes_teste)

print("Acurácia no conjunto de teste:", acuracia_teste)

# Outras métricas
print("\nOutras métricas:")
print(classification_report(y_teste, predicoes_teste))

# Matriz de confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(y_teste, predicoes_teste))
