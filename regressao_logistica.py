import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

df = pd.read_excel('base_train.xlsx')

# #trás a palavra até a raiz dela ex; atuar, atuando e etc

X = df['news'].values
Y = df['rotulo'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

X_train,X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=42)

lr = LogisticRegression()
lr.fit(X_train,Y_train)

pred = lr.predict(X_test)

acc = accuracy_score(Y_test,pred)

print(f'Acuracia: {acc*100:.2f}')