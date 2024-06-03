import streamlit as st
import re
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier

with open('data.pickle', "rb") as file:
    data = pickle.load(file)
clf = pickle.load('clf.pickle')
count_vect = pickle.load('count_vect.pickle')
tfidf_transformer = pickle.load('tfidf_transformer.pickle')

st.title('Рекомендация статьи')

text_input = st.text_area("Введите название", "")

def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление пунктуации
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Простая токенизация с использованием регулярного выражения
    tokens = re.findall(r'\b\w+\b', text)
    # Склеивание токенов обратно в текст
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def for_sampling(docs):

  n_neighbors = 20
  ans = 0
  name = []
  # Формируем вектор признаков для нашего ввода
  X_new_counts = count_vect.transform(docs)
  # Мы вызываем transform вместо fit_transform, потому что он уже был обучен
  X_new_tfidf = tfidf_transformer.transform(X_new_counts)
  while ans == 0:
      
    distances, indices = clf.kneighbors(X_new_tfidf, n_neighbors=n_neighbors)
    closest_articles_indices = indices.flatten()
    closest_articles_distances = distances.flatten()
    
    counter = Counter(closest_articles_distances)
    for i in counter.most_common():
      if i[0] == 1:
        continue
      elif i[1] != 1:
        ans = i[0]
        break
      else:
        n_neighbors += 20
        ans = 0
        break
    if ans != 0:
      pos = np.where(closest_articles_distances==ans)
      name = closest_articles_indices[pos[0]]
      #name = data[name[0]]
  return name[0]

def give_link(name):
  return data[name][2]


if st.button("Рекомендовать"):
    # Предобработка введенного пользователем текста
    name = for_sampling(text_input)
    link = give_link(name)
    # Отображение предсказания
    st.write(f"Ссылка на статью: {link}")
