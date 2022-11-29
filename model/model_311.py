#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

from attacut import tokenize, Tokenizer

import umap
import umap.plot
from sklearn.datasets import load_digits

import tweepy

from sklearn.feature_extraction.text import CountVectorizer
import emoji
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords

from wordcloud import WordCloud
import matplotlib.pyplot as plt 


# ### Define Tokenizer Function

# In[3]:


def tokenizer_text(sentence):
    words = tokenize(sentence)
    space_sentence = " ".join(words)
    return space_sentence


# ### Preprocess

# In[33]:


#df = pd.read_excel('dataset314.xlsx')
df = pd.read_excel('C:\\Users\\jira\\เดสก์ท็อป\\dataset.xlsx', engine = 'openpyxl')
df_text = pd.read_excel('C:\\Users\\jira\\เดสก์ท็อป\\dataset.xlsx', engine = 'openpyxl', sheet_name='all')


# In[6]:


df.dtypes


# In[7]:


df['TSIC CODE'] = df['TSIC CODE'].astype('object')
df['ประกอบธุรกิจ'] = df['ประกอบธุรกิจ'].astype('string')
df['product_type'] = df['product_type'].astype('string')


# In[8]:


df = df.fillna('Unknown Unknown')
df


# In[368]:


df['x_roberta'] = df['ประกอบธุรกิจ'] +''+ df['product_type']
df['x_tfidf'] = df['x_roberta'].apply(tokenizer_text)
df['target'] = df['TSIC CODE'].astype('category').cat.codes


# In[369]:


df


# In[370]:


y = df['target'].values


# ### Embedding with TFIDF

# In[371]:


vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['x_tfidf'].values)


# In[372]:


X_tfidf.toarray()


# #### umap & plot 

# In[373]:


mapper = umap.UMAP().fit(X_tfidf)
umap.plot.points(mapper, labels=y)


# #### Plot interactive

# In[374]:


p = umap.plot.interactive(mapper, labels=y, hover_data=df, point_size=10)
umap.plot.show(p)


# ### Embedding with xlm-roberta-base 

# In[375]:


model = SentenceTransformer('xlm-roberta-base')


# In[376]:


X_roberta = model.encode(df['x_roberta'].values)


# In[377]:


X_roberta


# #### umap & plot 

# In[378]:


mapper = umap.UMAP().fit(X_roberta)
umap.plot.points(mapper, labels=y)


# #### Plot interactive

# In[379]:


p = umap.plot.interactive(mapper, labels=y, hover_data=df, point_size=10)
umap.plot.show(p)


# In[380]:


df_label = df[['TSIC CODE', 'target']]
df_label = df_label.rename(columns={'TSIC CODE': 'item', 'target': 'label' })
df_label


# In[43]:


def cleanText(text):
    text = str(text)
    stop_word = list(thai_stopwords())
    sentence = word_tokenize(text)
    result = [word for word in sentence if word not in stop_word and " " not in word]
    return text
cleaning = []
for txt in df_text["TEXT"]:
    cleaning.append(cleanText(txt))
cleaning[:10]


# In[44]:


df_cleaning = cleaning


# In[31]:


def cleanText(text):
    text = str(text)
    stop_word = list(thai_stopwords())
    sentence = word_tokenize(text, engine="multi_cut")
    # sentence = word_tokenize(text)
    result = [word for word in sentence if word not in stop_word and " " not in word]
    return ",".join(result)

def tokenize(d):  
    result = d.split(",")
    result = list(filter(None, result))
    return result

new_text = []
for txt in df_cleaning:
    new_text.append(cleanText(txt))


vectorizer = CountVectorizer(tokenizer=tokenize)
transformed_data = vectorizer.fit_transform(new_text)
count_data = zip(vectorizer.get_feature_names(), np.ravel(transformed_data.sum(axis=0)))
keyword_df = pd.DataFrame(columns = ['word', 'count'])
keyword_df['word'] = vectorizer.get_feature_names()
keyword_df['count'] = np.ravel(transformed_data.sum(axis=0))   
keyword_df.sort_values(by=['count'], ascending=False).head(10)


# In[ ]:




