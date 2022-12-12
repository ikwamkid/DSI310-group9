#!/usr/bin/env python
# coding: utf-8

# In[110]:


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

# In[111]:


def tokenizer_text(sentence):
    words = tokenize(sentence)
    space_sentence = " ".join(words)
    return space_sentence


# ### Preprocess

# In[129]:


df = pd.read_excel('C:\\Users\\jira\\เดสก์ท็อป\\dataset314.xlsx', engine = 'openpyxl')


# In[130]:


df


# In[114]:


df = df.fillna('-')


# In[115]:


df.info()


# In[116]:


#df['TSIC CODE'] = df['TSIC CODE'].astype('object')
df['ประกอบธุรกิจ'] = df['ประกอบธุรกิจ'].astype('string')
df['product_type'] = df['product_type'].astype('string')
df['กลุ่มผลิตภัณฑ์'] = df['กลุ่มผลิตภัณฑ์'].astype('string')
df['bussiness_type'] = df['bussiness_type'] .astype('string')


# In[117]:


df['details'] = df['ประกอบธุรกิจ'] +''+ df['product_type']+ df['กลุ่มผลิตภัณฑ์']+ df['bussiness_type']
df['All'] = df['details'].apply(tokenizer_text)
df['target'] = df['TSIC_Group'].astype('category').cat.codes


# In[118]:


y = df['target'].values


# ### Embedding with TFIDF

# In[119]:


vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['All'].values)


# In[120]:


X_tfidf.toarray()


# #### umap & plot 

# In[121]:


mapper = umap.UMAP().fit(X_tfidf)
#umap.plot.points(mapper, labels=y)
hover_df = pd.DataFrame(df, columns=['TSIC_Group'])
f = umap.plot.points(mapper, labels=hover_df['TSIC_Group'])


# In[122]:


p = umap.plot.interactive(mapper, labels=y[:30000],hover_data = df, point_size=5)
umap.plot.show(p)


# ### Embedding with xlm-roberta-base 

# In[123]:


model = SentenceTransformer('xlm-roberta-base')


# In[124]:


X_roberta = model.encode(df['details'].values)


# In[125]:


X_roberta


# #### umap & plot 

# In[126]:


mapper = umap.UMAP().fit(X_roberta)
#umap.plot.points(mapper, labels=y)
hover_df = pd.DataFrame(df, columns=['TSIC_Group'])
f = umap.plot.points(mapper, labels=hover_df['TSIC_Group'])


# #### Plot interactive

# In[127]:


p = umap.plot.interactive(mapper, labels=y[:30000],hover_data = df, point_size=5)
umap.plot.show(p)


# In[128]:


df_label = df[['TSIC_Group', 'target']]
df_label = df_label.rename(columns={'TSIC_Group': 'item', 'target': 'label' })
df_label


# In[ ]:




