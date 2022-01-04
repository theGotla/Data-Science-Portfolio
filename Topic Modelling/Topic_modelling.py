# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 15:30:17 2022

@author: Taro
"""

import csv
from nltk.corpus import stopwords
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



file = pd.read_excel(r'C:/Users/Taro/OneDrive - purdue.edu/Desktop/Assignment 2/Assignment 2 text.xlsx',  names=['id', 'review','label'])



reviews = file['review'].tolist()
#Tokenizing the reviews
token_reviews = [nltk.word_tokenize(review.lower()) for review in reviews]
print(token_reviews)


stop_words_removed1 = [[token for token in token_doc if not token in stopwords.words('english') if token.isalpha()] for
                      token_doc in token_reviews]

#Lemmatizng the tokens
lemmatizer = WordNetLemmatizer()
lemmatized_token_reviews = [[lemmatizer.lemmatize(word) for word in token_doc] for token_doc in stop_words_removed1]
print(lemmatized_token_reviews)
#Removing all the stop words from the tokens
stop_words_removed = [[token for token in token_doc if not token in stopwords.words('english') if token.isalpha()] for
                      token_doc in lemmatized_token_reviews]
#Additional HTML Keywords that are removed
additional_stop = ['gt','lt','quot','li','ul','ol']
stop_words_removed2 = [[token for token in token_doc if not token in additional_stop if token.isalpha()] for
                      token_doc in stop_words_removed]
print(stop_words_removed)



#Untokenising the reveiws
untokenised_reviews = [" ".join(token_doc) for token_doc in stop_words_removed2]
# untokenised_reviews1 = [[" ".join(token_doc)] for token_doc in stop_words_removed]


#Using 2 grams wuth min doc freq as 3 for calculating Tf-Idf
vectorizer3 = TfidfVectorizer(ngram_range=(1, 2), min_df=5)

vectorizer3.fit(untokenised_reviews)
v3 = vectorizer3.transform(untokenised_reviews)
a = (v3.toarray())
terms = vectorizer3.get_feature_names()


#LDA for topic modelling
from sklearn.decomposition import LatentDirichletAllocation  
lda = LatentDirichletAllocation(n_components=6).fit(a) 
lda_output = lda.transform(a)

#Top 5 words in each topic
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx))
    print(" ".join([terms[i] for i in topic.argsort()[:-5-1:-1]]))  
    
#topic document matrix
topicnames = ['Topic    ' + str(i) for i in range(lda.n_components)]
docnames = ['Doc ' + str(i) for i in range(len(a))]
# FInding the probabilties of each topic to a corresponding document
df_document_topic = pd.DataFrame(np.round(lda_output, 3), columns=topicnames, index=docnames)
