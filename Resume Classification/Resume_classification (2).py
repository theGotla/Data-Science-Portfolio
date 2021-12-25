# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 19:05:00 2021

@author: Taronish
"""

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
import re

# In[2]:


df=pd.read_csv(r"C:/Users/Taro/OneDrive - purdue.edu/Documents/Purdue/AUD/Resume Classification/Final_Resume.csv")
df.columns 


# In[3]:


df['Category'].value_counts(dropna=True)
df.shape


# In[4]:


def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 = str1+" "+ele  
    
    # return string  
    return str1 
#Function to clean the resumes by removing stop words
def cleanResume(resumeText):
    resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    # resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText) 
    # resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
    resumeText = resumeText.replace('\s+', ' ')
    return resumeText

# In[76]:


df.head()
list1=df['Category'].value_counts().index
print(list1)

df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))

# In[6]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df['target'] = labelencoder.fit_transform(df['Category'])


# In[7]:


df['target'].value_counts().sort_index()
y=df[['target']]
X=df[['Resume']]


# In[8]:

#splitting into train and test(validation)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=13)


# In[9]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[10]:
####################3working on Train Data

#Transforming the data into training data 
#Tokenize the collection 
token_list=[]
for i in X_train['Resume']:
    token_list.append(nltk.word_tokenize(i))
#print(len(token_list))
#Lammetize the words 
lammetize_list=[]
for i in token_list:
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemmatized_token=[lemmatizer.lemmatize(token) for token in i]
    lammetize_list.append(lemmatized_token)
#Remove stop words
from nltk.corpus import stopwords
stop_list=[]
for i in lammetize_list:
    stop_words_removed = [token for token in i if not token in stopwords.words('english') if token.isalpha()]
    stop_list.append(stop_words_removed)


# Using the Count Vectoriser for creating a term document matrix 

final_list=[]
for i in stop_list:
    sentence_list=listToString(i)
    final_list.append(sentence_list)
#TFIDF min_df=3 and include 2-gram 
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(min_df=5)
v1=vectorizer.fit(final_list)
v2=vectorizer.transform(final_list)
print(len(v1.vocabulary_.keys()))
X_train=pd.DataFrame(v2.toarray(),columns=v1.vocabulary_.keys())
X_train.head()


# ## Preprocessing Done 

# ### Preparing the test data 



# In[12]:


#Transforming the data into testing data 
#Tokenize the collection 
token_list_test=[]
for i in X_test['Resume']:
    token_list_test.append(nltk.word_tokenize(i))
#print(len(token_list))
#Lammetize the words 
lammetize_list_test=[]
for i in token_list_test:
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemmatized_token=[lemmatizer.lemmatize(token) for token in i]
    lammetize_list_test.append(lemmatized_token)
#Remove stop words
from nltk.corpus import stopwords
stop_list_test=[]
for i in lammetize_list_test:
    stop_words_removed = [token for token in i if not token in stopwords.words('english') if token.isalpha()]
    stop_list_test.append(stop_words_removed)


# In[13]:


final_list_test=[]
for i in stop_list_test:
    sentence_list=listToString(i)
    final_list_test.append(sentence_list)
#Changing it w.r.t Tfidf vector 
v_test=vectorizer.transform(final_list_test)
X_test=pd.DataFrame(v_test.toarray(),columns=v1.vocabulary_.keys())
X_test.head()


# ### Model Development 

# ### Naive Bayes

# In[14]:


from sklearn.metrics import accuracy_score
## Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
# training
NBmodel.fit(X_train, Y_train)
y_pred_NB = NBmodel.predict(X_test)
# evaluation
acc_NB = accuracy_score(Y_test, y_pred_NB)
print("Naive Bayes model Accuracy:: {:.4f}%".format(acc_NB*100))
y_pred_prob_nb = NBmodel.predict_proba(X_test)
print(y_pred_prob_nb[0])


# ### Logistic Regression

# In[15]:


from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()
# training
Logitmodel.fit(X_train, Y_train)
y_pred_logit = Logitmodel.predict(X_test)
# evaluation
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(Y_test, y_pred_logit)
print("Logit model Accuracy:: {:.4f}%".format(acc_logit*100))
y_pred_prob_logit = Logitmodel.predict_proba(X_test)
print(y_pred_prob_logit[0])


# ### Decision Tree

# In[16]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score
DTmodel = DecisionTreeClassifier(min_samples_leaf=15,random_state=0) ## number of trees and number of layers/depth
#training
DTmodel.fit(X_train, Y_train)
y_pred_DT = DTmodel.predict(X_test)
#evaluation
acc_DT = accuracy_score(Y_test, y_pred_DT)
print("Decision Tree Model Accuracy: {:.4f}%".format(acc_DT*100))
y_pred_prob_dt = DTmodel.predict_proba(X_test)
print(y_pred_prob_dt[0])


# ### Random Forest Classifier 

# In[17]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RFmodel = RandomForestClassifier(n_estimators=100, max_depth=15, bootstrap=True, random_state=0) ## number of trees and number of layers/depth
#training
RFmodel.fit(X_train, Y_train)
y_pred_RF = RFmodel.predict(X_test)
#evaluation
acc_RF = accuracy_score(Y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.4f}%".format(acc_RF*100))
y_pred_prob_rf = RFmodel.predict_proba(X_test)
print(y_pred_prob_rf)


# ### Xgboost Classifier 

# In[18]:


from xgboost import XGBClassifier
xgmodel = XGBClassifier(random_state=0) ## number of trees and number of layers/depth
#training
xgmodel.fit(X_train, Y_train)
y_pred_xg = xgmodel.predict(X_test)
#evaluation
acc_xg = accuracy_score(Y_test, y_pred_xg)
print("Xgboost Classifier Model Accuracy: {:.4f}%".format(acc_xg*100))
y_pred_prob_xg = xgmodel.predict_proba(X_test)
print(y_pred_prob_xg[0])


# ### Light GBM Classifier 

# In[19]:


import lightgbm as lgb
lgbmodel = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=0) ## number of trees and number of layers/depth
#training
lgbmodel.fit(X_train, Y_train, verbose=20,eval_metric='logloss')
y_pred_lgb = lgbmodel.predict(X_test)
#evaluation
acc_lgb = accuracy_score(Y_test, y_pred_lgb)
print("Light GMB Classifier Model Accuracy: {:.4f}%".format(acc_lgb*100))
y_pred_prob_lgb = lgbmodel.predict_proba(X_test)
print(y_pred_prob_lgb[0])


# ### CAT Boost Classifier 

# In[20]:


from catboost import CatBoostClassifier
catmodel = CatBoostClassifier(random_state=0) ## number of trees and number of layers/depth
#training
catmodel.fit(X_train, Y_train, verbose=20)
y_pred_cat = catmodel.predict(X_test)
#evaluation
acc_cat = accuracy_score(Y_test, y_pred_cat)
print("Cat Boost Classifier Model Accuracy: {:.4f}%".format(acc_cat*100))
y_pred_prob_cat = catmodel.predict_proba(X_test)
print(y_pred_prob_cat[0])


# ### ANN

# In[43]:


## Neural Network and Deep Learning
from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(150,1200), random_state=1)
# training
DLmodel.fit(X_train, Y_train)
y_pred_DL= DLmodel.predict(X_test)
# evaluation
acc_DL = accuracy_score(Y_test, y_pred_DL)
print("DL model Accuracy: {:.4f}%".format(acc_DL*100))
y_pred_prob_dl = DLmodel.predict_proba(X_test)
print(y_pred_prob_dl[0])


# ### LSTM 

# In[92]:


#Splitting the data as this needs different inputs 
y_lstm=df[['Category']]
X_lstm=df[['Resume']]
from sklearn.model_selection import train_test_split
X_train_lstm,X_test_lstm,Y_train_lstm,Y_test_lstm=train_test_split(X_lstm,y_lstm,test_size=0.3,random_state=13)


# In[93]:


#encoding each label and transforming both train and test datasets 
from numpy import array
from sklearn.preprocessing import LabelEncoder
tokenized_list= [nltk.word_tokenize(doc.lower()) for doc in df['Resume']]
tokenized_list_test= [nltk.word_tokenize(doc.lower()) for doc in X_test_lstm['Resume']]
tokenized_list_train= [nltk.word_tokenize(doc.lower()) for doc in X_train_lstm['Resume']]
# A set for all possible words
words = [j for i in tokenized_list for j in i]
total_words=len(words)
index_encoder = LabelEncoder()
index_encoder = index_encoder.fit(words) # define vocabulary
X_train_lstm = [index_encoder.transform(doc) for doc in tokenized_list_train]
X_test_lstm = [index_encoder.transform(doc) for doc in tokenized_list_test]


# In[94]:


# build model
from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import Dense, Embedding, Flatten
from keras.layers import LSTM
max_features = total_words
maxlen = 2000
batch_size = 50
# padding
x_train_lstm = sequence.pad_sequences(X_train_lstm, maxlen=maxlen)
x_test_lstm = sequence.pad_sequences(X_test_lstm, maxlen=maxlen)
# model architecture
model = Sequential()
model.add(Embedding(max_features, 40, input_length=maxlen))
model.add(LSTM(1000, dropout=0.20, recurrent_dropout=0.20))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_lstm, Y_train_lstm, batch_size=batch_size, epochs=100, validation_data=(x_test_lstm, Y_test_lstm))


# In[ ]:





# ### Craiglist data - test data

# In[65]:


test=pd.read_csv(r"C:\Users\Taro\Desktop\Files\Resume\CraigList_Test.csv",dtype={'selection1_selection2':"category"})
test1['selection1_selection2']=test1['selection1_selection2'].astype('str')

test1['selection1_selection2'] = test1['selection1_selection2'].apply(lambda x: cleanResume(x))

# In[71]:


#Tokenize the collection 
token_test=[]
j=0
for i in test1['selection1_selection2']:
    token_test.append(nltk.word_tokenize(i))
#print(len(token_list))
#Lammetize the words 
lammetize_test=[]
for i in token_test:
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemmatized_token=[lemmatizer.lemmatize(token) for token in i]
    lammetize_test.append(lemmatized_token)
#Remove stop words
from nltk.corpus import stopwords
stop_test=[]
for i in lammetize_test:
    stop_words_removed = [token for token in i if not token in stopwords.words('english') if token.isalpha()]
    stop_test.append(stop_words_removed)
print(len(token_test))


# In[72]:


final_test=[]
for i in stop_test:
    sentence_list=listToString(i)
    final_test.append(sentence_list)
#Changing it w.r.t Tfidf vector 
v_test_c=vectorizer.transform(final_test)
test=pd.DataFrame(v_test_c.toarray(),columns=v1.vocabulary_.keys())
test.head()


# In[77]:


#Naive Bayes 
y_pred_prob_nb = NBmodel.predict_proba(test)
y_pred_prob_nb=pd.DataFrame(y_pred_prob_nb,columns=list1)
y_pred_prob_nb.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\y_pred_prob_nb.csv")


# In[78]:


#Logistic Regression 
y_pred_prob_log = Logitmodel.predict_proba(test)
y_pred_prob_log=pd.DataFrame(y_pred_prob_log,columns=list1)
y_pred_prob_log.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\y_pred_prob_log.csv")


# In[79]:


#Decision Tree
y_pred_prob_dt = DTmodel.predict_proba(test)
y_pred_prob_dt=pd.DataFrame(y_pred_prob_dt,columns=list1)
y_pred_prob_dt.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\y_pred_prob_dt.csv")


# In[80]:


#Random Forest
y_pred_prob_rf = RFmodel.predict_proba(test)
y_pred_prob_rf=pd.DataFrame(y_pred_prob_rf,columns=list1)
y_pred_prob_rf.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\y_pred_prob_rf.csv")


# In[81]:


#Xgboost 
y_pred_prob_xg = xgmodel.predict_proba(test)
y_pred_prob_xg=pd.DataFrame(y_pred_prob_xg,columns=list1)
y_pred_prob_xg.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\y_pred_prob_xg.csv")


# In[82]:


#CAT Boost
y_pred_prob_cat = catmodel.predict_proba(test)
y_pred_prob_cat=pd.DataFrame(y_pred_prob_cat,columns=list1)
y_pred_prob_cat.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\y_pred_prob_cat.csv")


# In[83]:


#Light GBM
y_pred_prob_lgb = lgbmodel.predict_proba(test)
y_pred_prob_lgb=pd.DataFrame(y_pred_prob_lgb,columns=list1)
y_pred_prob_lgb.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\y_pred_prob_lgb.csv")


# In[84]:


#ANN DLmodel
y_pred_prob_ann = DLmodel.predict_proba(test)
y_pred_prob_ann=pd.DataFrame(y_pred_prob_ann,columns=list1)
y_pred_prob_ann.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\y_pred_prob_ann.csv")


### TFIDF Vectorizer
#!/usr/bin/env python
# coding: utf-8

# In[75]:


import pandas as pd
import numpy as np
import nltk


# In[78]:
    
#####################################Same code but using TFIDF as vectorising method


df=pd.read_csv(r"C:\Users\Taro\Desktop\Files\Resume\Resume\Final_Resume.csv")
df.columns 
list1=df['Category'].value_counts().index
df.columns 


# In[18]:


df['Category'].value_counts(dropna=True)
df.shape


# In[19]:


def listToString(s): 
    
    # initialize an empty string
    str1 = "" 
    
    # traverse in the string  
    for ele in s: 
        str1 = str1+" "+ele  
    
    # return string  
    return str1 


# In[20]:


df.head()
df['Category'].value_counts()


# In[21]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df['target'] = labelencoder.fit_transform(df['Category'])


# In[22]:


df['target'].value_counts().sort_index()
y=df[['target']]
X=df[['Resume']]


# In[23]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=13)


# In[24]:


print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


# In[25]:


#Transforming the data into training data 
#Tokenize the collection 
token_list=[]
for i in X_train['Resume']:
    token_list.append(nltk.word_tokenize(i))
#print(len(token_list))
#Lammetize the words 
lammetize_list=[]
for i in token_list:
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemmatized_token=[lemmatizer.lemmatize(token) for token in i]
    lammetize_list.append(lemmatized_token)
#Remove stop words
from nltk.corpus import stopwords
stop_list=[]
for i in lammetize_list:
    stop_words_removed = [token for token in i if not token in stopwords.words('english') if token.isalpha()]
    stop_list.append(stop_words_removed)


# In[26]:


final_list=[]
for i in stop_list:
    sentence_list=listToString(i)
    final_list.append(sentence_list)
#TFIDF min_df=3 and include 2-gram 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(min_df=5)
v1=vectorizer.fit(final_list)
v2=vectorizer.transform(final_list)
print(len(v1.vocabulary_.keys()))
X_train=pd.DataFrame(v2.toarray(),columns=v1.vocabulary_.keys())
X_train.head()


# ## Preprocessing Done 

# ### Preparing the test data 

# In[27]:


#Transforming the data into training data 
#Tokenize the collection 
token_list_test=[]
for i in X_test['Resume']:
    token_list_test.append(nltk.word_tokenize(i))
#print(len(token_list))
#Lammetize the words 
lammetize_list_test=[]
for i in token_list_test:
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemmatized_token=[lemmatizer.lemmatize(token) for token in i]
    lammetize_list_test.append(lemmatized_token)
#Remove stop words
from nltk.corpus import stopwords
stop_list_test=[]
for i in lammetize_list_test:
    stop_words_removed = [token for token in i if not token in stopwords.words('english') if token.isalpha()]
    stop_list_test.append(stop_words_removed)


# In[28]:


final_list_test=[]
for i in stop_list_test:
    sentence_list=listToString(i)
    final_list_test.append(sentence_list)
#Changing it w.r.t Tfidf vector 
v_test=vectorizer.transform(final_list_test)
X_test=pd.DataFrame(v_test.toarray(),columns=v1.vocabulary_.keys())
X_test.head()


# ### Model Development 

# ### Naive Bayes

# In[29]:


from sklearn.metrics import accuracy_score
## Naive Bayes
from sklearn.naive_bayes import MultinomialNB
NBmodel = MultinomialNB()
# training
NBmodel.fit(X_train, Y_train)
y_pred_NB = NBmodel.predict(X_test)
# evaluation
acc_NB = accuracy_score(Y_test, y_pred_NB)
print("Naive Bayes model Accuracy:: {:.4f}%".format(acc_NB*100))
y_pred_prob_nb = NBmodel.predict_proba(X_test)
print(y_pred_prob_nb[0])


# ### Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression
Logitmodel = LogisticRegression()
# training
Logitmodel.fit(X_train, Y_train)
y_pred_logit = Logitmodel.predict(X_test)
# evaluation
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(Y_test, y_pred_logit)
print("Logit model Accuracy:: {:.4f}%".format(acc_logit*100))
y_pred_prob_logit = Logitmodel.predict_proba(X_test)
print(y_pred_prob_logit[0])


# ### Decision Tree

# In[31]:


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score
DTmodel = DecisionTreeClassifier(min_samples_leaf=15,random_state=0) ## number of trees and number of layers/depth
#training
DTmodel.fit(X_train, Y_train)
y_pred_DT = DTmodel.predict(X_test)
#evaluation
acc_DT = accuracy_score(Y_test, y_pred_DT)
print("Decision Tree Model Accuracy: {:.4f}%".format(acc_DT*100))
y_pred_prob_dt = DTmodel.predict_proba(X_test)
print(y_pred_prob_dt[0])


# ### Random Forest Classifier 

# In[41]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
RFmodel = RandomForestClassifier(n_estimators=300, max_depth=20, bootstrap=True, random_state=0) ## number of trees and number of layers/depth
#training
RFmodel.fit(X_train, Y_train)
y_pred_RF = RFmodel.predict(X_test)
#evaluation
acc_RF = accuracy_score(Y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.4f}%".format(acc_RF*100))
y_pred_prob_rf = RFmodel.predict_proba(X_test)
print(y_pred_prob_rf)


# ### Xgboost Classifier 

# In[33]:


from xgboost import XGBClassifier
xgmodel = XGBClassifier(random_state=0) ## number of trees and number of layers/depth
#training
xgmodel.fit(X_train, Y_train)
y_pred_xg = xgmodel.predict(X_test)
#evaluation
acc_xg = accuracy_score(Y_test, y_pred_xg)
print("Xgboost Classifier Model Accuracy: {:.4f}%".format(acc_xg*100))
y_pred_prob_xg = xgmodel.predict_proba(X_test)
print(y_pred_prob_xg[0])


# ### Light GBM Classifier 

# In[34]:


import lightgbm as lgb
lgbmodel = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=0) ## number of trees and number of layers/depth
#training
lgbmodel.fit(X_train, Y_train, verbose=20,eval_metric='logloss')
y_pred_lgb = lgbmodel.predict(X_test)
#evaluation
acc_lgb = accuracy_score(Y_test, y_pred_lgb)
print("Light GMB Classifier Model Accuracy: {:.4f}%".format(acc_lgb*100))
y_pred_prob_lgb = lgbmodel.predict_proba(X_test)
print(y_pred_prob_lgb[0])


# ### CAT Boost Classifier 

# In[35]:


from catboost import CatBoostClassifier
catmodel = CatBoostClassifier(random_state=0) ## number of trees and number of layers/depth
#training
catmodel.fit(X_train, Y_train, verbose=20)
y_pred_cat = catmodel.predict(X_test)
#evaluation
acc_cat = accuracy_score(Y_test, y_pred_cat)
print("Light GMB Classifier Model Accuracy: {:.4f}%".format(acc_cat*100))
y_pred_prob_cat = catmodel.predict_proba(X_test)
print(y_pred_prob_cat[0])


# ### ANN

# In[43]:


## Neural Network and Deep Learning
from sklearn.neural_network import MLPClassifier
DLmodel = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(150,1000), random_state=1)
# training
DLmodel.fit(X_train, Y_train)
y_pred_DL= DLmodel.predict(X_test)
# evaluation
acc_DL = accuracy_score(Y_test, y_pred_DL)
print("DL model Accuracy: {:.4f}%".format(acc_DL*100))
y_pred_prob_dl = DLmodel.predict_proba(X_test)
print(y_pred_prob_dl[0])


# ### LSTM 

# In[79]:


#Splitting the data as this needs different inputs 
y_lstm=df[['Category']]
X_lstm=df[['Resume']]
from sklearn.model_selection import train_test_split
X_train_lstm,X_test_lstm,Y_train_lstm,Y_test_lstm=train_test_split(X_lstm,y_lstm,test_size=0.3,random_state=13)


# In[80]:


#encoding each label and transforming both train and test datasets 
from numpy import array
from sklearn.preprocessing import LabelEncoder
tokenized_list= [nltk.word_tokenize(doc.lower()) for doc in df['Resume']]
tokenized_list_test= [nltk.word_tokenize(doc.lower()) for doc in X_test_lstm['Resume']]
tokenized_list_train= [nltk.word_tokenize(doc.lower()) for doc in X_train_lstm['Resume']]
# A set for all possible words
words = [j for i in tokenized_list for j in i]
total_words=len(words)
index_encoder = LabelEncoder()
index_encoder = index_encoder.fit(words) # define vocabulary
X_train_lstm = [index_encoder.transform(doc) for doc in tokenized_list_train]
X_test_lstm = [index_encoder.transform(doc) for doc in tokenized_list_test]


# In[82]:


# build model
from keras.preprocessing import sequence 
from keras.models import Sequential 
from keras.layers import Dense, Embedding, Flatten
from keras.layers import LSTM
max_features = total_words
maxlen = 2000
batch_size = 50
# padding
x_train_lstm = sequence.pad_sequences(X_train_lstm, maxlen=maxlen)
x_test_lstm = sequence.pad_sequences(X_test_lstm, maxlen=maxlen)
# model architecture
model = Sequential()
model.add(Embedding(max_features, 40, input_length=maxlen))
model.add(LSTM(1000, dropout=0.20, recurrent_dropout=0.20))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train_lstm, Y_train_lstm, batch_size=batch_size, epochs=100, validation_data=(x_test_lstm, Y_test_lstm))


# ### Cariglist Data 

# In[68]:


test=pd.read_csv(r"C:\Users\Taro\Desktop\Files\Resume\CraigList_Test.csv",dtype={'selection1_selection2':"category"})
test1=test.dropna()
test1['selection1_selection2']=test1['selection1_selection2'].astype('str')
test1.shape


# In[69]:


test1['selection1_selection2'].to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\Resume.csv")
test1['selection1_selection2'] = test1['selection1_selection2'].apply(lambda x: cleanResume(x))


# In[70]:


#Tokenize the collection 
token_test=[]
j=0
for i in test1['selection1_selection2']:
    token_test.append(nltk.word_tokenize(i))
#print(len(token_list))
#Lammetize the words 
lammetize_test=[]
for i in token_test:
    lemmatizer=nltk.stem.WordNetLemmatizer()
    lemmatized_token=[lemmatizer.lemmatize(token) for token in i]
    lammetize_test.append(lemmatized_token)
#Remove stop words
from nltk.corpus import stopwords
stop_test=[]
for i in lammetize_test:
    stop_words_removed = [token for token in i if not token in stopwords.words('english') if token.isalpha()]
    stop_test.append(stop_words_removed)
print(len(token_test))


# In[71]:


final_test=[]
for i in stop_test:
    sentence_list=listToString(i)
    final_test.append(sentence_list)
#Changing it w.r.t Tfidf vector 
v_test_c=vectorizer.transform(final_test)
test=pd.DataFrame(v_test_c.toarray(),columns=v1.vocabulary_.keys())
test.head()


# In[54]:


#Naive Bayes 
y_pred_prob_nb = NBmodel.predict_proba(test)
y_pred_prob_nb=pd.DataFrame(y_pred_prob_nb,columns=list1)
y_pred_prob_nb.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\TFIDF\y_pred_prob_nb.csv")


# In[55]:


#Logistic Regression 
y_pred_prob_log = Logitmodel.predict_proba(test)
y_pred_prob_log=pd.DataFrame(y_pred_prob_log,columns=list1)
y_pred_prob_log.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\TFIDF\y_pred_prob_log.csv")


# In[56]:


#Decision Tree
y_pred_prob_dt = DTmodel.predict_proba(test)
y_pred_prob_dt=pd.DataFrame(y_pred_prob_dt,columns=list1)
y_pred_prob_dt.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\TFIDF\y_pred_prob_dt.csv")


# In[57]:


#Random Forest
y_pred_prob_rf = RFmodel.predict_proba(test)
y_pred_prob_rf=pd.DataFrame(y_pred_prob_rf,columns=list1)
y_pred_prob_rf.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\TFIDF\y_pred_prob_rf.csv")


# In[58]:


#Xgboost 
y_pred_prob_xg = xgmodel.predict_proba(test)
y_pred_prob_xg=pd.DataFrame(y_pred_prob_xg,columns=list1)
y_pred_prob_xg.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\TFIDF\y_pred_prob_xg.csv")


# In[87]:


#CAT Boost
y_pred_prob_cat = catmodel.predict_proba(test)
y_pred_prob_cat=pd.DataFrame(y_pred_prob_cat,columns=list1)
f=test1.merge(y_pred_prob_cat,left_index=True,right_index=True,how="inner")
f.columns
#test.columns
f.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\TFIDF\f.csv")


# In[60]:


#Light GBM
y_pred_prob_lgb = lgbmodel.predict_proba(test)
y_pred_prob_lgb=pd.DataFrame(y_pred_prob_lgb,columns=list1)
y_pred_prob_lgb.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\TFIDF\y_pred_prob_lgb.csv")


# In[61]:


#ANN DLmodel
y_pred_prob_ann = DLmodel.predict_proba(test)
y_pred_prob_ann=pd.DataFrame(y_pred_prob_ann,columns=list1)
y_pred_prob_ann.to_csv(r"C:\Users\Taro\OneDrive - purdue.edu\Mod 2\Analyzing Unstructured Data\Group Project\Results\TFIDF\y_pred_prob_ann.csv")


# In[ ]:




