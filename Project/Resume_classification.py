# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:32:16 2021

@author: Taro
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
nltk.download('perluniprops')
# from nltk.tokenize import MosesDetokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

Resume_data = pd.read_csv(r'C:/Users/Taro/OneDrive - purdue.edu/Documents/Purdue/AUD/Project/UpdatedResumeDataSet.csv')

# Resume_data.Category.value_counts()
# sns.countplot(y="Category", data=Resume_data)

Resume_data['Category'].value_counts().sort_index().plot(kind='bar', figsize=(12, 6))
plt.show()



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



# Resume_data['cleaned_resume'] = Resume_data.Resume.apply(lambda x: nltk.word_tokenize(x)  )
Resume_data['Resume1'] = Resume_data['Resume'].apply(lambda x: cleanResume(x))
Resume_data['Resume'] = Resume_data['Resume'].str.encode('ascii', 'ignore').str.decode('ascii')


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
Resume_data['text_lemmatized'] = Resume_data['Resume'].apply(lemmatize_text)
Resume_data['text_lemmatized'] = Resume_data.text_lemmatized.apply(lambda x: " ".join(x))
Resume_data['text_lemmatized'] = Resume_data['text_lemmatized'].str.lower()
X = Resume_data['text_lemmatized']
y = Resume_data['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

word_vectorizer = TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=5000,ngram_range=(1, 2), min_df=2)
word_vectorizer.fit(X_train)
WordFeatures_train = word_vectorizer.transform(X_train)
WordFeatures_test = word_vectorizer.transform(X_test)
Train_transformed = pd.DataFrame(WordFeatures_train.toarray())
Test_transformed = pd.DataFrame(WordFeatures_test.toarray())


from sklearn.ensemble import RandomForestClassifier
RFmodel = RandomForestClassifier(n_estimators=250, max_depth=25,bootstrap=True, random_state=0)

# RFmodel = RFmodel(RFmodel, n_jobs=-1)
# RFmodel.fit(X, Y).predict(X)

RFmodel.fit(Train_transformed, y_train)
y_pred_RF = RFmodel.predict(Test_transformed)
#Give Probability for each class
probability_class_1 = RFmodel.predict_proba(Test_transformed)[:, ]


from sklearn.metrics import accuracy_score 
acc_RF = accuracy_score(y_test, y_pred_RF)
print("Random Forest Model Accuracy: {:.2f}%".format(acc_RF*100))



# Logit
from sklearn.linear_model import LogisticRegression 
Logitmodel = LogisticRegression()
# training
Logitmodel.fit(Train_transformed, y_train)
y_pred_logit = Logitmodel.predict(Test_transformed)
# evaluation
from sklearn.metrics import accuracy_score
acc_logit = accuracy_score(y_test, y_pred_logit)
print("Logit model Accuracy:: {:.2f}%".format(acc_logit*100))


from xgboost import XGBClassifier
xgmodel = XGBClassifier(random_state=0) ## number of trees and number of layers/depth
#training
xgmodel.fit(Train_transformed, y_train)
y_pred_xg = xgmodel.predict(Test_transformed)
y_pred_xg_train = xgmodel.predict(Train_transformed)
#evaluation
acc_xg = accuracy_score(y_test, y_pred_xg)
acc_xg_train = accuracy_score(y_train, y_pred_xg_train)
print("Xgboost Classifier Model Accuracy: {:.4f}%".format(acc_xg*100))
print("Xgboost Classifier Model Accuracy: {:.4f}%".format(acc_xg_train*100))
y_pred_prob_xg = xgmodel.predict_proba(X_test)
print(y_pred_prob_xg[0])


#################SfBay craiglist resume testing


SfData = pd.read_csv(r'C:/Users/Taro/Downloads/run_results_sfbay.csv')


SfData['selection1_selection2'] = SfData['selection1_selection2'].astype('str')
SfData['selection1_selection2'] = SfData['selection1_selection2'].apply(lambda x: cleanResume(x))
SfData['selection1_selection2'] = SfData['selection1_selection2'].str.encode('ascii', 'ignore').str.decode('ascii')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
SfData['text_lemmatized'] = SfData['selection1_selection2'].apply(lemmatize_text)
SfData['text_lemmatized'] = SfData.text_lemmatized.apply(lambda x: " ".join(x))
SfData['text_lemmatized'] = SfData['text_lemmatized'].str.lower()
Sf = SfData['text_lemmatized']
WordFeatures_Sf = word_vectorizer.transform(Sf)

Sf_transformed = pd.DataFrame(WordFeatures_Sf.toarray())
Sf_pred_RF = RFmodel.predict(Sf_transformed)
#probabilties 
probability_Sf = pd.DataFrame(RFmodel.predict_proba(Sf_transformed)[:, ])