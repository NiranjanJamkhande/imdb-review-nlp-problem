# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 23:53:38 2021

@author: Admin
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\Admin\Downloads\IMDB Dataset.csv', nrows = 2000)


import nltk
import re

nltk.download('stopwords')


!pip install wordcloud

from nltk.corpus import stopwords
stops = stopwords.words('english')

from wordcloud import STOPWORDS
stops = set(STOPWORDS)
stops.add("br")

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()

nltk.download('wordnet')


corpus = []
for i in range(0, 2000):
    review = re.sub('[^a-zA-Z]', ' ', df['review'][i])
    # meaning of above line is,...
    # by using re.sub, apart from a-z and A-Z, replace everything by ' ',
    # and do it for all sentences.
    # ^ says = not in or apart from
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stops)]
    review = ' '.join(review)
    corpus.append(review)
    
# Generating Word Cloud
corp_str = str(corpus)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wordcloud = WordCloud(relative_scaling=1.0).generate(corp_str)
plt.imshow(wordcloud)
plt.axis("off")
plt.show()



## creating TF- IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1]
y = pd.get_dummies(y, drop_first=True)


pickle.dump(cv, open('tranform.pkl', 'wb'))


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=99, stratify = y)



from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(random_state = 000, n_jobs = -1)
logreg.fit(X_train,y_train)

y_pred = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:,1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob))
###################################################################################

filename = 'nlp_model.pkl'
pickle.dump(logreg, open(filename, 'wb'))
