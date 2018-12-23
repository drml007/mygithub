# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#del[a,fulldataset,train,test]
import re
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline

os.getcwd()
os.chdir("D:\Analytics_Vidhya_Research\Twitter_Sentiment_Analysis")
os.getcwd()

train = pd.read_csv("train_E6oV3lV.csv")
train['source'] = 'train'
test = pd.read_csv("test_tweets_anuFYb8.csv")
test['source'] = 'test'

a = train.head()

fulldataset = pd.concat([train, test], axis = 0)

def pattern_replace(inp, pattern):
    r = re.findall(pattern, inp)
    for i in r:
        inp = re.sub(i,'',inp)
        
    return inp

# remove @user patterns from the twitter texts
    
fulldataset['clean_tweet'] = np.vectorize(pattern_replace)(fulldataset['tweet'],"@[\w]*")

# removing punctuations numbers and special characters
fulldataset['clean_tweet'] = fulldataset.clean_tweet.str.replace("[^a-zA-Z#]"," ")

# removing short words
fulldataset['clean_tweet'] = fulldataset['clean_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


#tokenization
tokenized_tweet = fulldataset['clean_tweet'].apply(lambda x: x.split())
tokenized_tweet[0]

# stemming
from nltk.stem.porter import *

stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
tokenized_tweet.head()

type(tokenized_tweet)


for i in range(len(tokenized_tweet)):
    s = ""
    for j in tokenized_tweet.iloc[i]:
        s += ''.join(j)+' '
    tokenized_tweet.iloc[i] = s.rstrip()



        

fulldataset['clean_tweet'] = tokenized_tweet

train_final = fulldataset.loc[fulldataset.source == 'train'].copy()
test_final = fulldataset.loc[fulldataset.source == 'test'].copy()

all_words = ' '.join([text for text in train_final['clean_tweet']])
from wordcloud import WordCloud


wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)  
    
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


normal_words =' '.join([text for text in train_final['clean_tweet'][train_final['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


negative_words =' '.join([text for text in train_final['clean_tweet'][train_final['label'] == 1]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()



# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags

# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(train_final['clean_tweet'][train_final['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(train_final['clean_tweet'][train_final['label'] == 1])


# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])



a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
train_bow = bow_vectorizer.fit_transform(train_final['clean_tweet'])
test_bow = bow_vectorizer.fit_transform(test_final['clean_tweet'])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
train_tfidf = tfidf_vectorizer.fit_transform(train_final['clean_tweet'])
test_tfidf = tfidf_vectorizer.fit_transform(test_final['clean_tweet'])


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'], random_state=42, test_size=0.3)
lreg = LogisticRegression()
lreg.fit(xtrain_bow, ytrain) # training the model
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.2 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int) # calculating f1 score


test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file


xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int)

test_pred = lreg.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_tf.csv', index=False) # writing data to a CSV file
