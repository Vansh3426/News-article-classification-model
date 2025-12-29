import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import spacy




df = pd.read_json("News-article-classification\dataset\dataset-news.json")



train = df["text"]
target = df["category"]


Xtrain ,Xtest , ytrain ,ytest  = train_test_split(train,target , test_size=0.2)


pipe =Pipeline([("vectorizer",CountVectorizer(ngram_range=(1,2))),
                ("model", MultinomialNB())])

model = pipe.fit(Xtrain , ytrain)

pred = pipe.predict(Xtest)

print(classification_report(ytest,pred))

# raw_text =[
#     "Elon musk buys twitter for 42 billion$",
#     "sunil chettri turns the match to his side by scoring 3 goals at second half",
#     "light takes 8 min to reach earth form the sun",
#     "A theft broke the glass and perform day time robbery in washington"
# ]
# pred =model.predict(raw_text)

# print(pred)