import pandas as pd 
import numpy as np 
import spacy
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report



true_csv = pd.read_csv("News-article-classification\dataset\True.csv")
fake_csv = pd.read_csv("News-article-classification\dataset\Fake.csv")

true_csv["label"] = 1
fake_csv["label"] =0

# print(true_csv.shape)
# print(fake_csv.shape)
# print(fake_csv[:5])

mix_csv  = pd.concat((true_csv[:2000],fake_csv[:2000]),axis=0)
# print(mix_csv[:5])


main_csv= shuffle(mix_csv, random_state=42).reset_index(drop=True)
# print(main_csv[:5])

nlp =spacy.load("en_core_web_md")

df = pd.DataFrame()

# print(df)

df['vector'] = main_csv["text"].apply(lambda x :nlp(x).vector)


train = df["vector"]
target = main_csv["label"]


Xtrain ,Xtest , ytrain ,ytest  = train_test_split(train,target, test_size=0.2)

Xtrain_1 = np.stack(Xtrain)
Xtest_1 = np.stack(Xtest)


pipe =Pipeline([("Scaling",MinMaxScaler()),
                ("model", MultinomialNB())])


pipe.fit(Xtrain_1,ytrain)

pred = pipe.predict(Xtest_1)



print(classification_report(ytest,pred))