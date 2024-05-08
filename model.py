from time import time
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import nltk
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

from sklearn.svm import LinearSVC
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier

# import matplotlib.pyplot as plt
# import seaborn as sns

def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split(' ') if word not in stop_words]
    return ' '.join(text)

def stemming(text):
    stemmer = PorterStemmer()
    text = [stemmer.stem(word) for word in text.split(' ')]
    return ' '.join(text)

def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = stemming(text)
    return text

dataset = pd.read_csv('spam.csv', encoding='ISO-8859-1', usecols=[0,1])
dataset.columns = ['label', 'message']
dataset['label'] = dataset['label'].map({'spam': 1, 'ham': 0})

dataset['message_cleaned'] = dataset['message'].apply(preprocess_text)
dataset['message_length'] = dataset['message'].apply(len)
dataset['has_punctuations'] = dataset['message'].apply(lambda x: any(char in string.punctuation for char in x))
dataset['has_uppercase'] = dataset['message'].apply(lambda x: any(char.isupper() for char in x))
dataset['has_lowercase'] = dataset['message'].apply(lambda x: any(char.islower() for char in x))
dataset['has_titlecase'] = dataset['message'].apply(lambda x: any(char.istitle() for char in x))
dataset['has_special_characters'] = dataset['message'].apply(lambda x: any(char in ['$', '#', '@'] for char in x))
dataset['has_numerical_digits'] = dataset['message'].apply(lambda x: any(char.isdigit() for char in x))
dataset['contains_discount'] = dataset['message_cleaned'].str.contains('discount', case=False).astype(int)
dataset['contains_free'] = dataset['message_cleaned'].str.contains('free', case=False).astype(int)
dataset['contains_please'] = dataset['message_cleaned'].str.contains('please', case=False).astype(int)
dataset['contains_account'] = dataset['message_cleaned'].str.contains('account', case=False).astype(int)
dataset['contains_customer'] = dataset['message_cleaned'].str.contains('customer', case=False).astype(int)
dataset['contains_card'] = dataset['message_cleaned'].str.contains('card', case=False).astype(int)
dataset['contains_email'] = dataset['message_cleaned'].str.contains('email', case=False).astype(int)
dataset['contains_details'] = dataset['message_cleaned'].str.contains('details', case=False).astype(int)
dataset['contains_update'] = dataset['message_cleaned'].str.contains('update', case=False).astype(int)
dataset['contains_online'] = dataset['message_cleaned'].str.contains('online', case=False).astype(int)
dataset['contains_bank'] = dataset['message_cleaned'].str.contains('bank', case=False).astype(int)
dataset['contains_link'] = dataset['message_cleaned'].str.contains('link', case=False).astype(int)
dataset['contains_refund'] = dataset['message_cleaned'].str.contains('refund', case=False).astype(int)
dataset['contains_due'] = dataset['message_cleaned'].str.contains('due', case=False).astype(int)

tfidf = TfidfVectorizer(max_features=1000, analyzer='word', ngram_range=(1,1))
tfidf_features = tfidf.fit_transform(dataset['message_cleaned']).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=tfidf.get_feature_names_out())
dataset = pd.concat([dataset, tfidf_df], axis=1)

dataset.drop(['message', 'message_cleaned'], axis=1, inplace=True)
X, y = dataset.drop('label', axis=1), dataset['label']

# Split training and texting data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234, 
    shuffle= True
)

# Build the models

# LinearSVC
svc = LinearSVC()

# Linear Regression
lr = LinearRegression()

# Random Forest
rf = RandomForestClassifier(n_estimators=500,
                            max_features=0.06,
                            n_jobs=6,
                            random_state=1234)

df_results = pd.DataFrame(columns=['F1_score', 'Precision', 'Recall', 'Accuracy', 'Execution time'])

models = [svc, lr, rf]
model_names = [i.__class__.__name__ for i in models]

start = timer()

for m, n in zip(models, model_names):
    start_time = time()
    m.fit(X_train, y_train)

    run_time = time() - start_time
    accuracy = np.mean(m.predict(X_test) == y_test)

    df_results.loc[n] = [None, None, None, accuracy, run_time]
    del m
print(df_results)

