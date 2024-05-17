from time import time
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import nltk

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import re
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, f1_score, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import pickle

# Define functions for text preprocessing
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

# Load the dataset
dataset = pd.read_csv('data/spam.csv', encoding='ISO-8859-1', usecols=[0,1])
dataset.columns = ['label', 'message']
# Map 'spam' to 1 and 'ham' to 0
dataset['label'] = dataset['label'].map({'spam': 1, 'ham': 0})

# Define functions to count characters, uppercase letters, lowercase letters, digits, and specific keywords in a message
countChar = lambda l1,l2: sum([1 for x in l1 if x in l2])
countCapital = lambda l1: sum([1 for x in l1 if x.isupper()])
countLower = lambda l1: sum([1 for x in l1 if x.islower()])
countDigit = lambda l1: sum([1 for x in l1 if x.isdigit()])

# Apply preprocessing functions to the message column
dataset['message_cleaned'] = dataset['message'].apply(preprocess_text)

# Calculate additional features based on the message content
dataset['message_length'] = dataset['message'].apply(len)
dataset['count_punctuations'] = dataset['message'].apply(lambda s: countChar(s, string.punctuation))
dataset['count_uppercase'] = dataset['message'].apply(lambda s: countCapital(s))
dataset['count_lowercase'] = dataset['message'].apply(lambda s: countLower(s))
dataset['count_numerical_digits'] = dataset['message'].apply(lambda s: countDigit(s))
dataset['contains_discount'] = dataset['message'].str.contains('discount', case=False).astype(int)
dataset['contains_free'] = dataset['message'].str.contains('free', case=False).astype(int)
dataset['contains_please'] = dataset['message'].str.contains('please', case=False).astype(int)
dataset['contains_account'] = dataset['message'].str.contains('account', case=False).astype(int)
dataset['contains_customer'] = dataset['message'].str.contains('customer', case=False).astype(int)
dataset['contains_card'] = dataset['message'].str.contains('card', case=False).astype(int)
dataset['contains_email'] = dataset['message'].str.contains('email', case=False).astype(int)
dataset['contains_details'] = dataset['message'].str.contains('details', case=False).astype(int)
dataset['contains_update'] = dataset['message'].str.contains('update', case=False).astype(int)
dataset['contains_online'] = dataset['message'].str.contains('online', case=False).astype(int)
dataset['contains_bank'] = dataset['message'].str.contains('bank', case=False).astype(int)
dataset['contains_link'] = dataset['message'].str.contains('link', case=False).astype(int)
dataset['contains_refund'] = dataset['message'].str.contains('refund', case=False).astype(int)
dataset['contains_due'] = dataset['message'].str.contains('due', case=False).astype(int)

# Use TF-IDF vectorization to convert text data into numerical features
vectoriser = TfidfVectorizer(max_features=1000, analyzer='word', ngram_range=(1,1))
tfidf_features = vectoriser.fit_transform(dataset['message_cleaned']).toarray()
# Dump the vectorizer
pickle.dump(vectoriser, open('models/tfidf.pkl', 'wb'))
tfidf_df = pd.DataFrame(tfidf_features, columns=vectoriser.get_feature_names_out())
dataset = pd.concat([dataset, tfidf_df], axis=1)

dataset.drop(['message','message_cleaned'], axis=1, inplace=True)
X, y = dataset.drop('label', axis=1), dataset['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234,
    shuffle= True
)

# Define machine learning models

# LinearSVC
svc = LinearSVC()

# # Logistic Regression
lr = LogisticRegression()

# Random Forest
rf = RandomForestClassifier(n_estimators=500,
                            max_features=0.06,
                            n_jobs=6,
                            random_state=1234)

# Adaboost
base_estim = DecisionTreeClassifier(max_depth=1, max_features=0.06)
ab = AdaBoostClassifier(estimator=base_estim,
                        n_estimators=500,
                        learning_rate=0.5,
                        random_state=1234)

# Gradient Boosted Decision Trees
gbm = GradientBoostingClassifier(n_estimators=2000,
                                 subsample=0.67,
                                 max_features=0.06,
                                 validation_fraction=0.1,
                                 n_iter_no_change=15,
                                 verbose=0,
                                 random_state=1234)

#XGBoost
xgb = XGBClassifier(n_estimators=2000,
                    tree_method='hist',
                    subsample=0.67,
                    colsample_level=0.06,
                    verbose=0,
                    n_jobs=6,
                    random_state=1234)

#CatBoost
cb = CatBoostClassifier(n_estimators=2000,
                        colsample_bylevel=0.06,
                        max_leaves=31,
                        subsample=0.67,
                        thread_count=6,
                        verbose=0,
                        random_state=1234)

# Intialise results dataframe
df_results = pd.DataFrame(columns=['F1_score',
                                   'Precision',
                                   'Recall',
                                   'Accuracy',
                                   'AUC',
                                   'Training time'])

models = [svc, lr, rf, ab, xgb, cb]
model_names = [i.__class__.__name__ for i in models]

esp_models = ['XGBClassifier',
              'CatBoostClassifier']

# Train the models and evaluate their performance
start = timer()
for m, n in zip(models, model_names):
    if n in esp_models:
        start_time = time()
        m.fit(X_train,
              y_train,
              eval_set=[(X_test, y_test)],
              early_stopping_rounds=15,
              verbose=0)
    else:
        start_time = time()
        m.fit(X_train, y_train)

    run_time = time() - start_time
    accuracy = np.mean(m.predict(X_test) == y_test)
    precision = precision_score(y_test, m.predict(X_test))
    recall = recall_score(y_test, m.predict(X_test))
    p, r, t = precision_recall_curve(y_test, m.predict(X_test))
    auc_score = auc(r, p)
    f1 = f1_score(y_test, m.predict(X_test))
    df_results.loc[n] = [f1, precision, recall, accuracy, auc_score, run_time]

    del m

# Dump selected pre-trained model
pickle.dump(rf, open('models/model.pkl', 'wb'))

pd.set_option('display.max_columns', None)
print(df_results)

# GRAPHS
metrics = ['F1_score', 'Precision', 'Recall', 'Accuracy', 'AUC']
df_results[metrics].plot(kind='bar', figsize=(10, 6), ylim=(0.8, 1.0))
plt.title('Model Performance Metrics')
plt.xlabel('Models')
plt.ylabel('Scores')
plt.legend(loc='lower right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

df_results['Training time'].plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Training Time Comparison')
plt.xlabel('Models')
plt.ylabel('Training Time (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# PRECISION RECALL CURVE
best_model = RandomForestClassifier(n_estimators=500,
                                    max_features=0.06,
                                    n_jobs=6,
                                    random_state=1234)

best_model.fit(X_train, y_train)

y_scores = best_model.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label='RandomForestClassifier')
plt.title('Precision-Recall Curve: RandomForestClassifier')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.grid(True)
plt.tight_layout()
plt.show()