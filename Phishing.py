import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string

from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Load the dataset
data = pd.read_csv('spam.csv', encoding='ISO-8859-1')

# See the first few rows of the dataset
# print(data.head())

# The dataset has 5 columns?, but we are only interested in the first two. The first column is the label and the second column is the message.

# Drop the columns that are not required
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)

# Rename the columns to something better
data = data.rename(columns={"v1":"label", "v2":"message"})
print(data.head())

# Convert the labels to binary values
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Let's take a look at the first few rows
print(data.head())

# Checking the balance of the labels
print(data['label'].value_counts())

# There is a class imbalance in the dataset. More ham than spam messages.

# Check the average length of messages for both spam and ham
print(data.groupby('label').describe())

#Further data exploration. Explore relationships between the features.

#pairwise relationships in the dataset
sns.pairplot(data) 
plt.show()

#Identify patterns in the dataset
sns.countplot(data['label'], label = "Count")
plt.show()

# WordCloud for spam messages
spam_words = ' '.join(list(data[data['label'] == 1]['message']))
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

# WordCloud for ham messages
ham_words = ' '.join(list(data[data['label'] == 0]['message']))
ham_wc = WordCloud(width = 512,height = 512).generate(ham_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(ham_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

#Visualise data distribution
plt.figure(figsize=(10, 6))
data['message_length'] = data['message'].apply(len)
data['message_length'].plot(bins=50, kind='hist')
plt.xlabel('Message Length')
plt.show()

#message length distribution
data.hist(column='message_length', by='label', bins=50,figsize=(12,4))
plt.show()

#Feature Engineering

#Punctuation count

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['punct%'] = data['message'].apply(lambda x: count_punct(x))
print(data.head())

#Punctuation percentage
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['punct%'] = data['message'].apply(lambda x: count_punct(x))
print(data.head())

# Add a new feature 'message_length'
data['message_length'] = data['message'].apply(len)

#text data processing

#tokenization
data['tokenized_message'] = data['message'].apply(lambda x: x.split())

#stop words removal
stop_words = set(stopwords.words('english'))
data['stopwords_removed'] = data['tokenized_message'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

#stemming
ps = PorterStemmer()
data['stemmed'] = data['stopwords_removed'].apply(lambda x: [ps.stem(word) for word in x])

#lemmatization
wn = WordNetLemmatizer()
data['lemmatized'] = data['stopwords_removed'].apply(lambda x: [wn.lemmatize(word) for word in x])

#tf-idf vectorization with n-grams
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word', stop_words= 'english',ngram_range=(1,1))
vect = tfidf.fit_transform(data['message'])

# Display the modified dataset
print(data.head())

# Message length
data['message_length'] = data['message'].apply(len)

# Additional Features

# Presence of punctuations
data['has_punctuations'] = data['message'].apply(lambda x: any(char in string.punctuation for char in x))

# Presence of uppercase words
data['has_uppercase'] = data['message'].apply(lambda x: any(char.isupper() for char in x))

# Presence of lowercase words
data['has_lowercase'] = data['message'].apply(lambda x: any(char.islower() for char in x))

# Presence of title case words
data['has_titlecase'] = data['message'].apply(lambda x: any(char.istitle() for char in x))

# Presence of stop words
stop_words = set(STOPWORDS)
data['has_stopwords'] = data['message'].apply(lambda x: any(word in stop_words for word in x.split()))

# Presence of special characters
data['has_special_characters'] = data['message'].apply(lambda x: any(char in ['$', '#', '@'] for char in x))

# Presence of numerical digits
data['has_numerical_digits'] = data['message'].apply(lambda x: any(char.isdigit() for char in x))

# Presence of specific keywords
data['contains_discount'] = data['message'].str.contains('discount', case=False).astype(int)
data['contains_free'] = data['message'].str.contains('free', case=False).astype(int)

# Extract sender domain from email address
data['sender_domain'] = data['message'].str.extract(r'@([^\s]+)')
data['sender_domain'] = data['sender_domain'].fillna('unknown')

# Extract sender name from email address
data['sender_name'] = data['message'].str.extract(r'([a-zA-Z0-9._%+-]+)@')
data['sender_name'] = data['sender_name'].fillna('unknown')

# Extract the subject from the message
data['subject'] = data['message'].str.extract(r'Subject: ([^\n]+)')
data['subject'] = data['subject'].fillna('unknown')

# Perform PCA to reduce dimensionality
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(vect.toarray())
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Concatenate PCA features with existing data
data = pd.concat([data, principalDf], axis=1)

# Display the modified dataset
print(data.head())

# Dimensionality Reduction 
# Apply TF-IDF to reduce dimensionality

tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_features = tfidf_vectorizer.fit_transform(data['message'])

# Perform PCA to reduce dimensionality
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(vect.toarray())
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])

# Concatenate PCA features with existing data
data = pd.concat([data, principalDf], axis=1)

# Convert sparse matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Concatenate TF-IDF features with existing data
data = pd.concat([data, tfidf_df], axis=1)

# Display the modified dataset
print(data.head())


# Drop non-numeric columns
data.drop(['message', 'tokenized_message', 'stopwords_removed', 'stemmed', 'lemmatized'], axis=1, inplace=True)

# One-hot encode categorical columns (sender_domain, sender_name, subject)
data = pd.get_dummies(data, columns=['sender_domain', 'sender_name', 'subject'], drop_first=True)

# Split the data into features (X) and target variable (y)
X = data.drop('label', axis=1)
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Convert predictions to binary values
lr_predictions = [1 if x > 0.5 else 0 for x in lr_predictions]

# Evaluate Linear Regression
lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)

print("Linear Regression:")
print("Accuracy:", lr_accuracy)
print("Precision:", lr_precision)

# Train and test SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

# Evaluate SVM
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions)

print("\nSupport Vector Machine:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
rf_predictions = rf_classifier.predict(X_test)

# Evaluate Random Forest Classifier
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print("\nRandom Forest Classifier Performance:")
print("Accuracy:", rf_accuracy)
print("Precision:", rf_precision)
print("Recall:", rf_recall)
print("F1 Score:", rf_f1)

# Train and test Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict(X_test)

# Evaluate Gradient Boosting Classifier
gb_accuracy = accuracy_score(y_test, gb_predictions)
gb_precision = precision_score(y_test, gb_predictions)

print("\nGradient Boosting Classifier:")
print("Accuracy:", gb_accuracy)
print("Precision:", gb_precision)