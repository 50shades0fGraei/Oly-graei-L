# Oly-graei-L
Generating integrations

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load data
df = pd.read_csv('your_data.csv')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Define hyperparameter tuning space for traditional models
param_grid_nb = {'alpha': [0.1, 0.5, 1.0]}
param_grid_svm = {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf']}
param_grid_rf = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}

# Perform hyperparameter tuning for traditional models
grid_nb = GridSearchCV(MultinomialNB(), param_grid_nb, cv=5)
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)

grid_nb.fit(X_train_vectorized, y_train)
grid_svm.fit(X_train_vectorized, y_train)
grid_rf.fit(X_train_vectorized, y_train)

# Evaluate traditional models
y_pred_nb = grid_nb.best_estimator_.predict(X_test_vectorized)
y_pred_svm = grid_svm.best_estimator_.predict(X_test_vectorized)
y_pred_rf = grid_rf.best_estimator_.predict(X_test_vectorized)

print("Naive Bayes - Accuracy:", accuracy_score(y_test, y_pred_nb))
print("SVM - Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Random Forest - Accuracy:", accuracy_score(y_test, y_pred_rf))

# Prepare data for CNN and LSTM
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=1000)

# Split padded sequences into training and testing sets
X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_pad, df['label'], test_size=0.2, random_state=42)

# One-hot encode labels
y_train_oh = to_categorical(y_train_seq)
y_test_oh = to_categorical(y_test_seq)

# Define CNN model
def create_cnn_model(input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=128, input_length=1000))
    model.add(Conv1D(64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size)
