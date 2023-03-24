import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.svm import SVC

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load datasets
train_data = pd.read_csv('data/archive/atis_intents_train.csv', names=['intent', 'text'])
test_data = pd.read_csv('data/archive/atis_intents_test.csv', names=['intent', 'text'])
intent_mapping = pd.read_csv("data/archive/atis_intents.csv", names=["intent"])

# Map intent labels to indices
intent_to_index = {row["intent"]: index for index, row in intent_mapping.iterrows()}
print(type(intent_to_index))
print(list(intent_to_index.values())[10])
print(list(intent_to_index.keys())[10])
print(list(intent_to_index.values())[12])
print(list(intent_to_index.keys())[12])
print(list(intent_to_index.values())[15])
print(list(intent_to_index.keys())[15])

print(train_data)
#train_data["intent"] = train_data["intent"].map(intent_to_index)

print(train_data)
#test_data["intent"] = test_data["intent"].map(intent_to_index)

# Drop rows with NaN values in 'intent'
train_data = train_data.dropna(subset=['intent'])
test_data = test_data.dropna(subset=['intent'])

def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove punctuations
    text = re.sub(r'[^\w\s]', ' ', text)

    # Tokenize
    words = nltk.word_tokenize(text)

    return ' '.join(words)



train_data['cleaned_text'] = train_data['text'].apply(preprocess_text)
test_data['cleaned_text'] = test_data['text'].apply(preprocess_text)

X_train = train_data['text']
X_test = test_data['text']
y_train = train_data['intent']
y_test = test_data['intent']

# Feature extraction
vectorizer = CountVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
lr_classifier = LogisticRegression()
lr_classifier.fit(X_train_tfidf, y_train)

# Train Multinomial Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Train SVM
svm_classifier = SVC(kernel='linear', C=1.0, random_state=0)
svm_classifier.fit(X_train_tfidf, y_train)

# Predict on test data
lr_predictions = lr_classifier.predict(X_test_tfidf)
nb_predictions = nb_classifier.predict(X_test_tfidf)
svm_predictions = svm_classifier.predict(X_test_tfidf)


# Evaluate performance
lr_accuracy = accuracy_score(y_test, lr_predictions)
nb_accuracy = accuracy_score(y_test, nb_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)


lr_precision = precision_score(y_test, lr_predictions, average='weighted')
nb_precision = precision_score(y_test, nb_predictions, average='weighted')
svm_precision = precision_score(y_test, svm_predictions, average='weighted')



lr_recall = recall_score(y_test, lr_predictions, average='weighted')
nb_recall = recall_score(y_test, nb_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')


lr_f1 = f1_score(y_test, lr_predictions, average='weighted')
nb_f1 = f1_score(y_test, nb_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

print("Logistic Regression Performance:")
print(f"Accuracy: {lr_accuracy:.4f}, Precision: {lr_precision:.4f}, Recall: {lr_recall:.4f}, F1-score: {lr_f1:.4f}")

print("\nMultinomial Naive Bayes Performance:")
print(f"Accuracy: {nb_accuracy:.4f}, Precision: {nb_precision:.4f}, Recall: {nb_recall:.4f}, F1-score: {nb_f1:.4f}")

print("\nSVM Performance:")
print(f"Accuracy: {svm_accuracy:.4f}, Precision: {svm_precision:.4f}, Recall: {svm_recall:.4f}, F1-score: {svm_f1:.4f}")

# Print classification report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))

print("\nMultinomial Naive Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))