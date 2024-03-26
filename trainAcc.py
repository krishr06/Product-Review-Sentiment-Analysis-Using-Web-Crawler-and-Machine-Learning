import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('bigds.csv')

reviews = data['review'].astype(str)
sentiments = data['sentiment'].astype(int)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(reviews, sentiments, 
                                                    train_size=450000, 
                                                    test_size=data.shape[0] - 450000, 
                                                    random_state=42)

print(f"Number of rows used for training: {len(X_train)}/{len(data)}")

# Vectorization
vectorizer = TfidfVectorizer(max_features=10000)  
X_train_vectorized = vectorizer.fit_transform(X_train)

# Model Training
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_vectorized, y_train)

# Evaluate accuracy on test set
X_test_vectorized = vectorizer.transform(X_test)
predictions = rf_classifier.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy on the test set: {accuracy}")

# Save trained model and vectorizer
model_filename = 'random_forest_model.pkl'
joblib.dump(rf_classifier, model_filename)
print(f"Trained model saved to {model_filename}")

vectorizer_filename = 'tfidf_vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_filename)
print(f"Vectorizer saved to {vectorizer_filename}")
