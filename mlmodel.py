import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

loaded_model = joblib.load('random_forest_model.pkl')

vectorizer = joblib.load('tfidf_vectorizer.pkl')

review = input("Enter a review: ")

review_vectorized = vectorizer.transform([review])

prediction = loaded_model.predict(review_vectorized)

sentiment = 'Negative' if prediction == 1 else 'Positive'

print(f"Review: {review}")
print(f"Predicted Sentiment: {sentiment}")
