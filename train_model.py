import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

def train():
    os.makedirs("models", exist_ok=True)
    train_data = pd.read_csv("data/imdb_train.csv")
    test_data = pd.read_csv("data/imdb_test.csv")

    # Train model
    pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())
    pipeline.fit(train_data["text"], train_data["label"])

    # Evaluate model
    predictions = pipeline.predict(test_data["text"])
    accuracy = accuracy_score(test_data["label"], predictions)
    print(f"Model Accuracy: {accuracy}")

    # Save model
    joblib.dump(pipeline, "models/sentiment_model.pkl")

if __name__ == "__main__":
    train()
