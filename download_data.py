from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os

def download_data(train_ratio=0.8):
    os.makedirs("data", exist_ok=True)
    dataset = load_dataset("imdb")
    # Convert to DataFrame
    df = pd.DataFrame(dataset["train"])
    train_data, test_data = train_test_split(df, train_size=train_ratio, random_state=42)

    # Save train and test data
    train_data.to_csv("data/imdb_train.csv", index=False)
    test_data.to_csv("data/imdb_test.csv", index=False)

    print(f"Data split with {train_ratio*100}% for training and {(1-train_ratio)*100}% for testing.")

if __name__ == "__main__":
    download_data(train_ratio=0.8) 
