import os, sys
import pandas as pd
import numpy as np
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------------------------------------------------
# Step 1: Load Data
# ------------------------------------------------------------
#   Function Name : load_data
#   Description : Load the from csv
#   Input : Name of csv file
#   Output : Data frame
# ------------------------------------------------------------
def load_data(file_name):
    df = pd.read_csv(file_name)

    print("Dataset loaded successfully")
    print(df.head())

    return df

# ------------------------------------------------------------
# Step 2: Data Pre-processing
# ------------------------------------------------------------
#   Function Name : preprocess_data
#   Description : It perform data analysis
#   Input : Data frame
#   Output : Updated Data frame
# ------------------------------------------------------------
def preprocess_data(df):
    df = df.drop(columns = [
        "Passengerid", 
        "zero",
        "name"
    ],
    errors = "ignore"
    )

    # Handle missing value
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Convert catrgorical to numeric data
    df = pd.get_dummies(
        df,
        columns = ["Embarked"],
        drop_first = True,
        dtype = int
    )

    print(df.head())

    print("Data preprocessing completed")

    return df

# ------------------------------------------------------------
# Step 3: Split Dataset
# ------------------------------------------------------------
#   Function Name : split_data
#   Description : It perform spliting data from training & testing
#   Input : Data frame
#   Output : Four subsets for training & testing
# ------------------------------------------------------------
def split_data(df):
    X = df.drop("Survived", axis = 1)
    Y = df["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size = 0.2,
        random_state = 42  
    )

    print("Dataset Splitting Completed Successfully")

    return X_train, X_test, Y_train, Y_test

# ------------------------------------------------------------
# Step 4: Train model
# ------------------------------------------------------------
#   Function Name : train_model
#   Description : It perform model training
#   Input : Training features & labels
#   Output : Trained model
# ------------------------------------------------------------
def train_model(X_train, Y_train):
    model = LogisticRegression(max_iter = 2000)

    model = model.fit(X_train, Y_train)

    print("Model trained successfully")

    return model

# ------------------------------------------------------------
# Step 5: Evaluate model
# ------------------------------------------------------------
#   Function Name : evaluate_model
#   Description : It perform model testing
#   Input : Model, testing data (features, labels)
#   Output : None
# ------------------------------------------------------------
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)

    print(f"Accuracy is: {accuracy * 100:.2f}")

    print(confusion_matrix(Y_test, Y_pred))

# ------------------------------------------------------------
# Step 6: Preserve model
# ------------------------------------------------------------
#   Function Name : preserve_model
#   Description : It perform model preservation into .pkl file
#   Input : Model
#   Output : None
# ------------------------------------------------------------
def preserve_model(model, file_name):
    joblib.dump(model, file_name)

    print(f"Model preserved with name {file_name}")

# ------------------------------------------------------------
#   Function Name : main
#   Description : Entry point function
#   Input : None
#   Output : None
# ------------------------------------------------------------
def main():
    if not os.path.exists("TitanicDataset.csv"):
        print("'TitanicDataset.csv' no such file or directory found")
        sys.exit(1)

    # Step 1
    df = load_data("TitanicDataset.csv")

    # Step 2
    df = preprocess_data(df)

    # Step 3
    X_train, X_test, Y_train, Y_test = split_data(df)

    # Step 4
    model = train_model(X_train, Y_train)

    # Step 5
    evaluate_model(model, X_test, Y_test)

    # Step 6
    preserve_model(model, "titanic_cs.pkl")

if __name__ == "__main__":
    main()