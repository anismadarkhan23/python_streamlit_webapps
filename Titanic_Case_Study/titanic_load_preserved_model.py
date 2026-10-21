import pandas as pd
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def load_model(file_name):
    model = joblib.load(file_name)

    print("Model loaded successfully")

    print(model.feature_names_in_)

    return model

def predict_passenger(model):
    print("Enter the information")

    pclass = int(input("Enter Pclass (1/2/3): "))
    sex = int(input("Enter Sex (0 - M / 1 - F): "))
    age = float(input("Enter Age: "))
    sibsp = int(input("Enter sibsp: "))
    parch = int(input("Enter Parch: "))
    fare = int(input("Enter Fare: "))
    embarked = int(input("Enter Embarked (0/1/2): "))

    passanger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "sibsp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked_1.0": 1 if embarked == 1 else 0,
        "Embarked_2.0": 1 if embarked == 2 else 0
    }])

    passanger = passanger[model.feature_names_in_]

    result = model.predict(passanger)[0]

    if result == 1:
        print("The passenger will survive")
    else:
        print("The passenger won't survive")

def main():
    model = load_model("titanic_cs.pkl")

    predict_passenger(model)

if __name__ == "__main__":
    main()