import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. load and clean dataset

df = pd.read_csv(r"D:\Hardik\auto-mpg.csv")

# replace '?' with NAN
df.replace("?", np.nan, inplace=True)

# convert horsepower to numeric
df["horsepower"] = pd.to_numeric(df["horsepower"])

# drop missing values
df = df.dropna()

# drop non-numeric column (car name)
if "car name" in df.columns:
    df = df.drop(columns=["car name"])

#2. Define Features and Target
X = df.drop(columns=["mpg"])
Y = df["mpg"]

# train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#3. Backward feature selection

selected_features = list(X.columns)

#Train full model first
model = LinearRegression()
model.fit(X_train[selected_features], Y_train)
y_pred = model.predict(X_test[selected_features])
best_score = r2_score(Y_test, y_pred)

print("Backward feature elimination process:\n")
print(f'Inintial R2 score (All features): {best_score:4f}\n')

while len(selected_features) > 1:

    scores = []

    for feature in selected_features:
        features_to_test = selected_features.copy()
        features_to_test.remove(feature)

        model = LinearRegression()
        model.fit(X_train[features_to_test], Y_train)

        y_pred = model.predict(X_test[features_to_test])
        score = r2_score(Y_test, y_pred)

        scores.append((score, feature))

    #Find removal that gives highest R2
    scores.sort(reverse=True)
    current_best_score, worst_feature = scores[0]

    if current_best_score >= best_score:
        best_score = current_best_score
        selected_features.remove(worst_feature)

        print(f"Removed: {worst_feature}, R2 score: {best_score:.4f}")
    else:
        break

print("\n Final selected Features: ")
print(selected_features)
