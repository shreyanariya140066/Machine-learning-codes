import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('D:\Hardik\Machine-learning-programs-main\Machine-learning-programs-main\Datasets\knn_dataset.csv')

print("Dataset Preview:")
print(df.head())

le = LabelEncoder()

df['CGPA'] = le.fit_transform(df['CGPA'])
df['Communication'] = le.fit_transform(df['Communication'])
df.head()

df['Aptitude'] = le.fit_transform(df['Aptitude'])
df['Pro_skill'] = le.fit_transform(df['Pro_skill'])
df.head()

df['Job_offered'] = le.fit_transform(df['Job_offered'])
     
print("\nEncoded Dataset:")
print(df.head())

#All Columns Except Last
X=df.iloc[:,:-1]

#LAST Column
Y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

print("\nTraining samples:", len(X_train))

max_k = len(X_train)

for k in range(1, max_k + 1):

    print(f"K = {k}")

    classifier = KNeighborsClassifier(n_neighbors=k)

    # Train model
    classifier.fit(X_train, y_train)

    # Prediction
    y_pred = classifier.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Accuracy
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"Accuracy = {acc:.2f}%")
